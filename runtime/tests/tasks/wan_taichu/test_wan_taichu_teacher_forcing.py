"""Wan Taichu block-causal teacher-forcing task contracts."""

from __future__ import annotations

from copy import deepcopy
from functools import partial

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from vforge.tasks.requests import TrainRequest
from vforge.tasks.wan_taichu.batch import build_encoded_batch_codec, stream_field
from vforge.tasks.wan_taichu.conditioning import ConditionDropoutPolicy
from vforge.tasks.wan_taichu.train import (
    _build_condition_processor,
    _resolve_request,
    _train_step,
)
from vforge.tasks.wan_taichu.variants import require_variant
from vforge.training.recipe import StepContext

PATCH_SIZE = (1, 2, 2)
LATENT_FRAMES = 3
LATENT_HEIGHT = 4
LATENT_WIDTH = 4
TOKENS_PER_HALF = LATENT_FRAMES * 4


@pytest.fixture
def cuda_ops_runtime():
    from vforge.distributed.parallel_state import ParallelConfig
    from vforge.distributed.runtime import destroy_runtime, initialize_runtime
    from vforge.ops.config import OpsConfig

    device = torch.device("cuda")
    initialize_runtime(
        ParallelConfig(),
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=device,
    )
    try:
        yield device
    finally:
        destroy_runtime()


def _encoded_batch(z_dim: int = 2) -> dict[str, torch.Tensor | list[str]]:
    shape = (1, z_dim, LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    return {
        "sample_id": ["sample"],
        "prompt": torch.ones(1, 3, 4),
        "frame_latents": torch.arange(
            int(torch.tensor(shape).prod()),
            dtype=torch.float32,
        ).reshape(shape),
        stream_field("normal"): torch.full(shape, 0.25),
        "first_frame_latents": torch.full(
            (1, z_dim, 1, LATENT_HEIGHT, LATENT_WIDTH),
            -7.0,
        ),
    }


def _processor(variant: str):
    spec, streams, extras = _resolve_request(
        TrainRequest(
            extras={
                "variant": variant,
                "noise_shift": 1.0,
                "patch_size": PATCH_SIZE,
            },
            precision="fp32",
        )
    )
    return _build_condition_processor(
        spec=spec,
        streams=streams,
        extras=extras,
        precision="fp32",
    )


def _process(processor, *, seed: int = 1234):
    return processor.process(
        _encoded_batch(),
        step=0,
        device=torch.device("cpu"),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        apply_dropouts=False,
        patch_size=PATCH_SIZE,
    )


def test_causal_teacher_doubles_clean_and_noisy_inputs() -> None:
    processed = _process(_processor("wan22_ti2v_5b_causal_teacher"))
    hidden = processed["model_inputs"]["hidden_states"]
    timestep = processed["model_inputs"]["timestep"]
    encoded = _encoded_batch()

    assert hidden.shape == (1, 4, 2 * LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    clean, noisy = hidden[:, :, :LATENT_FRAMES], hidden[:, :, LATENT_FRAMES:]
    torch.testing.assert_close(clean[:, :2, 0:1], encoded["first_frame_latents"])
    torch.testing.assert_close(noisy[:, :2, 0:1], encoded["first_frame_latents"])
    torch.testing.assert_close(clean[:, 2:], noisy[:, 2:])

    assert timestep.shape == (1, 2 * TOKENS_PER_HALF)
    assert torch.all(timestep[:, :TOKENS_PER_HALF] == 0)
    assert torch.all(timestep[:, TOKENS_PER_HALF : TOKENS_PER_HALF + 4] == 0)
    assert torch.all(timestep[:, TOKENS_PER_HALF + 4 :] > 0)


def test_causal_teacher_reuses_bidirectional_noise_target_and_loss() -> None:
    causal = _process(_processor("wan22_ti2v_5b_causal_teacher"))
    ordinary = _process(_processor("wan22_ti2v_5b"))

    torch.testing.assert_close(causal["targets"], ordinary["targets"], rtol=0, atol=0)
    torch.testing.assert_close(
        causal["loss_weights"],
        ordinary["loss_weights"],
        rtol=0,
        atol=0,
    )
    noisy_half = causal["model_inputs"]["hidden_states"][:, :, LATENT_FRAMES:]
    torch.testing.assert_close(
        noisy_half,
        ordinary["model_inputs"]["hidden_states"],
        rtol=0,
        atol=0,
    )


def test_a14b_causal_teacher_doubles_without_a_first_frame_exception() -> None:
    encoded = _encoded_batch()
    shape = encoded["frame_latents"].shape
    encoded["condition_latents_normal"] = torch.full(
        (shape[0], 6, shape[2], shape[3], shape[4]),
        0.25,
    )
    processor = _processor("wan22_a14b_causal_teacher")
    processed = processor.process(
        encoded,
        step=0,
        device=torch.device("cpu"),
        generator=torch.Generator(device="cpu").manual_seed(1234),
        apply_dropouts=False,
        patch_size=PATCH_SIZE,
    )

    hidden = processed["model_inputs"]["hidden_states"]
    timestep = processed["model_inputs"]["timestep"]
    clean, noisy = hidden[:, :, :LATENT_FRAMES], hidden[:, :, LATENT_FRAMES:]
    assert hidden.shape == (1, 8, 2 * LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    torch.testing.assert_close(clean[:, :2], encoded["frame_latents"])
    torch.testing.assert_close(clean[:, 2:], noisy[:, 2:])
    assert torch.all(timestep[:, :TOKENS_PER_HALF] == 0)
    assert torch.all(timestep[:, TOKENS_PER_HALF:] > 0)


@pytest.mark.gpu(gpus=1)
def test_causal_teacher_runs_through_the_wan_taichu_train_step(
    cuda_ops_runtime,
) -> None:
    from vforge.distributed.parallel_state import get_parallel_state

    device = cuda_ops_runtime
    spec = require_variant("wan22_ti2v_5b_causal_teacher")
    request = TrainRequest(
        extras={
            "variant": spec.name,
            "condition_streams": ["normal"],
            "noise_shift": 1.0,
            "patch_size": PATCH_SIZE,
        },
        precision="fp32",
    )
    _, streams, extras = _resolve_request(request)
    model = _causal_teacher_model(
        patch_size=PATCH_SIZE,
        num_attention_heads=2,
        attention_head_dim=12,
        in_channels=96,
        out_channels=48,
        text_dim=4,
        freq_dim=16,
        ffn_dim=16,
        num_layers=1,
        rope_max_seq_len=32,
    ).to(device)
    processor = _build_condition_processor(
        spec=spec,
        streams=streams,
        extras=extras,
        precision=request.precision,
    )
    output = _train_step(
        model,
        _encoded_batch(z_dim=48),
        0,
        StepContext(device=device, parallel_state=get_parallel_state()),
        request=request,
        spec=spec,
        streams=streams,
        codec=build_encoded_batch_codec(spec, streams),
        processor=processor,
        dropout=ConditionDropoutPolicy(spec=spec, streams=streams, extras=extras),
        encoder=None,
    )
    output.loss.backward()

    assert torch.isfinite(output.loss)
    assert output.workload["local_tokens"] == TOKENS_PER_HALF
    assert any(parameter.grad is not None for parameter in model.parameters())


def _causal_teacher_model(**config):
    from vforge.models.wan_taichu.model import (
        WanTaichuModel,
        WanTaichuTransformer3DModel,
    )

    return WanTaichuModel(
        variant="wan22_ti2v_5b_causal_teacher",
        architecture="wan22_ti2v_5b",
        attention="block_causal_teacher",
        transformers={
            "": WanTaichuTransformer3DModel(
                architecture="wan22_ti2v_5b",
                attention="block_causal_teacher",
                **config,
            )
        },
    )


def _tiny_causal_teacher(device: torch.device):
    return _causal_teacher_model(
        patch_size=PATCH_SIZE,
        num_attention_heads=2,
        attention_head_dim=12,
        in_channels=4,
        out_channels=2,
        text_dim=4,
        freq_dim=16,
        ffn_dim=16,
        num_layers=1,
        rope_max_seq_len=32,
        frame_per_block=1,
    ).to(device=device, dtype=torch.bfloat16)


def _backward_snapshot(model, hidden, timestep, prompt):
    output = model(
        hidden_states=hidden,
        timestep=timestep,
        encoder_hidden_states=prompt,
    )
    loss = output.float().square().mean()
    loss.backward()
    gradients = {
        name: parameter.grad.detach().float().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output.detach(), loss.detach(), gradients


@pytest.mark.gpu(gpus=1)
@pytest.mark.parametrize("use_reentrant", (True, False))
def test_bf16_checkpoint_backward_keeps_per_forward_mask(
    cuda_ops_runtime,
    use_reentrant: bool,
) -> None:
    """CUDA gate for output/loss/grad parity and stale-mask isolation."""
    device = cuda_ops_runtime
    torch.manual_seed(20260831)
    direct = _tiny_causal_teacher(device)
    replayed = deepcopy(direct)
    replayed.gradient_checkpointing = True
    replayed._gradient_checkpointing_func = partial(
        checkpoint,
        use_reentrant=use_reentrant,
    )

    hidden = torch.randn(
        1,
        4,
        2 * LATENT_FRAMES,
        LATENT_HEIGHT,
        LATENT_WIDTH,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    timestep = torch.full(
        (1, 2 * TOKENS_PER_HALF),
        500.0,
        device=device,
    )
    timestep[:, :TOKENS_PER_HALF] = 0.0
    prompt = torch.randn(1, 5, 4, device=device, dtype=torch.bfloat16)
    expected = _backward_snapshot(
        direct,
        hidden.detach().clone().requires_grad_(True),
        timestep,
        prompt,
    )

    long_output = replayed(
        hidden_states=hidden,
        timestep=timestep,
        encoder_hidden_states=prompt,
    )
    short_frames = 2
    short_tokens = short_frames * 4
    replayed(
        hidden_states=torch.randn(
            1,
            4,
            2 * short_frames,
            LATENT_HEIGHT,
            LATENT_WIDTH,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        ),
        timestep=torch.cat(
            [
                torch.zeros(1, short_tokens, device=device),
                torch.full((1, short_tokens), 400.0, device=device),
            ],
            dim=1,
        ),
        encoder_hidden_states=prompt,
    )
    long_loss = long_output.float().square().mean()
    long_loss.backward()
    actual_gradients = {
        name: parameter.grad.detach().float().clone()
        for name, parameter in replayed.named_parameters()
        if parameter.grad is not None
    }

    torch.testing.assert_close(long_output.detach(), expected[0], rtol=0, atol=0)
    torch.testing.assert_close(long_loss.detach(), expected[1], rtol=0, atol=0)
    assert actual_gradients.keys() == expected[2].keys()
    for name in expected[2]:
        torch.testing.assert_close(
            actual_gradients[name],
            expected[2][name],
            rtol=2e-2,
            atol=2e-3,
        )
