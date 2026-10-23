"""Contracts for the Wan22 TI2V teacher-forcing recipe.

What separates teacher forcing from the bidirectional plugin it subclasses: a
doubled ``[clean | noisy]`` input, a timestep that is zero across the clean
half, and a block-causal mask that must actually stop a noisy token from
reading its own answer or the future. Everything else -- noising, targets, loss
weights, cache format -- is asserted to be *unchanged*, because that shared
path is what makes the two recipes comparable.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

Z_DIM = 2
LATENT_FRAMES = 3
LATENT_H = 4
LATENT_W = 4
PATCH_SIZE = (1, 2, 2)
TOKENS_PER_FRAME = (LATENT_H // PATCH_SIZE[1]) * (LATENT_W // PATCH_SIZE[2])
TOKENS_PER_HALF = LATENT_FRAMES * TOKENS_PER_FRAME


@pytest.fixture
def cpu_ops_runtime():
    """vforge attention refuses to dispatch until a runtime exists."""
    from vforge.distributed.parallel_state import ParallelConfig
    from vforge.distributed.runtime import destroy_runtime, initialize_runtime
    from vforge.ops.config import OpsConfig

    initialize_runtime(
        ParallelConfig(),
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=torch.device("cpu"),
    )
    try:
        yield
    finally:
        destroy_runtime()


def _encoded_batch() -> dict[str, torch.Tensor]:
    latent_shape = (1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)
    return {
        "prompt": torch.ones(1, 3, 4),
        "frame_latents": torch.arange(
            int(torch.tensor(latent_shape).prod()),
            dtype=torch.float32,
        ).reshape(latent_shape),
        "condition_latents_normal": torch.full(latent_shape, 0.25),
        "first_frame_latents": torch.full((1, Z_DIM, 1, LATENT_H, LATENT_W), -7.0),
    }


def _extras(**overrides) -> dict:
    return {
        "noise_shift": 1.0,
        "text_drop_prob": 0.0,
        "normal_drop_prob": 0.0,
        "patch_size": PATCH_SIZE,
        **overrides,
    }


def _tf_processor(**overrides):
    from vforge.tasks.wan.v22.ti2v_5b.teacher_forcing.train import _TfConditionProcessor

    return _TfConditionProcessor(extras=_extras(**overrides), precision="fp32")


def _bidirectional_processor(**overrides):
    from vforge.tasks.wan.v22.ti2v_5b.ct.train import _ConditionProcessor

    return _ConditionProcessor(extras=_extras(**overrides), precision="fp32")


def _process(processor, *, seed: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return processor.process(
        _encoded_batch(),
        step=0,
        device=torch.device("cpu"),
        generator=generator,
        apply_dropouts=False,
        patch_size=PATCH_SIZE,
    )


def _tiny_tf_transformer(**overrides):
    from vforge.models.wan.v22.ti2v_5b.teacher_forcing.transformer import (
        WanTeacherForcingTransformer3DModel,
    )

    config = {
        "patch_size": PATCH_SIZE,
        "num_attention_heads": 2,
        "attention_head_dim": 12,
        "in_channels": 2 * Z_DIM,
        "out_channels": Z_DIM,
        "text_dim": 4,
        "freq_dim": 16,
        "ffn_dim": 16,
        "num_layers": 1,
        "rope_max_seq_len": 32,
    }
    config.update(overrides)
    return WanTeacherForcingTransformer3DModel(**config)


def _forward_loss_and_parameter_gradients(model, hidden, timestep, prompt):
    output = model(
        hidden_states=hidden,
        timestep=timestep,
        encoder_hidden_states=prompt,
    )
    loss = output.float().square().mean()
    loss.backward()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output.detach(), loss.detach(), gradients


# --------------------------------------------------------------------------
# Doubled input
# --------------------------------------------------------------------------
def test_doubled_input_contract() -> None:
    processed = _process(_tf_processor())
    hidden = processed["model_inputs"]["hidden_states"]
    batch = _encoded_batch()

    assert hidden.shape == (1, 2 * Z_DIM, 2 * LATENT_FRAMES, LATENT_H, LATENT_W)
    clean, noisy = hidden[:, :, :LATENT_FRAMES], hidden[:, :, LATENT_FRAMES:]
    torch.testing.assert_close(clean[:, :Z_DIM, 1:], batch["frame_latents"][:, :, 1:])
    torch.testing.assert_close(clean[:, :Z_DIM, 0:1], batch["first_frame_latents"])
    torch.testing.assert_close(noisy[:, :Z_DIM, 0:1], batch["first_frame_latents"])
    torch.testing.assert_close(clean[:, Z_DIM:], noisy[:, Z_DIM:])
    assert not torch.allclose(noisy[:, :Z_DIM, 1:], batch["frame_latents"][:, :, 1:])

    timestep = processed["model_inputs"]["timestep"]
    assert timestep.shape == (1, 2 * TOKENS_PER_HALF)
    clean_timestep, noisy_timestep = (
        timestep[0, :TOKENS_PER_HALF],
        timestep[0, TOKENS_PER_HALF:],
    )
    assert torch.all(clean_timestep == 0)
    assert torch.all(noisy_timestep[:TOKENS_PER_FRAME] == 0)
    assert torch.all(noisy_timestep[TOKENS_PER_FRAME:] > 0)
    assert noisy_timestep[TOKENS_PER_FRAME:].unique().numel() == 1

    assert processed["targets"].shape == (1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)
    assert processed["loss_weights"].shape == processed["targets"].shape
    assert torch.all(processed["loss_weights"][:, :, 0] == 0.0)


def test_noising_is_bit_identical_to_the_bidirectional_recipe() -> None:
    """Both recipes must draw the same noise from the same seed.

    This is what makes a teacher-forcing run comparable to a bidirectional one
    step for step; a drift here would silently invalidate every A/B.
    """
    tf = _process(_tf_processor(), seed=1234)
    bidirectional = _process(_bidirectional_processor(), seed=1234)

    torch.testing.assert_close(tf["targets"], bidirectional["targets"], rtol=0, atol=0)
    torch.testing.assert_close(tf["loss_weights"], bidirectional["loss_weights"], rtol=0, atol=0)
    tf_noisy = tf["model_inputs"]["hidden_states"][:, :, LATENT_FRAMES:]
    torch.testing.assert_close(
        tf_noisy,
        bidirectional["model_inputs"]["hidden_states"],
        rtol=0,
        atol=0,
    )


def test_per_token_timestep_reuses_the_bidirectional_layout_for_its_noisy_half() -> None:
    from vforge.models.wan.v22.ti2v_5b.ct.inference import build_per_token_timestep
    from vforge.models.wan.v22.ti2v_5b.teacher_forcing.inference import build_tf_per_token_timestep

    latent = torch.zeros(1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)
    scalar = torch.tensor(500.0)

    doubled = build_tf_per_token_timestep(latent, scalar, PATCH_SIZE)
    single = build_per_token_timestep(latent, scalar, PATCH_SIZE)
    torch.testing.assert_close(doubled[:, TOKENS_PER_HALF:], single, rtol=0, atol=0)


# --------------------------------------------------------------------------
# Transformer
# --------------------------------------------------------------------------
def test_transformer_returns_one_half_of_the_frames_it_was_given(cpu_ops_runtime) -> None:
    torch.manual_seed(0)
    model = _tiny_tf_transformer().eval()
    hidden = torch.randn(1, 2 * Z_DIM, 2 * LATENT_FRAMES, LATENT_H, LATENT_W)

    output = model(
        hidden_states=hidden,
        timestep=torch.full((1, 2 * TOKENS_PER_HALF), 500.0),
        encoder_hidden_states=torch.randn(1, 5, 4),
    )
    assert output.shape == (1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)


@pytest.mark.parametrize(
    ("frames", "timestep", "attention_mask", "message"),
    [
        (3, torch.full((1, 2 * TOKENS_PER_HALF), 500.0), None, "doubled"),
        (2 * LATENT_FRAMES, torch.tensor([500.0]), None, "per-token timestep"),
        (
            2 * LATENT_FRAMES,
            torch.full((1, 2 * TOKENS_PER_HALF), 500.0),
            torch.ones(1, 1, dtype=torch.bool),
            "derives its own attention mask",
        ),
    ],
)
def test_transformer_rejects_invalid_teacher_forcing_inputs(
    cpu_ops_runtime,
    frames,
    timestep,
    attention_mask,
    message,
) -> None:
    model = _tiny_tf_transformer().eval()
    with pytest.raises(ValueError, match=message):
        model(
            hidden_states=torch.randn(1, 2 * Z_DIM, frames, LATENT_H, LATENT_W),
            timestep=timestep,
            encoder_hidden_states=torch.randn(1, 5, 4),
            attention_mask=attention_mask,
        )


def test_teacher_forcing_tiles_one_half_rope_without_recomputing_positions(
    cpu_ops_runtime,
    monkeypatch,
) -> None:
    import vforge.models.wan.v22.ti2v_5b.teacher_forcing.transformer as tf_module

    model = _tiny_tf_transformer().eval()
    hidden = torch.randn(1, 2 * Z_DIM, 2 * LATENT_FRAMES, LATENT_H, LATENT_W)
    expected_cos, expected_sin = model.rope(hidden[:, :, :LATENT_FRAMES])
    captured = {}
    original = tf_module.run_transformer_blocks

    def capture(model, states, context, timestep_proj, rotary_emb, **kwargs):
        captured["rotary_emb"] = rotary_emb
        return original(model, states, context, timestep_proj, rotary_emb, **kwargs)

    monkeypatch.setattr(tf_module, "run_transformer_blocks", capture)
    model(
        hidden_states=hidden,
        timestep=torch.full((1, 2 * TOKENS_PER_HALF), 500.0),
        encoder_hidden_states=torch.randn(1, 5, 4),
    )

    actual_cos, actual_sin = captured["rotary_emb"]
    torch.testing.assert_close(
        actual_cos,
        torch.cat([expected_cos, expected_cos], dim=1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual_sin,
        torch.cat([expected_sin, expected_sin], dim=1),
        rtol=0,
        atol=0,
    )


def test_clean_context_reaches_later_blocks_but_not_its_own(cpu_ops_runtime) -> None:
    """The point of the mask, stated as a behaviour rather than a shape.

    With ``frame_per_block=1`` each latent frame is its own causal block, so a
    perturbation of clean frame 1 must reach the prediction for frame 2 (which
    may read strictly-earlier clean context) and must not reach the prediction
    for frame 1 (which would be reading its own answer).
    """
    torch.manual_seed(0)
    model = _tiny_tf_transformer(frame_per_block=1).eval()
    hidden = torch.randn(1, 2 * Z_DIM, 2 * LATENT_FRAMES, LATENT_H, LATENT_W)
    prompt = torch.randn(1, 5, 4)
    timestep = torch.full((1, 2 * TOKENS_PER_HALF), 500.0)
    timestep[:, :TOKENS_PER_HALF] = 0.0

    def run(states):
        return model(
            hidden_states=states,
            timestep=timestep,
            encoder_hidden_states=prompt,
        )

    baseline = run(hidden)
    perturbed_states = hidden.clone()
    perturbed_states[:, :Z_DIM, 1] += 10.0  # clean frame 1
    perturbed = run(perturbed_states)

    torch.testing.assert_close(perturbed[:, :, 1], baseline[:, :, 1], rtol=0, atol=0)
    assert not torch.allclose(perturbed[:, :, 2], baseline[:, :, 2])


def test_transformer_is_weight_compatible_with_the_bidirectional_model() -> None:
    """A teacher-forcing run must start from a bidirectional checkpoint."""
    from vforge.models.wan.v22.ti2v_5b.ct.transformer import WanTransformer3DModel

    torch.manual_seed(0)
    config = {
        "patch_size": PATCH_SIZE,
        "num_attention_heads": 2,
        "attention_head_dim": 12,
        "in_channels": 2 * Z_DIM,
        "out_channels": Z_DIM,
        "text_dim": 4,
        "freq_dim": 16,
        "ffn_dim": 16,
        "num_layers": 1,
        "rope_max_seq_len": 32,
    }
    bidirectional = WanTransformer3DModel(**config)
    teacher = _tiny_tf_transformer()

    missing, unexpected = teacher.load_state_dict(bidirectional.state_dict(), strict=True)
    assert not missing and not unexpected


def test_incremental_block_matches_full_recompute_for_tiny_transformer(
    cpu_ops_runtime,
) -> None:
    from vforge.models.causal import IncrementalKVExecutor, IncrementalKVTransformer
    from vforge.models.wan.v22.ti2v_5b.teacher_forcing.inference import (
        build_active_block_tf_per_token_timestep,
    )

    torch.manual_seed(17)
    model = _tiny_tf_transformer(frame_per_block=1).eval()
    executor = model._incremental_kv_executor()
    assert isinstance(executor, IncrementalKVExecutor)
    assert isinstance(executor, IncrementalKVTransformer)
    clean = torch.randn(1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)
    condition = torch.randn_like(clean)
    prompt = torch.randn(1, 5, 4)
    active = torch.randn(1, Z_DIM, 1, LATENT_H, LATENT_W)
    full_rope = model.full_clip_rotary_emb(torch.cat([clean, condition], dim=1))
    cache = model.init_kv_cache(
        batch_size=1,
        num_cache_tokens=TOKENS_PER_HALF,
        device="cpu",
        dtype=torch.float32,
    )

    model.forward_kv_block(
        torch.cat([clean[:, :, :1], condition[:, :, :1]], dim=1),
        torch.zeros(1, TOKENS_PER_FRAME),
        prompt,
        kv_cache=cache,
        full_rotary_emb=full_rope,
        kv_write_start=0,
        commit_kv=True,
    )
    incremental = model.forward_kv_block(
        torch.cat([active, condition[:, :, 1:2]], dim=1),
        torch.full((1, TOKENS_PER_FRAME), 500.0),
        prompt,
        kv_cache=cache,
        full_rotary_emb=full_rope,
        kv_write_start=TOKENS_PER_FRAME,
        commit_kv=False,
    )

    noisy = clean.clone()
    noisy[:, :, 1:2] = active
    full_hidden = torch.cat(
        [
            torch.cat([clean, condition], dim=1),
            torch.cat([noisy, condition], dim=1),
        ],
        dim=2,
    )
    full_timestep = build_active_block_tf_per_token_timestep(
        clean,
        torch.tensor(500.0),
        PATCH_SIZE,
        active_block=(1, 2),
    )
    full = model(
        hidden_states=full_hidden,
        timestep=full_timestep,
        encoder_hidden_states=prompt,
    )

    torch.testing.assert_close(incremental, full[:, :, 1:2], rtol=1e-5, atol=1e-6)
    assert cache.committed_tokens == TOKENS_PER_FRAME


@pytest.mark.parametrize("use_reentrant", (True, False))
def test_checkpointed_forward_loss_and_parameter_gradients_match_direct(
    cpu_ops_runtime,
    use_reentrant: bool,
) -> None:
    torch.manual_seed(31)
    direct = _tiny_tf_transformer(frame_per_block=1)
    replayed = deepcopy(direct)
    replayed.gradient_checkpointing = True
    replayed._gradient_checkpointing_func = partial(
        checkpoint,
        use_reentrant=use_reentrant,
    )

    hidden = torch.randn(
        1,
        2 * Z_DIM,
        2 * LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        requires_grad=True,
    )
    timestep = torch.full((1, 2 * TOKENS_PER_HALF), 500.0)
    timestep[:, :TOKENS_PER_HALF] = 0.0
    prompt = torch.randn(1, 5, 4)

    expected = _forward_loss_and_parameter_gradients(
        direct,
        hidden.detach().clone().requires_grad_(True),
        timestep,
        prompt,
    )
    actual = _forward_loss_and_parameter_gradients(
        replayed,
        hidden.detach().clone().requires_grad_(True),
        timestep,
        prompt,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert actual[2].keys() == expected[2].keys()
    for name in expected[2]:
        torch.testing.assert_close(
            actual[2][name],
            expected[2][name],
            rtol=1e-6,
            atol=1e-8,
        )


@pytest.mark.parametrize("use_reentrant", (True, False))
def test_checkpoint_recompute_keeps_the_originating_forward_sparsity(
    cpu_ops_runtime,
    use_reentrant: bool,
) -> None:
    """A later, shorter forward must not overwrite the first backward mask."""
    torch.manual_seed(37)
    reference = _tiny_tf_transformer(frame_per_block=1)
    replayed = deepcopy(reference)
    replayed.gradient_checkpointing = True
    replayed._gradient_checkpointing_func = partial(
        checkpoint,
        use_reentrant=use_reentrant,
    )

    long_hidden = torch.randn(
        1,
        2 * Z_DIM,
        2 * LATENT_FRAMES,
        LATENT_H,
        LATENT_W,
        requires_grad=True,
    )
    long_timestep = torch.full((1, 2 * TOKENS_PER_HALF), 500.0)
    long_timestep[:, :TOKENS_PER_HALF] = 0.0
    prompt = torch.randn(1, 5, 4)
    expected = _forward_loss_and_parameter_gradients(
        reference,
        long_hidden.detach().clone().requires_grad_(True),
        long_timestep,
        prompt,
    )

    long_output = replayed(
        hidden_states=long_hidden,
        timestep=long_timestep,
        encoder_hidden_states=prompt,
    )
    short_frames = 2
    short_tokens_per_half = short_frames * TOKENS_PER_FRAME
    replayed(
        hidden_states=torch.randn(
            1,
            2 * Z_DIM,
            2 * short_frames,
            LATENT_H,
            LATENT_W,
            requires_grad=True,
        ),
        timestep=torch.cat(
            [
                torch.zeros(1, short_tokens_per_half),
                torch.full((1, short_tokens_per_half), 400.0),
            ],
            dim=1,
        ),
        encoder_hidden_states=prompt,
    )
    long_loss = long_output.float().square().mean()
    long_loss.backward()
    actual_gradients = {
        name: parameter.grad.detach().clone()
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
            rtol=1e-6,
            atol=1e-8,
        )


def test_sequence_parallel_shards_the_doubled_sequence_without_padding(
    cpu_ops_runtime,
    monkeypatch,
) -> None:
    import vforge.distributed.ulysses as usp_ops
    import vforge.models.wan.v22.ti2v_5b.teacher_forcing.transformer as tf_module

    sliced: list[tuple[int, ...]] = []

    def record_slice(tensor, *, dim, group=None, scale_grad=False):
        assert dim == 1
        sliced.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(usp_ops, "slice_sequence", record_slice)
    monkeypatch.setattr(
        usp_ops,
        "gather_sequence",
        lambda tensor, *, dim, group=None, scale_grad=False: tensor,
    )
    monkeypatch.setattr(
        tf_module,
        "get_parallel_state",
        lambda: SimpleNamespace(
            config=SimpleNamespace(usp_size=2),
            ulysses_group=None,
            usp_group=None,
        ),
    )

    torch.manual_seed(0)
    head_dim = 12
    inner_dim = 2 * head_dim
    model = _tiny_tf_transformer().eval()
    model(
        hidden_states=torch.randn(1, 2 * Z_DIM, 2 * LATENT_FRAMES, LATENT_H, LATENT_W),
        timestep=torch.full((1, 2 * TOKENS_PER_HALF), 500.0),
        encoder_hidden_states=torch.randn(1, 5, 4),
    )

    doubled = 2 * TOKENS_PER_HALF
    # Hidden states, per-token modulation and both RoPE tables, all at 2S.
    assert sliced[0] == (1, doubled, inner_dim)
    assert (1, doubled, 6, inner_dim) in sliced
    assert sliced.count((1, doubled, 1, head_dim)) == 2
    # ``temb`` is never sharded: the output head reads it after the gather.
    assert all(shape[1] == doubled for shape in sliced)


def test_sequence_parallel_refuses_a_sequence_it_cannot_split(
    cpu_ops_runtime,
    monkeypatch,
) -> None:
    """Padding would describe a mask the sparsity metadata is not built for."""
    import vforge.models.wan.v22.ti2v_5b.teacher_forcing.transformer as tf_module

    monkeypatch.setattr(
        tf_module,
        "get_parallel_state",
        lambda: SimpleNamespace(
            config=SimpleNamespace(usp_size=7),
            ulysses_group=None,
            usp_group=None,
        ),
    )
    model = _tiny_tf_transformer().eval()
    with pytest.raises(ValueError, match="divide by usp_size"):
        model(
            hidden_states=torch.randn(1, 2 * Z_DIM, 2 * LATENT_FRAMES, LATENT_H, LATENT_W),
            timestep=torch.full((1, 2 * TOKENS_PER_HALF), 500.0),
            encoder_hidden_states=torch.randn(1, 5, 4),
        )


# --------------------------------------------------------------------------
# Task wiring
# --------------------------------------------------------------------------
def test_train_step_drives_a_real_teacher_forcing_transformer(cpu_ops_runtime) -> None:
    from vforge.tasks.wan.v22.ti2v_5b.ct.train import _wan22_train_step
    from vforge.tasks.wan.v22.ti2v_5b.shared import _codec
    from vforge.training.recipe import StepContext

    torch.manual_seed(0)
    model = _tiny_tf_transformer()
    context = StepContext(
        device=torch.device("cpu"),
        parallel_state=SimpleNamespace(
            config=SimpleNamespace(usp_size=1),
            usp_rank=0,
            global_rank=0,
            ulysses_group=None,
            usp_group=None,
        ),
    )
    output = _wan22_train_step(
        model,
        {**_encoded_batch(), "sample_id": ["sample"]},
        0,
        context,
        latent_cache_mode="offline",
        seed=42,
        codec=_codec({}),
        processor=_tf_processor(),
        encoder=None,
    )

    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    # Workload is counted off the target, which is one half.
    assert output.workload["local_tokens"] == TOKENS_PER_HALF


def test_task_rejects_a_degenerate_block_layout() -> None:
    from vforge.tasks.wan.v22.ti2v_5b.teacher_forcing.train import _validate_tf

    with pytest.raises(ValueError, match="frame_per_block"):
        _validate_tf(extras=_extras(frame_per_block=0))


def test_registry_routes_tf_eval_to_the_causal_builder_only() -> None:
    from vforge.tasks.registry import require_registered

    tasks = require_registered("wan22_ti2v_5b_tf")
    assert tasks.train == (
        "vforge.tasks.wan.v22.ti2v_5b.teacher_forcing.train:build_train_components"
    )
    assert tasks.eval == ("vforge.tasks.wan.v22.ti2v_5b.teacher_forcing.eval:build_eval_components")
    assert tasks.encode == require_registered("wan22_ti2v_5b").encode


# --------------------------------------------------------------------------
# Causal full-recompute eval
# --------------------------------------------------------------------------
class _CausalEvalTransformer(torch.nn.Module):
    config = SimpleNamespace(in_channels=2 * Z_DIM, patch_size=PATCH_SIZE)

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.active_cache: str | None = None
        self.cache_entries: list[str] = []
        self.allocated_kv_caches: list[object] = []

    @property
    def dtype(self):
        return torch.float32

    @contextmanager
    def cache_context(self, name: str):
        self.cache_entries.append(name)
        self.active_cache = name
        try:
            yield
        finally:
            self.active_cache = None

    def full_clip_rotary_emb(self, hidden_states):
        tokens = hidden_states.shape[2] * TOKENS_PER_FRAME
        positions = torch.arange(tokens, dtype=hidden_states.dtype).view(1, tokens, 1, 1)
        return positions, torch.zeros_like(positions)

    def init_kv_cache(self, **_kwargs):
        cache = object()
        self.allocated_kv_caches.append(cache)
        return cache


class _CausalEvalScheduler:
    def __init__(self) -> None:
        self.reset_count = 0

    def set_timesteps(self, steps, device):
        self.reset_count += 1
        self.timesteps = [torch.tensor(float(step), device=device) for step in range(steps, 0, -1)]

    def step(self, prediction, _timestep, latents, return_dict):
        assert return_dict is False
        return (latents + prediction,)


def _causal_eval_pipeline(
    *,
    guidance_scale: float = 1.0,
    rollout_mode="full_recompute",
    first_frame_own_block: bool = True,
):
    from vforge.models.wan.v22.ti2v_5b.teacher_forcing.inference import (
        Wan22Ti2v5bCausalPipeline,
    )

    transformer = _CausalEvalTransformer()
    scheduler = _CausalEvalScheduler()
    calls: list[dict] = []

    def engine_forward(*, hidden_states, timestep, **_kwargs):
        calls.append(
            {
                "cache": transformer.active_cache,
                "hidden_states": hidden_states.clone(),
                "timestep": timestep.clone(),
                "kv_cache": _kwargs.get("kv_cache"),
                "kv_write_start": _kwargs.get("kv_write_start"),
                "commit_kv": _kwargs.get("commit_kv"),
                "full_rotary_emb": _kwargs.get("full_rotary_emb"),
            }
        )
        output_frames = (
            hidden_states.shape[2] if _kwargs.get("kv_cache") is not None else LATENT_FRAMES
        )
        prediction = torch.ones(
            hidden_states.shape[0],
            Z_DIM,
            output_frames,
            LATENT_H,
            LATENT_W,
        )
        return (prediction,)

    pipeline = Wan22Ti2v5bCausalPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=None,
        transformer=transformer,
        engine_forward=engine_forward,
        scheduler=scheduler,
        frame_per_block=1,
        first_frame_own_block=first_frame_own_block,
        rollout_mode=rollout_mode,
        vae_scale_factor_temporal=4,
        vae_scale_factor_spatial=16,
    )
    prompt = torch.ones(1, 3, 4)
    condition = torch.full(
        (1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W),
        9.0,
    )
    z0 = torch.full((1, Z_DIM, 1, LATENT_H, LATENT_W), 5.0)
    negative = torch.zeros_like(prompt) if guidance_scale > 1 else None
    result = pipeline.call_with_latent(
        prompt,
        y=condition,
        first_frame_latents=z0,
        height=LATENT_H * 16,
        width=LATENT_W * 16,
        num_frames=(LATENT_FRAMES - 1) * 4 + 1,
        num_inference_steps=2,
        guidance_scale=guidance_scale,
        latents=torch.zeros(1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W),
        negative_prompt_embeds=negative,
        decode_output=False,
    )
    return result, transformer, scheduler, calls


def test_tiny_transformer_incremental_and_full_rollouts_match(
    cpu_ops_runtime,
) -> None:
    from vforge.models.wan.v22.ti2v_5b.teacher_forcing.inference import (
        Wan22Ti2v5bCausalPipeline,
    )
    from vforge.models.wan.v22.ti2v_5b.teacher_forcing.model import Wan22Ti2v5bTfModel

    torch.manual_seed(23)
    model = Wan22Ti2v5bTfModel(_tiny_tf_transformer(frame_per_block=1).eval())
    common = {
        "tokenizer": None,
        "text_encoder": None,
        "vae": None,
        "transformer": model,
        "engine_forward": model,
        "frame_per_block": 1,
        "first_frame_own_block": True,
        "vae_scale_factor_temporal": 4,
        "vae_scale_factor_spatial": 16,
    }
    full_pipeline = Wan22Ti2v5bCausalPipeline(
        **common,
        scheduler=_CausalEvalScheduler(),
        rollout_mode="full_recompute",
    )
    incremental_pipeline = Wan22Ti2v5bCausalPipeline(
        **common,
        scheduler=_CausalEvalScheduler(),
        rollout_mode="incremental_kv",
    )
    prompt = torch.randn(1, 3, 4)
    condition = torch.randn(1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)
    z0 = torch.randn(1, Z_DIM, 1, LATENT_H, LATENT_W)
    initial_noise = torch.randn(1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W)
    call_kwargs = {
        "y": condition,
        "first_frame_latents": z0,
        "height": LATENT_H * 16,
        "width": LATENT_W * 16,
        "num_frames": (LATENT_FRAMES - 1) * 4 + 1,
        "num_inference_steps": 2,
        "guidance_scale": 1.0,
        "latents": initial_noise,
        "decode_output": False,
    }

    full = full_pipeline.call_with_latent(prompt, **call_kwargs)
    incremental = incremental_pipeline.call_with_latent(prompt, **call_kwargs)

    torch.testing.assert_close(incremental, full, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(incremental[:, :, :1], z0, rtol=0, atol=0)


@pytest.mark.parametrize("rollout_mode", ["full_recompute", "incremental_kv"])
def test_causal_eval_metadata_records_actual_mode_and_block_layout(
    monkeypatch,
    rollout_mode,
) -> None:
    from vforge.models.wan.v22.ti2v_5b.teacher_forcing.inference import (
        Wan22CausalInferencePipeline,
    )

    pipeline_result, transformer, _scheduler, _calls = _causal_eval_pipeline(
        rollout_mode=rollout_mode
    )
    inner = SimpleNamespace(
        align_num_frames=lambda value: value,
        vae_scale_factor_temporal=4,
        rollout_mode=rollout_mode,
        frame_per_block=1,
        first_frame_own_block=True,
        block_layout=lambda _frames: ((0, 1), (1, 2), (2, 3)),
        call_with_latent=lambda *_args, **_kwargs: pipeline_result,
    )
    wrapper = Wan22CausalInferencePipeline(
        model=transformer,
        encode_condition=lambda _sample: {},
        pipeline=inner,
        stream_fields={"normal": "condition_latents_normal"},
        num_inference_steps=2,
        guidance_scale=1.0,
    )
    monkeypatch.setattr(
        "vforge.distributed.parallel_state.get_parallel_state",
        lambda: SimpleNamespace(is_batch_leader=True),
    )
    result = wrapper.infer_sample(
        {
            "sample_id": "causal",
            "__height__": LATENT_H * 16,
            "__width__": LATENT_W * 16,
            "__num_frames__": (LATENT_FRAMES - 1) * 4 + 1,
            "__encoded__": {
                "prompt": torch.ones(1, 3, 4),
                "condition_latents_normal": torch.zeros(
                    1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W
                ),
                "first_frame_latents": torch.ones(1, Z_DIM, 1, LATENT_H, LATENT_W),
                # A validation/cache target may still be present in a manually
                # prepared sample; the wrapper must never pass it to rollout.
                "frame_latents": torch.full((1, Z_DIM, LATENT_FRAMES, LATENT_H, LATENT_W), 99.0),
            },
        },
        seed=3,
    )

    assert result["metadata"]["rollout_mode"] == rollout_mode
    assert result["metadata"]["block_layout"]["ranges"] == [[0, 1], [1, 2], [2, 3]]


def test_incremental_kv_overwrites_then_context_repasses_clean_blocks() -> None:
    _result, transformer, scheduler, calls = _causal_eval_pipeline(rollout_mode="incremental_kv")

    assert scheduler.reset_count == 2
    assert len(transformer.allocated_kv_caches) == 1
    cache = transformer.allocated_kv_caches[0]
    assert [call["kv_cache"] for call in calls] == [cache] * 7
    assert [call["commit_kv"] for call in calls] == [
        True,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    assert [call["kv_write_start"] for call in calls] == [
        0,
        TOKENS_PER_FRAME,
        TOKENS_PER_FRAME,
        TOKENS_PER_FRAME,
        2 * TOKENS_PER_FRAME,
        2 * TOKENS_PER_FRAME,
        2 * TOKENS_PER_FRAME,
    ]


def test_incremental_kv_cfg_allocates_strictly_isolated_caches() -> None:
    _result, transformer, _scheduler, calls = _causal_eval_pipeline(
        guidance_scale=3.0,
        rollout_mode="incremental_kv",
    )

    cond_cache, uncond_cache = transformer.allocated_kv_caches
    assert cond_cache is not uncond_cache
    assert [call["cache"] for call in calls] == ["cond", "uncond"] * 7
    assert {call["kv_cache"] for call in calls if call["cache"] == "cond"} == {cond_cache}
    assert {call["kv_cache"] for call in calls if call["cache"] == "uncond"} == {uncond_cache}


def test_incremental_kv_pipeline_rejects_usp_before_rollout() -> None:
    from vforge.models.wan.v22.ti2v_5b.teacher_forcing.inference import (
        Wan22Ti2v5bCausalPipeline,
    )

    with pytest.raises(NotImplementedError, match="usp_size=1"):
        Wan22Ti2v5bCausalPipeline(
            tokenizer=None,
            text_encoder=None,
            vae=None,
            transformer=_CausalEvalTransformer(),
            engine_forward=lambda **_kwargs: None,
            scheduler=_CausalEvalScheduler(),
            frame_per_block=1,
            first_frame_own_block=True,
            rollout_mode="incremental_kv",
            usp_size=2,
        )


def test_causal_rollout_requires_the_first_frame_to_own_its_block() -> None:
    with pytest.raises(ValueError, match="first_frame_own_block=true"):
        _causal_eval_pipeline(first_frame_own_block=False)
