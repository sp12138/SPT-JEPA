"""Small forward and lifecycle contracts for Wan Taichu model roots."""

from __future__ import annotations

import ast
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def ops_runtime():
    from vforge.distributed.parallel_state import ParallelConfig
    from vforge.distributed.runtime import destroy_runtime, initialize_runtime
    from vforge.ops.config import OpsConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    initialize_runtime(
        ParallelConfig(),
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=device,
    )
    try:
        yield device
    finally:
        destroy_runtime()


def _common_config(*, in_channels: int, out_channels: int) -> dict:
    return {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 12,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "text_dim": 4,
        "freq_dim": 16,
        "ffn_dim": 16,
        "num_layers": 1,
        "rope_max_seq_len": 32,
    }


def test_family_import_is_lightweight() -> None:
    code = (
        "import sys; import vforge.models.wan_taichu; "
        "assert 'diffusers' not in sys.modules; "
        "assert 'vforge.models.wan_taichu.model' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parents[3],
        check=True,
    )


def test_model_package_never_imports_task_or_training_layers() -> None:
    import vforge.models.wan_taichu

    root = Path(vforge.models.wan_taichu.__file__).parent
    forbidden = (
        "vforge.tasks",
        "vforge.training",
        "vforge.compat",
        "vforge.models.wan.",
    )
    violations = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.startswith(forbidden) for name in names):
                violations.append(path.relative_to(root))
    assert violations == []


@pytest.mark.parametrize(
    ("variant", "architecture", "config", "per_token_timestep"),
    [
        (
            "wan21_i2v_14b",
            "wan21",
            {**_common_config(in_channels=4, out_channels=2), "use_vae_condition": False},
            False,
        ),
        (
            "wan22_ti2v_5b",
            "wan22_ti2v_5b",
            _common_config(in_channels=4, out_channels=2),
            True,
        ),
    ],
)
def test_single_root_delegates_to_package_local_transformer_without_numerical_change(
    ops_runtime,
    variant: str,
    architecture: str,
    config: dict,
    per_token_timestep: bool,
) -> None:
    from vforge.distributed.parallelize import _resolve_fsdp_execution_roots
    from vforge.models.wan_taichu.model import (
        WanTaichuModel,
        WanTaichuTransformer3DModel,
    )

    device = ops_runtime
    torch.manual_seed(17)
    transformer = (
        WanTaichuTransformer3DModel(
            architecture=architecture,
            attention="bidirectional",
            **config,
        )
        .eval()
        .to(device)
    )
    direct = deepcopy(transformer)
    normal = (
        WanTaichuModel(
            variant=variant,
            architecture=architecture,
            attention="bidirectional",
            transformers={"": transformer},
        )
        .eval()
        .to(device)
    )

    hidden_states = torch.randn(1, config["in_channels"], 1, 4, 4, device=device)
    encoder_hidden_states = torch.randn(1, 3, config["text_dim"], device=device)
    token_count = 4
    timestep = (
        torch.full((1, token_count), 700.0, device=device)
        if per_token_timestep
        else torch.full((1,), 700.0, device=device)
    )
    with torch.no_grad():
        expected = direct(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        actual = normal(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual.shape == (1, config["out_channels"], 1, 4, 4)
    assert normal.fsdp_execution_roots() == (normal.transformer,)
    assert _resolve_fsdp_execution_roots(normal) == (normal.transformer,)
    assert all(key.startswith("transformer.") for key in normal.state_dict())


def test_single_root_uses_package_rope_and_model_checkpointing(ops_runtime) -> None:
    from vforge.distributed.parallelize import (
        FSDP2Config,
        ParallelizeConfig,
        parallelize_model,
    )
    from vforge.models.wan_taichu.model import (
        WanTaichuModel,
        WanTaichuTransformer3DModel,
    )
    from vforge.models.wan_taichu.transformer import WanRotaryPosEmbed

    model = WanTaichuModel(
        variant="wan21_i2v_14b",
        architecture="wan21",
        attention="bidirectional",
        transformers={
            "": WanTaichuTransformer3DModel(
                architecture="wan21",
                attention="bidirectional",
                **_common_config(in_channels=4, out_channels=2),
                use_vae_condition=False,
            )
        },
    ).to(ops_runtime)
    assert isinstance(model.transformer.rope, WanRotaryPosEmbed)

    parallelize_model(
        model,
        ParallelizeConfig(
            fsdp2=FSDP2Config(enable=False),
            gradient_checkpointing=True,
        ),
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )
    assert model.transformer.gradient_checkpointing is True
    assert model.transformer._gradient_checkpointing_func.keywords == {"use_reentrant": True}


def test_ref_rope_zeroes_time_and_preserves_spatial_coordinates(ops_runtime) -> None:
    from vforge.models.wan_taichu.transformer import WanRotaryPosEmbed

    rope = WanRotaryPosEmbed(12, (1, 2, 2), 32).to(ops_runtime)
    ref = torch.zeros(1, 2, 2, 4, 4, device=ops_runtime)
    cos, sin = rope.build_ref(ref)
    regular_cos, regular_sin = rope(ref)

    assert torch.count_nonzero(cos[..., : rope.t_dim]) == 0
    assert torch.count_nonzero(sin[..., : rope.t_dim]) == 0
    torch.testing.assert_close(cos[..., rope.t_dim :], regular_cos[..., rope.t_dim :])
    torch.testing.assert_close(sin[..., rope.t_dim :], regular_sin[..., rope.t_dim :])


def test_transformer_casts_rope_to_patch_token_dtype(ops_runtime) -> None:
    from vforge.models.wan_taichu.model import WanTaichuTransformer3DModel

    transformer = (
        WanTaichuTransformer3DModel(
            architecture="wan22_ti2v_5b",
            attention="bidirectional",
            **_common_config(in_channels=4, out_channels=2),
        )
        .to(device=ops_runtime, dtype=torch.bfloat16)
        .eval()
    )
    # Reproduce FSDP2 meta materialization retaining the non-persistent RoPE
    # tables at a wider dtype than the BF16 patch-token stream.
    transformer.rope.freqs_cos = transformer.rope.freqs_cos.float()
    transformer.rope.freqs_sin = transformer.rope.freqs_sin.float()
    observed: list[tuple[torch.dtype, torch.dtype]] = []

    def record_block_inputs(_module, args) -> None:
        rotary_emb = args[3]
        observed.append((rotary_emb[0].dtype, rotary_emb[1].dtype))

    handle = transformer.blocks[0].register_forward_pre_hook(record_block_inputs)
    try:
        with torch.no_grad():
            transformer(
                hidden_states=torch.randn(
                    1,
                    4,
                    1,
                    4,
                    4,
                    device=ops_runtime,
                    dtype=torch.bfloat16,
                ),
                timestep=torch.full(
                    (1, 4),
                    700.0,
                    device=ops_runtime,
                    dtype=torch.float32,
                ),
                encoder_hidden_states=torch.randn(
                    1,
                    3,
                    4,
                    device=ops_runtime,
                    dtype=torch.bfloat16,
                ),
            )
    finally:
        handle.remove()

    assert observed == [(torch.bfloat16, torch.bfloat16)]


def test_ref_tokens_preserve_video_shape_and_fsdp_dummy_dependency(ops_runtime) -> None:
    from vforge.models.wan_taichu.model import (
        WanTaichuModel,
        WanTaichuTransformer3DModel,
    )

    config = _common_config(in_channels=4, out_channels=2)
    transformer = WanTaichuTransformer3DModel(
        architecture="wan22_ti2v_5b",
        attention="bidirectional",
        use_ref_token=True,
        in_channels_ref=2,
        **config,
    )
    model = WanTaichuModel(
        variant="wan22_ti2v_5b_ref",
        architecture="wan22_ti2v_5b",
        attention="bidirectional",
        transformers={"": transformer},
    ).to(ops_runtime)
    hidden = torch.randn(1, 4, 3, 4, 4, device=ops_runtime)
    # Production ref sidecars may retain one pre-crop row. Ref tokens own an
    # independent patch grid, so they need not match the target latent H/W.
    ref = torch.randn(1, 2, 2, 5, 4, device=ops_runtime)
    timestep = torch.full((1, 12), 700.0, device=ops_runtime)
    prompt = torch.randn(1, 3, 4, device=ops_runtime)

    with_ref = model(
        hidden_states=hidden,
        timestep=timestep,
        encoder_hidden_states=prompt,
        ref_hidden_states=ref,
        global_has_ref=True,
    )
    assert with_ref.shape == (1, 2, 3, 4, 4)
    with_ref.square().mean().backward()
    assert model.transformer.patch_embedding_ref.weight.grad is not None

    model.zero_grad(set_to_none=True)
    without_local_ref = model(
        hidden_states=hidden,
        timestep=timestep,
        encoder_hidden_states=prompt,
        ref_hidden_states=None,
        global_has_ref=True,
    )
    without_local_ref.square().mean().backward()
    dummy_grad = model.transformer.patch_embedding_ref.weight.grad
    assert dummy_grad is not None
    assert torch.count_nonzero(dummy_grad) == 0

    model.zero_grad(set_to_none=True)
    model(
        hidden_states=hidden,
        timestep=timestep,
        encoder_hidden_states=prompt,
        ref_hidden_states=None,
        global_has_ref=False,
    ).square().mean().backward()
    no_ref_grad = model.transformer.patch_embedding_ref.weight.grad
    assert no_ref_grad is not None
    assert torch.count_nonzero(no_ref_grad) == 0


def test_audio_speak_and_listen_residuals_share_the_taichu_model(ops_runtime) -> None:
    from vforge.models.wan_taichu.model import WanTaichuTransformer3DModel

    model = WanTaichuTransformer3DModel(
        architecture="wan21",
        attention="bidirectional",
        use_vae_condition=False,
        use_audio_condition=True,
        audio_dim=4,
        audio_num_layers=3,
        num_audio_token=2,
        audio_inject_layers_s=[0],
        audio_inject_layers_l=[0],
        **_common_config(in_channels=4, out_channels=2),
    ).to(ops_runtime)
    hidden = torch.randn(1, 4, 2, 4, 4, device=ops_runtime)
    prompt = torch.randn(1, 3, 4, device=ops_runtime)
    timestep = torch.full((1,), 700.0, device=ops_runtime)
    speak = torch.randn(1, 3, 4, 5, device=ops_runtime)
    listen = torch.randn(1, 3, 4, 5, device=ops_runtime)

    model.zero_audio_output_projections()
    without_audio = model(
        hidden_states=hidden,
        timestep=timestep,
        encoder_hidden_states=prompt,
    )
    with_audio = model(
        hidden_states=hidden,
        timestep=timestep,
        encoder_hidden_states=prompt,
        audio_input_s=speak,
        audio_input_l=listen,
    )
    torch.testing.assert_close(with_audio, without_audio, rtol=0.0, atol=0.0)

    with torch.no_grad():
        model.audio_injector.injector_s[0].o.weight.fill_(0.01)
        model.audio_injector.injector_l[0].o.weight.fill_(0.01)
    active = model(
        hidden_states=hidden,
        timestep=timestep,
        encoder_hidden_states=prompt,
        audio_input_s=speak,
        audio_input_l=listen,
    )
    assert not torch.equal(active, without_audio)


def test_causal_teacher_root_delegates_to_package_local_transformer(
    ops_runtime,
) -> None:
    from vforge.distributed.parallelize import _resolve_fsdp_execution_roots
    from vforge.models.wan_taichu.model import (
        WanTaichuModel,
        WanTaichuTransformer3DModel,
    )

    config = _common_config(in_channels=4, out_channels=2)
    torch.manual_seed(23)
    transformer = (
        WanTaichuTransformer3DModel(
            architecture="wan22_ti2v_5b",
            attention="block_causal_teacher",
            **config,
        )
        .eval()
        .to(ops_runtime)
    )
    direct = deepcopy(transformer)
    migrated = (
        WanTaichuModel(
            variant="wan22_ti2v_5b_causal_teacher",
            architecture="wan22_ti2v_5b",
            attention="block_causal_teacher",
            transformers={"": transformer},
        )
        .eval()
        .to(ops_runtime)
    )

    half_frames = 2
    tokens_per_half = half_frames * 4
    hidden_states = torch.randn(
        1,
        config["in_channels"],
        2 * half_frames,
        4,
        4,
        device=ops_runtime,
    )
    timestep = torch.full((1, 2 * tokens_per_half), 700.0, device=ops_runtime)
    timestep[:, :tokens_per_half] = 0.0
    encoder_hidden_states = torch.randn(1, 3, config["text_dim"], device=ops_runtime)
    with torch.no_grad():
        expected = direct(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
        actual = migrated(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actual.shape == (1, config["out_channels"], half_frames, 4, 4)
    assert migrated.fsdp_execution_roots() == (migrated.transformer,)
    assert _resolve_fsdp_execution_roots(migrated) == (migrated.transformer,)
    assert all(key.startswith("transformer.") for key in migrated.state_dict())


class _Expert(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(value))

    def forward(self, hidden_states, *, timestep, **kwargs):
        return hidden_states + self.weight


def test_a14b_root_preserves_expert_paths_and_routes_forward(ops_runtime) -> None:
    from vforge.models.wan_taichu.model import WanTaichuModel

    model = WanTaichuModel(
        variant="wan22_a14b",
        architecture="wan22_a14b",
        attention="bidirectional",
        transformers={"wan": _Expert(1.0), "wan_2": _Expert(2.0)},
        boundary_ratio=0.9,
        num_train_timesteps=1000,
    ).to(ops_runtime)
    hidden = torch.zeros(1, device=ops_runtime)

    assert torch.equal(
        model(hidden, timestep=torch.tensor([900.0], device=ops_runtime)),
        torch.ones(1, device=ops_runtime),
    )
    assert torch.equal(
        model(hidden, timestep=torch.tensor([899.0], device=ops_runtime)),
        torch.full((1,), 2.0, device=ops_runtime),
    )
    assert set(model.state_dict()) == {"wan.weight", "wan_2.weight"}
    assert model.fsdp_execution_roots() == (model.wan, model.wan_2)


def test_a14b_causal_teacher_routes_on_the_noisy_timestep_half(ops_runtime) -> None:
    from vforge.models.wan_taichu.model import WanTaichuModel

    model = WanTaichuModel(
        variant="wan22_a14b_causal_teacher",
        architecture="wan22_a14b",
        attention="block_causal_teacher",
        transformers={"wan": _Expert(1.0), "wan_2": _Expert(2.0)},
        boundary_ratio=0.9,
        num_train_timesteps=1000,
    ).to(ops_runtime)
    hidden = torch.zeros(1, device=ops_runtime)

    assert torch.equal(
        model(
            hidden,
            timestep=torch.tensor([[0.0, 0.0, 900.0, 900.0]], device=ops_runtime),
        ),
        torch.ones(1, device=ops_runtime),
    )
    assert torch.equal(
        model(
            hidden,
            timestep=torch.tensor([[0.0, 0.0, 899.0, 899.0]], device=ops_runtime),
        ),
        torch.full((1,), 2.0, device=ops_runtime),
    )
