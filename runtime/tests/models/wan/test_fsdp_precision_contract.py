"""Native Wan FSDP precision contract: paths, wrap flags, and time-path dtype."""

from __future__ import annotations

import torch
import torch.nn as nn

import vforge.distributed.parallelize as parallelize_module
from vforge.distributed.parallelize import (
    FSDP2Config,
    MixedPrecisionConfig,
    _apply_fsdp2,
    _resolve_fsdp_precision_contract,
)
from vforge.models.wan.transformer import wan_fsdp_precision_contract

_TINY = {
    "patch_size": (1, 2, 2),
    "num_attention_heads": 2,
    "attention_head_dim": 12,
    "text_dim": 4,
    "freq_dim": 16,
    "ffn_dim": 16,
    "num_layers": 2,
    "rope_max_seq_len": 32,
}


_LEGACY_MODULE_PATHS = (
    "condition_embedder.time_embedder",
    "blocks.*.norm1",
    "blocks.*.norm2",
    "blocks.*.norm3",
)
_EXPECTED_INPUT_PATHS = (".", "blocks.*", "norm_out")


def _expanded_legacy_paths(transformer: nn.Module) -> set[str]:
    from vforge.distributed.parallelize import _expand_path_pattern

    return {
        concrete
        for pattern in _LEGACY_MODULE_PATHS
        for concrete in _expand_path_pattern(transformer, pattern)
    }


def _assert_wan_contract(transformer: nn.Module) -> None:
    contract = transformer.get_fsdp_precision_contract()
    assert contract.preserve_forward_input_paths == _EXPECTED_INPUT_PATHS
    # The preserved modules now come from ``_keep_in_fp32_modules``; what
    # matters is that they still name exactly what the hand-written list named.
    resolved = _resolve_fsdp_precision_contract(transformer, config=MixedPrecisionConfig())
    assert set(resolved.preserve_module_paths) == _expanded_legacy_paths(transformer)


def test_an_undeclared_fork_preserves_no_modules() -> None:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(2)
            self.scale_shift_table = nn.Parameter(torch.zeros(1, 2))

    class Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([Block()])

        def get_fsdp_precision_contract(self) -> object:
            return wan_fsdp_precision_contract()

    resolved = _resolve_fsdp_precision_contract(Root(), config=MixedPrecisionConfig())
    # Reusing Wan's contract does not import Wan's fp32 list: a fork that never
    # declares ``_keep_in_fp32_modules`` is exempt from nothing.
    assert resolved.preserve_module_paths == ()
    assert resolved.preserve_forward_input_paths == _EXPECTED_INPUT_PATHS


def test_wan_keep_in_fp32_modules_match_diffusers_for_parameters() -> None:
    from diffusers.models.transformers.transformer_wan import (
        WanTransformer3DModel as DiffusersWan,
    )

    from vforge.models.loading import cast_module_parameters
    from vforge.models.wan.v21.transformer import WanTransformer3DModel as V21
    from vforge.models.wan.v22.ti2v_5b.ct.transformer import (
        WanTransformer3DModel as Ti2v,
    )
    from vforge.models.wan.v22.ti2v_5b.teacher_forcing.transformer import (
        WanTeacherForcingTransformer3DModel,
    )

    expected = ("time_embedder", "scale_shift_table", "norm1", "norm2", "norm3")
    # Diffusers may keep more than this: its `.to(dtype)` also reaches buffers,
    # so a buffer-only fragment like `rope` earns a slot there while
    # ``cast_module_parameters`` never walks buffers. Keeping a fragment
    # Diffusers downcasts is the one direction that would diverge.
    assert set(expected) <= set(DiffusersWan._keep_in_fp32_modules)
    assert tuple(V21._keep_in_fp32_modules) == expected
    assert tuple(Ti2v._keep_in_fp32_modules) == expected
    assert tuple(WanTeacherForcingTransformer3DModel._keep_in_fp32_modules) == expected

    model = V21(
        **_TINY,
        in_channels=16,
        out_channels=16,
        use_vae_condition=False,
    )
    cast_module_parameters(model, torch.bfloat16)
    assert next(model.condition_embedder.time_embedder.parameters()).dtype is torch.float32
    assert model.scale_shift_table.dtype is torch.float32
    assert model.blocks[0].scale_shift_table.dtype is torch.float32
    assert model.patch_embedding.weight.dtype is torch.bfloat16
    assert model.proj_out.weight.dtype is torch.bfloat16
    assert model.condition_embedder.time_proj.weight.dtype is torch.bfloat16


def test_v21_transformer_declares_the_wan_precision_contract() -> None:
    from vforge.models.wan.v21.transformer import WanTransformer3DModel

    _assert_wan_contract(
        WanTransformer3DModel(
            **_TINY,
            in_channels=16,
            out_channels=16,
            use_vae_condition=False,
        )
    )


def test_ti2v_and_teacher_forcing_share_the_same_contract_paths() -> None:
    from vforge.models.wan.v22.ti2v_5b.ct.transformer import (
        WanTransformer3DModel as CtTransformer,
    )
    from vforge.models.wan.v22.ti2v_5b.teacher_forcing.transformer import (
        WanTeacherForcingTransformer3DModel,
    )

    ct = CtTransformer(**_TINY, in_channels=4, out_channels=2)
    tf = WanTeacherForcingTransformer3DModel(**_TINY, in_channels=4, out_channels=2)
    _assert_wan_contract(ct)
    _assert_wan_contract(tf)
    assert (
        WanTeacherForcingTransformer3DModel.get_fsdp_precision_contract
        is CtTransformer.get_fsdp_precision_contract
    )


def test_time_embedder_type_as_matches_diffusers_when_text_is_bf16() -> None:
    from vforge.models.wan.v21.transformer import WanTimeTextImageEmbedding
    from vforge.models.wan.v22.ti2v_5b.ct.transformer import WanTimeTextEmbedding

    timestep = torch.tensor([0.25, 0.75])
    text = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    v21 = WanTimeTextImageEmbedding(dim=8, time_freq_dim=16, time_proj_dim=48, text_embed_dim=4)
    ti2v = WanTimeTextEmbedding(dim=8, time_freq_dim=16, time_proj_dim=48, text_embed_dim=4)

    for embedder in (v21, ti2v):
        embedder.text_embedder.to(dtype=torch.bfloat16)
        embedder.time_proj.to(dtype=torch.bfloat16)
        temb, timestep_proj, encoder_hidden_states = embedder(timestep, text)[:3]
        assert next(embedder.time_embedder.parameters()).dtype is torch.float32
        assert temb.dtype is torch.bfloat16
        assert timestep_proj.dtype is torch.bfloat16
        assert encoder_hidden_states.dtype is torch.bfloat16


class _WanBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Linear(4, 4)
        self.norm1 = nn.LayerNorm(4)
        self.norm2 = nn.LayerNorm(4)
        self.norm3 = nn.LayerNorm(4, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6, 4))


class _ConditionEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.time_embedder = nn.Linear(4, 4, dtype=torch.float32)
        self.time_proj = nn.Linear(4, 4, dtype=torch.float32)
        self.text_embedder = nn.Linear(4, 4)


class _FakeWanTransformer(nn.Module):
    _no_split_modules = ["_WanBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "norm1",
        "norm2",
        "norm3",
    ]

    def __init__(self) -> None:
        super().__init__()
        self.condition_embedder = _ConditionEmbedder()
        self.blocks = nn.ModuleList([_WanBlock(), _WanBlock()])
        self.norm_out = nn.LayerNorm(4)
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 2, 4))

    def get_fsdp_precision_contract(self):
        return wan_fsdp_precision_contract()


def test_apply_fsdp2_keeps_time_embedder_and_does_not_precast_blocks(monkeypatch) -> None:
    model = _FakeWanTransformer()
    calls: list[tuple[nn.Module, dict]] = []

    def fake_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    import torch.distributed._composable.fsdp as composable_fsdp

    monkeypatch.setattr(composable_fsdp, "fully_shard", fake_fully_shard)
    monkeypatch.setattr(
        parallelize_module,
        "get_parallel_state",
        lambda: type("State", (), {"fsdp_mesh": object()})(),
    )

    _apply_fsdp2(model, FSDP2Config(mixed_precision=MixedPrecisionConfig(enable=True)))

    kwargs_by_module = {id(module): kwargs for module, kwargs in calls}
    time_embedder = model.condition_embedder.time_embedder
    time_proj = model.condition_embedder.time_proj
    assert "mp_policy" not in kwargs_by_module[id(time_embedder)]
    assert id(time_proj) not in kwargs_by_module
    assert "mp_policy" not in kwargs_by_module[id(model.blocks[0].norm1)]
    assert "mp_policy" not in kwargs_by_module[id(model.blocks[0].norm2)]
    assert kwargs_by_module[id(model.blocks[0])]["mp_policy"].cast_forward_inputs is False
    assert kwargs_by_module[id(model.blocks[1])]["mp_policy"].cast_forward_inputs is False
    assert kwargs_by_module[id(model)]["mp_policy"].cast_forward_inputs is False
    assert kwargs_by_module[id(model.blocks[0])]["mp_policy"].param_dtype is torch.bfloat16
    assert kwargs_by_module[id(model)]["mp_policy"].param_dtype is torch.bfloat16
    assert id(model.blocks[0].attn) not in kwargs_by_module
    assert id(model.condition_embedder.text_embedder) not in kwargs_by_module
    assert id(model.condition_embedder) not in kwargs_by_module

    # The AdaLN tables have no group of their own, so they follow the policy
    # dtype of the group that owns them.
    assert all("ignored_params" not in kwargs for kwargs in kwargs_by_module.values())
    assert isinstance(model.blocks[0].scale_shift_table, nn.Parameter)
