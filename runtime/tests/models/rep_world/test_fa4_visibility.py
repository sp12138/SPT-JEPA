"""Exact visibility and unchanged SDPA contracts, independent of CUDA kernels."""

import copy
from unittest.mock import patch

import pytest
import torch

from vforge.models.rep_world.conditioner import CausalViT
from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, WorldConfig
from vforge.models.rep_world.layers import (
    _build_full_history_teacher_forcing_attn_mask,
    _build_prefix_block_causal_attn_mask,
)
from vforge.models.rep_world.sparsity import (
    prefix_sparsity,
    slice_query_sparsity,
    state_causal_sparsity,
    teacher_forcing_sparsity,
)
from vforge.models.rep_world.world import WorldDiT
from vforge.ops import OpsConfig, initialize_ops, reset_ops
from vforge.ops._attention.segmented import segmented_dense_mask


@pytest.fixture(autouse=True)
def ops():
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False),
        device=torch.device("cpu"),
        parallel_state=None,
    )
    yield
    reset_ops()


@pytest.mark.parametrize("tokens", [3, 73, 137, 204])
def test_state_visibility_and_rectangular_cache(tokens):
    for past in (0, tokens, 2 * tokens):
        spec = state_causal_sparsity(3 * tokens, tokens, past)
        query = torch.arange(past, past + 3 * tokens) // tokens
        keys = torch.arange(past + 3 * tokens) // tokens
        expected = keys[None, :] <= query[:, None]
        assert torch.equal(segmented_dense_mask(spec)[0, 0], expected)
        # A query shard can start/end in the middle of a state.
        sliced = slice_query_sparsity(spec, 1, tokens + 1)
        assert torch.equal(segmented_dense_mask(sliced)[0, 0], expected[1 : tokens + 2])


@pytest.mark.parametrize("tokens", [3, 73, 137, 204])
def test_prefix_and_tf_match_existing_dense_reference(tokens):
    expected = _build_prefix_block_causal_attn_mask(
        5 * tokens, 1, 2 * tokens, torch.device("cpu"), torch.float32
    )
    assert torch.equal(
        segmented_dense_mask(prefix_sparsity(5 * tokens, 2 * tokens)),
        torch.isfinite(expected),
    )
    expected = _build_full_history_teacher_forcing_attn_mask(
        6 * tokens, 1, tokens, torch.device("cpu"), torch.float32
    )
    assert torch.equal(
        segmented_dense_mask(teacher_forcing_sparsity(6 * tokens, tokens)),
        torch.isfinite(expected),
    )


def compare(model, inputs, kwargs, module):
    reference = copy.deepcopy(model)
    torch.manual_seed(61)
    expected = reference(inputs, **kwargs)
    weight = torch.randn_like(expected)
    (expected * weight).sum().backward()
    with patch(module + ".use_segmented_attention", return_value=True):
        torch.manual_seed(61)
        actual = model(inputs, **kwargs)
        (actual * weight).sum().backward()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for (name, actual), (_, expected) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        if expected.grad is None:
            assert actual.grad is None, name
        else:
            torch.testing.assert_close(actual.grad, expected.grad, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize(
    "domain,patch_size,ratio", [("rgb", 2, 0), ("rgb", 4, 0), ("latent", 2, 0), ("latent", 2, 0.8)]
)
def test_conditioner_segment_compilation_preserves_output_and_all_gradients(
    domain, patch_size, ratio
):
    geometry = CodecGeometry("test", "test", 4, 4, 2, grid=4)
    config = ConditionerConfig(
        input=domain,
        input_size=32,
        temporal_patch=patch_size,
        width=48,
        heads=4,
        depth=2,
        feature_layers=(0, 1),
        context_mask_ratio=ratio,
    )
    model = CausalViT(config, geometry)
    source = torch.randn(2, 3, 12, 32, 32) if domain == "rgb" else torch.randn(2, 8, 3, 4, 4)
    compare(model, source, {}, "vforge.models.rep_world.conditioner")


@pytest.mark.parametrize("mode", ["prefix", "teacher_forcing", "causal"])
def test_world_segment_compilation_preserves_output_and_all_gradients(mode):
    model = WorldDiT(WorldConfig(4, 2, grid=4, width=48, heads=4, depth=2, condition_width=0))
    with torch.no_grad():
        model.final_layer.linear.weight.normal_(std=0.02)
        model.shared_mod[-1].weight.normal_(std=0.02)
    frames = 8 if mode == "teacher_forcing" else 4
    kwargs = dict(timestep=torch.rand(2))
    kwargs[
        {
            "prefix": "prefix_frames",
            "teacher_forcing": "teacher_forcing_planes",
            "causal": "causal_state_planes",
        }[mode]
    ] = 2
    compare(model, torch.randn(2, 4, frames, 4, 4), kwargs, "vforge.models.rep_world.world")


@pytest.mark.parametrize(
    "experiment", ["rep_world", "rep_world_1d", "rep_world_eupe", "rep_world_vjepa"]
)
def test_sdpa_default_and_explicit_fa4_override(experiment):
    from hydra import compose, initialize_config_module

    from vforge.tasks.common import build_and_snapshot

    with initialize_config_module(config_module="vforge.conf", version_base=None):
        config = compose(config_name="train", overrides=[f"experiment={experiment}"])
        assert config.attention.backend == "sdpa"
        assert config.attention.allow_fallback is False
        build_and_snapshot(config, world_size=1, write_snapshot=False)
        config = compose(
            config_name="train",
            overrides=[
                f"experiment={experiment}",
                "attention.backend=flash_attention_4",
                "runtime.deterministic=true",
            ],
        )
        assert config.attention.backend == "flash_attention_4"
        assert config.runtime.deterministic is True
        build_and_snapshot(config, world_size=1, write_snapshot=False)


@pytest.mark.parametrize("segmented_mask", [False, True])
def test_cached_parallel_prefill_keeps_local_output_length(monkeypatch, segmented_mask):
    from vforge.models.rep_world import attention as module

    layer = module.SelfAttention(48, 4, qk_norm=False)
    local = torch.randn(1, 2, 48)
    spec = state_causal_sparsity(4, 2)
    dense = torch.tensor(
        [
            [True, True, False, False],
            [True, True, False, False],
            [True, True, True, True],
            [True, True, True, True],
        ]
    )
    mask = spec if segmented_mask else dense

    def token_range(global_length):
        assert global_length == 4
        return 2, 2

    def attend(q, k, v, **kwargs):
        assert q.shape == (1, 2, 4, 12)
        assert k.shape == v.shape == (1, 4, 4, 12)
        assert kwargs["distributed"] is False
        if segmented_mask:
            actual = kwargs["sparsity"]
            assert actual.q_segment_ids == spec.q_segment_ids[2:]
            assert actual.kv_segment_ids == spec.kv_segment_ids
        else:
            torch.testing.assert_close(kwargs["attn_mask"], dense[2:])
        return torch.zeros_like(q)

    monkeypatch.setattr(module, "token_range", token_range)
    monkeypatch.setattr(module, "gather_tokens", lambda x, dim: torch.cat((x, x), dim=dim))
    monkeypatch.setattr(module, "attention", attend)
    output, current = layer(
        local, rope=lambda x: x, mask=mask, return_kv=True, sequence_parallel=True
    )
    assert output.shape == local.shape
    assert current[0].shape == current[1].shape == (1, 4, 4, 12)
