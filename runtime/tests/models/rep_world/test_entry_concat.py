"""Spatial concat tests: topology, exact position recovery, gradients and causal masking."""

import dataclasses

import pytest
import torch

from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, WorldConfig
from vforge.models.rep_world.model import FlowConfig, RepWorld
from vforge.ops import OpsConfig, initialize_ops, reset_ops


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


def make_model(mode="entry_concat", ratio=0.8):
    geo = CodecGeometry("test", "test", 4, 8, 2, grid=4)
    enc = ConditionerConfig(
        input="latent", width=32, depth=2, heads=4, feature_layers=(1,), context_mask_ratio=ratio
    )
    world = WorldConfig(
        channels=8,
        state_planes=2,
        grid=4,
        width=24,
        depth=2,
        heads=3,
        condition_width=32,
        condition_dropout=0,
        condition_mode=mode,
        adaln_lora_dim=8,
    )
    return RepWorld(geo, world, enc, FlowConfig())


def test_concat_does_not_instantiate_cross_attention_or_unused_patch_projection():
    model = make_model()
    assert all(block.cross_attn is None for block in model.net.blocks)
    assert isinstance(model.net.x_embedder.proj, torch.nn.Identity)
    assert model.net.semantic_entry_fusion[0].in_features == 8 + 24
    assert all(block.cross_attn is not None for block in make_model("cross_attention").net.blocks)


def test_concat_scatter_is_exact_and_masks_after_projection():
    model = make_model().train()
    world = model.net
    torch.manual_seed(12)
    condition = torch.randn(2, 6, 32, requires_grad=True)
    positions = torch.tensor([[0, 3, 6, 17, 22, 31], [1, 5, 9, 16, 18, 30]])
    noisy = torch.randn(2, 8, 2, 4, 4)
    captured = []
    handle = world.semantic_entry_fusion.register_forward_pre_hook(
        lambda m, args: captured.append(args[0])
    )
    world(noisy, torch.ones(2, 1) * 0.5, condition=condition, condition_positions=positions)
    handle.remove()
    fused = captured[0]
    assert torch.equal(fused[..., :8], noisy.flatten(2).transpose(1, 2))
    projected = world.semantic_raw_projector(condition)
    expected = projected.new_zeros(2, 32, 24).scatter(
        1, positions[..., None].expand(-1, -1, 24), projected
    )
    torch.testing.assert_close(fused[..., 8:], expected, rtol=0, atol=0)
    fused[..., 8:].sum().backward()
    assert condition.grad is not None and condition.grad.abs().sum() > 0


def test_mask_indices_preserve_state_local_positions_and_no_future_leak():
    enc = make_model().causal_semantic_encoder.train()
    source = torch.randn(2, 16, 4, 4, 4)
    torch.manual_seed(11)
    features, idx = enc.training_context(source, return_indices=True)
    assert features.shape == (2, 3, 6, 32)
    assert idx.shape == (2, 3, 6)
    assert torch.all(idx[..., :3] < 16) and torch.all(idx[..., 3:] >= 16)
    altered = source.clone()
    altered[:, :, 2:] += 100
    torch.manual_seed(11)
    other, other_idx = enc.training_context(altered, return_indices=True)
    assert torch.equal(idx, other_idx)
    torch.testing.assert_close(features[:, :2], other[:, :2], atol=0, rtol=0)
    enc.eval()
    dense, idx = enc.training_context(source, return_indices=True)
    assert dense.shape == (2, 3, 32, 32) and idx is None


@pytest.mark.parametrize("checkpoint", [False, True])
def test_full_masked_loss_backward_and_dense_inference(checkpoint):
    model = make_model().train()
    if checkpoint:
        model.gradient_checkpointing_enable()
    # Zero output head is intentional; open it to check all conditioner paths.
    torch.nn.init.normal_(model.net.final_layer.linear.weight, std=0.02)
    latents = torch.randn(2, 16, 4, 4, 4)
    loss = model(latents=latents)
    assert torch.isfinite(loss)
    loss.backward()
    for p in [
        model.net.semantic_entry_fusion[0].weight,
        model.net.semantic_raw_projector[1].weight,
        model.causal_semantic_encoder.latent_patch_embed.weight,
    ]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    model.eval()
    condition = model(operation="condition", source=latents[:, :, :1])
    assert condition.shape == (2, 32, 32)
    result = model(
        operation="velocity",
        noisy=torch.randn(2, 8, 2, 4, 4),
        timestep=torch.full((2, 1), 0.5),
        condition=condition,
    )
    assert result.shape == (2, 8, 2, 4, 4) and torch.isfinite(result).all()


def test_concat_rejects_missing_or_duplicate_positions():
    world = make_model().net
    q = torch.randn(1, 8, 2, 4, 4)
    t = torch.ones(1, 1)
    c = torch.randn(1, 6, 32)
    with pytest.raises(ValueError, match="dense tokens"):
        world(q, t, condition=c)
    with pytest.raises(ValueError, match="unique"):
        world(q, t, condition=c, condition_positions=torch.zeros(1, 6, dtype=torch.long))


def test_concat_config_rejects_nonspatial_or_prev():
    model = make_model()
    with pytest.raises(ValueError, match="no_zprev"):
        RepWorld(
            model.geometry,
            model.net.config,
            model.causal_semantic_encoder.config,
            FlowConfig(mode="zprev"),
        )
    with pytest.raises(ValueError, match="patch_size"):
        dataclasses.replace(model.net.config, patch_size=2)
