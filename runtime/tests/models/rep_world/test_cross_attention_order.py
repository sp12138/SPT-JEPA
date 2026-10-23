"""Cross-attention ordering, prefix visibility and gradients."""

from dataclasses import replace

import pytest
import torch

from vforge.models.rep_world.config import WorldConfig
from vforge.models.rep_world.world import WorldBlock, WorldDiT
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


def config(position="after_self"):
    return WorldConfig(
        channels=8,
        state_planes=1,
        grid=2,
        width=24,
        depth=3,
        heads=3,
        condition_width=24,
        adaln_mode="fully_shared",
        cross_attention_every=3,
        cross_attention_position=position,
    )


@pytest.mark.parametrize(
    "position,expected", [("before_self", ["CA", "SA", "FFN"]), ("after_self", ["SA", "CA", "FFN"])]
)
@pytest.mark.parametrize("prefix", [0, 2])
def test_order_prefix_and_conditioned_kv(position, expected, prefix):
    torch.manual_seed(42)
    block = WorldBlock(config(position), cross_attention=True)
    calls = []
    for name, module in [("CA", block.cross_attn), ("SA", block.attn), ("FFN", block.mlp)]:
        module.register_forward_pre_hook(lambda m, a, n=name: calls.append(n))
    x = torch.randn(2, 6, 24, requires_grad=True)
    cond = torch.randn(2, 4, 24, requires_grad=True)
    emb = torch.zeros(2, 1, 24)
    shared = torch.randn(2, 1, 144)
    seen = {}
    block.cross_attn.register_forward_pre_hook(lambda m, a: seen.update(ca_input=a[0]))
    block.attn.register_forward_pre_hook(lambda m, a: seen.update(sa_input=a[0]))
    y, kv = block(
        x, emb, shared, rope=lambda q: q, context=cond, prefix_tokens=prefix, return_kv=True
    )
    assert calls == expected
    assert seen["ca_input"].shape[1] == 6 - prefix
    assert y.shape == x.shape and kv[0].shape == (2, 3, 6, 8)
    y.square().sum().backward()
    assert cond.grad is not None and cond.grad.abs().sum() > 0
    assert torch.isfinite(x.grad).all()
    # Direct CA preserves prefix exactly; before-SA cache includes conditioner effects
    # only in target tokens (QKV are computed independently per token).
    z = block._apply_cross_attention(x.detach(), cond.detach(), prefix)
    assert torch.equal(z[:, :prefix], x[:, :prefix])
    _, other_kv = block(
        x.detach(),
        emb,
        shared,
        rope=lambda q: q,
        context=cond.detach() + torch.randn_like(cond),
        prefix_tokens=prefix,
        return_kv=True,
    )
    if position == "before_self":
        assert not torch.equal(kv[0][:, :, prefix:], other_kv[0][:, :, prefix:])
        assert torch.equal(kv[0][:, :, :prefix], other_kv[0][:, :, :prefix])
    else:
        assert torch.equal(kv[0], other_kv[0])


def test_only_first_block_has_ca_and_parameter_layout_unchanged():
    before = WorldDiT(config("before_self"))
    after = WorldDiT(config("after_self"))
    assert [i for i, b in enumerate(before.blocks) if b.cross_attn is not None] == [0]
    assert {k: v.shape for k, v in before.state_dict().items()} == {
        k: v.shape for k, v in after.state_dict().items()
    }
    after.load_state_dict(before.state_dict(), strict=True)


def test_order_does_not_change_blocks_without_ca():
    torch.manual_seed(5)
    a = WorldBlock(config("before_self"), cross_attention=False)
    b = WorldBlock(config("after_self"), cross_attention=False)
    b.load_state_dict(a.state_dict())
    x, emb, shared = torch.randn(2, 4, 24), torch.zeros(2, 1, 24), torch.randn(2, 1, 144)
    torch.testing.assert_close(
        a(x, emb, shared, rope=lambda q: q), b(x, emb, shared, rope=lambda q: q), rtol=0, atol=0
    )


def test_reject_unknown_order():
    with pytest.raises(ValueError, match="cross_attention_position"):
        replace(config(), cross_attention_position="between")


@pytest.mark.parametrize("checkpoint", [False, True])
def test_before_self_full_masked_model_training_and_inference(checkpoint):
    from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig
    from vforge.models.rep_world.model import FlowConfig, RepWorld

    geometry = CodecGeometry("test", "test", 4, 8, 1, grid=2)
    enc = ConditionerConfig(
        input="latent", width=24, depth=2, heads=3, feature_layers=(1,), context_mask_ratio=0.5
    )
    model = RepWorld(
        geometry, replace(config("before_self"), condition_dropout=0), enc, FlowConfig()
    ).train()
    if checkpoint:
        model.gradient_checkpointing_enable()
    torch.nn.init.normal_(model.net.final_layer.linear.weight, std=0.02)
    latent = torch.randn(2, 8, 3, 2, 2)
    loss = model(latents=latent)
    assert torch.isfinite(loss)
    loss.backward()
    for parameter in [
        model.net.blocks[0].cross_attn.to_kv.weight,
        model.causal_semantic_encoder.latent_patch_embed.weight,
    ]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    model.eval()
    cond = model(operation="condition", source=latent[:, :, :1])
    result = model(
        operation="velocity",
        noisy=latent[:, :, :1],
        timestep=torch.full((2, 1), 0.5),
        condition=cond,
    )
    assert result.shape == (2, 8, 1, 2, 2) and torch.isfinite(result).all()
