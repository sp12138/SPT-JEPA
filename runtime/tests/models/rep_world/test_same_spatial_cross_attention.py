"""Same-spatial CA: independent dense mask reference and routing contracts."""

import copy
import dataclasses

import pytest
import torch

from vforge.models.rep_world.attention import CrossAttention
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


def reference(module, x, context, spatial, positions):
    batch, length, width = x.shape
    q = module.q_norm(module.to_q(x).reshape(batch, length, module.num_heads, module.head_dim))
    k, v = module.to_kv(context).reshape(batch, -1, 2, module.num_heads, module.head_dim).unbind(2)
    k = module.k_norm(k)
    if positions is None:
        positions = torch.arange(context.shape[1], device=x.device).expand(batch, -1)
    allowed = (
        torch.arange(length, device=x.device)[None, :, None] % spatial
        == positions[:, None, :] % spatial
    )
    nonempty = allowed.any(-1, keepdim=True)
    safe = allowed | (~nonempty & (torch.arange(context.shape[1], device=x.device) == 0))
    # Independent explicit dense softmax; no grouped attention implementation.
    logits = torch.einsum("blhd,bnhd->bhln", q.to(v.dtype), k.to(v.dtype)) / module.head_dim**0.5
    probabilities = (
        logits.float().masked_fill(~safe[:, None], float("-inf")).softmax(-1).to(v.dtype)
    )
    out = torch.einsum("bhln,bnhd->blhd", probabilities, v).reshape(batch, length, width)
    return module.proj(out) * nonempty.to(out.dtype)


@pytest.mark.parametrize("sparse", [False, True])
def test_matches_independent_dense_mask_output_and_gradients(sparse):
    torch.manual_seed(9)
    module = CrossAttention(32, 4, True)
    other = copy.deepcopy(module)
    positions = torch.tensor([[0, 4, 8, 1, 5], [2, 6, 10, 3, 7]]) if sparse else None
    n = 5 if sparse else 12
    x = torch.randn(2, 12, 32, requires_grad=True)
    context = torch.randn(2, n, 32, requires_grad=True)
    xx, cc = x.detach().clone().requires_grad_(), context.detach().clone().requires_grad_()
    actual = module(x, context, spatial_size=4, context_positions=positions)
    expected = reference(other, xx, cc, 4, positions)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    upstream = torch.randn_like(actual)
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    for a, b in [(x.grad, xx.grad), (context.grad, cc.grad)] + [
        (a.grad, b.grad) for a, b in zip(module.parameters(), other.parameters(), strict=True)
    ]:
        torch.testing.assert_close(a, b, rtol=2e-4, atol=1e-5)
    if sparse:
        assert torch.equal(actual[0, [2, 3, 6, 7, 10, 11]], torch.zeros(6, 32))


def test_forbidden_keys_have_zero_influence_and_zero_gradient():
    torch.manual_seed(1)
    module = CrossAttention(32, 4, True)
    x = torch.randn(1, 12, 32)
    c = torch.randn(1, 12, 32, requires_grad=True)
    first = module(x, c, spatial_size=4)
    changed = c.detach().clone()
    changed[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]] += torch.randn(1, 9, 32) * 30
    second = module(x, changed, spatial_size=4)
    torch.testing.assert_close(first[:, [0, 4, 8]], second[:, [0, 4, 8]], rtol=0, atol=0)
    first[:, [0, 4, 8]].square().sum().backward()
    assert torch.count_nonzero(c.grad[:, [1, 2, 3, 5, 6, 7, 9, 10, 11]]) == 0
    assert c.grad[:, [0, 4, 8]].abs().sum() > 0


def make_model(scope):
    geo = CodecGeometry("test", "test", 4, 8, 4, grid=4)
    enc = ConditionerConfig(
        input="latent", width=32, depth=2, heads=4, feature_layers=(1,), context_mask_ratio=0.8
    )
    world = WorldConfig(
        channels=8,
        state_planes=4,
        grid=4,
        width=24,
        depth=2,
        heads=3,
        condition_width=32,
        adaln_lora_dim=8,
        cross_attention_every=2,
        cross_attention_position="before_self",
        cross_attention_scope=scope,
    )
    return RepWorld(geo, world, enc, FlowConfig())


def test_no_parameter_or_initialization_change():
    torch.manual_seed(42)
    global_model = make_model("global")
    torch.manual_seed(42)
    local_model = make_model("same_spatial")
    assert global_model.state_dict().keys() == local_model.state_dict().keys()
    for name, value in global_model.state_dict().items():
        torch.testing.assert_close(value, local_model.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("checkpoint", [False, True])
def test_sparse_training_dense_inference_and_null(checkpoint):
    torch.manual_seed(42)
    model = make_model("same_spatial").train()
    if checkpoint:
        model.gradient_checkpointing_enable()
    torch.nn.init.normal_(model.net.final_layer.linear.weight, std=0.02)
    latents = torch.randn(2, 32, 3, 4, 4)
    captured = []
    h = model.net.blocks[0].cross_attn.register_forward_pre_hook(
        lambda m, args, kw: captured.append(kw), with_kwargs=True
    )
    loss = model(latents=latents, condition_drop=torch.tensor([False, True, False, True]))
    loss.backward()
    h.remove()
    assert torch.isfinite(loss)
    assert captured[0]["context_positions"].shape == (4, 12)
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    assert model.causal_semantic_encoder.latent_patch_embed.weight.grad.abs().sum() > 0
    model.eval()
    condition = model(operation="condition", source=latents[:, :, :1])
    noisy = torch.randn(2, 8, 4, 4, 4)
    t = torch.full((2, 1), 0.5)
    for c in [condition, None]:
        assert torch.isfinite(
            model(operation="velocity", noisy=noisy, timestep=t, condition=c)
        ).all()
    drop = torch.ones(2, dtype=torch.bool)
    a = model(
        operation="velocity", noisy=noisy, timestep=t, condition=condition, condition_drop=drop
    )
    b = model(
        operation="velocity", noisy=noisy, timestep=t, condition=condition * 17, condition_drop=drop
    )
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_sparse_permutation_uses_original_coordinates():
    module = CrossAttention(32, 4, True)
    x, c = torch.randn(1, 12, 32), torch.randn(1, 5, 32)
    pos = torch.tensor([[0, 4, 8, 1, 5]])
    order = torch.tensor([3, 0, 4, 2, 1])
    a = module(x, c, spatial_size=4, context_positions=pos)
    b = module(x, c[:, order], spatial_size=4, context_positions=pos[:, order])
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_bad_coordinates_and_unsupported_config_fail_loudly():
    model = make_model("same_spatial")
    args = (torch.randn(1, 8, 4, 4, 4), torch.ones(1, 1))
    c = torch.randn(1, 4, 32)
    for positions, match in [
        (torch.zeros(1, 4, dtype=torch.long), "unique"),
        (torch.full((1, 4), 64), "outside"),
    ]:
        with pytest.raises(ValueError, match=match):
            model.net(*args, condition=c, condition_positions=positions)
    with pytest.raises(ValueError, match="explicit"):
        model.net(*args, condition=c)
    with pytest.raises(ValueError, match="patch_size"):
        dataclasses.replace(model.net.config, patch_size=2)
    with pytest.raises(ValueError, match="spatially ordered"):
        RepWorld(
            dataclasses.replace(model.geometry, spatially_ordered=False),
            model.net.config,
            model.causal_semantic_encoder.config,
            FlowConfig(),
        )



def test_bfloat16_autocast_sparse_scatter_and_backward():
    module = CrossAttention(32, 4, True)
    x = torch.randn(2, 12, 32, requires_grad=True)
    c = torch.randn(2, 5, 32, requires_grad=True)
    pos = torch.tensor([[0, 4, 8, 1, 5], [2, 6, 10, 3, 7]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = module(x, c, spatial_size=4, context_positions=pos)
    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()
    out.float().square().mean().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(c.grad).all()
