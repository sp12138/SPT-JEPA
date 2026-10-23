from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, WorldConfig
from vforge.models.rep_world.model import FlowConfig, RepWorld, pack_states, unpack_states
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


def build(mode, domain="rgb", patch=2):
    geo = CodecGeometry("vrae", "test", 4, 4, 2, grid=2)
    vit = ConditionerConfig(
        input=domain,
        input_size=32,
        width=24,
        depth=2,
        heads=4,
        temporal_patch=patch,
        feature_layers=(1,),
    )
    world = WorldConfig(
        4,
        2,
        grid=2,
        width=48,
        heads=4,
        depth=2,
        condition_width=0 if mode == "pure_tf" else 24,
        condition_dropout=0,
    )
    model = RepWorld(geo, world, None if mode == "pure_tf" else vit, FlowConfig(mode=mode))
    with torch.no_grad():
        model.net.final_layer.linear.weight.normal_(std=0.02)
    return model


@pytest.mark.parametrize(
    "mode,domain,patch",
    [
        ("no_zprev", "rgb", 2),
        ("no_zprev", "rgb", 4),
        ("no_zprev", "latent", 2),
        ("zprev", "rgb", 2),
        ("pure_tf", "rgb", 2),
    ],
)
def test_all_target_fm_batching_and_live_conditioner(mode, domain, patch):
    torch.manual_seed(923)
    model = build(mode, domain, patch)
    batch, length = 2, 4
    states = torch.randn(batch, length, 4, 2, 2, 2)
    latents = pack_states(states).requires_grad_()
    rgb = torch.randn(batch, 3, length * 4, 32, 32, requires_grad=True)
    if mode == "pure_tf":
        noise = torch.randn(batch, 4, length - 1, 2, 2, 2)
        uniform = torch.rand(batch, length - 1).clamp(0.01, 0.99)
    else:
        noise = torch.randn(batch * (length - 1), 4, 2, 2, 2)
        uniform = torch.rand(batch * (length - 1), 1).clamp(0.01, 0.99)
    loss = model(latents, rgb, noise=noise, uniform=uniform)
    assert loss.ndim == 0 and torch.isfinite(loss)
    t = model.draw_timesteps(uniform.shape, latents, uniform)
    if mode != "pure_tf":
        source = rgb if domain == "rgb" else latents
        context = model.causal_semantic_encoder.training_context(source)
        context = context.reshape(batch * (length - 1), -1, 24)
        targets = states[:, 1:].reshape_as(noise)
        noisy = (1 - t.view(-1, 1, 1, 1, 1)) * targets + t.view(-1, 1, 1, 1, 1) * noise
        previous = states[:, :-1].reshape_as(noise) if mode == "zprev" else None
        reference = model.velocity(noisy, t, context, previous)
        expected = (targets - noise - reference).square().mean()
        torch.testing.assert_close(loss, expected, atol=0, rtol=0)
    loss.backward()
    assert latents.grad is None
    if mode == "pure_tf":
        assert model.causal_semantic_encoder is None and rgb.grad is None
    else:
        assert model.causal_semantic_encoder.encoder.blocks[0].attn.qkv.weight.grad.abs().sum() > 0
        if domain == "rgb":
            assert rgb.grad[:, :, :-4].abs().sum() > 0
            assert torch.count_nonzero(rgb.grad[:, :, -4:]) == 0
        else:
            assert rgb.grad is None
            assert model.causal_semantic_encoder.latent_patch_embed.weight.grad.abs().sum() > 0
    assert model.net.final_layer.linear.weight.grad.abs().sum() > 0


def test_logit_normal_shift_and_exact_pack_roundtrip():
    model = build("no_zprev")
    x = torch.randn(2, 4, 4, 2, 2, 2)
    torch.testing.assert_close(unpack_states(pack_states(x), model.geometry), x, atol=0, rtol=0)
    u = torch.tensor([0.01, 0.2, 0.5, 0.9, 0.99])
    warped = torch.sigmoid(0.8 + 0.8 * torch.special.ndtri(u.double())).float()
    shift = 2.667 * (4 / 600) ** 0.5
    expected = shift * warped / (1 + (shift - 1) * warped)
    torch.testing.assert_close(model.draw_timesteps(u.shape, x, u), expected, atol=0, rtol=0)
    model.flow = replace(model.flow, std=0)
    torch.testing.assert_close(
        model.draw_timesteps(u.shape, x, u), shift * u / (1 + (shift - 1) * u), atol=0, rtol=0
    )


def test_invalid_latent_zprev_is_rejected_instead_of_changing_baseline():
    with pytest.raises(ValueError, match="no_zprev only"):
        build("zprev", "latent")


@pytest.mark.parametrize(
    "mode,domain",
    [("no_zprev", "rgb"), ("no_zprev", "latent"), ("zprev", "rgb"), ("pure_tf", "rgb")],
)
def test_activation_checkpoint_matches_loss_gradients_and_passes_native_context(mode, domain):
    torch.manual_seed(53)
    original = build(mode, domain)
    checked = deepcopy(original)
    events = []

    @contextmanager
    def record(name):
        events.append(name)
        yield

    checked.gradient_checkpointing_enable(
        dict(context_fn=lambda: (record("forward"), record("recompute")))
    )
    latents = pack_states(torch.randn(1, 3, 4, 2, 2, 2))
    rgb = torch.randn(1, 3, 12, 32, 32)
    outputs = []
    for model in (original, checked):
        torch.manual_seed(93)
        value = model(latents, rgb)
        value.backward()
        outputs.append(value)
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    for (name, a), (_, b) in zip(
        original.named_parameters(), checked.named_parameters(), strict=True
    ):
        if a.grad is None:
            assert b.grad is None, name
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0, msg=name)
    assert "forward" in events and "recompute" in events
    checked.gradient_checkpointing_disable()
    assert not checked.net.gradient_checkpointing
