"""K=4 future-state mosaics: exact source coordinates and temporal RoPE."""

import copy
from dataclasses import replace

import pytest
import torch

from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, WorldConfig
from vforge.models.rep_world.horizon import (
    quadrant_future_targets,
    sample_quadrant_offsets,
    target_gap_embedding,
)
from vforge.models.rep_world.model import FlowConfig, RepWorld, unpack_states
from vforge.ops import OpsConfig, initialize_ops, reset_ops


@pytest.fixture(autouse=True)
def cpu_ops():
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa_math", allow_fallback=False),
        device=torch.device("cpu"),
        parallel_state=None,
    )
    yield
    reset_ops()


def make_model(mode="entry_concat", *, mask=0.0, planes=4, patch_size=1):
    geo = CodecGeometry("test", "test", 16, 8, planes, grid=4)
    enc = ConditionerConfig(
        input="latent",
        width=24,
        depth=2,
        heads=3,
        feature_layers=(1,),
        context_mask_ratio=mask,
    )
    cfg = WorldConfig(
        channels=8,
        state_planes=planes,
        grid=4,
        width=24,
        depth=2,
        heads=3,
        condition_width=24,
        condition_dropout=0,
        adaln_lora_dim=8,
        patch_size=patch_size,
        condition_mode="entry_concat" if mode == "entry_concat" else "cross_attention",
        cross_attention_scope="same_spatial" if mode == "same_spatial" else "global",
        cross_attention_3d_rope=mode in {"global", "same_spatial"},
        cross_attention_position="before_self",
        cross_attention_every=2,
    )
    return RepWorld(geo, cfg, enc, FlowConfig(target_layout="quadrant_mosaic"))


def test_per_origin_permutations_and_rng_replay():
    torch.manual_seed(47)
    state = torch.get_rng_state()
    a = sample_quadrant_offsets((8, 6), "cpu")
    assert torch.equal(a.sort(-1).values, torch.arange(4).expand(8, 6, 4))
    assert a.flatten(0, 1).unique(dim=0).shape[0] > 1
    torch.set_rng_state(state)
    assert torch.equal(a, sample_quadrant_offsets((8, 6), "cpu"))


@pytest.mark.parametrize("patch", [1, 2])
def test_source_gather_keeps_every_channel_plane_and_spatial_coordinate(patch):
    # Each scalar uniquely labels its input coordinates; no spatial resizing or relocation.
    states = torch.arange(2 * 7 * 3 * 4 * 4 * 8).reshape(2, 7, 3, 4, 4, 8)
    assignment = torch.tensor(
        [
            [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2]],
            [[2, 0, 3, 1], [1, 2, 3, 0], [3, 1, 2, 0]],
        ]
    )
    actual, token_offsets, pixel_offsets = quadrant_future_targets(states, assignment, patch)
    expected = torch.empty_like(actual)
    expected_offsets = torch.empty(2, 3, 4, 8, dtype=torch.long)
    for b in range(2):
        for i in range(3):
            for q in range(4):
                y = slice((q // 2) * 2, (q // 2 + 1) * 2)
                x = slice((q % 2) * 4, (q % 2 + 1) * 4)
                gap = int(assignment[b, i, q])
                expected[b, i, :, :, y, x] = states[b, i + 1 + gap, :, :, y, x]
                expected_offsets[b, i, y, x] = gap
    assert torch.equal(actual, expected)
    assert torch.equal(pixel_offsets, expected_offsets.flatten(0, 1))
    expected_tokens = (
        expected_offsets[..., ::patch, ::patch][:, :, None]
        .expand(2, 3, 4, 4 // patch, 8 // patch)
        .reshape(6, -1)
    )
    assert torch.equal(token_offsets, expected_tokens)


@pytest.mark.parametrize(
    "assignment",
    [
        torch.tensor([[[0, 0, 2, 3]]]),
        torch.tensor([[[-1, 1, 2, 3]]]),
        torch.tensor([[[0, 1, 2, 4]]]),
        torch.tensor([[[0.0, 1.0, 2.0, 3.0]]]),
        torch.tensor([[0, 1, 2, 3]]),
    ],
)
def test_invalid_quadrant_assignments(assignment):
    with pytest.raises(ValueError, match="permutations"):
        sample_quadrant_offsets((1, 1), "cpu", offsets=assignment)


def test_invalid_layout_geometry_and_context_budget():
    with pytest.raises(ValueError, match="target_layout"):
        FlowConfig(target_layout="other")
    with pytest.raises(ValueError, match="distant_target_prob"):
        FlowConfig(target_layout="quadrant_mosaic", distant_target_prob=0.25)
    with pytest.raises(ValueError, match="no_zprev"):
        FlowConfig(target_layout="quadrant_mosaic", mode="pure_tf")
    model = make_model()
    assert model.flow.target_count(10) == 6
    with pytest.raises(ValueError, match="at least 5"):
        model(latents=torch.randn(1, 32, 4, 4, 4))
    with pytest.raises(ValueError, match="spatially ordered"):
        RepWorld(
            replace(model.geometry, spatially_ordered=False),
            model.net.config,
            model.causal_semantic_encoder.config,
            model.flow,
        )
    with pytest.raises(ValueError, match="even DiT"):
        RepWorld(
            replace(model.geometry, grid=3),
            replace(model.net.config, grid=3),
            model.causal_semantic_encoder.config,
            model.flow,
        )
    with pytest.raises(ValueError, match="even DiT"):
        quadrant_future_targets(torch.randn(1, 5, 2, 1, 6, 6), torch.tensor([[[0, 1, 2, 3]]]), 2)


@pytest.mark.parametrize("mode", ["entry_concat", "global", "same_spatial", "no_ca_rope"])
@pytest.mark.parametrize("mask", [0.0, 0.8])
def test_mosaic_fm_uses_correct_targets_and_causal_prefix(mode, mask):
    model = make_model(mode, mask=mask).train()
    torch.manual_seed(9)
    latent = torch.randn(2, 32, 7, 4, 4)
    assignment = torch.tensor(
        [
            [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2]],
            [[2, 0, 3, 1], [1, 2, 3, 0], [3, 1, 2, 0]],
        ]
    )
    target, offsets, _ = quadrant_future_targets(unpack_states(latent, model.geometry), assignment)
    target = target.flatten(0, 1)
    noise = torch.randn_like(target)
    uniform = torch.full((6, 1), 0.5)
    captured = []
    h = model.net.register_forward_pre_hook(
        lambda m, a, kw: captured.append((a, kw)), with_kwargs=True
    )
    torch.manual_seed(51)
    result = model(
        latents=latent,
        quadrant_offsets=assignment,
        noise=noise,
        uniform=uniform,
        return_metrics=True,
    )
    args, kw = captured[0]
    t = model.draw_timesteps((6, 1), target, uniform).reshape(-1, 1, 1, 1, 1)
    torch.testing.assert_close(args[0], (1 - t) * target + t * noise, rtol=0, atol=0)
    assert torch.equal(kw["target_offsets"], offsets)
    torch.testing.assert_close(result["loss"], (target - noise).square().mean(), rtol=0, atol=0)
    metrics = result["metrics"]
    for gap in range(1, 5):
        assert metrics[f"target_horizon_{gap}_fraction"] == 0.25
    torch.testing.assert_close(
        sum(metrics[f"loss_fm_horizon_{g}_contribution"] for g in range(1, 5)), result["loss"]
    )
    modified = latent.clone()
    modified[:, :, 1:] += 100
    torch.manual_seed(51)
    model(latents=modified, quadrant_offsets=assignment, noise=noise, uniform=uniform)
    c0 = captured[0][1]["condition"].reshape(2, 3, -1, 24)[:, 0]
    c1 = captured[1][1]["condition"].reshape(2, 3, -1, 24)[:, 0]
    torch.testing.assert_close(c0, c1, rtol=0, atol=0)
    h.remove()


def independent_rotation(value, indices, time, side):
    # Independent complex rotations, using actual t/h/w coordinates.
    dim = value.shape[-1]
    spatial = 2 * ((dim // 3) // 2)
    widths = (dim - 2 * spatial, spatial, spatial)
    coordinates = (time, indices // side % side, indices % side)
    parts = []
    start = 0
    for width, coord in zip(widths, coordinates, strict=True):
        z = torch.view_as_complex(
            value[..., start : start + width].reshape(*value.shape[:-1], width // 2, 2).contiguous()
        )
        freq = 10000.0 ** (-torch.arange(width // 2).float() * 2 / width)
        angle = coord.float()[..., None] * freq
        phase = torch.polar(torch.ones_like(angle), angle)[:, None]
        parts.append(torch.view_as_real(z * phase).flatten(-2))
        start += width
    return torch.cat(parts, -1)


@pytest.mark.parametrize("mode", ["entry_concat", "global", "same_spatial", "no_ca_rope"])
def test_all_sa_and_ca_rope_match_true_quadrant_time_and_gap_embedding(mode):
    model = make_model(mode).eval()
    x = torch.randn(1, 8, 4, 4, 4)
    c = torch.randn(1, 64, 24)
    _, offsets, _ = quadrant_future_targets(
        torch.randn(1, 5, 8, 4, 4, 4), torch.tensor([[[2, 0, 3, 1]]])
    )
    blocks = []
    hooks = [
        b.register_forward_pre_hook(
            lambda m, a, kw: blocks.append((a[0].detach().clone(), kw)), with_kwargs=True
        )
        for b in model.net.blocks
    ]
    base = []
    hb = model.net.blocks[0].register_forward_pre_hook(
        lambda m, a: base.append(a[0].detach().clone())
    )
    model(operation="velocity", noisy=x, timestep=torch.ones(1, 1) * 0.5, condition=c)
    base_hidden = base[-1]
    blocks.clear()
    model(
        operation="velocity",
        noisy=x,
        timestep=torch.ones(1, 1) * 0.5,
        condition=c,
        target_offsets=offsets,
    )
    for h in hooks + [hb]:
        h.remove()
    expected_gap = target_gap_embedding(offsets, 24, blocks[0][0].dtype)
    torch.testing.assert_close(blocks[0][0] - base_hidden, expected_gap, rtol=1e-5, atol=3e-7)
    q = torch.randn(1, 3, 64, 8)
    ids = torch.arange(64).expand(1, -1)
    times = offsets * 4 + ids // 16
    assert times.max() == 15 and times.min() == 0
    expected = independent_rotation(q, ids, times, 4)
    for _, kwargs in blocks:
        torch.testing.assert_close(kwargs["rope"](q), expected, rtol=2e-5, atol=2e-6)
        if model.net.config.cross_attention_3d_rope:
            qr, kr = kwargs["cross_rope"]
            expected_q = independent_rotation(q, ids, times + 4, 4)
            torch.testing.assert_close(qr(q), expected_q, rtol=2e-5, atol=2e-6)
            expected_k = independent_rotation(q, ids, ids // 16, 4)
            torch.testing.assert_close(kr(q), expected_k, rtol=2e-5, atol=2e-6)
    # Ignoring quadrant times must measurably change cross-quadrant QK scores.
    local = model.net.feat_rope(q, num_frames_override=4)
    actual = blocks[0][1]["rope"](q)
    assert (actual @ actual.transpose(-1, -2) - local @ local.transpose(-1, -2)).abs().max() > 1e-3


@pytest.mark.parametrize("mode", ["entry_concat", "global", "same_spatial"])
def test_checkpoint_recompute_gradients_and_ordinary_rollout(mode):
    torch.manual_seed(18)
    a = make_model(mode, mask=0.8).train()
    torch.nn.init.normal_(a.net.final_layer.linear.weight, std=0.02)
    b = copy.deepcopy(a)
    b.gradient_checkpointing_enable()
    latent = torch.randn(2, 32, 7, 4, 4)
    for m in [a, b]:
        torch.manual_seed(83)
        loss = m(latents=latent)
        loss.backward()
        grad = m.causal_semantic_encoder.latent_patch_embed.weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0, msg=name)
    a.eval()
    c = a(operation="condition", source=latent[:, :, :1])
    value = a(
        operation="velocity",
        noisy=torch.randn(2, 8, 4, 4, 4),
        timestep=torch.ones(2, 1) * 0.5,
        condition=c,
    )
    assert value.shape == (2, 8, 4, 4, 4) and torch.isfinite(value).all()


@pytest.mark.parametrize("mode", ["entry_concat", "global", "same_spatial", "no_ca_rope"])
def test_zero_offset_map_is_ordinary_prediction(mode):
    torch.manual_seed(63)
    model = make_model(mode).eval()
    torch.nn.init.normal_(model.net.final_layer.linear.weight, std=0.02)
    x = torch.randn(2, 8, 4, 4, 4)
    condition = torch.randn(2, 64, 24)
    t = torch.full((2, 1), 0.5)
    plain = model(operation="velocity", noisy=x, timestep=t, condition=condition)
    zero = model(
        operation="velocity",
        noisy=x,
        timestep=t,
        condition=condition,
        target_offsets=torch.zeros(2, 64, dtype=torch.long),
    )
    torch.testing.assert_close(zero, plain, rtol=0, atol=0)
