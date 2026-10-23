"""K=2 and left/right future targets with a matched history budget."""

from dataclasses import replace

import pytest
import torch

from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, WorldConfig
from vforge.models.rep_world.horizon import (
    half_future_targets,
    sample_half_offsets,
    sample_target_offsets,
    sample_token_offsets,
    token_future_targets,
)
from vforge.models.rep_world.model import FlowConfig, RepWorld, unpack_states
from vforge.ops import OpsConfig, initialize_ops, reset_ops


@pytest.fixture(autouse=True)
def cpu_ops(request):
    reset_ops()
    backend = "flash_attention_4" if request.node.get_closest_marker("gpu") else "sdpa_math"
    device = torch.device("cuda" if backend == "flash_attention_4" else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    initialize_ops(
        OpsConfig(backend=backend, allow_fallback=False), device=device, parallel_state=None
    )
    yield
    reset_ops()


def flow(arm):
    return FlowConfig(
        future_target_choices=2,
        target_origin_reserve=2,
        distant_target_prob=0.5 if arm == "random" else 0,
        target_layout={"halves": "left_right_mosaic", "tokens": "random_token_mosaic"}.get(
            arm, "single_state"
        ),
    )


def model_for(arm, split=False, width=24):
    geo = CodecGeometry("test", "test", 16, 8, 4, grid=4)
    enc = ConditionerConfig(
        input="latent",
        width=width,
        depth=2,
        heads=3,
        feature_layers=(1,),
        architecture="split_ec" if split else "causal",
    )
    world = WorldConfig(
        channels=8,
        state_planes=4,
        grid=4,
        width=width,
        depth=2,
        heads=3,
        condition_width=width,
        condition_dropout=0,
        cross_attention_every=2,
        cross_attention_position="before_self",
        cross_attention_3d_rope=True,
        adaln_lora_dim=8,
    )
    return RepWorld(geo, world, enc, flow(arm))


def test_binary_sampling_and_replay():
    u = torch.arange(10000).float() / 10000
    out = sample_target_offsets(u.shape, 0.5, "cpu", choices=2, uniform=u)
    assert out.bincount().tolist() == [5000, 5000]
    state = torch.get_rng_state()
    assert sample_target_offsets((2, 8), 0, "cpu", choices=2).eq(0).all()
    assert torch.equal(state, torch.get_rng_state())
    a = sample_half_offsets((100, 8), "cpu")
    assert torch.equal(a.sort(-1).values, torch.arange(2).expand(100, 8, 2))
    assert set(a[..., 0].flatten().tolist()) == {0, 1}
    torch.set_rng_state(state)
    assert torch.equal(a, sample_half_offsets((100, 8), "cpu"))


@pytest.mark.parametrize("patch", [1, 2])
def test_half_targets_preserve_source_channel_plane_and_spatial_coordinates(patch):
    states = torch.arange(2 * 10 * 3 * 4 * 4 * 8).reshape(2, 10, 3, 4, 4, 8)
    left = torch.arange(16).reshape(2, 8) % 2
    assignment = torch.stack((left, 1 - left), -1)
    actual, tokens, pixels = half_future_targets(states, assignment, patch)
    expected = torch.empty_like(actual)
    expected_pixels = torch.empty(2, 8, 4, 8, dtype=torch.long)
    for b in range(2):
        for i in range(8):
            for side in range(2):
                x = slice(side * 4, (side + 1) * 4)
                offset = int(assignment[b, i, side])
                expected[b, i, :, :, :, x] = states[b, i + 1 + offset, :, :, :, x]
                expected_pixels[b, i, :, x] = offset
    assert torch.equal(actual, expected)
    assert torch.equal(pixels, expected_pixels.flatten(0, 1))
    expected_tokens = (
        expected_pixels[..., ::patch, ::patch][:, :, None]
        .expand(2, 8, 4, 4 // patch, 8 // patch)
        .reshape(16, -1)
    )
    assert torch.equal(tokens, expected_tokens)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("arm", ["baseline", "random", "halves", "tokens"])
def test_all_arms_see_exactly_same_eight_prefixes_and_no_future_leakage(arm, split):
    m = model_for(arm, split).eval()
    z = torch.randn(1, 32, 10, 4, 4)
    captured, source = [], []
    h1 = m.net.register_forward_pre_hook(
        lambda mod, a, kw: captured.append(kw["condition"].detach().clone()),
        with_kwargs=True,
    )
    h2 = m.causal_semantic_encoder.register_forward_pre_hook(
        lambda mod, a: source.append(a[0].detach().clone()),
    )
    fixed = dict(noise=torch.zeros(8, 8, 4, 4, 4), uniform=torch.full((8, 1), 0.5))
    result = m(latents=z, return_metrics=True, **fixed)
    assert captured[0].shape == (8, 64, 24)
    assert source[0].shape[2] == 8 and torch.equal(source[0], z[:, :, :8])
    changed = z.clone()
    changed[:, :, 1:] += 100
    m(latents=changed, **fixed)
    torch.testing.assert_close(captured[0][0], captured[1][0], rtol=0, atol=0)
    for h in [h1, h2]:
        h.remove()
    if arm in {"halves", "tokens"}:
        assert result["metrics"]["target_horizon_1_fraction"] == 0.5
        assert result["metrics"]["target_horizon_2_fraction"] == 0.5
    assert m.flow.target_count(10) == 8


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("arm", ["random", "halves", "tokens"])
def test_tuning_matches_baseline_loss_gradients_and_rng_without_changing_origins(arm, split):
    reference = model_for("baseline", split).train()
    torch.manual_seed(8)
    for p in reference.parameters():
        torch.nn.init.normal_(p, std=0.02)
    actual = model_for(arm, split).train()
    actual.load_state_dict(reference.state_dict(), strict=True)
    z = torch.randn(1, 32, 10, 4, 4)
    fixed = dict(latents=z, noise=torch.randn(8, 8, 4, 4, 4), uniform=torch.full((8, 1), 0.5))
    rng = torch.get_rng_state()
    a = actual(**fixed, next_state_only=True)
    a.backward()
    end = torch.get_rng_state()
    torch.set_rng_state(rng)
    b = reference(**fixed)
    b.backward()
    assert torch.equal(end, torch.get_rng_state())
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for (name, p), q in zip(actual.named_parameters(), reference.parameters(), strict=True):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


@pytest.mark.parametrize("arm", ["halves", "tokens"])
def test_half_rope_preserves_spatial_coordinates_and_true_query_key_time_difference(arm):
    m = model_for(arm).eval()
    z = torch.randn(1, 32, 10, 4, 4)
    assignment = torch.tensor([[[0, 1], [1, 0]] * 4])
    if arm == "tokens":
        assignment = sample_token_offsets((1, 8), 64, "cpu")
        _, offsets, _ = token_future_targets(unpack_states(z, m.geometry), assignment)
        assignments = {"token_assignments": assignment}
    else:
        _, offsets, _ = half_future_targets(unpack_states(z, m.geometry), assignment)
        assignments = {"half_offsets": assignment}
    blocks = []
    hooks = [
        b.register_forward_pre_hook(lambda mod, a, kw: blocks.append(kw), with_kwargs=True)
        for b in m.net.blocks
    ]
    m(latents=z, **assignments)
    for h in hooks:
        h.remove()
    q = torch.randn(8, 3, 64, 8)
    ids = torch.arange(64).expand(8, -1)

    def rotate(t):
        # Independent complex-plane reference for t,h,w, head_dim=8 => 4,2,2.
        result, start = [], 0
        for size, pos in zip((4, 2, 2), (t, ids // 4 % 4, ids % 4), strict=True):
            value = torch.view_as_complex(
                q[..., start : start + size].reshape(8, 3, 64, size // 2, 2).contiguous()
            )
            freq = 10000.0 ** (-torch.arange(size // 2).float() * 2 / size)
            angle = pos.float()[..., None] * freq
            result.append(
                torch.view_as_real(
                    value * torch.polar(torch.ones_like(angle), angle)[:, None]
                ).flatten(-2)
            )
            start += size
        return torch.cat(result, -1)

    for kw in blocks:
        torch.testing.assert_close(
            kw["rope"](q), rotate(offsets * 4 + ids // 16), rtol=2e-5, atol=2e-6
        )
        qr, kr = kw["cross_rope"]
        torch.testing.assert_close(
            qr(q), rotate((offsets + 1) * 4 + ids // 16), rtol=2e-5, atol=2e-6
        )
        torch.testing.assert_close(kr(q), rotate(ids // 16), rtol=2e-5, atol=2e-6)


def test_config_rejects_invalid_reserves_and_assignments():
    with pytest.raises(ValueError, match="choices"):
        FlowConfig(future_target_choices=4)
    with pytest.raises(ValueError, match="reserve"):
        replace(flow("random"), target_origin_reserve=1)
    with pytest.raises(ValueError, match="at least 3"):
        flow("halves").target_count(2)
    with pytest.raises(ValueError, match="permutations"):
        sample_half_offsets((1, 1), "cpu", offsets=torch.zeros(1, 1, 2, dtype=torch.long))


@pytest.mark.gpu(gpus=1)
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("arm", ["baseline", "random", "halves", "tokens"])
def test_h200_fa4_forward_backward_main_and_tuning(arm, split, monkeypatch):
    from vforge.ops._attention import dispatch

    def no_sdpa(*args, **kwargs):
        raise AssertionError("SDPA forbidden")

    monkeypatch.setattr(dispatch, "_run_sdpa", no_sdpa)
    m = model_for(arm, split, 192).cuda().train()
    m.gradient_checkpointing_enable()
    z = torch.randn(1, 32, 10, 4, 4, device="cuda")
    for tuning in (False, True):
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(latents=z, next_state_only=tuning)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
