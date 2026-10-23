"""Random future horizons: DuET probabilities, causality, coordinates and legacy parity."""

import copy

import pytest
import torch

from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, WorldConfig
from vforge.models.rep_world.horizon import sample_target_offsets, target_gap_embedding
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


def make_model(mode="entry_concat", *, rope=False, mask=0.0, probability=0.25):
    geometry = CodecGeometry("test", "test", 4, 8, 2, grid=3)
    encoder = ConditionerConfig(
        input="latent",
        width=24,
        depth=2,
        heads=3,
        feature_layers=(1,),
        context_mask_ratio=mask,
    )
    world = WorldConfig(
        channels=8,
        state_planes=2,
        grid=3,
        width=24,
        depth=2,
        heads=3,
        condition_width=24,
        adaln_lora_dim=8,
        condition_dropout=0.0,
        condition_mode="entry_concat" if mode == "entry_concat" else "cross_attention",
        cross_attention_every=2,
        cross_attention_position="before_self",
        cross_attention_scope="same_spatial" if mode == "same_spatial" else "global",
        cross_attention_3d_rope=rope,
    )
    return RepWorld(geometry, world, encoder, FlowConfig(distant_target_prob=probability))


def test_duet_probability_thresholds_and_disabled_rng():
    u = torch.tensor([[0.0, 0.7499, 0.75, 0.8749, 0.875, 0.9999]])
    actual = sample_target_offsets(u.shape, 0.25, u.device, uniform=u)
    assert actual.tolist() == [[0, 0, 1, 1, 2, 2]]
    state = torch.get_rng_state().clone()
    assert sample_target_offsets(u.shape, 0, u.device).eq(0).all()
    assert torch.equal(state, torch.get_rng_state())
    assert sample_target_offsets(u.shape, 1, u.device, uniform=u).tolist() == [[1, 2, 2, 2, 2, 2]]
    # Exact stratified draws verify probabilities without a flaky statistical test.
    u = (torch.arange(10000) + 0.5) / 10000
    counts = sample_target_offsets(u.shape, 0.25, u.device, uniform=u).bincount()
    assert counts.tolist() == [7500, 1250, 1250]


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_probability(bad):
    with pytest.raises(ValueError, match="distant_target_prob"):
        FlowConfig(distant_target_prob=bad)


@pytest.mark.parametrize("mode", ["pure_tf", "zprev"])
def test_unsupported_modes_are_explicit(mode):
    with pytest.raises(ValueError, match="no_zprev"):
        FlowConfig(mode=mode, distant_target_prob=0.25)


def test_state_budget_and_uniform_validation():
    model = make_model()
    assert model.flow.target_count(10) == 7
    assert FlowConfig().target_count(10) == 9
    with pytest.raises(ValueError, match="at least 4"):
        model(latents=torch.randn(1, 16, 3, 3, 3))
    for bad in [torch.zeros(2), torch.ones(1, 3), torch.full((1, 3), float("nan"))]:
        with pytest.raises(ValueError, match="uniforms"):
            sample_target_offsets((1, 3), 0.25, "cpu", uniform=bad)


@pytest.mark.parametrize(
    "mode,rope",
    [
        ("entry_concat", False),
        ("global", False),
        ("global", True),
        ("same_spatial", False),
        ("same_spatial", True),
    ],
)
@pytest.mark.parametrize("mask", [0.0, 0.8])
def test_targets_gather_per_sample_causal_context_and_metrics(mode, rope, mask):
    torch.manual_seed(31)
    model = make_model(mode, rope=rope, mask=mask).train()
    latents = torch.randn(2, 16, 6, 3, 3)
    # Each sample chooses different targets for the same origin.
    u = torch.tensor([[0.1, 0.8, 0.9], [0.9, 0.1, 0.8]])
    offsets = torch.tensor([[0, 1, 2], [2, 0, 1]])
    indices = torch.arange(3)[None] + 1 + offsets
    states = unpack_states(latents, model.geometry)
    targets = states[torch.arange(2)[:, None], indices].flatten(0, 1)
    noise = torch.randn_like(targets)
    uniform = torch.full((6, 1), 0.5)
    captured = []
    hook = model.net.register_forward_pre_hook(
        lambda m, a, kw: captured.append((a, kw)), with_kwargs=True
    )
    torch.manual_seed(91)
    result = model(
        latents=latents, noise=noise, uniform=uniform, target_uniform=u, return_metrics=True
    )
    args, kw = captured[0]
    t = model.draw_timesteps((6, 1), targets, uniform).reshape(-1, 1, 1, 1, 1)
    torch.testing.assert_close(args[0], (1 - t) * targets + t * noise, rtol=0, atol=0)
    assert torch.equal(kw["target_offsets"], offsets.flatten())
    # Zero initialized output head makes this an independent target/velocity-sign check.
    torch.testing.assert_close(result["loss"], (targets - noise).square().mean(), rtol=0, atol=0)
    assert sum(result["metrics"][f"target_horizon_{gap}_fraction"] for gap in (1, 2, 3)) == 1
    contribution = sum(
        result["metrics"][f"loss_fm_horizon_{gap}_contribution"] for gap in (1, 2, 3)
    )
    torch.testing.assert_close(contribution, result["loss"])
    changed = latents.clone()
    changed[:, :, 1:] += 100
    torch.manual_seed(91)
    model(latents=changed, noise=noise, uniform=uniform, target_uniform=u)
    # Origin 0 must never read states 1/2 while predicting state 3.
    c0 = kw["condition"].reshape(2, 3, -1, 24)[:, 0]
    c1 = captured[1][1]["condition"].reshape(2, 3, -1, 24)[:, 0]
    torch.testing.assert_close(c0, c1, rtol=0, atol=0)
    if kw["condition_positions"] is not None:
        assert torch.equal(kw["condition_positions"], captured[1][1]["condition_positions"])
    hook.remove()


@pytest.mark.parametrize(
    "mode,rope", [("entry_concat", False), ("global", True), ("same_spatial", True)]
)
def test_gap_changes_network_with_finite_gradients_and_checkpoint_recompute(mode, rope):
    torch.manual_seed(8)
    model = make_model(mode, rope=rope, mask=0.8)
    torch.nn.init.normal_(model.net.final_layer.linear.weight, std=0.02)
    other = copy.deepcopy(model)
    other.gradient_checkpointing_enable()
    latent = torch.randn(2, 16, 6, 3, 3)
    for current in (model, other):
        torch.manual_seed(17)
        loss = current(latents=latent)
        loss.backward()
        p = current.causal_semantic_encoder.latent_patch_embed.weight
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    for (name, a), (_, b) in zip(model.named_parameters(), other.named_parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
    model.eval()
    context = model(operation="condition", source=latent[:, :, :1])
    x = torch.randn(2, 8, 2, 3, 3)
    t = torch.full((2, 1), 0.5)
    plain = model(operation="velocity", noisy=x, timestep=t, condition=context)
    next_state = model(
        operation="velocity",
        noisy=x,
        timestep=t,
        condition=context,
        target_offsets=torch.zeros(2, dtype=torch.long),
    )
    torch.testing.assert_close(plain, next_state, rtol=0, atol=0)
    future = model(
        operation="velocity",
        noisy=x,
        timestep=t,
        condition=context,
        target_offsets=torch.tensor([1, 2]),
    )
    assert (future - plain).abs().max() > 1e-5
    assert torch.isfinite(future).all()


def test_duet_gap_embedding_reference_and_no_parameters_added():
    offsets = torch.tensor([0, 1, 2])
    width = 25
    freq = 10000.0 ** (-torch.arange(width // 2).double() / (width // 2 - 1))
    angles = (offsets.double()[:, None] + 1) * freq
    expected = torch.cat((angles.sin() - freq.sin(), angles.cos() - freq.cos()), -1)
    expected = torch.nn.functional.pad(expected, (0, 1)).float()
    actual = target_gap_embedding(offsets, width, torch.float32)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-7)
    assert torch.count_nonzero(actual[0]) == 0
    torch.manual_seed(12)
    disabled = make_model(probability=0)
    torch.manual_seed(12)
    enabled = make_model()
    assert disabled.state_dict().keys() == enabled.state_dict().keys()
    for name, value in disabled.state_dict().items():
        assert torch.equal(value, enabled.state_dict()[name]), name


def test_disabled_path_matches_original_rng_loss_and_gradients():
    torch.manual_seed(14)
    model = make_model(probability=0, mask=0.8)
    torch.nn.init.normal_(model.net.final_layer.linear.weight, std=0.02)
    other = copy.deepcopy(model)
    latents = torch.randn(2, 16, 4, 3, 3)
    torch.manual_seed(47)
    actual = model(latents=latents)
    rng = torch.get_rng_state().clone()
    actual.backward()
    torch.manual_seed(47)
    contexts, positions = other.causal_semantic_encoder.training_context(
        latents, return_indices=True
    )
    target = unpack_states(latents, other.geometry)[:, 1:].flatten(0, 1)
    t = other.draw_timesteps((6, 1), target)
    noise = torch.randn_like(target)
    tv = t.reshape(-1, 1, 1, 1, 1)
    prediction = other.velocity(
        (1 - tv) * target + tv * noise,
        t,
        contexts.flatten(0, 1),
        condition_positions=positions.flatten(0, 1),
    )
    expected = (target - noise - prediction).square().mean()
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.backward()
    for (name, a), (_, b) in zip(model.named_parameters(), other.named_parameters(), strict=True):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)


def test_batched_query_rope_encodes_actual_gap_and_masked_keys_stay_at_origin():
    model = make_model("global", rope=True, mask=0.8)
    captured = []
    handle = model.net.blocks[0].cross_attn.register_forward_pre_hook(
        lambda m, a, kw: captured.append(kw), with_kwargs=True
    )
    model(
        latents=torch.randn(1, 16, 6, 3, 3),
        target_uniform=torch.tensor([[0.1, 0.8, 0.9]]),
    )
    handle.remove()
    q = torch.randn(3, 3, 18, 8)
    actual = captured[0]["q_rope"](q)
    expected = torch.cat(
        [
            model.net.feat_rope(q[i : i + 1], num_frames_override=2, start_frame=(i + 1) * 2)
            for i in range(3)
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    positions = captured[0]["k_rope"].keywords["token_position_ids"]
    assert positions.min() >= 0 and positions.max() < 18


@pytest.mark.parametrize(
    "bad", [torch.tensor([-1]), torch.tensor([3]), torch.tensor([1.0]), torch.tensor([[1]])]
)
def test_invalid_prediction_offset_rejected(bad):
    m = make_model()
    with pytest.raises(ValueError, match="offsets"):
        m(
            operation="velocity",
            noisy=torch.randn(1, 8, 2, 3, 3),
            timestep=torch.ones(1, 1),
            condition=torch.randn(1, 18, 24),
            target_offsets=bad,
        )
