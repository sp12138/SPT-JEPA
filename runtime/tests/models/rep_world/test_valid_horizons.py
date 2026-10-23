"""Valid-only future target sampling preserves all origin states and true CA time."""

import copy

import pytest
import torch

from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, WorldConfig
from vforge.models.rep_world.horizon import sample_available_target_offsets
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


def make_model(flow):
    geometry = CodecGeometry("test", "test", 4, 8, 2, grid=3)
    encoder = ConditionerConfig(input="latent", width=24, depth=2, heads=3, feature_layers=(1,))
    world = WorldConfig(
        channels=8,
        state_planes=2,
        grid=3,
        width=24,
        depth=2,
        heads=3,
        condition_width=24,
        adaln_lora_dim=8,
        condition_dropout=0,
        cross_attention_every=2,
        cross_attention_position="before_self",
        cross_attention_scope="global",
        cross_attention_3d_rope=True,
    )
    return RepWorld(geometry, world, encoder, flow)


@pytest.mark.parametrize(
    "k,expected",
    [
        (1, [0] * 9),
        (2, [1] * 8 + [0]),
        (3, [2] * 7 + [1, 0]),
        (4, [3] * 6 + [2, 1, 0]),
    ],
)
def test_available_horizons_at_tail(k, expected):
    u = torch.full((2, 9), 0.99)
    offsets = sample_available_target_offsets((2, 9), 10, k, "cpu", uniform=u)
    assert offsets.tolist() == [expected, expected]
    assert all(i + 1 + offset < 10 for i, offset in enumerate(expected))
    assert (
        FlowConfig(
            future_target_policy="valid_uniform", future_target_choices=k, target_origin_reserve=1
        ).target_count(10)
        == 9
    )


def test_uniform_draw_is_per_origin_and_k1_keeps_rng():
    rng = torch.get_rng_state().clone()
    assert sample_available_target_offsets((3, 9), 10, 1, "cpu").eq(0).all()
    assert torch.equal(rng, torch.get_rng_state())
    u = torch.tensor([[0.0, 0.24, 0.26, 0.49, 0.51, 0.74, 0.76, 0.99, 0.99]])
    expected = [[0, 0, 1, 1, 2, 2, 2, 1, 0]]
    assert sample_available_target_offsets((1, 9), 10, 4, "cpu", uniform=u).tolist() == expected


@pytest.mark.parametrize("bad", [0, 5, True])
def test_bad_horizon_choice_rejected(bad):
    with pytest.raises(ValueError, match="choices"):
        FlowConfig(
            future_target_policy="valid_uniform", future_target_choices=bad, target_origin_reserve=1
        )


def test_k4_gathers_real_target_and_rope_offsets_without_leakage():
    torch.manual_seed(17)
    model = make_model(
        FlowConfig(
            future_target_policy="valid_uniform", future_target_choices=4, target_origin_reserve=1
        )
    ).eval()
    latents = torch.randn(1, 16, 10, 3, 3)
    states = unpack_states(latents, model.geometry)
    u = torch.full((1, 9), 0.99)
    offsets = torch.tensor([3, 3, 3, 3, 3, 3, 2, 1, 0])
    chosen = states[0, torch.arange(9) + 1 + offsets]
    noise = torch.randn_like(chosen)
    t_uniform = torch.full((9, 1), 0.5)
    calls, blocks = [], []
    h1 = model.net.register_forward_pre_hook(
        lambda _, a, kw: calls.append((a, kw)), with_kwargs=True
    )
    h2 = model.net.blocks[0].register_forward_pre_hook(
        lambda _, a, kw: blocks.append(kw), with_kwargs=True
    )
    model(latents=latents, noise=noise, uniform=t_uniform, target_uniform=u)
    timestep = model.draw_timesteps((9, 1), chosen, t_uniform).reshape(-1, 1, 1, 1, 1)
    torch.testing.assert_close(
        calls[0][0][0], (1 - timestep) * chosen + timestep * noise, rtol=0, atol=0
    )
    assert torch.equal(calls[0][1]["target_offsets"], offsets)
    q_rope = blocks[0]["cross_rope"][0]
    assert q_rope.keywords["num_frames_override"] == 10
    assert torch.equal(q_rope.keywords["token_position_ids"][0], torch.arange(18) + 72)
    modified = latents.clone()
    modified[:, :, 1:] += 100
    model(latents=modified, noise=noise, uniform=t_uniform, target_uniform=u)
    torch.testing.assert_close(
        calls[0][1]["condition"][0], calls[1][1]["condition"][0], rtol=0, atol=0
    )
    h1.remove()
    h2.remove()


def test_k1_matches_existing_next_state_baseline():
    torch.manual_seed(19)
    baseline = make_model(FlowConfig()).train()
    torch.nn.init.normal_(baseline.net.final_layer.linear.weight, std=0.02)
    valid = make_model(
        FlowConfig(
            future_target_policy="valid_uniform",
            future_target_choices=1,
            target_origin_reserve=1,
        )
    ).train()
    valid.load_state_dict(copy.deepcopy(baseline.state_dict()), strict=True)
    latents = torch.randn(1, 16, 10, 3, 3)
    noise = torch.randn(9, 8, 2, 3, 3)
    time = torch.full((9, 1), 0.5)
    rng = torch.get_rng_state().clone()
    a = valid(latents=latents, noise=noise, uniform=time)
    a.backward()
    after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    b = baseline(latents=latents, noise=noise, uniform=time)
    b.backward()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert torch.equal(after, torch.get_rng_state())
    for (name, current), reference in zip(
        valid.named_parameters(), baseline.parameters(), strict=True
    ):
        if current.grad is None:
            assert reference.grad is None, name
        else:
            torch.testing.assert_close(current.grad, reference.grad, rtol=0, atol=0, msg=name)
