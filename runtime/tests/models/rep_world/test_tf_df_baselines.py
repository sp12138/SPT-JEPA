"""Contracts for conditioner-free TF/DF with the same DiT and data window."""

import pytest
import torch
from torch import nn

from vforge.models.rep_world.config import CodecGeometry, WorldConfig
from vforge.models.rep_world.model import (
    FlowConfig,
    RepWorld,
    pack_states,
    sample_incremental_timesteps,
)
from vforge.tasks.rep_world.shared import parse_config


def test_incremental_df_bins_are_monotone_and_cover_all_states():
    torch.manual_seed(42)
    values = sample_incremental_timesteps(1024, 8, 50, torch.device("cpu"))
    assert values.shape == (1024, 8)
    assert values.min() >= 0 and values.max() <= 49
    assert bool((values[:, 1:] >= values[:, :-1]).all())
    assert bool((values[:, 1:] > values[:, :-1]).any())


class RecordingWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.last = None

    def forward(self, latents, timesteps, **kwargs):
        self.last = (latents.detach().clone(), timesteps.detach().clone(), kwargs)
        return torch.zeros_like(latents)


def test_df_keeps_first_state_clean_and_losses_only_eight_targets():
    geometry = CodecGeometry("test", "tiny", 4, 4, 2, grid=2)
    world = WorldConfig(4, 2, grid=2, width=48, depth=1, heads=4, condition_width=0)
    model = RepWorld(geometry, world, None, FlowConfig(mode="pure_df"))
    recorder = RecordingWorld()
    model.net = recorder
    states = torch.randn(2, 9, 4, 2, 2, 2)
    packed = pack_states(states)
    noise = torch.randn(2, 4, 8, 2, 2, 2)
    uniform = torch.linspace(0.1, 0.9, 8).expand(2, -1)
    loss = model(latents=packed, noise=noise, uniform=uniform)
    received, times, options = recorder.last
    expected_times = model.draw_timesteps((2, 8), states, uniform)
    torch.testing.assert_close(received[:, :, :2], states[:, 0].permute(0, 1, 2, 3, 4))
    torch.testing.assert_close(times[:, :2], torch.zeros_like(times[:, :2]))
    torch.testing.assert_close(times[:, 2:], expected_times.repeat_interleave(2, dim=1))
    assert options["causal_state_planes"] == 2
    assert received.shape == (2, 4, 18, 2, 2)
    expected_loss = (states[:, 1:].permute(0, 2, 1, 3, 4, 5) - noise).square().mean()
    torch.testing.assert_close(loss, expected_loss)


def test_tf_df_share_exact_trainable_topology():
    geometry = CodecGeometry("test", "tiny", 4, 4, 2, grid=2)
    world = WorldConfig(4, 2, grid=2, width=48, depth=2, heads=4, condition_width=0)
    tf = RepWorld(geometry, world, None, FlowConfig(mode="pure_tf"))
    df = RepWorld(geometry, world, None, FlowConfig(mode="pure_df"))
    assert [(name, tuple(p.shape)) for name, p in tf.named_parameters()] == [
        (name, tuple(p.shape)) for name, p in df.named_parameters()
    ]
    with pytest.raises(ValueError, match="nondecreasing"):
        df(
            latents=pack_states(torch.randn(1, 9, 4, 2, 2, 2)),
            uniform=torch.arange(8, 0, -1)[None].float(),
        )


def test_random_nine_state_view_accepts_conditioner_free_baselines():
    extras = {
        "codec": {"family": "videorae", "variant": "3d"},
        "world": {"condition_mode": "cross_attention"},
        "flow": {"mode": "pure_df"},
        "states": 9,
        "random_clip": False,
        "cached_state_view": "random_9_from_10",
    }
    config = parse_config(extras)
    assert config.conditioner is None and config.cached_state_view == "random_9_from_10"
