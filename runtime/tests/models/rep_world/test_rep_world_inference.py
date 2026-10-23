from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, WorldConfig
from vforge.models.rep_world.inference import RepWorldInference, RolloutConfig
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


class ToyCodec:
    def __init__(self, geometry, display_value=0.25):
        self.geometry = geometry
        self.display_value = display_value
        self.encoded_rgb = []

    def encode(self, rgb):
        self.encoded_rgb.append(rgb.clone())
        geo = self.geometry
        return rgb.new_full(
            (rgb.shape[0], geo.packed_channels, rgb.shape[2] // geo.rgb_frames, geo.grid, geo.grid),
            0.1,
        )

    def stream(self):
        def step(packed):
            return packed.new_full(
                (packed.shape[0], 3, packed.shape[2] * self.geometry.rgb_frames, 32, 32),
                self.display_value,
            )

        return SimpleNamespace(prime=lambda _: None, step=step)


class SpyEngine:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["operation"] == "condition":
            source = kwargs["source"]
            return source.mean().expand(source.shape[0], 4, 24).clone()
        assert kwargs["operation"] == "velocity"
        if kwargs.get("condition_drop") is not None:
            return torch.zeros_like(kwargs["noisy"])
        return torch.full_like(kwargs["noisy"], kwargs["condition"].mean())


@pytest.mark.parametrize(
    "mode,domain", [("no_zprev", "rgb"), ("zprev", "rgb"), ("no_zprev", "latent")]
)
def test_engine_boundary_no_gt_leak_seed_and_generated_history(mode, domain):
    geo = CodecGeometry("videorae", "test", 4, 2, 2, grid=2)
    cfg = RolloutConfig(4, history_window_states=2, steps=2, cfg_scale=2)
    vit = ConditionerConfig(input=domain)
    engine, codec = SpyEngine(), ToyCodec(geo)
    infer = RepWorldInference(engine, codec, geo, FlowConfig(mode), vit, 1.5, cfg)
    rgb = torch.randn(1, 3, 16, 32, 32)
    result = infer.infer_sample({"sample_id": "test", "rgb": rgb}, seed=71)
    assert result["latents"].shape == (1, 4, 4, 2, 2)
    assert result["prediction"].shape == rgb.shape
    torch.testing.assert_close(result["prediction"][:, :, :4], rgb[:, :, :4], atol=0, rtol=0)
    assert len(codec.encoded_rgb) == 1
    torch.testing.assert_close(codec.encoded_rgb[0], rgb[:, :, :4], atol=0, rtol=0)
    conditions = [call for call in engine.calls if call["operation"] == "condition"]
    assert [call["source"].shape[2] for call in conditions] == (
        [4, 8, 8] if domain == "rgb" else [1, 2, 2]
    )
    velocity = [call for call in engine.calls if call["operation"] == "velocity"]
    assert len(velocity) == 3 * 2 * 2
    assert all((call["previous"] is not None) == (mode == "zprev") for call in velocity)
    if domain == "latent":
        torch.testing.assert_close(conditions[1]["source"], result["latents"][:, :, :2])
    else:
        assert (conditions[1]["source"][:, :, 4:] == codec.display_value).all()
    torch.manual_seed(384)
    torch.randn(117)
    changed = rgb.clone()
    changed[:, :, 4:] += 100
    repeated = infer.infer_sample({"sample_id": "test", "rgb": changed}, seed=71)
    torch.testing.assert_close(repeated["latents"], result["latents"], atol=0, rtol=0)
    # Changing display/decoder output must influence RGB conditioning, but
    # must not influence the generated latent trajectory for latent input.
    codec.display_value = -0.7
    new_display = infer.infer_sample({"sample_id": "test", "rgb": rgb}, seed=71)
    if domain == "latent":
        torch.testing.assert_close(new_display["latents"], result["latents"], atol=0, rtol=0)
    else:
        assert not torch.allclose(new_display["latents"], result["latents"])


def test_pure_tf_cache_matches_uncached_full_history_euler():
    torch.manual_seed(99)
    geo = CodecGeometry("videorae", "test", 4, 4, 2, grid=2)
    model = RepWorld(
        geo,
        WorldConfig(4, 2, grid=2, width=48, heads=4, depth=2, condition_width=0),
        None,
        FlowConfig(mode="pure_tf"),
    ).eval()
    with torch.no_grad():
        model.net.final_layer.linear.weight.normal_(std=0.02)
    cfg = RolloutConfig(4, history_states=2, steps=3, cfg_scale=1)
    infer = RepWorldInference(
        model, ToyCodec(geo), geo, model.flow, None, model.timestep_shift, cfg
    )
    observed = torch.randn(1, 8, 2, 2, 2)
    actual = infer._pure_tf(observed, torch.Generator().manual_seed(456))
    noise = torch.randn(1, 4, 8, 2, 2, generator=torch.Generator().manual_seed(456))
    history = unpack_states(observed, geo).permute(0, 2, 1, 3, 4, 5).flatten(2, 3).contiguous()
    schedule = infer._schedule(observed)
    for state in range(2, 4):
        z = noise[:, :, state * 2 : (state + 1) * 2].clone()
        for now, following in zip(schedule[:-1], schedule[1:], strict=True):
            value = torch.cat((history, z), 2)
            time = torch.cat((history.new_zeros(1, history.shape[2]), now.expand(1, 2)), 1)
            v = model(operation="world", noisy=value, timestep=time, causal_state_planes=2)[
                :, :, -2:
            ]
            z = z - (following - now) * v
        history = torch.cat((history, z), 2)
    expected = pack_states(history.reshape(1, 4, 4, 2, 2, 2).permute(0, 2, 1, 3, 4, 5))
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    with pytest.raises(ValueError, match="no semantic CFG"):
        RepWorldInference(model, infer.codec, geo, model.flow, None, 1, replace(cfg, cfg_scale=2))
    with pytest.raises(ValueError, match="trained state window"):
        RepWorldInference(
            model, infer.codec, geo, model.flow, None, 1, replace(cfg, trained_states=3)
        )


def test_pure_tf_decoder_sees_only_the_current_generated_prefix(monkeypatch):
    geo = CodecGeometry("vrae", "test", 4, 2, 1, grid=2)
    codec = ToyCodec(geo)
    prefix = []

    def prime(observed):
        prefix.append(observed)

    def step(states):
        prefix.append(states)
        # A full-attention decoder can depend on every state in its prefix.
        return states.new_full(
            (1, 3, states.shape[2] * 4, 32, 32), torch.cat(prefix, 2).mean().item()
        )

    monkeypatch.setattr(codec, "stream", lambda: SimpleNamespace(prime=prime, step=step))
    infer = RepWorldInference(
        None,
        codec,
        geo,
        FlowConfig("pure_tf"),
        None,
        1,
        RolloutConfig(3, cfg_scale=1),
    )
    packed = torch.tensor([0.1, 0.2, 0.9]).view(1, 1, 3, 1, 1).expand(1, 2, 3, 2, 2)
    monkeypatch.setattr(infer, "_pure_tf", lambda *_: packed)
    result = infer.infer_sample(
        {"sample_id": "causal", "rgb": torch.zeros(1, 3, 4, 32, 32)}, seed=1
    )
    torch.testing.assert_close(result["prediction"][:, :, 4:8], torch.full((1, 3, 4, 32, 32), 0.15))
    torch.testing.assert_close(result["prediction"][:, :, 8:12], torch.full((1, 3, 4, 32, 32), 0.4))


@pytest.mark.parametrize("guidance", [1, 2])
def test_bf16_velocity_keeps_source_fp32_cfg_and_euler_arithmetic(guidance):
    geo = CodecGeometry("videorae", "test", 4, 2, 2, grid=2)
    cfg = RolloutConfig(3, steps=7, cfg_scale=guidance)
    conditional = torch.full((1, 2, 2, 2, 2), 0.123, dtype=torch.bfloat16)
    unconditional = torch.full_like(conditional, -0.321)

    def engine(**kwargs):
        return unconditional if kwargs.get("condition_drop") is not None else conditional

    infer = RepWorldInference(engine, None, geo, FlowConfig(), ConditionerConfig(), 1.742, cfg)
    condition = torch.ones(1, 4, 24, dtype=torch.float32)
    noise = torch.ones(1, 2, 2, 2, 2, dtype=torch.float32)
    expected = noise.clone()
    schedule = infer._schedule(condition)
    for now, following in zip(schedule[:-1], schedule[1:], strict=True):
        scale = now.new_full((1, 1, 1, 1, 1), guidance if now > 0.2 else 1)
        velocity = (
            unconditional + scale * (conditional - unconditional) if guidance > 1 else conditional
        )
        expected -= (following - now).view(1, 1, 1, 1, 1) * velocity
    torch.testing.assert_close(infer._denoise(noise, condition, None), expected, atol=0, rtol=0)
