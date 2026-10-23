from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vforge.tasks.wan_taichu.batch import fallback_stream_field, stream_field
from vforge.tasks.wan_taichu.conditioning import ConditionDropoutPolicy
from vforge.tasks.wan_taichu.variants import require_variant


def test_stream_dropout_never_silently_uses_zero_tensor(monkeypatch) -> None:
    spec = require_variant("wan21_i2v_14b")
    policy = ConditionDropoutPolicy(
        spec=spec,
        streams=("normal",),
        extras={"normal_drop_prob": 1.0},
    )
    bundle = {
        "prompt": torch.randn(1, 4, 8),
        stream_field("normal"): torch.randn(1, 20, 1, 2, 2),
    }
    monkeypatch.setattr("random.random", lambda: 0.0)

    with pytest.raises(RuntimeError, match="zero tensors are not a valid fallback"):
        policy.apply(bundle, encoder=None)


def test_stream_dropout_uses_cached_uniform_fallback(monkeypatch) -> None:
    spec = require_variant("wan21_i2v_14b")
    policy = ConditionDropoutPolicy(
        spec=spec,
        streams=("normal",),
        extras={"normal_drop_prob": 1.0},
    )
    condition = torch.randn(1, 20, 1, 2, 2)
    fallback = torch.full_like(condition, 0.25)
    bundle = {
        "prompt": torch.randn(1, 4, 8),
        stream_field("normal"): condition,
        fallback_stream_field("normal"): fallback,
    }
    monkeypatch.setattr("random.random", lambda: 0.0)

    output = policy.apply(bundle, encoder=None)

    assert torch.equal(output[stream_field("normal")], fallback)


def test_missing_production_stream_uses_cached_uniform_fallback() -> None:
    spec = require_variant("wan22_ti2v_5b_ref")
    policy = ConditionDropoutPolicy(
        spec=spec,
        streams=("grid", "normal"),
        extras={},
    )
    normal = torch.randn(1, 48, 3, 2, 2)
    fallback = torch.full_like(normal, 0.25)

    output = policy.complete_missing_streams(
        {
            "frame_latents": torch.randn(1, 48, 3, 2, 2),
            stream_field("grid"): None,
            stream_field("normal"): normal,
            fallback_stream_field("grid"): fallback,
        },
        encoder=None,
    )

    assert torch.equal(output[stream_field("grid")], fallback)
    assert output[stream_field("normal")] is normal


def test_live_vae_fallback_uses_declared_spatial_scale(monkeypatch) -> None:
    class FakeEncoder:
        training_dtype = torch.float32

        def __init__(self) -> None:
            self.vae = SimpleNamespace(
                config=SimpleNamespace(
                    z_dim=48,
                    scale_factor_temporal=4,
                    scale_factor_spatial=16,
                ),
                # This legacy-derived value is 8 and must not override the
                # VAE's explicit spatial compression contract.
                temperal_downsample=(False, True, True),
            )
            self.pixel_shape = None

        def scale_pixels(self, pixels):
            return pixels

        def encode_image(self, pixels):
            self.pixel_shape = tuple(pixels.shape)
            return torch.full((1, 48, 3, 2, 3), 0.25)

    spec = require_variant("wan22_ti2v_5b_ref")
    policy = ConditionDropoutPolicy(
        spec=spec,
        streams=("normal",),
        extras={"normal_drop_prob": 1.0},
    )
    encoder = FakeEncoder()
    condition = torch.randn(1, 48, 3, 2, 3)
    monkeypatch.setattr("random.random", lambda: 0.0)

    output = policy.apply(
        {
            "prompt": torch.randn(1, 4, 8),
            stream_field("normal"): condition,
        },
        encoder=encoder,
    )

    assert encoder.pixel_shape == (1, 3, 9, 32, 48)
    assert tuple(output[stream_field("normal")].shape) == tuple(condition.shape)


def test_audio_dropout_is_independent_per_sample(monkeypatch) -> None:
    spec = require_variant("wan21_i2v_14b_audio")
    policy = ConditionDropoutPolicy(
        spec=spec,
        streams=("rgb",),
        extras={"audio_drop_prob": 0.5},
    )
    audio = torch.ones(2, 25, 4, 3)
    draws = iter((0.25, 0.75))
    monkeypatch.setattr("random.random", lambda: next(draws))

    output = policy.apply(
        {
            "prompt": torch.randn(2, 4, 8),
            stream_field("rgb"): torch.randn(2, 20, 1, 2, 2),
            "audio_features": audio,
        },
        encoder=None,
    )

    assert torch.count_nonzero(output["audio_features"][0]) == 0
    assert torch.equal(output["audio_features"][1], audio[1])


def test_dropout_draw_order_and_rgb_first_frame_match_taichu(monkeypatch) -> None:
    spec = require_variant("wan22_a14b")
    policy = ConditionDropoutPolicy(
        spec=spec,
        streams=("rgb", "grid", "normal"),
        extras={
            "rgb_drop_prob": 0.5,
            "rgb_cond_drop_prob": 0.5,
            "normal_drop_prob": 0.5,
            "grid_drop_prob": 0.5,
            "text_drop_prob": 0.5,
        },
    )
    shape = (1, 20, 3, 2, 2)
    conditions = {
        name: torch.full(shape, value)
        for name, value in (("rgb", 1.0), ("grid", 2.0), ("normal", 3.0))
    }
    fallbacks = {
        name: torch.full(shape, value)
        for name, value in (("rgb", 11.0), ("grid", 12.0), ("normal", 13.0))
    }
    draws = iter(
        (
            0.1,  # rgb stream: drop
            0.9,  # rgb image side branch: keep
            0.2,  # normal stream: drop
            0.8,  # grid stream: keep
            0.3,  # text: drop
        )
    )
    monkeypatch.setattr("random.random", lambda: next(draws))
    image = torch.ones(1, 2, 4)
    output = policy.apply(
        {
            "prompt": torch.ones(1, 4, 8),
            "first_frame_latents": torch.ones(1, 16, 1, 2, 2),
            "image_condition_latent_vae": image,
            **{stream_field(name): value for name, value in conditions.items()},
            **{fallback_stream_field(name): value for name, value in fallbacks.items()},
        },
        encoder=None,
    )

    assert output["first_frame_latents"] is None
    assert torch.count_nonzero(output["prompt"]) == 0
    assert output["image_condition_latent_vae"] is image
    assert torch.equal(output[stream_field("rgb")], fallbacks["rgb"])
    assert output[stream_field("grid")] is conditions["grid"]
    assert torch.equal(output[stream_field("normal")], fallbacks["normal"])
    with pytest.raises(StopIteration):
        next(draws)
