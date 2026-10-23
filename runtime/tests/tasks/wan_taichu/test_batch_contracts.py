from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vforge.tasks.wan_taichu.batch import (
    build_encoded_batch_codec,
    canonicalize_encoded_batch,
    canonicalize_training_bundle,
    collate_encoded_bundles,
    collate_raw_samples,
    stream_field,
)
from vforge.tasks.wan_taichu.variants import require_variant


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _common(*, z_dim: int, batch: int = 2) -> dict[str, torch.Tensor]:
    device = _device()
    return {
        "prompt": torch.randn(batch, 8, 12, device=device),
        "frame_latents": torch.randn(batch, z_dim, 3, 4, 5, device=device),
    }


def test_v3_explicit_streams_preserve_declared_order() -> None:
    variant = require_variant("wan22_a14b")
    bundle = _common(z_dim=16)
    bundle[stream_field("grid")] = torch.full((2, 20, 3, 4, 5), 1.0, device=_device())
    bundle[stream_field("normal")] = torch.full((2, 20, 3, 4, 5), 2.0, device=_device())

    canonical = canonicalize_encoded_batch(
        bundle,
        variant=variant,
        streams=("grid", "normal"),
    )

    assert torch.equal(canonical[stream_field("grid")], bundle[stream_field("grid")])
    assert torch.equal(canonical[stream_field("normal")], bundle[stream_field("normal")])
    assert "condition_latents_normal" in canonical


def test_v3_codec_requires_each_configured_stream_and_native_manifest() -> None:
    variant = require_variant("wan22_a14b")
    codec = build_encoded_batch_codec(variant, ("grid", "normal"))

    assert codec.required_keys == (
        "prompt",
        "frame_latents",
        "condition_latents_grid",
        "condition_latents_normal",
    )
    assert codec.accept_legacy_manifest is False


def test_v2_packed_cache_is_rejected() -> None:
    variant = require_variant("wan22_a14b")
    bundle = _common(z_dim=16)
    grid = torch.full((2, 20, 3, 4, 5), 3.0, device=_device())
    normal = torch.full((2, 20, 3, 4, 5), 7.0, device=_device())
    bundle["condition_latents_normal"] = torch.cat([grid, normal], dim=1)

    with pytest.raises(ValueError, match="condition_latents_grid"):
        canonicalize_encoded_batch(
            bundle,
            variant=variant,
            streams=("grid", "normal"),
        )


def test_v1_single_stream_legacy_field_is_rejected() -> None:
    variant = require_variant("wan21_i2v_14b")
    bundle = _common(z_dim=16)
    legacy = torch.randn(2, 20, 3, 4, 5, device=_device())
    bundle["condition_latents_normal"] = legacy

    with pytest.raises(ValueError, match="condition_latents_rgb"):
        canonicalize_encoded_batch(
            bundle,
            variant=variant,
            streams=("rgb",),
        )


def test_v3_stream_width_must_match_the_variant_block_width() -> None:
    variant = require_variant("wan22_a14b")
    bundle = _common(z_dim=16)
    bundle["condition_latents_grid"] = torch.randn(2, 20, 3, 4, 5, device=_device())
    bundle["condition_latents_normal"] = torch.randn(2, 40, 3, 4, 5, device=_device())

    with pytest.raises(ValueError, match="condition_latents_normal must have shape"):
        canonicalize_encoded_batch(
            bundle,
            variant=variant,
            streams=("grid", "normal"),
        )


def test_ti2v_requires_clean_first_frame_latent() -> None:
    variant = require_variant("wan22_ti2v_5b")
    bundle = _common(z_dim=48)
    bundle[stream_field("normal")] = torch.randn(2, 48, 3, 4, 5, device=_device())

    with pytest.raises(ValueError, match="requires first_frame_latents"):
        canonicalize_encoded_batch(bundle, variant=variant, streams=("normal",))


def test_ref_variant_accepts_independent_spatial_geometry() -> None:
    variant = require_variant("wan22_ti2v_5b_ref")
    bundle = _common(z_dim=48)
    for name in ("grid", "normal"):
        bundle[stream_field(name)] = torch.randn(2, 48, 3, 4, 5, device=_device())
    bundle["first_frame_latents"] = bundle["frame_latents"][:, :, 0:1].clone()
    bundle["ref_frame_latents"] = torch.randn(2, 48, 2, 5, 5, device=_device())

    canonical = canonicalize_encoded_batch(
        bundle,
        variant=variant,
        streams=("grid", "normal"),
    )
    assert torch.equal(canonical["ref_frame_latents"], bundle["ref_frame_latents"])

    bundle["ref_frame_latents"] = torch.randn(2, 47, 2, 4, 5, device=_device())
    with pytest.raises(ValueError, match="batch/channel geometry"):
        canonicalize_encoded_batch(
            bundle,
            variant=variant,
            streams=("grid", "normal"),
        )


def test_non_ref_variant_rejects_ref_latents() -> None:
    variant = require_variant("wan22_ti2v_5b")
    bundle = _common(z_dim=48)
    bundle[stream_field("normal")] = torch.randn(2, 48, 3, 4, 5, device=_device())
    bundle["first_frame_latents"] = bundle["frame_latents"][:, :, 0:1].clone()
    bundle["ref_frame_latents"] = torch.randn(2, 48, 1, 4, 5, device=_device())

    with pytest.raises(ValueError, match="does not accept ref_frame_latents"):
        canonicalize_encoded_batch(bundle, variant=variant, streams=("normal",))


def test_audio_variant_requires_and_validates_stage_b_features() -> None:
    variant = require_variant("wan21_i2v_14b_audio")
    bundle = _common(z_dim=16)
    bundle[stream_field("rgb")] = torch.randn(2, 20, 3, 4, 5, device=_device())

    with pytest.raises(ValueError, match="requires audio_features"):
        canonicalize_encoded_batch(bundle, variant=variant, streams=("rgb",))

    bundle["audio_features"] = torch.randn(2, 25, 1024, 121, device=_device())
    canonical = canonicalize_encoded_batch(bundle, variant=variant, streams=("rgb",))
    assert torch.equal(canonical["audio_features"], bundle["audio_features"])

    bundle["audio_features"] = torch.randn(2, 24, 1024, 121, device=_device())
    with pytest.raises(ValueError, match=r"L=25,D=1024"):
        canonicalize_encoded_batch(bundle, variant=variant, streams=("rgb",))


def test_non_audio_variant_rejects_audio_features() -> None:
    variant = require_variant("wan21_i2v_14b")
    bundle = _common(z_dim=16)
    bundle[stream_field("normal")] = torch.randn(2, 20, 3, 4, 5, device=_device())
    bundle["audio_features"] = torch.randn(2, 25, 1024, 121, device=_device())

    with pytest.raises(ValueError, match="does not accept"):
        canonicalize_encoded_batch(bundle, variant=variant, streams=("normal",))


def test_collate_promotes_only_one_missing_batch_dimension() -> None:
    variant = require_variant("wan21_i2v_14b")
    codec = build_encoded_batch_codec(variant, ("normal",))
    samples = []
    for index in range(2):
        samples.append(
            {
                "sample_id": f"sample-{index}",
                "prompt": torch.randn(8, 12),
                "frame_latents": torch.randn(16, 3, 4, 5),
                "condition_latents_normal": torch.randn(20, 3, 4, 5),
            }
        )

    batch = collate_encoded_bundles(codec, samples)

    assert batch["prompt"].shape == (2, 8, 12)
    assert batch["frame_latents"].shape == (2, 16, 3, 4, 5)
    assert batch["condition_latents_normal"].shape == (2, 20, 3, 4, 5)
    assert batch["sample_id"] == ["sample-0", "sample-1"]


class _Tokenizer:
    def __call__(self, prompts, **kwargs):
        length = int(kwargs["max_length"])
        shape = (len(prompts), length)
        return SimpleNamespace(
            input_ids=torch.zeros(shape, dtype=torch.long),
            attention_mask=torch.ones(shape, dtype=torch.long),
        )


def _bound_samples(kind: str, bounds: tuple[tuple[int, int], ...]):
    if kind == "raw":
        return [
            {
                "sample_id": f"sample-{index}",
                "prompt": "prompt",
                "video_frames": torch.zeros(3, 3, 4, 5),
                "min_latents": minimum,
                "max_latents": maximum,
            }
            for index, (minimum, maximum) in enumerate(bounds)
        ]
    return [
        {
            "sample_id": f"sample-{index}",
            "prompt": torch.zeros(8, 12),
            "frame_latents": torch.zeros(16, 38, 4, 5),
            "condition_latents_normal": torch.zeros(20, 38, 4, 5),
            "min_latents": minimum,
            "max_latents": maximum,
        }
        for index, (minimum, maximum) in enumerate(bounds)
    ]


def _collate_bounds(kind: str, bounds: tuple[tuple[int, int], ...]):
    samples = _bound_samples(kind, bounds)
    if kind == "raw":
        return collate_raw_samples(samples, tokenizer=_Tokenizer())
    variant = require_variant("wan21_i2v_14b")
    codec = build_encoded_batch_codec(variant, ("normal",))
    return collate_encoded_bundles(codec, samples)


@pytest.mark.parametrize("kind", ("raw", "offline"))
def test_collate_preserves_one_source_bound_pair(kind: str) -> None:
    batch = _collate_bounds(kind, ((21, 38), (21, 38)))

    assert (batch["min_latents"], batch["max_latents"]) == (21, 38)


@pytest.mark.parametrize("kind", ("raw", "offline"))
def test_collate_rejects_mixed_source_bounds(kind: str) -> None:
    with pytest.raises(ValueError, match="mixes latent bounds"):
        _collate_bounds(kind, ((21, 38), (38, 150)))


def test_online_canonicalization_reattaches_source_metadata() -> None:
    variant = require_variant("wan21_i2v_14b")
    codec = build_encoded_batch_codec(variant, ("normal",))
    encoded = _bound_samples("offline", ((21, 38), (21, 38)))
    encoded_batch = collate_encoded_bundles(codec, encoded)

    canonical = canonicalize_training_bundle(
        codec,
        {
            key: encoded_batch[key]
            for key in ("prompt", "frame_latents", "condition_latents_normal")
        },
        metadata=encoded_batch,
    )

    assert canonical["sample_id"] == ["sample-0", "sample-1"]
    assert (canonical["min_latents"], canonical["max_latents"]) == (21, 38)
