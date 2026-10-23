from __future__ import annotations

import pytest
import torch

from vforge.models.gen_engine import (
    AudioGeometry,
    LatentCodecConfig,
    RuntimePacker,
    TargetGeometry,
    VisualGeometry,
    pack_runtime_sample,
    patchify_audio,
    patchify_visual,
)


def _h3_test_codec() -> LatentCodecConfig:
    return LatentCodecConfig.from_split_dicts(
        {
            "schema": "vforge.video_latent_codec",
            "schema_version": 1,
            "latent_channels": 1,
            "spatial_compression": 16,
            "posterior_sampling": {
                "type": "diagonal_gaussian",
                "logvar_min": -30.0,
                "logvar_max": 20.0,
            },
            "normalization": {
                "type": "channelwise_mean_std",
                "mean": [10.0],
                "std": [2.0],
            },
            "temporal": {
                "input_rate_unit": "frames_per_second",
                "mapping": {
                    "type": "periodic_offsets",
                    "input_period": 17,
                    "latent_offsets": [0, 1, 5, 9, 13],
                    "valid_length_remainder": 5,
                },
            },
        },
        {
            "schema": "vforge.audio_latent_codec",
            "schema_version": 1,
            "latent_channels": 2,
            "planes": 2,
            "normalization": {
                "type": "channelwise_mean_std",
                "mean": [0.0, 0.0],
                "std": [1.0, 1.0],
            },
            "temporal": {
                "input_rate_unit": "samples_per_second",
                "mapping": {"type": "stride", "input_stride": 800},
            },
        },
    )


def _visual_moments(
    mean: float,
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    posterior_mean = torch.full((1, *shape), mean, dtype=dtype)
    logvar = torch.full_like(posterior_mean, -30.0)
    return torch.cat((posterior_mean, logvar), dim=0)


def test_visual_patchification_is_h3_frame_major() -> None:
    latent = torch.arange(2 * 2 * 4 * 4).reshape(2, 2, 4, 4)

    rows = patchify_visual(latent)

    expected_first = latent[:, 0, :2, :2].reshape(-1)
    expected_second = latent[:, 0, :2, 2:].reshape(-1)
    assert rows.shape == (8, 8)
    assert torch.equal(rows[0], expected_first)
    assert torch.equal(rows[1], expected_second)


def test_audio_patchification_is_channel_then_time_ordered() -> None:
    latent = torch.arange(2 * 3 * 4).reshape(2, 3, 4)

    rows = patchify_audio(latent)

    assert rows.shape == (8, 3)
    assert torch.equal(
        rows,
        torch.stack([latent[channel, :, time] for channel in range(2) for time in range(4)]),
    )


def test_runtime_pack_keeps_reference_and_target_rows_separate() -> None:
    image = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
    av_video = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2) + 10
    av_audio = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3) + 30
    target_video = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2) + 40
    target_audio = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2) + 50
    sample = {
        "mme_embedding": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "references": [
            {"type": "image", "visual_latent": image},
            {
                "type": "video_audio",
                "visual_latent": av_video,
                "visual_effective_fps": 1.0,
                "audio_latent": av_audio,
                "audio_effective_fps": 1.0,
            },
        ],
        "target": {
            "visual_latent": target_video,
            "visual_effective_fps": 1.0,
            "audio_latent": target_audio,
            "audio_effective_fps": 1.0,
        },
    }

    packed = pack_runtime_sample(sample)

    for redundant_alias in (
        "prompt_embeds",
        "text_rows",
        "reference_visual_rows",
        "condition_video_rows",
        "target_visual_rows",
        "condition_audio_rows",
        "visual_indices",
        "target_start_time",
    ):
        assert redundant_alias not in packed
    assert torch.equal(
        packed["reference_video_rows"],
        torch.cat((patchify_visual(image), patchify_visual(av_video))),
    )
    assert torch.equal(packed["target_video_rows"], patchify_visual(target_video))
    assert torch.equal(packed["reference_audio_rows"], patchify_audio(av_audio))
    assert torch.equal(packed["target_audio_rows"], patchify_audio(target_audio))
    assert packed["num_reference_video_rows"] == 4
    assert packed["num_reference_audio_rows"] == 3
    assert packed["num_target_video_rows"] == 2
    assert packed["num_target_audio_rows"] == 2
    assert torch.equal(
        packed["video_loss_mask"],
        torch.tensor([False, False, False, False, True, True]),
    )
    assert torch.equal(
        packed["audio_loss_mask"],
        torch.tensor([False, False, False, True, True]),
    )
    assert int(packed["text_indices"][-1]) < int(packed["video_indices"][0])
    assert int(packed["video_indices"][-1]) < int(packed["audio_indices"][0])


def test_runtime_pack_accepts_dataset_nested_media_mappings() -> None:
    packed = pack_runtime_sample(
        {
            "mme_embedding": torch.ones(1, 3),
            "references": [
                {
                    "type": "image",
                    "image": {"latent": torch.ones(1, 1, 2, 2)},
                },
                {
                    "type": "audio",
                    "audio": {
                        "latent": torch.ones(2, 4, 2),
                        "effective_fps": 2.0,
                    },
                },
            ],
            "target": {
                "video": {
                    "latent": torch.ones(1, 2, 2, 2),
                    "effective_fps": 1.0,
                }
            },
        }
    )

    assert packed["reference_video_rows"].shape == (1, 4)
    assert packed["reference_audio_rows"].shape == (4, 4)
    assert torch.equal(
        packed["audio_plane_ids"],
        torch.tensor([0, 0, 0, 0, 1, 1, 2, 2]),
    )
    assert packed["target_video_rows"].shape == (2, 4)
    assert packed["target_audio_rows"].shape == (0, 4)


def test_inference_layout_has_target_rows_without_allocating_clean_latents() -> None:
    sample = {
        "mme_embedding": torch.ones(2, 5),
        "references": [
            {
                "type": "video",
                "visual_latent": torch.ones(1, 2, 2, 2),
                "visual_effective_fps": 1.0,
            }
        ],
        "target_spec": {
            "modalities": ["video", "audio"],
            "video": {
                "latent_shape": [1, 3, 2, 2],
                "effective_fps": 1.0,
            },
            "audio": {
                "latent_shape": [2, 4, 3],
                "effective_fps": 2.0,
            },
        },
    }

    packed = RuntimePacker().pack(sample)

    assert packed["is_inference"] is True
    assert packed["target_video_rows"].shape == (0, 4)
    assert packed["target_audio_rows"].shape == (0, 4)
    assert packed["num_target_video_rows"] == 3
    assert packed["num_target_audio_rows"] == 6
    assert packed["target_visual_shape"] == (1, 3, 2, 2)
    assert packed["target_audio_shape"] == (2, 4, 3)
    assert packed["video_indices"].numel() == 5
    assert packed["audio_indices"].numel() == 6
    assert torch.equal(
        packed["audio_plane_ids"],
        torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2]),
    )


@pytest.mark.parametrize("modality", ["video", "audio"])
def test_single_target_stream_has_stable_empty_other_stream(modality: str) -> None:
    sample = {
        "mme_embedding": torch.ones(1, 2),
        "references": [],
        "target_spec": {"modalities": [modality]},
    }
    if modality == "video":
        geometry = TargetGeometry(visual=VisualGeometry((1, 2, 2, 2), 1.0))
        expected = (0, 3)
        empty_key = "target_audio_rows"
    else:
        geometry = TargetGeometry(audio=AudioGeometry((1, 3, 2), 1.0))
        expected = (0, 4)
        empty_key = "target_video_rows"

    packed = pack_runtime_sample(
        sample,
        target_geometry=geometry,
        visual_row_width=4,
        audio_row_width=3,
    )

    assert packed[empty_key].shape == expected
    assert packed[empty_key].ndim == 2


def test_runtime_packer_rejects_latents_incompatible_with_model_width() -> None:
    sample = {
        "mme_embedding": torch.ones(1, 2),
        "references": [],
        "target": {
            "video": {
                "latent": torch.ones(1, 2, 2, 2),
                "effective_fps": 1.0,
            }
        },
    }

    with pytest.raises(ValueError, match="does not match model width"):
        RuntimePacker(visual_row_width=8).pack(sample)


def test_runtime_packer_rejects_mme_incompatible_with_model_width() -> None:
    sample = {
        "mme_embedding": torch.ones(1, 2),
        "references": [],
        "target": {
            "video": {
                "latent": torch.ones(1, 2, 2, 2),
                "effective_fps": 1.0,
            }
        },
    }

    with pytest.raises(ValueError, match="model text width"):
        RuntimePacker(text_width=3).pack(sample)


def test_runtime_packer_samples_normalizes_and_uses_codec_time_mapping() -> None:
    packed = RuntimePacker(latent_codec=_h3_test_codec()).pack(
        {
            "mme_embedding": torch.ones(1, 2),
            "references": [],
            "target": {
                "video": {
                    "latent": _visual_moments(
                        12.0,
                        (7, 2, 2),
                        dtype=torch.bfloat16,
                    ),
                    "effective_fps": 24.0,
                }
            },
        }
    )

    torch.testing.assert_close(packed["target_video_rows"], torch.ones(7, 4))
    assert packed["target_video_rows"].dtype == torch.float32
    torch.testing.assert_close(
        packed["position_ids"][packed["video_indices"], 0],
        torch.tensor([0, 1, 5, 9, 13, 17, 18], dtype=torch.float32) / 24,
    )


def test_runtime_packer_samples_and_normalizes_reference_moments() -> None:
    packed = RuntimePacker(latent_codec=_h3_test_codec()).pack(
        {
            "mme_embedding": torch.ones(1, 2),
            "references": [
                {
                    "type": "image",
                    "image": {"latent": _visual_moments(12.0, (1, 2, 2))},
                }
            ],
            "target": {
                "video": {
                    "latent": _visual_moments(12.0, (2, 2, 2)),
                    "effective_fps": 24.0,
                }
            },
        }
    )

    torch.testing.assert_close(packed["reference_video_rows"], torch.ones(1, 4))
    torch.testing.assert_close(packed["target_video_rows"], torch.ones(2, 4))


def test_raw_target_spec_requires_resolved_latent_geometry() -> None:
    with pytest.raises(ValueError, match="target_geometry"):
        pack_runtime_sample(
            {
                "mme_embedding": torch.ones(1, 2),
                "references": [],
                "target_spec": {
                    "modalities": ["video"],
                    "num_frames": 121,
                    "fps": 24,
                    "height": 768,
                    "width": 1344,
                },
            }
        )
