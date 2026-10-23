from __future__ import annotations

import json

import pytest
import torch

from vforge.models.gen_engine.codec import (
    LatentCodecConfig,
    PeriodicOffsetsTemporalMapping,
)


def _video_config() -> dict:
    return {
        "schema": "vforge.video_latent_codec",
        "schema_version": 1,
        "latent_channels": 2,
        "spatial_compression": 16,
        "posterior_sampling": {
            "type": "diagonal_gaussian",
            "logvar_min": -30.0,
            "logvar_max": 20.0,
        },
        "normalization": {
            "type": "channelwise_mean_std",
            "mean": [1.0, -1.0],
            "std": [2.0, 4.0],
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
    }


def _audio_config() -> dict:
    return {
        "schema": "vforge.audio_latent_codec",
        "schema_version": 1,
        "latent_channels": 3,
        "planes": 2,
        "normalization": {
            "type": "channelwise_mean_std",
            "mean": [1.0, 2.0, 3.0],
            "std": [2.0, 4.0, 5.0],
        },
        "temporal": {
            "input_rate_unit": "samples_per_second",
            "mapping": {"type": "stride", "input_stride": 800},
        },
    }


def _codec_config() -> LatentCodecConfig:
    return LatentCodecConfig.from_split_dicts(_video_config(), _audio_config())


def test_h3_periodic_mapping_aligns_frames_and_uses_input_clock() -> None:
    mapping = PeriodicOffsetsTemporalMapping(
        input_period=17,
        latent_offsets=(0, 1, 5, 9, 13),
        valid_length_remainder=5,
    )

    assert mapping.align_input_length(121) == 124
    assert mapping.latent_frames(121) == 37
    assert mapping.latent_frames(124) == 37
    torch.testing.assert_close(
        mapping.local_times(7, input_rate=24),
        torch.tensor([0, 1, 5, 9, 13, 17, 18], dtype=torch.float32) / 24,
    )
    assert mapping.local_times(37, input_rate=24)[-1].item() == 5.0


def test_h3_audio_mapping_ends_at_5_15_seconds_for_207_latents() -> None:
    codec = _codec_config()

    times = codec.audio.temporal.local_times(207, input_rate=32_000)

    torch.testing.assert_close(times[-1], torch.tensor(5.15))


def test_codec_normalizes_and_denormalizes_visual_and_audio_channels() -> None:
    codec = _codec_config()
    visual = torch.tensor(
        [
            [[[1.0]], [[3.0]]],
            [[[-1.0]], [[7.0]]],
        ]
    )
    audio = torch.tensor(
        [
            [[1.0, 3.0], [2.0, 6.0], [3.0, 8.0]],
            [[5.0, 1.0], [10.0, 2.0], [13.0, 3.0]],
        ]
    )

    normalized_visual = codec.visual.normalize(visual)
    normalized_audio = codec.audio.normalize(audio)

    torch.testing.assert_close(
        normalized_visual,
        torch.tensor(
            [
                [[[0.0]], [[1.0]]],
                [[[0.0]], [[2.0]]],
            ]
        ),
    )
    torch.testing.assert_close(
        normalized_audio,
        torch.tensor(
            [
                [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                [[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]],
            ]
        ),
    )
    torch.testing.assert_close(codec.visual.denormalize(normalized_visual), visual)
    torch.testing.assert_close(codec.audio.denormalize(normalized_audio), audio)


def test_visual_codec_samples_raw_posterior_moments_before_normalizing() -> None:
    codec = _codec_config()
    mean = torch.tensor(
        [
            [[[2.0]], [[3.0]]],
            [[[7.0]], [[11.0]]],
        ]
    )
    logvar = torch.zeros_like(mean)
    moments = torch.cat((mean, logvar), dim=0)
    mean_shift = torch.tensor([1.0, -1.0]).view(2, 1, 1, 1)
    std_scale = torch.tensor([2.0, 4.0]).view(2, 1, 1, 1)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        expected = (mean + torch.randn_like(mean) - mean_shift) / std_scale
        torch.manual_seed(7)
        actual = codec.visual.to_model_space(moments)
        second = codec.visual.to_model_space(moments)

    assert codec.visual.input_channels == 4
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(second, actual)


def test_codec_rejects_posterior_sampling_seed() -> None:
    config = _video_config()
    config["posterior_sampling"]["seed"] = 42
    with pytest.raises(ValueError, match="unsupported fields"):
        LatentCodecConfig.from_split_dicts(config, _audio_config())


def test_visual_codec_rejects_sampled_latent_when_raw_moments_are_required() -> None:
    codec = _codec_config()

    with pytest.raises(ValueError, match="expected 4"):
        codec.visual.to_model_space(torch.zeros(2, 1, 1, 1))


def test_codec_round_trips_split_configs(tmp_path) -> None:
    codec = _codec_config()

    assert (
        LatentCodecConfig.from_split_dicts(
            codec.video_config_dict(),
            codec.audio_config_dict(),
        )
        == codec
    )

    video_path = tmp_path / "video.json"
    audio_path = tmp_path / "audio.json"
    video_path.write_text(json.dumps(codec.video_config_dict()), encoding="utf-8")
    audio_path.write_text(json.dumps(codec.audio_config_dict()), encoding="utf-8")
    assert LatentCodecConfig.from_files(video_path, audio_path) == codec


def test_codec_rejects_invalid_channel_stats_and_temporal_offsets() -> None:
    invalid_std = _video_config()
    invalid_std["normalization"]["std"][0] = 0.0
    with pytest.raises(ValueError, match="positive"):
        LatentCodecConfig.from_split_dicts(invalid_std, _audio_config())

    invalid_offsets = _video_config()
    invalid_offsets["temporal"]["mapping"]["latent_offsets"] = [0, 5, 1]
    with pytest.raises(ValueError, match="strictly increasing"):
        LatentCodecConfig.from_split_dicts(invalid_offsets, _audio_config())

    invalid_logvar = _video_config()
    invalid_logvar["posterior_sampling"]["logvar_min"] = 20.0
    with pytest.raises(ValueError, match="logvar_min"):
        LatentCodecConfig.from_split_dicts(invalid_logvar, _audio_config())
