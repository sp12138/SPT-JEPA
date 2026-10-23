from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vforge.models.gen_engine.codec import LatentCodecConfig
from vforge.models.gen_engine.inference import (
    FlowMatchEulerScheduler,
    GenEngineInference,
    target_geometry_from_gen_config,
)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


class _Scheduler:
    def __init__(self, shift: float) -> None:
        self.shift = shift
        self.timesteps = torch.empty(0)
        self.predictions: list[torch.Tensor] = []

    def set_timesteps(self, steps: int, *, device: torch.device) -> None:
        self.timesteps = torch.linspace(0.25, 0.75, steps, device=device)

    def step(
        self,
        prediction: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> SimpleNamespace:
        del timestep
        self.predictions.append(prediction.detach().clone())
        return SimpleNamespace(prev_sample=sample - 0.1 * prediction)


class _SchedulerFactory:
    def __init__(self) -> None:
        self.schedulers: list[_Scheduler] = []

    def __call__(self, shift: float) -> _Scheduler:
        scheduler = _Scheduler(shift)
        self.schedulers.append(scheduler)
        return scheduler


class _EngineForward:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            sample=torch.ones_like(kwargs["hidden_states"]),
            audio_sample=torch.ones_like(kwargs["audio_hidden_states"]),
        )


def _generation_config() -> dict:
    return {
        "text_dim": 5,
        "in_channels": 1,
        "audio_in_channels": 3,
        "audio_channels": 2,
        "visual_spatial_compression": 2,
        "visual_temporal_compression": 2,
        "visual_effective_fps": 2.0,
        "audio_effective_fps": 2.0,
    }


def _latent_codec(
    *,
    visual_mean: float = 0.0,
    visual_std: float = 1.0,
) -> LatentCodecConfig:
    return LatentCodecConfig.from_split_dicts(
        {
            "schema": "vforge.video_latent_codec",
            "schema_version": 1,
            "latent_channels": 1,
            "spatial_compression": 2,
            "normalization": {
                "type": "channelwise_mean_std",
                "mean": [visual_mean],
                "std": [visual_std],
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
            "latent_channels": 3,
            "planes": 2,
            "normalization": {
                "type": "channelwise_mean_std",
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
            "temporal": {
                "input_rate_unit": "samples_per_second",
                "mapping": {"type": "stride", "input_stride": 800},
            },
        },
    )


def _target_spec(modalities: list[str]) -> dict:
    spec = {
        "modalities": modalities,
        "duration_seconds": 1.0,
    }
    if "video" in modalities:
        spec["video"] = {
            "height": 4,
            "width": 4,
            "fps": 4,
            "num_frames": 4,
        }
    if "audio" in modalities:
        spec["audio"] = {
            "sample_rate": 8,
            "num_samples": 8,
        }
    return spec


def test_default_flow_scheduler_integrates_data_ward_velocity() -> None:
    scheduler = FlowMatchEulerScheduler(shift=3.0)
    scheduler.set_timesteps(4, device=torch.device("cpu"))
    noise = torch.tensor([[-2.0]])
    clean = torch.tensor([[5.0]])
    sample = noise.clone()
    velocity = clean - noise

    for timestep in scheduler.timesteps:
        sample = scheduler.step(velocity, timestep, sample).prev_sample

    torch.testing.assert_close(sample, clean)
    assert scheduler.timesteps[0] == 0


def test_target_geometry_converts_pixel_and_waveform_units() -> None:
    geometry = target_geometry_from_gen_config(
        {
            "modalities": ["video", "audio"],
            "video": {
                "height": 8,
                "width": 12,
                "fps": 6,
                "num_frames": 9,
            },
            "audio": {
                "sample_rate": 8_000,
                "num_samples": 16_000,
            },
        },
        {
            "visual_channels": 4,
            "visual_spatial_compression": 4,
            "visual_temporal_compression": 3,
            "visual_effective_fps": 2,
            "audio_channels": 2,
            "audio_features": 7,
            "audio_effective_fps": 5,
        },
    )

    assert geometry.visual is not None
    assert geometry.visual.shape == (4, 3, 2, 3)
    assert geometry.visual.effective_fps == 2
    assert geometry.audio is not None
    assert geometry.audio.shape == (2, 7, 10)
    assert geometry.audio.effective_fps == 5


def test_target_geometry_accepts_explicit_latent_shapes() -> None:
    geometry = target_geometry_from_gen_config(
        {
            "modalities": ["video", "audio"],
            "video": {
                "height": 8,
                "width": 12,
                "fps": 6,
                "num_frames": 9,
                "latent_shape": [4, 5, 2, 3],
                "effective_fps": 2.5,
            },
            "audio": {
                "sample_rate": 8_000,
                "num_samples": 16_000,
                "latent_shape": [2, 7, 11],
                "effective_fps": 5.5,
            },
        },
        {
            "visual_channels": 4,
            "audio_features": 7,
        },
    )

    assert geometry.visual is not None
    assert geometry.visual.shape == (4, 5, 2, 3)
    assert geometry.visual.effective_fps == 2.5
    assert geometry.audio is not None
    assert geometry.audio.shape == (2, 7, 11)
    assert geometry.audio.effective_fps == 5.5


def test_target_geometry_rejects_nondivisible_pixels() -> None:
    with pytest.raises(ValueError, match="divisible"):
        target_geometry_from_gen_config(
            {
                "modalities": ["video"],
                "duration_seconds": 1,
                "video": {"height": 5, "width": 4, "fps": 4},
            },
            {
                "visual_channels": 1,
                "visual_spatial_compression": 2,
                "visual_temporal_compression": 2,
            },
        )


def test_codec_target_geometry_uses_h3_video_and_audio_sampling() -> None:
    geometry = target_geometry_from_gen_config(
        {
            "modalities": ["video", "audio"],
            "video": {
                "height": 32,
                "width": 64,
                "fps": 24,
                "num_frames": 124,
            },
            "audio": {
                "sample_rate": 32_000,
                "num_samples": 165_333,
            },
        },
        _generation_config(),
        _latent_codec(),
    )

    assert geometry.visual is not None
    assert geometry.visual.shape == (1, 37, 16, 32)
    assert geometry.visual.effective_fps == 24
    assert geometry.audio is not None
    assert geometry.audio.shape == (2, 3, 207)
    assert geometry.audio.effective_fps == 32_000


def test_inference_denormalizes_model_space_video_before_returning() -> None:
    def run(codec: LatentCodecConfig) -> torch.Tensor:
        inference = GenEngineInference(
            model=_Model(),
            engine_forward=_EngineForward(),
            generation_config=_generation_config(),
            latent_codec=codec,
            num_inference_steps=1,
            scheduler_factory=_SchedulerFactory(),
        )
        return inference.infer_sample(
            {
                "sample_id": "denormalize",
                "mme_embedding": torch.ones(2, 5),
                "references": [],
                "target_spec": {
                    "modalities": ["video"],
                    "video": {
                        "height": 4,
                        "width": 4,
                        "fps": 4,
                        "num_frames": 5,
                    },
                },
            },
            seed=19,
        )["video_latent"]

    normalized = run(_latent_codec())
    denormalized = run(_latent_codec(visual_mean=10.0, visual_std=2.0))

    torch.testing.assert_close(denormalized, normalized * 2.0 + 10.0)


def test_validation_target_latent_sets_geometry_but_is_not_used_as_noise() -> None:
    def run(clean: torch.Tensor) -> torch.Tensor:
        inference = GenEngineInference(
            model=_Model(),
            engine_forward=_EngineForward(),
            generation_config=_generation_config(),
            num_inference_steps=1,
            scheduler_factory=_SchedulerFactory(),
        )
        result = inference.infer_sample(
            {
                "sample_id": "validation",
                "mme_embedding": torch.ones(2, 5),
                "references": [],
                "target": {
                    "video": {
                        "latent": clean,
                        "effective_fps": 2.0,
                    }
                },
            },
            seed=9,
        )
        return result["video_latent"]

    zeros = run(torch.zeros(1, 2, 2, 2))
    ones = run(torch.ones(1, 2, 2, 2))

    torch.testing.assert_close(zeros, ones)


@pytest.mark.parametrize(
    ("modalities", "with_reference"),
    [
        (["video"], False),
        (["audio"], False),
        (["video", "audio"], True),
    ],
)
def test_denoises_video_audio_and_av_without_loading_vae(
    modalities: list[str],
    with_reference: bool,
) -> None:
    references = []
    if with_reference:
        references = [
            {
                "type": "video_audio",
                "video": {
                    "latent": torch.ones(1, 1, 2, 2),
                    "effective_fps": 2.0,
                },
                "audio": {
                    "latent": torch.ones(2, 3, 1),
                    "effective_fps": 2.0,
                },
            }
        ]
    sample = {
        "sample_id": "sample",
        "mme_embedding": torch.ones(2, 5),
        "references": references,
        "target_spec": _target_spec(modalities),
        "metadata": {"split": "test"},
    }
    engine_forward = _EngineForward()
    scheduler_factory = _SchedulerFactory()
    inference = GenEngineInference(
        model=_Model(),
        engine_forward=engine_forward,
        generation_config=_generation_config(),
        num_inference_steps=2,
        scheduler_factory=scheduler_factory,
    )

    result = inference.infer_sample(sample, seed=123)

    assert len(engine_forward.calls) == 2
    first = engine_forward.calls[0]
    assert first["encoder_hidden_states"].shape == (1, 2, 5)
    assert first["reference_modality_ids"].shape == first["reference_index_ids"].shape
    assert first["position_ids"].shape[0] == first["reference_modality_ids"].numel()
    assert first["hidden_states"].shape[-1] == 4
    assert first["audio_hidden_states"].shape[-1] == 3
    assert first["audio_plane_ids"].shape == first["position_ids"].shape[:1]
    assert len(scheduler_factory.schedulers) == len(modalities)

    if "video" in modalities:
        assert result["video_latent"].shape == (1, 2, 2, 2)
        video_scheduler = next(
            scheduler for scheduler in scheduler_factory.schedulers if scheduler.shift == 12.0
        )
        assert [prediction.shape[1] for prediction in video_scheduler.predictions] == [2, 2]
    else:
        assert "video_latent" not in result
        assert first["hidden_states"].shape == (1, 0, 4)

    if "audio" in modalities:
        assert result["audio_latent"].shape == (2, 3, 2)
        assert set(first["audio_plane_ids"][first["audio_indices"]].tolist()) == {1, 2}
        audio_scheduler = next(
            scheduler for scheduler in scheduler_factory.schedulers if scheduler.shift == 3.0
        )
        assert [prediction.shape[1] for prediction in audio_scheduler.predictions] == [4, 4]
    else:
        assert "audio_latent" not in result
        assert first["audio_hidden_states"].shape == (1, 0, 3)

    if with_reference:
        # The scheduler only sees target predictions after condition rows are stripped.
        assert first["hidden_states"].shape[1] == 3
        assert first["audio_hidden_states"].shape[1] == 6
        video_positions = first["position_ids"][first["video_indices"], 0]
        audio_positions = first["position_ids"][first["audio_indices"], 0]
        assert video_positions[1].item() == audio_positions[2].item() == 1.0


def test_native_transformer_runs_one_latent_inference_step() -> None:
    from vforge.distributed.parallel_state import ParallelState
    from vforge.models.gen_engine.dit import GenEngineTransformer3DModel
    from vforge.ops import initialize_ops, reset_ops
    from vforge.ops.config import OpsConfig

    model = GenEngineTransformer3DModel(
        architecture="mmdit_dense",
        num_attention_heads=1,
        attention_head_dim=8,
        hidden_size=8,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=16,
        in_channels=1,
        audio_in_channels=3,
        patch_size=(1, 1, 1),
        text_dim=2,
        freq_dim=4,
        time_embed_hidden_dim=8,
        time_embed_dim=4,
        rope_freq_dim=1,
        max_reference_index=4,
    ).eval()
    inference = GenEngineInference(
        model=model,
        engine_forward=lambda **kwargs: model(**kwargs),
        generation_config={
            "text_dim": 2,
            "in_channels": 1,
            "audio_in_channels": 3,
            "visual_spatial_compression": 1,
            "visual_effective_fps": 1.0,
            "audio_effective_fps": 1.0,
        },
        num_inference_steps=1,
        patch_size=(1, 1, 1),
    )
    reset_ops()
    initialize_ops(
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=torch.device("cpu"),
        parallel_state=ParallelState(),
    )
    try:
        result = inference.infer_sample(
            {
                "sample_id": "native-inference",
                "mme_embedding": torch.ones(1, 2),
                "references": [],
                "target_spec": {
                    "modalities": ["video", "audio"],
                    "duration_seconds": 1.0,
                    "video": {"height": 1, "width": 1, "fps": 1},
                    "audio": {"sample_rate": 1},
                },
            },
            seed=7,
        )
    finally:
        reset_ops()

    assert result["video_latent"].shape == (1, 1, 1, 1)
    assert result["audio_latent"].shape == (2, 3, 1)
    assert torch.isfinite(result["video_latent"]).all()
    assert torch.isfinite(result["audio_latent"]).all()
