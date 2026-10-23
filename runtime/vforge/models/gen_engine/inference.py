"""Latent-only inference for the precomputed multimodal generation contract."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import torch

from .codec import LatentCodecConfig
from .geometry import (
    _audio_features,
    _positive_integer,
    _positive_number,
    _required_config_value,
    _visual_channels,
    target_geometry_from_gen_config,
)
from .layout import AudioGeometry, TargetGeometry, VisualGeometry
from .packing import RuntimePacker
from .preparation import prepare_inference_sample, unpack_inference_sample
from .timesteps import build_row_timesteps

SchedulerFactory = Callable[[float], Any]
TimestepBuilder = Callable[..., tuple[torch.Tensor, torch.Tensor]]
Decoder = Callable[[dict[str, Any]], Mapping[str, Any]]


class FlowMatchEulerScheduler:
    """Minimal data-ward Euler sampler matching the training flow parameterization."""

    def __init__(self, shift: float = 1.0) -> None:
        self.shift = _positive_number(shift, label="scheduler shift")
        self.timesteps = torch.empty(0)
        self._next_timesteps = torch.empty(0)
        self._step_index = 0

    def set_timesteps(self, steps: int, *, device: torch.device) -> None:
        if isinstance(steps, bool) or int(steps) < 1:
            raise ValueError("scheduler steps must be at least 1")
        base_sigma = torch.linspace(1.0, 0.0, int(steps) + 1, device=device)
        sigma = self.shift * base_sigma / (1.0 + (self.shift - 1.0) * base_sigma)
        data_timesteps = 1.0 - sigma
        self.timesteps = data_timesteps[:-1]
        self._next_timesteps = data_timesteps[1:]
        self._step_index = 0

    def step(
        self,
        prediction: torch.Tensor,
        timestep: torch.Tensor | float,
        sample: torch.Tensor,
    ) -> SimpleNamespace:
        if self._step_index >= len(self.timesteps):
            raise RuntimeError("scheduler.step() called after the configured schedule")
        current = self.timesteps[self._step_index]
        supplied = torch.as_tensor(timestep, device=current.device, dtype=current.dtype)
        if not torch.allclose(supplied, current):
            raise ValueError(
                f"scheduler timestep mismatch: expected {float(current)}, got {float(supplied)}"
            )
        delta = self._next_timesteps[self._step_index] - current
        self._step_index += 1
        return SimpleNamespace(prev_sample=sample + delta.to(sample.dtype) * prediction)


def _default_scheduler_factory(shift: float) -> FlowMatchEulerScheduler:
    return FlowMatchEulerScheduler(shift=shift)


def _model_device(model: Any) -> torch.device:
    for values in (model.parameters(), model.buffers()):
        for value in values:
            if value.device.type != "meta":
                return value.device
    return torch.device("cpu")


def _previous_sample(step_output: Any) -> torch.Tensor:
    if isinstance(step_output, Mapping):
        value = step_output.get("prev_sample")
    else:
        value = getattr(step_output, "prev_sample", None)
    if not isinstance(value, torch.Tensor):
        raise TypeError("scheduler.step() must return an object containing prev_sample")
    return value


def _stream_output(output: Any, *, stream: str) -> torch.Tensor:
    names = ("sample", "video_sample", "video") if stream == "video" else ("audio_sample", "audio")
    value = None
    if isinstance(output, Mapping):
        for name in names:
            if isinstance(output.get(name), torch.Tensor):
                value = output[name]
                break
    else:
        for name in names:
            candidate = getattr(output, name, None)
            if isinstance(candidate, torch.Tensor):
                value = candidate
                break
    if value is None and isinstance(output, tuple):
        index = 0 if stream == "video" else 1
        if len(output) > index and isinstance(output[index], torch.Tensor):
            value = output[index]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"engine_forward output is missing the {stream} prediction tensor")
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3:
        raise ValueError(f"{stream} prediction must have rank 3, got {tuple(value.shape)}")
    return value


def _target_prediction(
    output: Any,
    *,
    stream: str,
    indices: torch.Tensor,
    sequence_length: int,
    reference_rows: int,
    target_rows: int,
) -> torch.Tensor:
    prediction = _stream_output(output, stream=stream)
    if prediction.shape[1] == sequence_length:
        prediction = prediction.index_select(1, indices.to(prediction.device))
    expected_rows = reference_rows + target_rows
    if prediction.shape[1] != expected_rows:
        raise ValueError(
            f"{stream} prediction has {prediction.shape[1]} rows, expected {expected_rows}"
        )
    return prediction[:, reference_rows:]


def unpatchify_visual(
    rows: torch.Tensor,
    *,
    geometry: VisualGeometry,
    patch_size: Sequence[int],
) -> torch.Tensor:
    """Invert visual patch rows into ``[B,C,T,H,W]`` latent tensors."""

    if rows.ndim == 2:
        rows = rows.unsqueeze(0)
    if rows.ndim != 3:
        raise ValueError(f"visual rows must have rank 3, got {tuple(rows.shape)}")
    patch_t, patch_h, patch_w = (int(value) for value in patch_size)
    grid_t, grid_h, grid_w = geometry.grid_shape((patch_t, patch_h, patch_w))
    expected_width = geometry.row_width((patch_t, patch_h, patch_w))
    if rows.shape[1:] != (grid_t * grid_h * grid_w, expected_width):
        raise ValueError(
            "visual rows do not match target latent geometry: "
            f"rows={tuple(rows.shape)} geometry={geometry.shape}"
        )
    channels = geometry.channels
    return (
        rows.reshape(
            rows.shape[0],
            grid_t,
            grid_h,
            grid_w,
            channels,
            patch_t,
            patch_h,
            patch_w,
        )
        .permute(0, 4, 1, 5, 2, 6, 3, 7)
        .reshape(rows.shape[0], channels, *geometry.shape[1:])
        .contiguous()
    )


def unpatchify_audio(rows: torch.Tensor, *, geometry: AudioGeometry) -> torch.Tensor:
    """Invert audio rows into ``[B,A,C,T]`` latent tensors."""

    if rows.ndim == 2:
        rows = rows.unsqueeze(0)
    if rows.ndim != 3 or rows.shape[1:] != (geometry.row_count, geometry.row_width):
        raise ValueError(
            "audio rows do not match target latent geometry: "
            f"rows={tuple(rows.shape)} geometry={geometry.shape}"
        )
    return (
        rows.reshape(
            rows.shape[0],
            geometry.num_channels,
            geometry.num_frames,
            geometry.row_width,
        )
        .permute(0, 1, 3, 2)
        .contiguous()
    )


class GenEngineInference:
    """Denoise precomputed bundles and return VAE-space generated latents."""

    def __init__(
        self,
        *,
        model: Any,
        engine_forward: Callable[..., Any],
        generation_config: Any,
        num_inference_steps: int,
        video_shift: float = 12.0,
        audio_shift: float = 3.0,
        patch_size: Sequence[int] = (1, 2, 2),
        latent_codec: LatentCodecConfig | None = None,
        reference_timestep: float = 1.0,
        text_timestep: float = 1.0,
        scheduler_factory: SchedulerFactory | None = None,
        timestep_builder: TimestepBuilder = build_row_timesteps,
        decoder: Decoder | None = None,
        fixed_seed: int | None = None,
    ) -> None:
        if isinstance(num_inference_steps, bool) or int(num_inference_steps) < 1:
            raise ValueError("num_inference_steps must be at least 1")
        self.model = model
        self.engine_forward = engine_forward
        self.generation_config = generation_config
        self.num_inference_steps = int(num_inference_steps)
        self.video_shift = _positive_number(video_shift, label="video_shift")
        self.audio_shift = _positive_number(audio_shift, label="audio_shift")
        self.reference_timestep = float(reference_timestep)
        self.text_timestep = float(text_timestep)
        if not 0 <= self.reference_timestep <= 1:
            raise ValueError("reference_timestep must be in [0, 1]")
        if not 0 <= self.text_timestep <= 1:
            raise ValueError("text_timestep must be in [0, 1]")
        self.scheduler_factory = scheduler_factory or _default_scheduler_factory
        self.timestep_builder = timestep_builder
        self.decoder = decoder
        self.fixed_seed = fixed_seed
        self.latent_codec = latent_codec
        normalized_patch_size = tuple(int(value) for value in patch_size)
        self.text_width = _positive_integer(
            _required_config_value(
                generation_config,
                ("text_dim", "mme_dim"),
                label="text_dim",
                sections=("text", "mme"),
                section_names=("dim", "hidden_size"),
            ),
            label="generation config text_dim",
        )
        self.visual_row_width = _visual_channels(generation_config) * math.prod(
            normalized_patch_size
        )
        self.audio_row_width = _audio_features(generation_config)
        self.packer = RuntimePacker(
            normalized_patch_size,
            text_width=self.text_width,
            visual_row_width=self.visual_row_width,
            audio_row_width=self.audio_row_width,
            latent_codec=latent_codec,
        )

    def resolve_seed(self, sample_id: str, default: int) -> int:
        del sample_id
        return int(self.fixed_seed) if self.fixed_seed is not None else int(default)

    def _pack(
        self,
        sample: Mapping[str, Any],
    ) -> tuple[dict[str, Any], TargetGeometry]:
        if "packed" not in sample:
            sample = prepare_inference_sample(
                sample,
                packer=self.packer,
                generation_config=self.generation_config,
            )
        return unpack_inference_sample(sample)

    @staticmethod
    def _reference_rows(
        rows: torch.Tensor,
        *,
        expected_width: int,
        device: torch.device,
    ) -> torch.Tensor:
        if rows.ndim != 2:
            raise ValueError(f"packed reference rows must have rank 2, got {tuple(rows.shape)}")
        if rows.shape[0] and rows.shape[1] != expected_width:
            raise ValueError(
                f"packed reference row width={rows.shape[1]} does not match model "
                f"width={expected_width}"
            )
        if not rows.shape[0]:
            rows = torch.empty((0, expected_width), dtype=torch.float32)
        return rows.unsqueeze(0).to(device=device, dtype=torch.float32)

    @staticmethod
    def _augment_reference(
        rows: torch.Tensor,
        *,
        timestep: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if not rows.numel() or timestep == 1.0:
            return rows
        noise = torch.randn(
            rows.shape,
            generator=generator,
            device=rows.device,
            dtype=rows.dtype,
        )
        return timestep * rows + (1.0 - timestep) * noise

    @torch.no_grad()
    def infer_sample(self, sample: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
        packed, target_geometry = self._pack(sample)
        layout = SimpleNamespace(**packed)
        device = _model_device(self.model)
        generator = torch.Generator(device=device).manual_seed(int(seed))

        reference_video = self._reference_rows(
            packed["reference_video_rows"],
            expected_width=self.visual_row_width,
            device=device,
        )
        reference_audio = self._reference_rows(
            packed["reference_audio_rows"],
            expected_width=self.audio_row_width,
            device=device,
        )
        reference_video = self._augment_reference(
            reference_video,
            timestep=self.reference_timestep,
            generator=generator,
        )
        reference_audio = self._augment_reference(
            reference_audio,
            timestep=self.reference_timestep,
            generator=generator,
        )
        target_video = (
            torch.randn(
                (1, layout.num_target_video_rows, self.visual_row_width),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            if layout.num_target_video_rows
            else torch.empty((1, 0, self.visual_row_width), device=device)
        )
        target_audio = (
            torch.randn(
                (1, layout.num_target_audio_rows, self.audio_row_width),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
            if layout.num_target_audio_rows
            else torch.empty((1, 0, self.audio_row_width), device=device)
        )

        video_scheduler = (
            self.scheduler_factory(self.video_shift) if layout.num_target_video_rows else None
        )
        audio_scheduler = (
            self.scheduler_factory(self.audio_shift) if layout.num_target_audio_rows else None
        )
        if video_scheduler is not None:
            video_scheduler.set_timesteps(self.num_inference_steps, device=device)
        if audio_scheduler is not None:
            audio_scheduler.set_timesteps(self.num_inference_steps, device=device)
        driving_scheduler = video_scheduler or audio_scheduler
        assert driving_scheduler is not None
        if (
            video_scheduler is not None
            and audio_scheduler is not None
            and len(video_scheduler.timesteps) != len(audio_scheduler.timesteps)
        ):
            raise ValueError("video and audio schedulers produced different timestep counts")

        structural = {
            key: packed[key].to(device)
            for key in (
                "token_tags",
                "position_ids",
                "video_indices",
                "audio_indices",
                "text_indices",
                "reference_modality_ids",
                "reference_index_ids",
                "audio_plane_ids",
            )
        }
        mme_embedding = packed["mme_embedding"].to(device)
        if mme_embedding.ndim == 2:
            mme_embedding = mme_embedding.unsqueeze(0)
        sequence_length = int(structural["position_ids"].shape[0])

        for step_index in range(len(driving_scheduler.timesteps)):
            video_timestep = (
                video_scheduler.timesteps[step_index]
                if video_scheduler is not None
                else audio_scheduler.timesteps[step_index]
            )
            audio_timestep = (
                audio_scheduler.timesteps[step_index]
                if audio_scheduler is not None
                else video_scheduler.timesteps[step_index]
            )
            video_value = float(video_timestep)
            audio_value = float(audio_timestep)
            timesteps, timestep_indices = self.timestep_builder(
                video_indices=structural["video_indices"],
                audio_indices=structural["audio_indices"],
                text_indices=structural["text_indices"],
                num_reference_video_rows=int(layout.num_reference_video_rows),
                num_reference_audio_rows=int(layout.num_reference_audio_rows),
                video_timestep=video_value,
                audio_timestep=audio_value,
                reference_timestep=self.reference_timestep,
                text_timestep=self.text_timestep,
            )
            model_video = torch.cat((reference_video, target_video), dim=1)
            model_audio = torch.cat((reference_audio, target_audio), dim=1)
            output = self.engine_forward(
                hidden_states=model_video,
                audio_hidden_states=model_audio,
                encoder_hidden_states=mme_embedding,
                timestep=timesteps.to(device),
                timestep_indices=timestep_indices.to(device),
                **structural,
            )
            if video_scheduler is not None:
                video_prediction = _target_prediction(
                    output,
                    stream="video",
                    indices=structural["video_indices"],
                    sequence_length=sequence_length,
                    reference_rows=int(layout.num_reference_video_rows),
                    target_rows=int(layout.num_target_video_rows),
                )
                target_video = _previous_sample(
                    video_scheduler.step(
                        video_prediction,
                        video_timestep,
                        target_video,
                    )
                )
            if audio_scheduler is not None:
                audio_prediction = _target_prediction(
                    output,
                    stream="audio",
                    indices=structural["audio_indices"],
                    sequence_length=sequence_length,
                    reference_rows=int(layout.num_reference_audio_rows),
                    target_rows=int(layout.num_target_audio_rows),
                )
                target_audio = _previous_sample(
                    audio_scheduler.step(
                        audio_prediction,
                        audio_timestep,
                        target_audio,
                    )
                )

        geometry: dict[str, Any] = {
            "target_start": float(layout.target_start),
            "target_end": float(layout.target_end),
        }
        if isinstance(sample.get("target_spec"), Mapping):
            geometry["requested"] = dict(sample["target_spec"])
        result: dict[str, Any] = {
            "sample_id": str(sample.get("sample_id") or "sample"),
            "seed": int(seed),
            "geometry": geometry,
            "metadata": dict(sample.get("metadata") or {}),
        }
        if layout.target_visual_shape is not None:
            visual = target_geometry.visual
            assert visual is not None
            video_latent = unpatchify_visual(
                target_video,
                geometry=visual,
                patch_size=self.packer.patch_size,
            )[0]
            if self.latent_codec is not None:
                video_latent = self.latent_codec.visual.denormalize(video_latent)
            result["video_latent"] = video_latent.detach().cpu()
            geometry["video"] = {
                "shape": list(visual.shape),
                "effective_fps": float(visual.effective_fps),
            }
        if layout.target_audio_shape is not None:
            audio_geometry = target_geometry.audio
            assert audio_geometry is not None
            audio_latent = unpatchify_audio(target_audio, geometry=audio_geometry)[0]
            if self.latent_codec is not None:
                audio_latent = self.latent_codec.audio.denormalize(audio_latent)
            result["audio_latent"] = audio_latent.detach().cpu()
            geometry["audio"] = {
                "shape": list(audio_geometry.shape),
                "effective_fps": float(audio_geometry.effective_fps),
            }
        if self.decoder is not None:
            decoded = self.decoder(result)
            if not isinstance(decoded, Mapping):
                raise TypeError("decoder callback must return a mapping")
            result["decoded"] = dict(decoded)
        return result


resolve_target_geometry = target_geometry_from_gen_config


__all__ = [
    "FlowMatchEulerScheduler",
    "GenEngineInference",
    "build_row_timesteps",
    "resolve_target_geometry",
    "target_geometry_from_gen_config",
    "unpatchify_audio",
    "unpatchify_visual",
]
