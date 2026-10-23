"""Frozen condition assets owned by the Wan Taichu model family.

This module keeps only the input semantics needed by Wan Taichu.  The VAE,
UMT5, CLIP and DINO implementations remain third-party frozen assets; their
construction belongs to the task layer.  No versioned official-Wan model is
imported here.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.video_processor import VideoProcessor

from vforge.models.wan_taichu.grid_renderer import camera_to_grid
from vforge.utils.inference_trace import get_inference_trace

WanTaichuEncoding = Literal["wan21", "ti2v"]

_VAE_CHANNELS_LAST_3D_BOUNDARIES = {"WanAttentionBlock", "WanResample"}


def _patchify_channels_last_3d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Keep Diffusers' Wan patch order while producing an NDHWC-contiguous view."""
    if patch_size == 1:
        return x
    if x.ndim != 5:
        raise ValueError(f"invalid Wan VAE input shape: {tuple(x.shape)}")
    batch, channels, frames, height, width = x.shape
    if height % patch_size or width % patch_size:
        raise ValueError(
            f"height={height} and width={width} must be divisible by patch_size={patch_size}"
        )
    patch_height = height // patch_size
    patch_width = width // patch_size
    x = x.view(
        batch,
        channels,
        frames,
        patch_height,
        patch_size,
        patch_width,
        patch_size,
    )
    x = x.permute(0, 2, 3, 5, 1, 6, 4).contiguous()
    x = x.view(
        batch,
        frames,
        patch_height,
        patch_width,
        channels * patch_size * patch_size,
    )
    return x.permute(0, 4, 1, 2, 3)


def _restore_channels_last_3d(
    _module: nn.Module,
    _inputs: tuple[Any, ...],
    output: torch.Tensor,
) -> torch.Tensor:
    return output.contiguous(memory_format=torch.channels_last_3d)


def _enable_vae_channels_last_3d(vae: nn.Module) -> list[Any]:
    """Mirror Taichu's production Wan VAE memory-layout path."""
    vae_module = importlib.import_module(vae.__class__.__module__)
    if hasattr(vae_module, "patchify"):
        vae_module.patchify = _patchify_channels_last_3d
    encoder = getattr(vae, "encoder", vae)
    return [
        module.register_forward_hook(_restore_channels_last_3d)
        for module in encoder.modules()
        if module.__class__.__name__ in _VAE_CHANNELS_LAST_3D_BOUNDARIES
    ]


def retrieve_latents(encoder_output: Any, sample_mode: str = "sample") -> torch.Tensor:
    """Extract the latent tensor from the output variants used by Diffusers VAEs."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample()
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("could not access latents of the VAE encoder output")


class WanTaichuConditionEncoder(nn.Module):
    """VAE/UMT5 encoder with architecture-selected Taichu bundle semantics."""

    def __init__(
        self,
        *,
        encoding: WanTaichuEncoding,
        vae: nn.Module,
        text_encoder: nn.Module,
        tokenizer: Any,
        video_processor: VideoProcessor,
        training_dtype: torch.dtype,
        streams: Sequence[str],
        raw_stream_keys: Mapping[str, str],
        mask_channels: int,
        use_vae_channels_last_3d: bool = False,
        use_vae_condition: bool = False,
        use_clip_condition: bool = False,
        image_encoder: nn.Module | None = None,
        image_processor: Any | None = None,
        use_dino_condition: bool = False,
        dino_encoder: nn.Module | None = None,
        dino_processor: Any | None = None,
        use_audio_condition: bool = False,
        audio_wav2vec: nn.Module | None = None,
        audio_processor: Any | None = None,
        audio_type: str = "wav2vec2",
        audio_video_rate: int = 25,
        audio_fps: int = 25,
        audio_batch_frames: int = 121,
        audio_bucket_m: int = 0,
        audio_sample_rate: int = 16000,
        grid_random_rot_deg: float = 0.0,
    ) -> None:
        super().__init__()
        if encoding not in {"wan21", "ti2v"}:
            raise ValueError(f"unsupported Wan Taichu encoding {encoding!r}")
        if encoding == "wan21" and mask_channels != 4:
            raise ValueError(f"Wan21-shaped encoding requires mask_channels=4; got {mask_channels}")
        if encoding == "ti2v" and mask_channels != 0:
            raise ValueError(f"TI2V encoding requires mask_channels=0; got {mask_channels}")

        self.encoding = encoding
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.training_dtype = training_dtype
        self.streams = tuple(streams)
        self.raw_stream_keys = dict(raw_stream_keys)
        self.mask_channels = int(mask_channels)
        self.use_vae_channels_last_3d = bool(use_vae_channels_last_3d)
        self._vae_channels_last_3d_hook_handles = (
            _enable_vae_channels_last_3d(vae) if self.use_vae_channels_last_3d else []
        )
        self.use_vae_condition = bool(use_vae_condition)
        self.use_clip_condition = bool(use_clip_condition)
        self.image_encoder = image_encoder
        self.image_processor = image_processor
        self.use_dino_condition = bool(use_dino_condition)
        self.dino_encoder = dino_encoder
        self.dino_processor = dino_processor
        self._dino_drop_prefix = 5
        self.use_audio_condition = bool(use_audio_condition)
        self.audio_wav2vec = audio_wav2vec
        self.audio_processor = audio_processor
        if audio_type not in {"wav2vec2", "wav2bert"}:
            raise ValueError(f"audio_type must be wav2vec2 or wav2bert; got {audio_type!r}")
        self.audio_type = audio_type
        self.audio_video_rate = int(audio_video_rate)
        self.audio_fps = int(audio_fps)
        self.audio_batch_frames = int(audio_batch_frames)
        self.audio_bucket_m = int(audio_bucket_m)
        self.audio_sample_rate = int(audio_sample_rate)
        self.grid_random_rot_deg = float(grid_random_rot_deg)
        self._constant_condition_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        if self.grid_random_rot_deg < 0.0:
            raise ValueError(
                f"grid_random_rot_deg must be non-negative; got {self.grid_random_rot_deg}"
            )

        missing_raw = sorted(
            name for name in self.streams if name != "rgb" and name not in self.raw_stream_keys
        )
        if missing_raw:
            raise ValueError(f"Wan Taichu condition streams have no raw mapping: {missing_raw}")
        if self.use_clip_condition and (image_encoder is None or image_processor is None):
            raise ValueError("use_clip_condition requires an image encoder and processor")
        if self.use_dino_condition and (dino_encoder is None or dino_processor is None):
            raise ValueError("use_dino_condition requires a DINO encoder and processor")
        if self.use_audio_condition and (audio_wav2vec is None or audio_processor is None):
            raise ValueError("use_audio_condition requires a Wav2Vec model and processor")

        for module in (vae, image_encoder, dino_encoder):
            if module is not None:
                self._freeze_bf16(module)
        if text_encoder is not None:
            # Freeze at the loaded precision instead of bf16. UMT5 is the one
            # frozen asset the reference pipeline runs in fp32, and its output
            # is cast to the training dtype at the end of encode_prompt
            # anyway, so downcasting the weights only costs prompt fidelity.
            self._freeze(text_encoder)
        if self.audio_wav2vec is not None:
            self.audio_wav2vec.to(dtype=torch.float32)
            self.audio_wav2vec.requires_grad_(False)
            self.audio_wav2vec.eval()

    @staticmethod
    def _freeze_bf16(module: nn.Module) -> None:
        module.to(dtype=torch.bfloat16)
        WanTaichuConditionEncoder._freeze(module)

    @staticmethod
    def _freeze(module: nn.Module) -> None:
        module.requires_grad_(False)
        module.eval()

    @staticmethod
    def scale_pixels(pixels: torch.Tensor) -> torch.Tensor:
        """Map ``[0, 1]`` pixels to ``[-1, 1]`` at the caller's precision.

        Callers cast to the compute dtype *before* calling this, not after:
        rounding ``2x - 1`` from fp32 lands on different bf16 values than
        doing the arithmetic in bf16, and the reference pipeline does the
        latter.
        """
        return (pixels * 2 - 1).to(pixels.dtype)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        vae_dtype = next(self.vae.parameters()).dtype
        device = next(self.vae.parameters()).device
        image = image.to(device=device, dtype=vae_dtype)
        if self.use_vae_channels_last_3d:
            image = image.contiguous(memory_format=torch.channels_last_3d)
        latents = retrieve_latents(
            self.vae.encode(image),
            sample_mode="argmax",
        )
        mean = torch.tensor(
            self.vae.config.latents_mean,
            device=device,
            dtype=torch.float32,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        reciprocal_std = 1.0 / torch.tensor(
            self.vae.config.latents_std,
            device=device,
            dtype=torch.float32,
        ).view(1, self.vae.config.z_dim, 1, 1, 1)
        return ((latents.float() - mean) * reciprocal_std).to(self.training_dtype)

    @torch.no_grad()
    def encode_prompt(
        self,
        text_inputs: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        *,
        max_sequence_length: int = 512,
    ) -> torch.Tensor:
        device = next(self.text_encoder.parameters()).device
        sequence_lengths = mask.gt(0).sum(dim=1).long()
        embeddings = self.text_encoder(
            text_inputs.to(device),
            mask.to(device),
        ).last_hidden_state.to(dtype=self.training_dtype, device=device)
        rows = [row[:length] for row, length in zip(embeddings, sequence_lengths, strict=True)]
        padded = torch.stack(
            [
                torch.cat([row, row.new_zeros(max_sequence_length - row.size(0), row.size(1))])
                for row in rows
            ],
            dim=0,
        )
        return padded.view(batch_size, padded.shape[1], -1)

    @torch.no_grad()
    def encode_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        if self.encoding == "ti2v":
            return self._encode_ti2v_batch(batch)
        return self._encode_wan21_batch(batch)

    @torch.no_grad()
    def encode_inference_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        """Encode generation conditions without requiring target-video latents."""
        if self.encoding != "ti2v":
            return self._encode_wan21_inference_batch(batch)
        device = next(self.vae.parameters()).device
        image = batch.get("image_condition")
        if not isinstance(image, torch.Tensor) or image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("image_condition must be [B,1,C,H,W] for Wan Taichu TI2V inference")
        frame_count = int(batch["num_frames"])
        if frame_count < 1:
            raise ValueError(f"num_frames must be positive, got {frame_count}")
        first_pixels = self.scale_pixels(image.to(device, self.training_dtype))
        frames = first_pixels.expand(-1, frame_count, -1, -1, -1).contiguous()
        prompt = self.encode_prompt(
            batch["text_inputs"].to(device).long(),
            batch["text_mask"].to(device).long(),
            int(image.shape[0]),
        )
        result: dict[str, Any] = {
            "prompt": prompt,
            "first_frame_latents": self.encode_image(first_pixels.permute(0, 2, 1, 3, 4)),
        }
        reference = batch.get("reference_condition")
        if reference is not None:
            if not isinstance(reference, torch.Tensor) or reference.ndim != 5:
                raise ValueError("reference_condition must be [B,K,C,H,W]")
            refs = self.scale_pixels(reference.to(device, self.training_dtype))
            batch_size, count, channels, height, width = refs.shape
            if count < 1:
                raise ValueError("reference_condition must contain at least one frame")
            encoded = self.encode_image(
                refs.reshape(batch_size * count, channels, 1, height, width)
            )
            result["ref_frame_latents"] = (
                encoded.reshape(
                    batch_size,
                    count,
                    encoded.shape[1],
                    encoded.shape[3],
                    encoded.shape[4],
                )
                .permute(0, 2, 1, 3, 4)
                .contiguous()
            )
        for name in self.streams:
            pixels = self._stream_pixels(name, batch, frames)
            result[_stream_field(name)] = self.encode_image(pixels.permute(0, 2, 1, 3, 4))
        for raw_field, latent_field in (
            ("normal_fg_condition", "normal_fg_body"),
            ("normal_bg_condition", "normal_bg_body"),
        ):
            pixels = batch.get(raw_field)
            if isinstance(pixels, torch.Tensor):
                scaled = self.scale_pixels(pixels.to(device, self.training_dtype))
                result[latent_field] = self.encode_image(scaled.permute(0, 2, 1, 3, 4))
        return result

    def _encode_wan21_inference_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        device = next(self.vae.parameters()).device
        dtype = self.training_dtype
        trace = get_inference_trace()
        for field in (
            "video_frames",
            "image_condition",
            "grid_condition",
            "normal_condition",
            "Ks",
            "c2ws",
        ):
            value = batch.get(field)
            if isinstance(value, torch.Tensor):
                trace.tensor(f"raw.{field}", value)
        image = batch.get("image_condition")
        if not isinstance(image, torch.Tensor) or image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("image_condition must be [B,1,C,H,W] for Wan Taichu A14B inference")
        frame_count = int(batch["num_frames"])
        if frame_count < 1:
            raise ValueError(f"num_frames must be positive, got {frame_count}")
        image = image.to(device, dtype)
        frames = image.expand(-1, frame_count, -1, -1, -1)
        prompt = self.encode_prompt(
            batch["text_inputs"].to(device).long(),
            batch["text_mask"].to(device).long(),
            int(image.shape[0]),
        )
        image_frame = image[:, 0]
        image_vae = None
        if self.use_vae_condition:
            image_vae_input = self._reference_video(self.scale_pixels(image), frame_count)
            image_vae = self.encode_image(image_vae_input.permute(0, 2, 1, 3, 4))

        result: dict[str, Any] = {
            "prompt": prompt,
            "image_condition_latent_vae": image_vae,
            "image_condition_latent_clip": (
                self._encode_clip(image_frame) if self.use_clip_condition else None
            ),
            "image_condition_latent_dino": (
                self._encode_dino(image_frame) if self.use_dino_condition else None
            ),
        }
        for name in self.streams:
            if name == "rgb":
                body = self._encode_first_frame_condition(image, frame_count)
                mask = self._first_frame_mask(int(image.shape[0]), frame_count, body)
            else:
                pixels = self._condition_pixels(name, batch, frames=frames)
                body = self.encode_image(pixels.permute(0, 2, 1, 3, 4))
                mask = torch.ones_like(body)[:, : self.mask_channels]
            result[_stream_field(name)] = torch.cat([mask, body], dim=1)
            trace.tensor(f"encoded.condition.{name}", result[_stream_field(name)])

        for raw_field, latent_field in (
            ("normal_fg_condition", "normal_fg_body"),
            ("normal_bg_condition", "normal_bg_body"),
        ):
            pixels = batch.get(raw_field)
            if isinstance(pixels, torch.Tensor):
                scaled = self.scale_pixels(pixels.to(device, dtype))
                result[latent_field] = self.encode_image(scaled.permute(0, 2, 1, 3, 4))
        return result

    @torch.no_grad()
    def encode_constant_condition(
        self,
        *,
        pixel_value: float,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode one neutral condition through the same frozen VAE."""
        cache_key = (
            float(pixel_value),
            int(batch_size),
            int(num_frames),
            int(height),
            int(width),
            dtype,
            torch.device(device),
        )
        cached = self._constant_condition_cache.get(cache_key)
        if cached is not None:
            return cached

        vae_device = next(self.vae.parameters()).device
        pixels = torch.full(
            (int(batch_size), 3, int(num_frames), int(height), int(width)),
            float(pixel_value),
            device=vae_device,
            dtype=self.training_dtype,
        )
        body = self.encode_image(self.scale_pixels(pixels))
        if self.mask_channels:
            mask = torch.ones(
                body.shape[0],
                self.mask_channels,
                *body.shape[2:],
                device=body.device,
                dtype=body.dtype,
            )
            body = torch.cat([mask, body], dim=1)
        result = body.to(device=device, dtype=dtype)
        self._constant_condition_cache[cache_key] = result
        return result

    def _encode_ti2v_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        device = next(self.vae.parameters()).device
        frames = self.scale_pixels(batch["video_frames"].to(device, self.training_dtype))
        prompt = self.encode_prompt(
            batch["text_inputs"].to(device).long(),
            batch["text_mask"].to(device).long(),
            int(frames.shape[0]),
        )
        frame_latents = self.encode_image(frames.permute(0, 2, 1, 3, 4))
        ref_frame_latents = None
        reference_frames = batch.get("reference_frames")
        if reference_frames is not None:
            if not isinstance(reference_frames, torch.Tensor) or reference_frames.ndim != 5:
                raise ValueError("reference_frames must be [B,K,C,H,W] for Wan Taichu ref encoding")
            refs = self.scale_pixels(reference_frames.to(device, self.training_dtype))
            batch_size, count, channels, height, width = refs.shape
            flattened = refs.reshape(batch_size * count, channels, 1, height, width)
            encoded = self.encode_image(flattened)
            _, latent_channels, _, latent_height, latent_width = encoded.shape
            ref_frame_latents = (
                encoded.reshape(
                    batch_size,
                    count,
                    latent_channels,
                    latent_height,
                    latent_width,
                )
                .permute(0, 2, 1, 3, 4)
                .contiguous()
            )
        result: dict[str, Any] = {
            "prompt": prompt,
            "frame_latents": frame_latents,
            "first_frame_latents": frame_latents[:, :, 0:1].clone(),
            "ref_frame_latents": ref_frame_latents,
        }
        for name in self.streams:
            pixels = self._stream_pixels(name, batch, frames)
            result[_stream_field(name)] = self.encode_image(pixels.permute(0, 2, 1, 3, 4))
        return result

    def _stream_pixels(
        self,
        name: str,
        batch: Mapping[str, Any],
        frames: torch.Tensor,
    ) -> torch.Tensor:
        if name == "rgb":
            reference = frames[:, 0:1]
            tail = reference.new_zeros(
                (reference.shape[0], frames.shape[1] - 1, *reference.shape[2:])
            )
            return torch.cat([reference, tail], dim=1)
        return self._condition_pixels(name, batch, frames=frames)

    def _condition_pixels(
        self,
        name: str,
        batch: Mapping[str, Any],
        *,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        raw_key = self.raw_stream_keys[name]
        pixels = batch.get(raw_key)
        if name == "grid" and pixels is None:
            pixels = self._render_grid_from_camera(batch, frames=frames)
            if pixels is None:
                pixels = torch.zeros_like(frames)
        if not isinstance(pixels, torch.Tensor):
            detail = " or paired 'c2ws'/'Ks'" if name == "grid" else ""
            raise ValueError(f"Wan Taichu stream {name!r} requires raw field {raw_key!r}{detail}")
        get_inference_trace().tensor(f"condition_pixels_raw.{name}", pixels)
        pixels = pixels.to(device=frames.device, dtype=frames.dtype)
        return self.scale_pixels(pixels)

    def _render_grid_from_camera(
        self,
        batch: Mapping[str, Any],
        *,
        frames: torch.Tensor,
    ) -> torch.Tensor | None:
        c2ws = batch.get("c2ws")
        intrinsics = batch.get("Ks")
        if c2ws is None or intrinsics is None:
            return None
        if not isinstance(c2ws, torch.Tensor) or not isinstance(intrinsics, torch.Tensor):
            raise TypeError("Wan Taichu camera fields c2ws and Ks must be batched tensors")
        if c2ws.ndim != 4 or c2ws.shape[2:] != (4, 4):
            raise ValueError(f"c2ws must be [B,T,4,4], got {tuple(c2ws.shape)}")
        if intrinsics.ndim != 4 or intrinsics.shape[2:] != (3, 3):
            raise ValueError(f"Ks must be [B,T,3,3], got {tuple(intrinsics.shape)}")
        batch_size, frame_count, _, height, width = frames.shape
        if c2ws.shape[:2] != (batch_size, frame_count):
            raise ValueError(
                "c2ws batch/time dimensions must match video_frames: "
                f"camera={tuple(c2ws.shape[:2])}, video={(batch_size, frame_count)}"
            )
        if intrinsics.shape[:2] != (batch_size, frame_count):
            raise ValueError(
                "Ks batch/time dimensions must match video_frames: "
                f"camera={tuple(intrinsics.shape[:2])}, video={(batch_size, frame_count)}"
            )
        grids = [
            camera_to_grid(
                c2ws[index],
                intrinsics[index],
                int(height),
                int(width),
                device=frames.device,
                random_rot_deg=self.grid_random_rot_deg,
            )
            for index in range(batch_size)
        ]
        return torch.stack(grids, dim=0).contiguous()

    def _encode_wan21_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        device = next(self.vae.parameters()).device
        dtype = self.training_dtype
        frames = self.scale_pixels(batch["video_frames"].to(device, dtype))
        prompt = self.encode_prompt(
            batch["text_inputs"].to(device).long(),
            batch["text_mask"].to(device).long(),
            int(frames.shape[0]),
        )
        image_raw = batch["image_condition"].to(device, dtype)
        image_frame = image_raw[:, 0]
        image_vae = None
        if self.use_vae_condition:
            image_vae_input = self.scale_pixels(image_raw)
            if self.streams == ("normal",):
                image_vae_input = image_vae_input.repeat(1, 4, 1, 1, 1)
            else:
                image_vae_input = self._reference_video(image_vae_input, int(frames.shape[1]))
            image_vae = self.encode_image(image_vae_input.permute(0, 2, 1, 3, 4))

        condition_streams: dict[str, torch.Tensor] = {}
        for name in self.streams:
            if name == "rgb":
                first_frame = batch["video_frames"][:, 0:1].to(device, dtype)
                body = self._encode_first_frame_condition(first_frame, int(frames.shape[1]))
                mask = self._first_frame_mask(int(first_frame.shape[0]), int(frames.shape[1]), body)
            else:
                pixels = self._condition_pixels(name, batch, frames=frames)
                body = self.encode_image(pixels.permute(0, 2, 1, 3, 4))
                mask = torch.ones_like(body)[:, : self.mask_channels]
            condition_streams[_stream_field(name)] = torch.cat([mask, body], dim=1)

        result = {
            "prompt": prompt,
            "frame_latents": self.encode_image(frames.permute(0, 2, 1, 3, 4)),
            "image_condition_latent_vae": image_vae,
            "image_condition_latent_clip": (
                self._encode_clip(image_frame) if self.use_clip_condition else None
            ),
            "image_condition_latent_dino": (
                self._encode_dino(image_frame) if self.use_dino_condition else None
            ),
            **condition_streams,
        }
        if self.use_audio_condition:
            audio = batch.get("audio")
            if audio is None:
                raise ValueError("Wan Taichu audio online encoding requires raw field 'audio'")
            if isinstance(audio, torch.Tensor):
                waveforms = list(audio)
            else:
                waveforms = list(audio)
            result["audio_features"] = torch.cat(
                [self.encode_audio_features(waveform) for waveform in waveforms],
                dim=0,
            ).to(device=device, dtype=dtype)
        return result

    @torch.no_grad()
    def encode_audio_features(self, audio_waveform: torch.Tensor) -> torch.Tensor:
        """Run frozen Wav2Vec stage A and Taichu's frame-bucket stage B."""
        if self.audio_wav2vec is None or self.audio_processor is None:
            raise RuntimeError("audio assets are not loaded")
        waveform = audio_waveform.detach().cpu().float().numpy()
        processed = self.audio_processor(
            waveform,
            sampling_rate=self.audio_sample_rate,
            return_tensors="pt",
        )
        values = (
            processed.input_values if self.audio_type == "wav2vec2" else processed.input_features
        )
        model = self.audio_wav2vec
        parameter = next(model.parameters())
        output = model(
            values.to(device=parameter.device, dtype=parameter.dtype),
            output_hidden_states=True,
        )
        features = torch.cat(output.hidden_states)
        features = self._linear_interp_audio(
            features,
            input_fps=50,
            output_fps=self.audio_video_rate,
        ).to(self.training_dtype)
        return self._bucket_audio_features(features).unsqueeze(0).contiguous()

    @staticmethod
    def _linear_interp_audio(
        features: torch.Tensor,
        *,
        input_fps: int,
        output_fps: int,
    ) -> torch.Tensor:
        if input_fps == 50 and output_fps == 25:
            return features[:, ::2, :]
        transposed = features.transpose(1, 2)
        output_length = int(transposed.shape[2] / float(input_fps) * output_fps)
        return F.interpolate(
            transposed,
            size=output_length,
            align_corners=True,
            mode="linear",
        ).transpose(1, 2)

    def _bucket_audio_features(self, audio_embed: torch.Tensor) -> torch.Tensor:
        layers, audio_frames, audio_dim = audio_embed.shape
        if layers <= 1:
            raise ValueError(f"audio encoder must return all hidden layers; got {layers}")
        scale = self.audio_video_rate / self.audio_fps
        minimum_batches = int(audio_frames / (self.audio_batch_frames * scale)) + 1
        bucket_count = minimum_batches * self.audio_batch_frames
        padded_audio = (
            math.ceil(
                minimum_batches * self.audio_batch_frames / self.audio_fps * self.audio_video_rate
            )
            - audio_frames
        )
        indices = self._sample_audio_indices(
            original_fps=self.audio_video_rate,
            total_frames=audio_frames + padded_audio,
            target_fps=self.audio_fps,
            count=bucket_count,
        )
        stride = int(self.audio_video_rate / self.audio_fps)
        buckets = []
        for frame_index in indices:
            if frame_index < audio_frames:
                chosen = range(
                    frame_index - self.audio_bucket_m * stride,
                    frame_index + (self.audio_bucket_m + 1) * stride,
                    stride,
                )
                chosen = [min(max(index, 0), audio_frames - 1) for index in chosen]
                bucket = audio_embed[:, chosen].flatten(start_dim=-2, end_dim=-1)
            else:
                bucket = audio_embed.new_zeros((layers, audio_dim * (2 * self.audio_bucket_m + 1)))
            buckets.append(bucket.unsqueeze(-1))
        return torch.cat(buckets, dim=-1)

    @staticmethod
    def _sample_audio_indices(
        *,
        original_fps: int,
        total_frames: int,
        target_fps: int,
        count: int,
    ) -> np.ndarray:
        required_duration = count / target_fps
        if required_duration > total_frames / original_fps:
            raise ValueError("requested audio duration exceeds the available feature duration")
        points = np.linspace(0.0, required_duration, count, endpoint=False)
        indices = np.round(points * original_fps).astype(int)
        return np.clip(indices, 0, total_frames - 1)

    @staticmethod
    def _reference_video(image: torch.Tensor, frames: int) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError(f"expected a [B,1,C,H,W] reference; got {tuple(image.shape)}")
        tail = image.new_zeros((image.shape[0], frames - 1, *image.shape[2:]))
        return torch.cat([image, tail], dim=1)

    def _encode_first_frame_condition(
        self,
        first_frame: torch.Tensor,
        frames: int,
    ) -> torch.Tensor:
        reference = self.scale_pixels(first_frame).permute(0, 2, 1, 3, 4)
        tail = reference.new_zeros(
            (reference.shape[0], reference.shape[1], frames - 1, *reference.shape[3:])
        )
        return self.encode_image(torch.cat([reference, tail], dim=2))

    def _first_frame_mask(
        self,
        batch_size: int,
        pixel_frames: int,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        _, _, latent_frames, height, width = latent.shape
        scale = int(getattr(self.vae.config, "scale_factor_temporal", 4))
        mask = torch.ones(
            batch_size,
            1,
            pixel_frames,
            height,
            width,
            dtype=latent.dtype,
            device=latent.device,
        )
        mask[:, :, 1:] = 0
        mask = torch.cat([mask[:, :, 0:1].repeat_interleave(scale, dim=2), mask[:, :, 1:]], dim=2)
        mask = mask.view(batch_size, -1, scale, height, width).transpose(1, 2)
        if mask.shape[2] != latent_frames:
            raise ValueError(
                f"Wan Taichu temporal mask has {mask.shape[2]} frames, expected {latent_frames}"
            )
        return mask

    def _encode_clip(self, image: torch.Tensor) -> torch.Tensor:
        assert self.image_encoder is not None and self.image_processor is not None
        device = next(self.image_encoder.parameters()).device
        dtype = next(self.image_encoder.parameters()).dtype
        processed = self.image_processor(
            images=image.float().clamp(0.0, 1.0),
            return_tensors="pt",
            do_rescale=False,
        )
        output = self.image_encoder(
            pixel_values=processed["pixel_values"].to(device=device, dtype=dtype),
            output_hidden_states=True,
        )
        return output.hidden_states[-2].to(self.training_dtype)

    def _encode_dino(self, image: torch.Tensor) -> torch.Tensor:
        assert self.dino_encoder is not None and self.dino_processor is not None
        device = next(self.dino_encoder.parameters()).device
        dtype = next(self.dino_encoder.parameters()).dtype
        processed = self.dino_processor(
            images=image.float().clamp(0.0, 1.0),
            return_tensors="pt",
            do_rescale=False,
        )
        output = self.dino_encoder(
            pixel_values=processed["pixel_values"].to(device=device, dtype=dtype)
        )
        return output.last_hidden_state[:, self._dino_drop_prefix :, :].to(self.training_dtype)


def _stream_field(name: str) -> str:
    return f"condition_latents_{name}"


__all__ = ["WanTaichuConditionEncoder", "WanTaichuEncoding", "retrieve_latents"]
