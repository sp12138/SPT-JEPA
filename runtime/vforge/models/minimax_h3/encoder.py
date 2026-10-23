"""Frozen MiniMax-H3 text, video, and audio condition encoding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_QWEN_PATCH_EMBED_MODES = frozenset({"default", "fp32", "last_channel"})


def resolve_qwen_patch_embed_mode(extras: Mapping[str, Any]) -> str:
    """Resolve the cache-relevant Qwen patch embedding implementation."""
    mode = str(extras.get("qwen_patch_embed_mode", "default"))
    if mode not in _QWEN_PATCH_EMBED_MODES:
        choices = ", ".join(sorted(_QWEN_PATCH_EMBED_MODES))
        raise ValueError(f"unknown qwen_patch_embed_mode={mode!r}; expected one of: {choices}")
    return mode


def _configure_qwen_patch_embed(text_encoder, *, mode: str) -> None:
    if mode == "default":
        return

    patch_embed = text_encoder.model.visual.patch_embed
    output_dtype = patch_embed.proj.weight.dtype

    if mode == "fp32":
        original_forward = patch_embed.forward
        patch_embed.proj.to(dtype=torch.float32)

        def forward(hidden_states: torch.Tensor) -> torch.Tensor:
            allow_tf32 = torch.backends.cudnn.allow_tf32
            torch.backends.cudnn.allow_tf32 = False
            try:
                with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                    output = original_forward(hidden_states)
            finally:
                torch.backends.cudnn.allow_tf32 = allow_tf32
            return output.to(dtype=output_dtype)

    elif mode == "last_channel":

        def forward(hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = hidden_states.view(
                -1,
                patch_embed.in_channels,
                patch_embed.temporal_patch_size,
                patch_embed.patch_size,
                patch_embed.patch_size,
            )
            hidden_states = hidden_states.to(dtype=output_dtype)
            hidden_states = hidden_states.contiguous(memory_format=torch.channels_last_3d)
            return patch_embed.proj(hidden_states).view(-1, patch_embed.embed_dim)

    else:
        choices = ", ".join(sorted(_QWEN_PATCH_EMBED_MODES))
        raise ValueError(f"unknown Qwen patch embedding mode {mode!r}; expected: {choices}")

    patch_embed.forward = forward


def _configure_video_vae_tile_batching(vae, *, batch_size: int) -> None:
    if batch_size <= 1:
        return

    def encode_clip(x: torch.Tensor) -> torch.Tensor:
        if not vae.use_tiling:
            return vae.quant_conv(vae.encoder(x))

        height, width = x.shape[-2:]
        y_indices, _, y_overlaps = vae._split_tiles(
            height,
            vae.tile_sample_min_height,
            vae.tile_sample_min_overlap_height,
        )
        x_indices, _, x_overlaps = vae._split_tiles(
            width,
            vae.tile_sample_min_width,
            vae.tile_sample_min_overlap_width,
        )
        tiles = [
            x[
                ...,
                y_pos : y_pos + vae.tile_sample_min_height,
                x_pos : x_pos + vae.tile_sample_min_width,
            ]
            for y_pos in y_indices
            for x_pos in x_indices
        ]
        encoded_tiles = []
        for start in range(0, len(tiles), batch_size):
            tile_batch = torch.cat(tiles[start : start + batch_size], dim=0)
            encoded_tiles.extend(vae.quant_conv(vae.encoder(tile_batch)).split(1, dim=0))
        rows = [
            encoded_tiles[start : start + len(x_indices)]
            for start in range(0, len(encoded_tiles), len(x_indices))
        ]
        ratio = vae.spatial_compression_ratio
        return vae._stitch_tiles(
            rows,
            [overlap // ratio for overlap in y_overlaps],
            [overlap // ratio for overlap in x_overlaps],
        )

    vae._encode_clip = encode_clip


def _resolve_canvas_size(
    aspect_width: float,
    aspect_height: float,
    canvas_multiple: int,
    short_edge: int,
    max_pixels: int,
    min_aspect_ratio: float = 0.25,
    max_aspect_ratio: float = 4.0,
) -> tuple[int, int]:
    """Resolve the official H3 canvas geometry without importing Modular Diffusers."""
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"H3 aspect ratio must be positive, got {aspect_width}:{aspect_height}")
    ratio = aspect_width / aspect_height
    if not min_aspect_ratio <= ratio <= max_aspect_ratio:
        raise ValueError(
            f"H3 supports aspect ratios from 1:{1 / min_aspect_ratio:g} to "
            f"{max_aspect_ratio:g}:1, got {aspect_width}:{aspect_height} ({ratio:g})"
        )
    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio
    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale
    return (
        max(canvas_multiple, round(height / canvas_multiple) * canvas_multiple),
        max(canvas_multiple, round(width / canvas_multiple) * canvas_multiple),
    )


def _patchify_video_latents(
    latents: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Pack channel-first H3 video latents into frame-major transformer rows."""
    patch_t, patch_h, patch_w = patch_size
    batch_size, channels, num_frames, height, width = latents.shape
    if num_frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            f"Latents of shape {tuple(latents.shape)} are not divisible by patch {patch_size}"
        )
    latents = latents.reshape(
        batch_size,
        channels,
        num_frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    return (
        latents.permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(-1, channels * patch_t * patch_h * patch_w)
        .contiguous()
    )


def _decode_video(path: str, *, num_frames: int) -> list[Image.Image]:
    try:
        import av
    except ImportError as exc:
        raise ImportError("MiniMax-H3 source encoding requires PyAV") from exc

    frames: list[Image.Image] = []
    with av.open(path) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_image().convert("RGB"))
            if len(frames) >= num_frames:
                break
    if not frames:
        raise ValueError(f"video has no decodable frames: {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames[:num_frames]


def _load_keyframe(path: str | None, fallback: Image.Image) -> Image.Image:
    if not path:
        return fallback.copy()
    with Image.open(Path(path)) as image:
        return image.convert("RGB")


def _resolve_canvas(
    keyframe: Image.Image,
    extras: Mapping[str, Any],
) -> tuple[int, int]:
    height = extras.get("height")
    width = extras.get("width")
    if (height is None) != (width is None):
        raise ValueError("H3 height and width must be set together")
    if height is None:
        height, width = _resolve_canvas_size(
            *keyframe.size,
            int(extras.get("canvas_multiple", 32)),
            int(extras.get("canvas_short_edge", 768)),
            int(extras.get("canvas_max_pixels", 768 * 1344)),
        )
    height, width = int(height), int(width)
    if height % 32 or width % 32:
        raise ValueError(f"H3 height/width must be multiples of 32, got {height}x{width}")
    return height, width


def _prepare_keyframes(
    first: Image.Image,
    last: Image.Image | None,
    *,
    height: int,
    width: int,
) -> tuple[list[Image.Image], tuple[str, ...]]:
    prepared = [first.resize((width, height), Image.Resampling.LANCZOS)]
    anchors = ["first"]
    if last is not None:
        scale = max(width / last.width, height / last.height)
        resized_size = (
            max(width, round(last.width * scale)),
            max(height, round(last.height * scale)),
        )
        left = max(0, (resized_size[0] - width) // 2)
        top = max(0, (resized_size[1] - height) // 2)
        resized = last.resize(resized_size, Image.Resampling.LANCZOS)
        prepared.append(resized.crop((left, top, left + width, top + height)))
        anchors.append("last")
    return prepared, tuple(anchors)


def _video_pixels(
    frames: list[Image.Image],
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    resized = [
        frame.resize((width, height), Image.Resampling.LANCZOS)
        if frame.size != (width, height)
        else frame
        for frame in frames
    ]
    array = np.stack([np.asarray(frame, dtype=np.uint8) for frame in resized])
    return torch.from_numpy(array.copy()).permute(3, 0, 1, 2).unsqueeze(0)


def _decode_audio(
    path: str,
    *,
    num_samples: int,
    sample_rate: int,
) -> tuple[torch.Tensor, bool]:
    try:
        import av
    except ImportError as exc:
        raise ImportError("MiniMax-H3 source encoding requires PyAV") from exc

    chunks: list[torch.Tensor] = []
    with av.open(path) as container:
        if not container.streams.audio:
            return torch.zeros(2, num_samples), False
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=sample_rate)
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                array = torch.from_numpy(resampled.to_ndarray()).float()
                if array.ndim == 1:
                    array = array.unsqueeze(0)
                if array.shape[0] == 1:
                    array = array.expand(2, -1)
                chunks.append(array[:2])
        for resampled in resampler.resample(None):
            chunks.append(torch.from_numpy(resampled.to_ndarray()).float()[:2])
    if not chunks:
        return torch.zeros(2, num_samples), False
    audio = torch.cat(chunks, dim=-1)
    if audio.shape[-1] < num_samples:
        audio = F.pad(audio, (0, num_samples - audio.shape[-1]))
    return audio[:, :num_samples].contiguous(), True


@dataclass
class MiniMaxH3PreparedSample:
    prompt: str
    keyframes: list[Image.Image]
    keyframe_anchors: tuple[str, ...]
    pixels: torch.Tensor
    audio: torch.Tensor
    has_audio: bool


@dataclass
class MiniMaxH3PreparedInference:
    prompt: str
    keyframes: list[Image.Image]
    keyframe_anchors: tuple[str, ...]
    height: int
    width: int
    num_frames: int


class MiniMaxH3ConditionEncoder:
    """Encode one raw MP4/prompt sample into the H3 packed training schema."""

    def __init__(
        self,
        *,
        text_encoder,
        tokenizer,
        processor,
        vae,
        audio_vae,
        extras: Mapping[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.vae = vae
        self.audio_vae = audio_vae
        self.extras = dict(extras)
        self.device = device
        self.dtype = dtype

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        extras: Mapping[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        vae=None,
        audio_vae=None,
    ) -> MiniMaxH3ConditionEncoder:
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
        from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            path,
            subfolder=str(extras.get("text_encoder_subfolder", "text_encoder")),
            torch_dtype=dtype,
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            subfolder=str(extras.get("tokenizer_subfolder", "tokenizer")),
        )
        processor = AutoProcessor.from_pretrained(
            path,
            subfolder=str(extras.get("processor_subfolder", "processor")),
        )
        if vae is None:
            vae = AutoencoderKLMiniMaxH3.from_pretrained(
                path,
                subfolder=str(extras.get("vae_subfolder", "vae")),
                torch_dtype=torch.float32,
            ).eval()
        if audio_vae is None:
            audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(
                path,
                subfolder=str(extras.get("audio_vae_subfolder", "audio_vae")),
                torch_dtype=torch.float32,
            ).eval()
        text_encoder.to(device)
        vae.to(device)
        audio_vae.to(device)
        _configure_qwen_patch_embed(
            text_encoder,
            mode=resolve_qwen_patch_embed_mode(extras),
        )
        _configure_video_vae_tile_batching(
            vae,
            batch_size=int(extras.get("video_vae_tile_batch_size", 1)),
        )
        return cls(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            vae=vae,
            audio_vae=audio_vae,
            extras=extras,
            device=device,
            dtype=dtype,
        )

    def prepare_sample(self, sample: dict[str, Any]) -> MiniMaxH3PreparedSample:
        """Decode and resize one sample on CPU for overlap with GPU encoding."""
        prompt = str(sample.get("prompt") or "")
        num_frames = int(self.extras.get("num_frames", 124))
        frames = _decode_video(str(sample["video"]), num_frames=num_frames)
        first_keyframe = _load_keyframe(sample.get("image"), frames[0])
        last_keyframe = (
            _load_keyframe(str(sample["last_image"]), frames[-1])
            if sample.get("last_image")
            else None
        )
        height, width = _resolve_canvas(first_keyframe, self.extras)
        keyframes, keyframe_anchors = _prepare_keyframes(
            first_keyframe,
            last_keyframe,
            height=height,
            width=width,
        )
        pixels = _video_pixels(frames, height=height, width=width)
        sample_rate = int(self.extras.get("audio_sample_rate", 32000))
        audio, has_audio = _decode_audio(
            str(sample["video"]),
            num_samples=round(num_frames / 24 * sample_rate),
            sample_rate=sample_rate,
        )
        return MiniMaxH3PreparedSample(
            prompt=prompt,
            keyframes=keyframes,
            keyframe_anchors=keyframe_anchors,
            pixels=pixels,
            audio=audio,
            has_audio=has_audio,
        )

    def prepare_inference_sample(
        self,
        sample: dict[str, Any],
        *,
        allow_video_fallback: bool = False,
    ) -> MiniMaxH3PreparedInference:
        """Load only the visual conditions and target generation geometry."""
        prompt = str(sample.get("prompt") or sample.get("text") or "")
        requested_frames = int(sample.get("num_frames") or self.extras.get("num_frames", 124))
        image_path = sample.get("image")
        if image_path:
            with Image.open(Path(str(image_path))) as image:
                first_keyframe = image.convert("RGB")
        elif allow_video_fallback and sample.get("video"):
            first_keyframe = _decode_video(
                str(sample["video"]),
                num_frames=1,
            )[0]
        else:
            raise ValueError("H3 inference requests require 'image'")
        last_keyframe = None
        if sample.get("last_image"):
            with Image.open(Path(str(sample["last_image"]))) as image:
                last_keyframe = image.convert("RGB")
        geometry_extras = dict(self.extras)
        if sample.get("height") is not None or sample.get("width") is not None:
            geometry_extras["height"] = sample.get("height")
            geometry_extras["width"] = sample.get("width")
        height, width = _resolve_canvas(first_keyframe, geometry_extras)
        keyframes, keyframe_anchors = _prepare_keyframes(
            first_keyframe,
            last_keyframe,
            height=height,
            width=width,
        )
        return MiniMaxH3PreparedInference(
            prompt=prompt,
            keyframes=keyframes,
            keyframe_anchors=keyframe_anchors,
            height=height,
            width=width,
            num_frames=requested_frames,
        )

    def _encode_prompt_and_conditions(
        self,
        *,
        prompt: str,
        keyframes: list[Image.Image],
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        from diffusers.modular_pipelines.minimax_h3.encoders import (
            encode_vae_condition,
            get_qwen3vl_prompt_embeds,
        )

        vision = self.processor.image_processor(images=keyframes, return_tensors="pt")
        image_grid_thw = vision["image_grid_thw"]
        token_ids: list[int] = []
        token_tags: list[int] = []
        merge_size = self.processor.image_processor.merge_size**2
        for index in range(len(keyframes)):
            num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
            label_ids = self.tokenizer(
                f"<Picture {index + 1}>: ",
                add_special_tokens=False,
            )["input_ids"]
            vision_ids = (
                [self.tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [self.tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                + [self.tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            token_ids += label_ids + vision_ids
            token_tags += [1] * len(label_ids) + [0] * len(vision_ids)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [1] * len(prompt_ids)
        prompt_embeds = get_qwen3vl_prompt_embeds(
            self.text_encoder,
            self.processor,
            token_ids,
            {
                "pixel_values": vision["pixel_values"],
                "image_grid_thw": image_grid_thw,
            },
            text_encoder_layer=int(self.extras.get("text_encoder_layer", 50)),
            device=self.device,
            dtype=self.dtype,
        ).cpu()
        condition_latents = [
            encode_vae_condition(
                self.vae,
                torch.from_numpy(np.asarray(keyframe, dtype=np.uint8).copy())
                .to(self.device)
                .permute(2, 0, 1)[None, :, None],
                tuple(self.extras.get("pixel_mean", (0.485, 0.456, 0.406))),
                tuple(self.extras.get("pixel_std", (0.229, 0.224, 0.225))),
                int(self.extras.get("encode_seed", 42)),
            )
            for keyframe in keyframes
        ]
        return prompt_embeds, torch.tensor(token_tags, dtype=torch.long), condition_latents

    @torch.no_grad()
    def encode_prepared(self, prepared: MiniMaxH3PreparedSample) -> dict[str, Any]:
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3PrepareLayoutStep,
        )
        from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition

        keyframes = prepared.keyframes
        keyframe_anchors = prepared.keyframe_anchors
        pixels = prepared.pixels.to(self.device)
        prompt_embeds, text_token_tags, condition_latents = self._encode_prompt_and_conditions(
            prompt=prepared.prompt,
            keyframes=keyframes,
        )
        video_latents = encode_vae_condition(
            self.vae,
            pixels,
            tuple(self.extras.get("pixel_mean", (0.485, 0.456, 0.406))),
            tuple(self.extras.get("pixel_std", (0.229, 0.224, 0.225))),
            int(self.extras.get("encode_seed", 42)),
        )

        posterior = self.audio_vae.encode(
            prepared.audio.to(self.device)[:, None],
            return_dict=False,
        )[0]
        audio_latents = posterior.mode().float().cpu().transpose(1, 2)
        audio_mean = torch.tensor(self.audio_vae.config.latents_mean).view(1, 1, -1)
        audio_std = torch.tensor(self.audio_vae.config.latents_std).view(1, 1, -1)
        audio_latents = (audio_latents - audio_mean) / audio_std
        target_audio_rows = audio_latents.reshape(-1, audio_latents.shape[-1]).contiguous()

        patch_size = tuple(self.extras.get("patch_size", (1, 2, 2)))
        target_video_rows = _patchify_video_latents(video_latents, patch_size)
        condition_video_rows = torch.cat(
            [_patchify_video_latents(latents, patch_size) for latents in condition_latents]
        )
        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            num_condition_video_rows,
            num_condition_audio_rows,
        ) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags,
            int(video_latents.shape[2]),
            int(video_latents.shape[3]),
            int(video_latents.shape[4]),
            int(audio_latents.shape[1]),
            patch_size,
            int(audio_latents.shape[0]),
            audio_tag=2,
            video_tag=0,
            keyframe_anchors=keyframe_anchors,
        )
        if condition_video_rows.shape[0] != num_condition_video_rows:
            raise ValueError(
                "H3 keyframe rows do not match the packed layout: "
                f"{condition_video_rows.shape[0]} != {num_condition_video_rows}"
            )
        return {
            "prompt_embeds": prompt_embeds,
            "condition_video_rows": condition_video_rows,
            "target_video_rows": target_video_rows,
            "target_audio_rows": target_audio_rows,
            "audio_loss_mask": torch.tensor([1.0 if prepared.has_audio else 0.0]),
            "token_tags": token_tags,
            "position_ids": position_ids,
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            "text_indices": text_indices,
            "num_condition_video_rows": num_condition_video_rows,
            "num_condition_audio_rows": num_condition_audio_rows,
            "num_latent_frames": int(video_latents.shape[2]),
            "latent_height": int(video_latents.shape[3]),
            "latent_width": int(video_latents.shape[4]),
            "num_audio_latents": int(audio_latents.shape[1]),
        }

    @torch.no_grad()
    def encode_inference_prepared(
        self,
        prepared: MiniMaxH3PreparedInference,
    ) -> dict[str, Any]:
        """Encode keyframes and build a target-free H3 inference layout."""
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3PrepareLayoutStep,
        )
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
            align_num_frames,
            audio_latent_num_frames,
            video_latent_num_frames,
        )

        prompt_embeds, text_token_tags, condition_latents = self._encode_prompt_and_conditions(
            prompt=prepared.prompt,
            keyframes=prepared.keyframes,
        )
        patch_size = tuple(self.extras.get("patch_size", (1, 2, 2)))
        condition_video_rows = torch.cat(
            [_patchify_video_latents(latents, patch_size) for latents in condition_latents]
        ).cpu()
        frames_per_chunk = int(self.vae.config.clip_length)
        latents_per_chunk = int(self.vae.tokens_chunk_size)
        num_frames = align_num_frames(
            prepared.num_frames,
            frames_per_chunk,
            latents_per_chunk,
        )
        fps = float(self.extras.get("fps", 24.0))
        duration = num_frames / fps
        if not 5.0 <= duration <= 15.0:
            raise ValueError(
                "H3 generation duration must be between 5 and 15 seconds; "
                f"got {num_frames} frames at {fps:g} fps ({duration:g}s)"
            )
        num_latent_frames = video_latent_num_frames(
            num_frames,
            frames_per_chunk,
            latents_per_chunk,
        )
        latent_height = prepared.height // int(self.vae.spatial_compression_ratio)
        latent_width = prepared.width // int(self.vae.spatial_compression_ratio)
        num_audio_latents = audio_latent_num_frames(num_frames)
        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            num_condition_video_rows,
            num_condition_audio_rows,
        ) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            patch_size,
            2,
            audio_tag=2,
            video_tag=0,
            keyframe_anchors=prepared.keyframe_anchors,
        )
        if condition_video_rows.shape[0] != num_condition_video_rows:
            raise ValueError(
                "H3 keyframe rows do not match the packed layout: "
                f"{condition_video_rows.shape[0]} != {num_condition_video_rows}"
            )
        return {
            "prompt_embeds": prompt_embeds,
            "condition_video_rows": condition_video_rows,
            "token_tags": token_tags,
            "position_ids": position_ids,
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            "text_indices": text_indices,
            "num_condition_video_rows": num_condition_video_rows,
            "num_condition_audio_rows": num_condition_audio_rows,
            "num_latent_frames": num_latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "num_audio_latents": num_audio_latents,
            "height": prepared.height,
            "width": prepared.width,
            "num_frames": num_frames,
        }

    @torch.no_grad()
    def encode_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        return self.encode_prepared(self.prepare_sample(sample))

    @torch.no_grad()
    def encode_inference_sample(
        self,
        sample: dict[str, Any],
        *,
        allow_video_fallback: bool = False,
    ) -> dict[str, Any]:
        return self.encode_inference_prepared(
            self.prepare_inference_sample(
                sample,
                allow_video_fallback=allow_video_fallback,
            )
        )


__all__ = [
    "MiniMaxH3ConditionEncoder",
    "MiniMaxH3PreparedInference",
    "MiniMaxH3PreparedSample",
]
