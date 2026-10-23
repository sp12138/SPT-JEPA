"""Wan-family assets and latent helpers for flow-matching inference."""

from __future__ import annotations

from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from vforge.data.raw_dataset import prompt_clean
from vforge.utils.inference_trace import get_inference_trace

WAN_DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)


@torch.no_grad()
def encode_wan_negative_prompt(
    tokenizer: Any,
    text_encoder: Any,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    text: str = "",
    max_sequence_length: int = 512,
) -> torch.Tensor:
    """Encode one padded Wan negative prompt batch on the output leader."""
    if tokenizer is None or text_encoder is None:
        raise RuntimeError("negative prompt encoding is only available on output leaders")
    text = prompt_clean(text)
    max_len = getattr(tokenizer, "model_max_length", max_sequence_length)
    if not isinstance(max_len, int) or max_len > 8192:
        max_len = max_sequence_length
    encoded = tokenizer(
        [text] * batch_size,
        padding="max_length",
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = encoded.input_ids.to(device)
    mask = encoded.attention_mask.to(device)
    trace = get_inference_trace()
    trace.event("negative.text", {"text": text})
    trace.tensor("negative.input_ids", input_ids)
    trace.tensor("negative.attention_mask", mask)
    seq_lens = mask.gt(0).sum(dim=1).long()
    embeds = text_encoder(input_ids, mask).last_hidden_state.to(
        dtype=dtype,
        device=device,
    )
    pieces = [row[:length] for row, length in zip(embeds, seq_lens, strict=True)]
    return torch.stack(
        [torch.cat([row, row.new_zeros(max_len - row.size(0), row.size(1))]) for row in pieces],
        dim=0,
    )


class WanFlowMatchPipelineBase:
    """Frozen assets and latent helpers shared by independent Wan implementations."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        text_encoder: Any,
        vae: Any,
        transformer: Any,
        scheduler: Any,
        vae_scale_factor_temporal: int,
        vae_scale_factor_spatial: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler
        self.vae_scale_factor_temporal = int(
            getattr(
                getattr(vae, "config", None),
                "scale_factor_temporal",
                vae_scale_factor_temporal,
            )
        )
        self.vae_scale_factor_spatial = int(
            getattr(
                getattr(vae, "config", None),
                "scale_factor_spatial",
                vae_scale_factor_spatial,
            )
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    @property
    def dtype(self) -> torch.dtype:
        declared = getattr(self.transformer, "dtype", None)
        if isinstance(declared, torch.dtype):
            return declared
        parameter = next(self.transformer.parameters(), None)
        if parameter is None:
            raise RuntimeError("Wan flow-match transformer must declare dtype or own parameters")
        return parameter.dtype

    @torch.no_grad()
    def encode_negative_prompt(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        text: str = "",
        max_sequence_length: int = 512,
    ) -> torch.Tensor:
        return encode_wan_negative_prompt(
            self.tokenizer,
            self.text_encoder,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            text=text,
            max_sequence_length=max_sequence_length,
        )

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def align_num_frames(self, num_frames: int) -> int:
        stride = self.vae_scale_factor_temporal
        if (num_frames - 1) % stride == 0:
            return num_frames
        return ((num_frames - 1) // stride) * stride + 1

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("VAE decode is only available on output leaders")
        declared = getattr(self.vae, "dtype", None)
        vae_dtype = (
            declared if isinstance(declared, torch.dtype) else next(self.vae.parameters()).dtype
        )
        latents = latents.to(vae_dtype)
        z_dim = int(self.vae.config.z_dim)
        mean = torch.tensor(
            self.vae.config.latents_mean,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, z_dim, 1, 1, 1)
        inverse_std = 1.0 / torch.tensor(
            self.vae.config.latents_std,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, z_dim, 1, 1, 1)
        return self.vae.decode(latents / inverse_std + mean, return_dict=False)[0]


__all__ = [
    "WAN_DEFAULT_NEGATIVE_PROMPT",
    "WanFlowMatchPipelineBase",
    "encode_wan_negative_prompt",
]
