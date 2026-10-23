"""Frozen VAE and UMT5 primitives shared by Wan condition encoders."""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.video_processor import VideoProcessor


def retrieve_latents(encoder_output, generator=None, sample_mode: str = "sample"):
    """Extract a latent tensor from the output variants used by Diffusers VAEs."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


class WanFrozenAssetEncoder(nn.Module):
    """Frozen Wan VAE + UMT5 assets and their normalization contract."""

    def __init__(
        self,
        vae: nn.Module | None,
        text_encoder: nn.Module | None,
        tokenizer,
        video_processor: VideoProcessor | None,
        training_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.video_processor = video_processor
        self.training_dtype = training_dtype

        if self.vae is not None:
            self.freeze_bf16(self.vae)
        if self.text_encoder is not None:
            self.freeze_bf16(self.text_encoder)

    @staticmethod
    def freeze_bf16(module: nn.Module) -> None:
        """Pin a frozen asset to bf16 so it matches the trainable transformer."""
        module.to(dtype=torch.bfloat16)
        module.requires_grad_(False)
        module.eval()

    @staticmethod
    def scale_pixels(pixels: torch.Tensor) -> torch.Tensor:
        """Map pixels from ``[0, 1]`` to ``[-1, 1]`` without changing dtype."""
        return (pixels * 2 - 1).to(pixels.dtype)

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode ``[B, C, T, H, W]`` pixels into normalized VAE latents."""
        if self.vae is None:
            raise RuntimeError(
                f"{type(self).__name__}.encode_image requires a VAE; this rank "
                "was built without frozen assets. Either the latent cache is "
                "incomplete (rebuild it, or drop latent_cache_dir so the live "
                "VAE forward is available) or a raw batch reached a follower rank."
            )
        vae = self.vae
        vae_dtype = next(vae.parameters()).dtype
        device = next(vae.parameters()).device
        image = image.to(device, dtype=vae_dtype)
        latents = retrieve_latents(vae.encode(image), sample_mode="argmax")
        latents_mean = torch.tensor(
            vae.config.latents_mean,
            device=device,
            dtype=torch.float32,
        ).view(1, vae.config.z_dim, 1, 1, 1)
        latents_std = 1.0 / torch.tensor(
            vae.config.latents_std,
            device=device,
            dtype=torch.float32,
        ).view(1, vae.config.z_dim, 1, 1, 1)
        return ((latents.float() - latents_mean) * latents_std).to(self.training_dtype)

    @torch.no_grad()
    def encode_prompt(
        self,
        texts_input: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        max_sequence_length: int = 512,
        num_videos_per_prompt: int = 1,
    ) -> torch.Tensor:
        """Run UMT5 and pad/truncate the result to ``max_sequence_length``."""
        if self.text_encoder is None:
            raise RuntimeError(
                f"{type(self).__name__}.encode_prompt requires a text encoder; "
                "this rank was built without frozen assets."
            )
        device = next(self.text_encoder.parameters()).device
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_embeds = self.text_encoder(
            texts_input.to(device), mask.to(device)
        ).last_hidden_state.to(dtype=self.training_dtype, device=device)
        prompt_embeds = [row[:length] for row, length in zip(prompt_embeds, seq_lens, strict=True)]
        prompt_embeds = torch.stack(
            [
                torch.cat([row, row.new_zeros(max_sequence_length - row.size(0), row.size(1))])
                for row in prompt_embeds
            ],
            dim=0,
        )
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        return prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)


__all__ = ["WanFrozenAssetEncoder", "retrieve_latents"]
