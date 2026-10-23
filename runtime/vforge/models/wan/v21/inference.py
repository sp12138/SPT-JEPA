"""Wan 2.1 inference helpers used by the generic evaluation runner."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from vforge.models.video_diffusion.flow_match import run_flow_match_denoising
from vforge.models.wan.inference import (
    DEFAULT_NEGATIVE_PROMPT,
    WanFlowMatchPipelineBase,
)


class Wan21Pipeline(WanFlowMatchPipelineBase):
    """Flow-matching I2V sampler with Wan2.1 CFG convention."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        text_encoder: Any,
        vae: Any,
        transformer: Any,
        engine_forward: Callable[..., Any],
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae_scale_factor_temporal: int = 4,
        vae_scale_factor_spatial: int = 8,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
        )
        self.engine_forward = engine_forward

    @torch.no_grad()
    def call_with_latent(
        self,
        encoder_hidden_state: torch.Tensor,
        *,
        y: torch.Tensor | None,
        encoder_hidden_states_image_vae: torch.Tensor | None = None,
        encoder_hidden_states_image_clip: torch.Tensor | None = None,
        encoder_hidden_states_image_dino: torch.Tensor | None = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        latents: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        decode_output: bool = True,
    ) -> torch.Tensor:
        """Wan2.1 CFG: uncond reuses image conditions, swaps text only."""
        ref = (
            encoder_hidden_states_image_vae
            if encoder_hidden_states_image_vae is not None
            else (
                encoder_hidden_states_image_clip
                if encoder_hidden_states_image_clip is not None
                else encoder_hidden_state
            )
        )
        device = ref.device
        batch_size = int(ref.shape[0])
        transformer_dtype = self.transformer.dtype
        num_frames = self.align_num_frames(num_frames)

        noise_channels = self.transformer.config.in_channels - (y.shape[1] if y is not None else 0)
        latents = self.prepare_latents(
            batch_size,
            noise_channels,
            height,
            width,
            num_frames,
            self.dtype,
            device,
            generator=generator,
            latents=latents,
        )

        encoder_hidden_state = encoder_hidden_state.to(device)
        if guidance_scale > 1:
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    device=device,
                    dtype=self.dtype,
                )
            else:
                negative_prompt_embeds = self.encode_negative_prompt(
                    batch_size,
                    device,
                    self.dtype,
                    text="",
                )
        else:
            negative_prompt_embeds = None

        def predict(
            current_latents: torch.Tensor,
            timestep: torch.Tensor,
            text_condition: torch.Tensor,
            cache_name: str,
        ) -> torch.Tensor:
            latent_model_input = current_latents.to(transformer_dtype)
            hidden_states = (
                torch.cat([latent_model_input, y], dim=1) if y is not None else latent_model_input
            )
            t = timestep.expand(current_latents.shape[0]).to(device)
            with self.transformer.cache_context(cache_name):
                output = self.engine_forward(
                    hidden_states=hidden_states,
                    timestep=t,
                    encoder_hidden_states=text_condition,
                    encoder_hidden_states_image_vae=encoder_hidden_states_image_vae,
                    encoder_hidden_states_image_clip=encoder_hidden_states_image_clip,
                    encoder_hidden_states_image_dino=encoder_hidden_states_image_dino,
                    attention_kwargs=None,
                    return_dict=False,
                )
            return output[0] if isinstance(output, tuple) else output

        latents = run_flow_match_denoising(
            scheduler=self.scheduler,
            latents=latents,
            num_inference_steps=num_inference_steps,
            conditioning=encoder_hidden_state,
            predict=predict,
            guidance_scale=guidance_scale,
            negative_conditioning=negative_prompt_embeds,
        )

        if not decode_output:
            return latents
        return self.decode_latents(latents)


@dataclass
class Wan21InferencePipeline:
    """Frozen assets + transformer used for one-sample inference."""

    model: nn.Module
    encode_condition: Callable[[dict[str, Any]], dict[str, Any]]
    pipeline: Wan21Pipeline
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT

    @torch.no_grad()
    def infer_sample(self, sample: dict[str, Any], *, seed: int) -> dict[str, Any]:
        """Run fixed-seed CFG sampling and return latents/video tensors."""
        device = next(self.model.parameters()).device
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

        encoded = sample.get("__encoded__")
        if encoded is None:
            encoded = self.encode_condition(sample)
        prompt = encoded["prompt"].to(device)
        condition = _to_device(encoded.get("condition_latents_normal"), device)
        image_vae = _to_device(encoded.get("image_condition_latent_vae"), device)
        image_clip = _to_device(encoded.get("image_condition_latent_clip"), device)
        image_dino = _to_device(encoded.get("image_condition_latent_dino"), device)
        frames = sample.get("video_frames")
        if "__height__" in sample:
            height = int(sample["__height__"])
            width = int(sample["__width__"])
            num_frames = int(sample["__num_frames__"])
        elif isinstance(frames, torch.Tensor):
            # Collated batch is [B, T, C, H, W]; raw single sample is [T, C, H, W].
            if frames.ndim == 5:
                num_frames = int(frames.shape[1])
            else:
                num_frames = int(frames.shape[0])
            height = int(frames.shape[-2])
            width = int(frames.shape[-1])
        else:
            # Latent-only offline eval: recover pixel grid from latent spatial size.
            frame_latents = encoded["frame_latents"]
            height = int(frame_latents.shape[-2] * self.pipeline.vae_scale_factor_spatial)
            width = int(frame_latents.shape[-1] * self.pipeline.vae_scale_factor_spatial)
            num_frames = (
                int(frame_latents.shape[-3]) - 1
            ) * self.pipeline.vae_scale_factor_temporal + 1

        neg = None
        if self.guidance_scale > 1:
            neg = sample.get("__negative_prompt_embeds__")
            if neg is None:
                neg = self.pipeline.encode_negative_prompt(
                    batch_size=int(prompt.shape[0]),
                    device=device,
                    dtype=prompt.dtype,
                    text=self.negative_prompt,
                )
            else:
                neg = neg.to(device)

        from vforge.distributed.parallel_state import get_parallel_state

        is_output_leader = get_parallel_state().is_batch_leader

        video = self.pipeline.call_with_latent(
            prompt,
            y=condition,
            encoder_hidden_states_image_vae=image_vae,
            encoder_hidden_states_image_clip=image_clip,
            encoder_hidden_states_image_dino=image_dino,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            negative_prompt_embeds=neg,
            decode_output=is_output_leader,
        )
        return {
            "video": video.detach().cpu() if is_output_leader else None,
            "sample_id": sample.get("sample_id"),
            "seed": int(seed),
            "metadata": {
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "height": height,
                "width": width,
                "num_frames": num_frames,
            },
        }


def _to_device(value: Any, device: torch.device) -> Any:
    return value.to(device) if isinstance(value, torch.Tensor) else value


def build_inference_pipeline(
    *,
    model: nn.Module,
    encode_condition: Callable[[dict[str, Any]], dict[str, Any]],
    tokenizer: Any,
    text_encoder: Any,
    vae: Any,
    engine_forward: Callable[..., Any],
    scheduler: Any,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> Wan21InferencePipeline:
    pipeline = Wan21Pipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=model,
        engine_forward=engine_forward,
        scheduler=scheduler,
    )
    return Wan21InferencePipeline(
        model=model,
        encode_condition=encode_condition,
        pipeline=pipeline,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
    )


__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "Wan21Pipeline",
    "Wan21InferencePipeline",
    "build_inference_pipeline",
]
