"""Wan 2.2 TI2V sampling used by the generic evaluation runner.

Mirrors the official TI2V I2V loop: the reference frame is written into latent
frame 0, every step is driven by a per-token timestep whose frame-0 tokens are
clean, and frame 0 is re-projected after each scheduler step so the sampler can
never drift away from the given first frame.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from vforge.models.wan.inference import (
    DEFAULT_NEGATIVE_PROMPT,
    WanFlowMatchPipelineBase,
)


def build_per_token_timestep(
    latent: torch.Tensor,
    scalar_timestep: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Expand a scalar timestep to ``[B, tokens]`` with a clean first frame.

    Token order is frame-major, matching ``hidden_states.flatten(2).transpose(1, 2)``
    inside the transformer, so every spatial token of a latent frame shares one
    timestep. Frame 0 gets ``0`` because it holds the in-painted reference.
    """
    batch, _, t_lat, h_lat, w_lat = latent.shape
    p_t, p_h, p_w = patch_size
    t_post = t_lat // p_t
    tokens_per_frame = (h_lat // p_h) * (w_lat // p_w)

    device = latent.device
    value = scalar_timestep.to(device).reshape(())
    per_frame = torch.full(
        (t_post,),
        float(value),
        device=device,
        dtype=torch.float32,
    )
    per_frame[0] = 0.0
    per_token = per_frame.repeat_interleave(tokens_per_frame)
    return per_token.unsqueeze(0).expand(batch, -1).contiguous()


class Wan22Ti2v5bPipeline(WanFlowMatchPipelineBase):
    """Flow-matching TI2V sampler with temporal first-frame in-painting."""

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
        vae_scale_factor_spatial: int = 16,
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

    def _denoise(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cache_name: str,
    ) -> torch.Tensor:
        with self.transformer.cache_context(cache_name):
            output = self.engine_forward(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                attention_kwargs=None,
                return_dict=False,
            )
        return output[0] if isinstance(output, tuple) else output

    @torch.no_grad()
    def call_with_latent(
        self,
        encoder_hidden_state: torch.Tensor,
        *,
        y: torch.Tensor | None,
        first_frame_latents: torch.Tensor,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 121,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        latents: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        decode_output: bool = True,
    ) -> torch.Tensor:
        """TI2V CFG: the uncond branch keeps normal + in-paint, swaps text only."""
        device = encoder_hidden_state.device
        batch_size = int(encoder_hidden_state.shape[0])
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

        z0 = first_frame_latents.to(device=device, dtype=latents.dtype)
        latents[:, :, 0:1] = z0

        encoder_hidden_state = encoder_hidden_state.to(device)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
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

        patch_size = tuple(int(value) for value in self.transformer.config.patch_size)
        for timestep in self.scheduler.timesteps:
            latent_model_input = latents.to(transformer_dtype)
            hidden_states = (
                torch.cat([latent_model_input, y], dim=1) if y is not None else latent_model_input
            )
            per_token_t = build_per_token_timestep(latents, timestep, patch_size)
            noise_pred = self._denoise(
                hidden_states,
                per_token_t,
                encoder_hidden_state,
                "cond",
            )
            if guidance_scale > 1:
                noise_uncond = self._denoise(
                    hidden_states,
                    per_token_t,
                    negative_prompt_embeds,
                    "uncond",
                )
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
            latents = self.scheduler.step(
                noise_pred,
                timestep,
                latents,
                return_dict=False,
            )[0]
            # Frame 0 is a given, not a prediction; re-project it every step.
            latents[:, :, 0:1] = z0

        if not decode_output:
            return latents
        return self.decode_latents(latents)


@dataclass
class Wan22InferencePipeline:
    """Frozen assets + transformer used for one-sample inference."""

    model: nn.Module
    encode_condition: Callable[[dict[str, Any]], dict[str, Any]]
    pipeline: Wan22Ti2v5bPipeline
    stream_fields: Mapping[str, str]
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT

    @torch.no_grad()
    def infer_sample(self, sample: dict[str, Any], *, seed: int) -> dict[str, Any]:
        """Run fixed-seed CFG sampling and return the decoded video."""
        device = next(self.model.parameters()).device
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

        encoded = sample.get("__encoded__")
        if encoded is None:
            encoded = self.encode_condition(sample)
        prompt = encoded["prompt"].to(device)
        condition = _concat_condition_streams(encoded, self.stream_fields, device)
        first_frame_latents = _to_device(encoded.get("first_frame_latents"), device)
        if first_frame_latents is None:
            raise ValueError(
                "Wan22 TI2V sampling requires first_frame_latents; the encoded "
                "bundle came from a cache written before TI2V support"
            )

        height, width, num_frames = _resolve_pixel_grid(sample, encoded, self.pipeline)

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

        is_usp_leader = get_parallel_state().usp_rank == 0

        video = self.pipeline.call_with_latent(
            prompt,
            y=condition,
            first_frame_latents=first_frame_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            negative_prompt_embeds=neg,
            decode_output=is_usp_leader,
        )
        return {
            "video": video.detach().cpu() if is_usp_leader else None,
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


def _resolve_pixel_grid(
    sample: dict[str, Any],
    encoded: dict[str, Any],
    pipeline: Wan22Ti2v5bPipeline,
) -> tuple[int, int, int]:
    if "__height__" in sample:
        return (
            int(sample["__height__"]),
            int(sample["__width__"]),
            int(sample["__num_frames__"]),
        )
    frames = sample.get("video_frames")
    if isinstance(frames, torch.Tensor):
        # Collated batch is [B, T, C, H, W]; a raw sample is [T, C, H, W].
        num_frames = int(frames.shape[1] if frames.ndim == 5 else frames.shape[0])
        return int(frames.shape[-2]), int(frames.shape[-1]), num_frames
    frame_latents = encoded["frame_latents"]
    height = int(frame_latents.shape[-2] * pipeline.vae_scale_factor_spatial)
    width = int(frame_latents.shape[-1] * pipeline.vae_scale_factor_spatial)
    num_frames = (int(frame_latents.shape[-3]) - 1) * pipeline.vae_scale_factor_temporal + 1
    return height, width, num_frames


def _to_device(value: Any, device: torch.device) -> Any:
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _concat_condition_streams(
    encoded: dict[str, Any],
    stream_fields: Mapping[str, str],
    device: torch.device,
) -> torch.Tensor | None:
    """Channel-concatenate the condition streams in declared order."""
    blocks: list[torch.Tensor] = []
    for name, field in stream_fields.items():
        block = encoded.get(field)
        if block is None:
            raise ValueError(
                f"Wan22 TI2V sampling needs the {name!r} condition stream, but "
                f"the encoded bundle has no {field!r}"
            )
        blocks.append(block.to(device))
    if not blocks:
        return None
    return blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=1)


def build_inference_pipeline(
    *,
    model: nn.Module,
    encode_condition: Callable[[dict[str, Any]], dict[str, Any]],
    tokenizer: Any,
    text_encoder: Any,
    vae: Any,
    engine_forward: Callable[..., Any],
    scheduler: Any,
    stream_fields: Mapping[str, str],
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> Wan22InferencePipeline:
    pipeline = Wan22Ti2v5bPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=model,
        engine_forward=engine_forward,
        scheduler=scheduler,
    )
    return Wan22InferencePipeline(
        model=model,
        encode_condition=encode_condition,
        pipeline=pipeline,
        stream_fields=stream_fields,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
    )


__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "Wan22InferencePipeline",
    "Wan22Ti2v5bPipeline",
    "build_inference_pipeline",
    "build_per_token_timestep",
]
