"""Opt-in primitives for single-latent video flow-matching inference.

This module deliberately stops at one model-specific prediction boundary.
Condition packing, timestep representation, cache ownership, expert routing,
and ``engine_forward`` remain owned by the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias

import torch

from vforge.models.video_tokens import FrameMajorPatchTokenLayout

FlowMatchPredictor: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor, Any, str],
    torch.Tensor,
]
FlowMatchTrace: TypeAlias = Callable[[str, int, torch.Tensor], None]


def build_first_frame_per_token_timestep(
    latent: torch.Tensor,
    scalar_timestep: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Expand a scalar timestep in frame-major order, keeping frame zero clean."""
    layout = FrameMajorPatchTokenLayout.from_latent(latent, patch_size)
    per_frame = torch.full(
        (layout.frames_per_sequence,),
        float(scalar_timestep.to(latent.device).reshape(())),
        device=latent.device,
        dtype=torch.float32,
    )
    per_frame[0] = 0.0
    return layout.expand_frame_values(per_frame)


@torch.no_grad()
def run_flow_match_denoising(
    *,
    scheduler: Any,
    latents: torch.Tensor,
    num_inference_steps: int,
    conditioning: Any,
    predict: FlowMatchPredictor,
    guidance_scale: float = 1.0,
    negative_conditioning: Any | None = None,
    frozen_prefix_latents: torch.Tensor | None = None,
    trace: FlowMatchTrace | None = None,
) -> torch.Tensor:
    """Run one scheduler traversal while delegating only model prediction.

    ``predict`` receives ``(current_latents, scheduler_timestep, conditioning,
    cache_name)``. The scheduler timestep is passed through unchanged so model
    roots may route experts from it. Persistent latents are never cast here:
    callers may cast model inputs inside ``predict`` while scheduler
    accumulation retains the scheduler's output dtype.
    """

    prefix_frames = 0
    if frozen_prefix_latents is not None:
        prefix_frames = int(frozen_prefix_latents.shape[2])
        latents[:, :, :prefix_frames] = frozen_prefix_latents

    scheduler.set_timesteps(num_inference_steps, device=latents.device)
    use_cfg = guidance_scale > 1

    for step_index, timestep in enumerate(scheduler.timesteps):
        if trace is not None:
            trace("timestep", step_index, timestep)
            trace("input_latents", step_index, latents)
        prediction = predict(latents, timestep, conditioning, "cond")
        if trace is not None:
            trace("cond_prediction", step_index, prediction)
        if use_cfg:
            if negative_conditioning is None:
                raise ValueError("CFG requires negative_prompt_embeds")
            unconditional = predict(latents, timestep, negative_conditioning, "uncond")
            if trace is not None:
                trace("uncond_prediction", step_index, unconditional)
            prediction = unconditional + guidance_scale * (prediction - unconditional)
        if trace is not None:
            trace("guided_prediction", step_index, prediction)
        latents = scheduler.step(
            prediction,
            timestep,
            latents,
            return_dict=False,
        )[0]
        if frozen_prefix_latents is not None:
            latents[:, :, :prefix_frames] = frozen_prefix_latents
        if trace is not None:
            trace("output_latents", step_index, latents)
    return latents


__all__ = [
    "FlowMatchPredictor",
    "FlowMatchTrace",
    "build_first_frame_per_token_timestep",
    "run_flow_match_denoising",
]
