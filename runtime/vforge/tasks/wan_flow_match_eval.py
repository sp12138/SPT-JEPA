"""Wan-family flow-match inference construction for prepared evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from vforge.models.video_diffusion.wan import encode_wan_negative_prompt
from vforge.tasks.prepared_eval import PreparedEvalShape
from vforge.tasks.requests import EvalRequest


@dataclass(frozen=True)
class WanFlowMatchEvalSettings:
    """Resolved sampling values shared by Wan preparation and inference."""

    num_inference_steps: int
    guidance_scale: float
    negative_prompt: str
    noise_shift: float


def resolve_wan_flow_match_eval_settings(
    request: EvalRequest,
    *,
    extras: Mapping[str, Any],
    default_negative_prompt: str,
    default_noise_shift: float,
) -> WanFlowMatchEvalSettings:
    """Resolve evaluation overrides once for both loader and pipeline."""
    return WanFlowMatchEvalSettings(
        num_inference_steps=(
            50 if request.num_inference_steps is None else int(request.num_inference_steps)
        ),
        guidance_scale=(5.0 if request.guidance_scale is None else float(request.guidance_scale)),
        negative_prompt=str(request.negative_prompt or default_negative_prompt),
        noise_shift=float(extras.get("noise_shift", default_noise_shift)),
    )


def resolve_wan_prepared_eval_shape(
    encoded: dict[str, Any],
    *,
    declared: PreparedEvalShape | None,
    vae: Any,
) -> PreparedEvalShape:
    """Remove training targets and resolve Wan latent geometry in pixel space."""
    frame_latents = encoded.pop("frame_latents", None)
    if declared is not None:
        return declared
    if isinstance(frame_latents, torch.Tensor):
        config = vae.config
        spatial = int(config.scale_factor_spatial)
        temporal = int(config.scale_factor_temporal)
        return PreparedEvalShape(
            height=int(frame_latents.shape[-2]) * spatial,
            width=int(frame_latents.shape[-1]) * spatial,
            num_frames=(int(frame_latents.shape[-3]) - 1) * temporal + 1,
        )
    try:
        return PreparedEvalShape(
            height=int(encoded.pop("__height__")),
            width=int(encoded.pop("__width__")),
            num_frames=int(encoded.pop("__num_frames__")),
        )
    except KeyError as exc:
        raise ValueError(
            "prepared Wan encoding must declare shape or contain "
            "frame_latents/encoded extent metadata"
        ) from exc


def prepare_wan_negative_prompt(
    encoded: Mapping[str, Any],
    *,
    tokenizer: Any,
    text_encoder: Any,
    settings: WanFlowMatchEvalSettings,
    device: torch.device,
) -> torch.Tensor | None:
    """Prepare the CFG unconditional branch for one encoded Wan sample."""
    if settings.guidance_scale <= 1:
        return None
    prompt = encoded.get("prompt")
    if not isinstance(prompt, torch.Tensor):
        raise TypeError("prepared Wan evaluation prompt must be a tensor")
    return encode_wan_negative_prompt(
        tokenizer,
        text_encoder,
        batch_size=int(prompt.shape[0]),
        device=device,
        dtype=prompt.dtype,
        text=settings.negative_prompt,
    )


def build_wan_flow_match_inference(
    *,
    model: Any,
    engine_forward: Callable[..., Any],
    pipeline_factory: Callable[..., Any],
    tokenizer: Any | None,
    text_encoder: Any | None,
    vae: Any | None,
    settings: WanFlowMatchEvalSettings,
    pipeline_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """Construct one Wan-compatible pipeline without coupling it to the loader."""
    return pipeline_factory(
        model=model,
        engine_forward=engine_forward,
        encode_condition=_prepared_sample_required,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=FlowMatchEulerDiscreteScheduler(shift=settings.noise_shift),
        num_inference_steps=settings.num_inference_steps,
        guidance_scale=settings.guidance_scale,
        negative_prompt=settings.negative_prompt,
        **dict(pipeline_kwargs or {}),
    )


def _prepared_sample_required(_sample: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError("prepared eval dataloader must yield encoded inference samples")


__all__ = [
    "WanFlowMatchEvalSettings",
    "build_wan_flow_match_inference",
    "prepare_wan_negative_prompt",
    "resolve_wan_flow_match_eval_settings",
    "resolve_wan_prepared_eval_shape",
]
