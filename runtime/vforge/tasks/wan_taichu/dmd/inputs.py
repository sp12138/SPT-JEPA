"""Wan-specific batch and noise translation for DMD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vforge.algorithms.dmd.model import (
    DmdNoiseLevel,
    FlowScheduler,
    PreparedDmdBatch,
)
from vforge.models.video_tokens import FrameMajorPatchTokenLayout


@dataclass(frozen=True)
class WanDmdInputs:
    """Conditioning tensors required by Wan generator and score towers."""

    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    condition_latents: torch.Tensor
    first_frame_latents: torch.Tensor | None = None


def prepare_wan_dmd_batch(
    encoded: dict[str, Any],
    *,
    negative_prompt_embeds: torch.Tensor,
    device: torch.device,
    condition_fields: tuple[str, ...] = ("condition_latents_normal",),
    require_first_frame: bool = True,
) -> PreparedDmdBatch[WanDmdInputs]:
    """Project an encoded Wan batch into frame-first DMD latents."""

    required = ("frame_latents", "prompt", *condition_fields)
    if require_first_frame:
        required = (*required, "first_frame_latents")
    missing = [name for name in required if not isinstance(encoded.get(name), torch.Tensor)]
    if missing:
        raise ValueError(f"Wan DMD encoded batch is missing tensors {missing}")
    frame_latents = encoded["frame_latents"].to(device)
    # y is the streams concatenated on the channel axis in declaration order,
    # the same order the ordinary Wan Taichu train path builds it in.
    condition = torch.cat([encoded[field].to(device) for field in condition_fields], dim=1)
    first_frame = encoded["first_frame_latents"].to(device) if require_first_frame else None
    prompt = encoded["prompt"].to(device)
    negative_prompt = negative_prompt_embeds.to(device)
    if frame_latents.ndim != 5 or condition.shape[0] != frame_latents.shape[0]:
        raise ValueError("Wan DMD latents must use [B, C, T, H, W]")
    if condition.shape[2:] != frame_latents.shape[2:]:
        raise ValueError("Wan DMD condition geometry must match frame_latents")
    if first_frame is not None and (
        first_frame.shape[:2] != frame_latents.shape[:2] or first_frame.shape[2] != 1
    ):
        raise ValueError("Wan DMD first_frame_latents must be [B, C, 1, H, W]")
    if negative_prompt.shape != prompt.shape:
        raise ValueError("Wan DMD negative prompt embedding must match prompt shape")
    return PreparedDmdBatch(
        clean={"latent": frame_latents.permute(0, 2, 1, 3, 4)},
        model_inputs=WanDmdInputs(
            prompt_embeds=prompt,
            negative_prompt_embeds=negative_prompt,
            condition_latents=condition,
            first_frame_latents=first_frame,
        ),
    )


def resolve_wan_noise_level(
    scheduler: FlowScheduler,
    normalized_level: torch.Tensor,
    batch: PreparedDmdBatch[WanDmdInputs],
    *,
    num_train_timesteps: int | None = None,
) -> DmdNoiseLevel[torch.Tensor]:
    """Resolve normalized levels to nearest Wan scheduler timesteps and sigmas."""

    if torch.any((normalized_level < 0) | (normalized_level > 1)):
        raise ValueError("normalized Wan DMD noise levels must lie in [0, 1]")
    latent = batch.clean["latent"]
    if normalized_level.shape == (latent.shape[0], 1):
        normalized_level = normalized_level.expand(latent.shape[:2])
    timestep_scale = (
        float(num_train_timesteps)
        if num_train_timesteps is not None
        else float(scheduler.timesteps.max().item())
    )
    requested = (
        normalized_level.to(
            device=batch.clean["latent"].device,
            dtype=torch.float64,
        )
        * timestep_scale
    )
    schedule = scheduler.timesteps.to(device=requested.device, dtype=torch.float64)
    sigmas = scheduler.sigmas.to(device=requested.device, dtype=torch.float64)
    if sigmas.numel() < schedule.numel():
        raise ValueError("scheduler.sigmas must align with scheduler.timesteps")
    indices = torch.argmin(
        (schedule.view(*(1 for _ in requested.shape), -1) - requested.unsqueeze(-1)).abs(),
        dim=-1,
    )
    # The nearest-neighbour lookup selects a *sigma*; it does not redefine the
    # level. The model is conditioned on the level the caller asked for, which
    # for the shifted flow-match schedule is not the tabulated neighbour: level
    # 938 sits next to a tabulated 938.1213, and conditioning on the neighbour
    # perturbs the timestep embedding and through it every block.
    timestep = requested.to(dtype=scheduler.timesteps.dtype)
    # Back to the scheduler's own dtype, the same treatment the timestep gets
    # above. float64 is here to pick the nearest tabulated level without ties
    # deciding it, not to widen the level itself: sigma goes on to multiply
    # latents, and carrying an artificially wide sigma into that arithmetic
    # rounds the result differently than the reference does. Sites that want a
    # wider intermediate ask for it explicitly.
    sigma = sigmas[indices].to(dtype=scheduler.sigmas.dtype)
    if sigma.ndim == 0:
        broadcast_sigma = sigma.reshape(*(1 for _ in latent.shape))
    elif sigma.shape == latent.shape[:2]:
        broadcast_sigma = sigma.view(*sigma.shape, *(1 for _ in latent.shape[2:]))
    elif sigma.shape == (latent.shape[0],):
        broadcast_sigma = sigma.view(latent.shape[0], *(1 for _ in latent.shape[1:]))
    else:
        raise ValueError(
            "Wan noise level must be scalar, per-batch, or per-frame; "
            f"got {tuple(sigma.shape)} for latent {tuple(latent.shape)}"
        )
    return DmdNoiseLevel(
        normalized=normalized_level,
        sigmas={"latent": broadcast_sigma},
        model_timestep=timestep,
    )


def wan_loss_valid_mask(
    batch: PreparedDmdBatch[WanDmdInputs],
    *,
    exclude_first_frame: bool = True,
) -> dict[str, torch.Tensor]:
    """Return a frame-broadcast mask over the frames the DMD loss may score.

    The TI2V first frame is an in-painted anchor, so it carries no generator
    signal and stays out of the loss. A14B has no anchor -- its reference frame
    arrives through y's mask channels -- so every frame is scored there.
    """

    latent = batch.clean["latent"]
    mask = torch.ones(
        latent.shape[:2] + (1,) * (latent.ndim - 2),
        dtype=torch.bool,
        device=latent.device,
    )
    if exclude_first_frame:
        mask[:, 0] = False
    return {"latent": mask}


def wan_score_noise_shape(
    batch: PreparedDmdBatch[WanDmdInputs],
) -> tuple[int, int]:
    """Sample one score timestep per video and broadcast it over frames."""

    return (int(batch.clean["latent"].shape[0]), 1)


def anchor_wan_score_noise(
    noisy: dict[str, torch.Tensor],
    clean: dict[str, torch.Tensor],
    level: DmdNoiseLevel[torch.Tensor],
) -> tuple[dict[str, torch.Tensor], DmdNoiseLevel[torch.Tensor]]:
    """Keep the TI2V first-frame anchor clean at score time."""

    anchored_latent = noisy["latent"].clone()
    anchored_latent[:, :1] = clean["latent"][:, :1]
    timestep = level.model_timestep.clone()
    timestep[:, 0] = 0
    normalized = level.normalized.clone()
    normalized[:, 0] = 0
    sigma = level.sigmas["latent"].clone()
    sigma[:, 0] = 0
    return (
        {"latent": anchored_latent},
        DmdNoiseLevel(
            normalized=normalized,
            sigmas={"latent": sigma},
            model_timestep=timestep,
        ),
    )


def wan_dmd_token_timesteps(
    model_timestep: torch.Tensor,
    latent: torch.Tensor,
    patch_size: tuple[int, int, int],
    *,
    frame_range: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Expand scalar, per-video, or per-frame values to Wan patch tokens."""

    layout = FrameMajorPatchTokenLayout.from_latent(latent, patch_size)
    if int(patch_size[0]) != 1:
        raise NotImplementedError("Wan DMD per-frame timesteps require temporal patch size 1")
    timestep = model_timestep.to(device=latent.device)
    if timestep.ndim == 0:
        frame_timestep = timestep.expand(layout.batch_size, layout.frames_per_sequence)
    elif timestep.shape == (layout.batch_size,):
        frame_timestep = timestep[:, None].expand(
            layout.batch_size,
            layout.frames_per_sequence,
        )
    elif timestep.ndim == 2 and timestep.shape[0] == layout.batch_size:
        if timestep.shape[1] == layout.frames_per_sequence:
            frame_timestep = timestep
        elif frame_range is not None and timestep.shape[1] >= frame_range[1]:
            frame_timestep = timestep[:, frame_range[0] : frame_range[1]]
        else:
            raise ValueError("Wan DMD timestep does not cover the selected frame range")
    else:
        raise ValueError("Wan DMD timestep must be scalar, per-video, or per-frame")
    return layout.expand_frame_values(frame_timestep)


__all__ = [
    "WanDmdInputs",
    "anchor_wan_score_noise",
    "prepare_wan_dmd_batch",
    "resolve_wan_noise_level",
    "wan_dmd_token_timesteps",
    "wan_loss_valid_mask",
    "wan_score_noise_shape",
]
