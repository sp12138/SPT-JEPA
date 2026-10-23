"""Wan doubled teacher-forcing adapters for consistency distillation."""

from __future__ import annotations

import torch
import torch.nn as nn

from vforge.algorithms.cd.model import CdTeacherForcingAdapter, PreparedCdBatch
from vforge.utils.training import resolve_dtype


class Wan5bCdTeacherForcingAdapter(CdTeacherForcingAdapter):
    def __init__(self, *, precision: str) -> None:
        self.dtype = resolve_dtype(precision)

    def predict_velocity(
        self,
        model: nn.Module,
        *,
        noisy_latent: torch.Tensor,
        clean_context: torch.Tensor,
        timestep: torch.Tensor,
        batch: PreparedCdBatch,
    ) -> torch.Tensor:
        hidden_states = build_cd_teacher_forcing_hidden_states(
            noisy_latent=noisy_latent,
            clean_context=clean_context,
            condition_latents=(
                batch.condition("grid"),
                batch.condition("normal"),
            ),
            dtype=self.dtype,
        )
        per_token_timestep = build_cd_teacher_forcing_timestep(
            noisy_latent,
            timestep,
            tuple(int(value) for value in model.config.patch_size),
        )
        output = model.forward_teacher_forcing(
            hidden_states=hidden_states,
            timestep=per_token_timestep,
            encoder_hidden_states=batch.prompt_embeds.to(
                device=noisy_latent.device,
                dtype=self.dtype,
            ),
        )
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError("Wan CD teacher-forcing model must return a tensor")
        if output.shape != noisy_latent.shape:
            raise ValueError(
                "Wan CD velocity shape must match noisy latent; "
                f"got {tuple(output.shape)} and {tuple(noisy_latent.shape)}"
            )
        return output


class WanA14bCdTeacherForcingAdapter(CdTeacherForcingAdapter):
    """A14B adapter with full-frame timesteps and high/low expert routing."""

    condition_streams = ("rgb", "grid", "normal")

    def __init__(self, *, precision: str) -> None:
        self.dtype = resolve_dtype(precision)

    def predict_velocity(
        self,
        model: nn.Module,
        *,
        noisy_latent: torch.Tensor,
        clean_context: torch.Tensor,
        timestep: torch.Tensor,
        batch: PreparedCdBatch,
    ) -> torch.Tensor:
        hidden_states = build_cd_teacher_forcing_hidden_states(
            noisy_latent=noisy_latent,
            clean_context=clean_context,
            condition_latents=tuple(batch.condition(stream) for stream in self.condition_streams),
            dtype=self.dtype,
        )
        per_token_timestep = build_cd_teacher_forcing_timestep(
            noisy_latent,
            timestep,
            tuple(int(value) for value in model.config.patch_size),
            zero_first_noisy_frame=False,
        )
        select_transformer = getattr(model, "select_transformer", None)
        expert = select_transformer(per_token_timestep) if callable(select_transformer) else model
        output = expert.forward_teacher_forcing(
            hidden_states=hidden_states,
            timestep=per_token_timestep,
            encoder_hidden_states=batch.prompt_embeds.to(
                device=noisy_latent.device,
                dtype=self.dtype,
            ),
        )
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            raise TypeError("Wan A14B CD teacher-forcing model must return a tensor")
        if output.shape != noisy_latent.shape:
            raise ValueError(
                "Wan A14B CD velocity shape must match noisy latent; "
                f"got {tuple(output.shape)} and {tuple(noisy_latent.shape)}"
            )
        return output


def build_cd_teacher_forcing_hidden_states(
    *,
    noisy_latent: torch.Tensor,
    clean_context: torch.Tensor,
    condition_latents: tuple[torch.Tensor, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if noisy_latent.shape != clean_context.shape:
        raise ValueError("Wan CD noisy and clean latent tensors must share one shape")
    if not condition_latents:
        raise ValueError("Wan CD requires at least one condition stream")
    for condition in condition_latents:
        if condition.ndim != noisy_latent.ndim:
            raise ValueError("Wan CD condition tensors must be rank five")
        if (
            condition.shape[0] != noisy_latent.shape[0]
            or condition.shape[2:] != noisy_latent.shape[2:]
        ):
            raise ValueError(
                "Wan CD conditions must match latent batch and spatiotemporal geometry"
            )
    converted = tuple(
        condition.to(device=noisy_latent.device, dtype=dtype) for condition in condition_latents
    )
    clean = torch.cat(
        [clean_context.to(dtype=dtype), *converted],
        dim=1,
    )
    noisy = torch.cat(
        [noisy_latent.to(dtype=dtype), *converted],
        dim=1,
    )
    return torch.cat([clean, noisy], dim=2)


def build_cd_teacher_forcing_timestep(
    latent: torch.Tensor,
    scalar_timestep: torch.Tensor,
    patch_size: tuple[int, ...],
    *,
    zero_first_noisy_frame: bool = True,
) -> torch.Tensor:
    if latent.ndim != 5 or len(patch_size) != 3:
        raise ValueError("Wan CD timestep requires [B,C,T,H,W] and a 3D patch size")
    batch, _, frames, height, width = latent.shape
    patch_t, patch_h, patch_w = patch_size
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError("Wan CD latent geometry must be divisible by patch size")
    post_frames = frames // patch_t
    tokens_per_frame = (height // patch_h) * (width // patch_w)
    clean = torch.zeros(post_frames, device=latent.device, dtype=torch.float32)
    noisy = torch.full(
        (post_frames,),
        float(scalar_timestep.reshape(())),
        device=latent.device,
        dtype=torch.float32,
    )
    if zero_first_noisy_frame:
        noisy[0] = 0.0
    per_token = torch.cat([clean, noisy]).repeat_interleave(tokens_per_frame)
    return per_token.unsqueeze(0).expand(batch, -1).contiguous()


__all__ = [
    "Wan5bCdTeacherForcingAdapter",
    "WanA14bCdTeacherForcingAdapter",
    "build_cd_teacher_forcing_hidden_states",
    "build_cd_teacher_forcing_timestep",
]
