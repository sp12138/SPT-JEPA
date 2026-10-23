"""Per-row diffusion timestep assignment for packed generation streams."""

from __future__ import annotations

import torch


def build_row_timesteps(
    *,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    text_indices: torch.Tensor,
    num_reference_video_rows: int,
    num_reference_audio_rows: int,
    video_timestep: float,
    audio_timestep: float,
    reference_timestep: float,
    text_timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return distinct timestep values and each packed row's lookup index."""
    for name, indices in (
        ("video_indices", video_indices),
        ("audio_indices", audio_indices),
        ("text_indices", text_indices),
    ):
        if indices.ndim != 1:
            raise ValueError(f"{name} must be rank one")
    if not 0 <= num_reference_video_rows <= video_indices.numel():
        raise ValueError("num_reference_video_rows is outside the visual stream")
    if not 0 <= num_reference_audio_rows <= audio_indices.numel():
        raise ValueError("num_reference_audio_rows is outside the audio stream")

    sequence_length = int(video_indices.numel() + audio_indices.numel() + text_indices.numel())
    if sequence_length < 1:
        raise ValueError("packed generation sequence must contain at least one row")
    row_timesteps = torch.empty(
        sequence_length,
        dtype=torch.float32,
        device=video_indices.device,
    )
    row_timesteps.index_fill_(0, text_indices, float(text_timestep))
    row_timesteps.index_fill_(
        0,
        video_indices[:num_reference_video_rows],
        float(reference_timestep),
    )
    row_timesteps.index_fill_(
        0,
        video_indices[num_reference_video_rows:],
        float(video_timestep),
    )
    row_timesteps.index_fill_(
        0,
        audio_indices[:num_reference_audio_rows],
        float(reference_timestep),
    )
    row_timesteps.index_fill_(
        0,
        audio_indices[num_reference_audio_rows:],
        float(audio_timestep),
    )
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


__all__ = ["build_row_timesteps"]
