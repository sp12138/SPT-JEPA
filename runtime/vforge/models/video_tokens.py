"""Model-independent helpers for frame-major video patch tokens."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FrameMajorPatchTokenLayout:
    """Patch-token geometry for ``[B, C, T, H, W]`` video latents.

    Frame values may contain one temporal sequence or multiple concatenated
    sequences, such as teacher forcing's ``[clean | noisy]`` pair.
    """

    batch_size: int
    frames_per_sequence: int
    tokens_per_frame: int

    @classmethod
    def from_latent(
        cls,
        latent: torch.Tensor,
        patch_size: tuple[int, int, int],
    ) -> FrameMajorPatchTokenLayout:
        if latent.ndim != 5:
            raise ValueError(f"latent must be [B, C, T, H, W], got {tuple(latent.shape)}")
        batch_size, _, temporal, height, width = latent.shape
        p_t, p_h, p_w = (int(value) for value in patch_size)
        if p_t < 1 or p_h < 1 or p_w < 1:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if temporal % p_t or height % p_h or width % p_w:
            raise ValueError(
                f"latent shape {tuple(latent.shape[-3:])} must divide by patch_size={patch_size}"
            )
        return cls(
            batch_size=int(batch_size),
            frames_per_sequence=int(temporal // p_t),
            tokens_per_frame=int((height // p_h) * (width // p_w)),
        )

    def expand_frame_values(self, frame_values: torch.Tensor) -> torch.Tensor:
        """Expand ``[..., frames]`` values to frame-major patch-token values."""
        if frame_values.ndim == 1:
            batched = frame_values.unsqueeze(0).expand(self.batch_size, -1)
        elif frame_values.ndim == 2:
            if frame_values.shape[0] != self.batch_size:
                raise ValueError(
                    "frame_values batch does not match latent batch: "
                    f"{frame_values.shape[0]} != {self.batch_size}"
                )
            batched = frame_values
        else:
            raise ValueError(
                f"frame_values must be [frames] or [B, frames], got {tuple(frame_values.shape)}"
            )
        if batched.shape[1] < 1 or batched.shape[1] % self.frames_per_sequence:
            raise ValueError(
                "frame_values must contain one or more complete frame sequences: "
                f"length={batched.shape[1]}, "
                f"frames_per_sequence={self.frames_per_sequence}"
            )
        return batched.repeat_interleave(self.tokens_per_frame, dim=1).contiguous()


__all__ = ["FrameMajorPatchTokenLayout"]
