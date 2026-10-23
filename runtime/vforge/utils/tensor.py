"""Tensor representation helpers shared across VForge domains."""

from __future__ import annotations

import torch


def to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return the local shard for DTensor-like values."""
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor


__all__ = ["to_local_tensor"]
