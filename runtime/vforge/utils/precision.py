"""Shared precision contexts for training and evaluation execution."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def autocast_context(
    device: torch.device,
    *,
    enabled: bool,
    precision: str,
):
    """Return the configured autocast context for one execution boundary."""
    if device.type != "cuda" or not enabled:
        return nullcontext()
    if precision in {"bf16", "bfloat16"}:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision in {"fp16", "float16"}:
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


__all__ = ["autocast_context"]
