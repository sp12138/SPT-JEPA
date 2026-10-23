"""Small model-agnostic helpers shared by native task plugins.

Every plugin has to answer the same handful of mechanical questions - what
torch dtype does ``runtime.precision`` mean, how do I build the AdamW the
request asked for, how many post-patch tokens are behind this latent batch.
None of the answers depend on the model, so they live here instead of being
copied into each ``vforge/tasks/<plugin>``.

Anything that does depend on the model (weight layout, condition contract,
checkpoint key mapping) stays in the plugin.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn


def resolve_dtype(precision: str) -> torch.dtype:
    """Map a ``runtime.precision`` string onto a torch dtype."""
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def build_optimizer_factory(
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> Callable[[nn.Module], torch.optim.Optimizer]:
    """Defer optimizer construction until after the model is sharded."""

    def factory(model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    return factory


def build_scheduler_factory(
    *,
    name: str,
    factor: float,
    warmup_steps: int,
    start_factor: float = 1.0,
    end_factor: float = 0.1,
    total_iters: int = 0,
) -> Callable[[torch.optim.Optimizer], Any] | None:
    """Return ``None`` for ``scheduler.name=none`` so the trainer skips stepping."""
    if name == "none":
        return None

    def factory(optimizer: torch.optim.Optimizer):
        if name == "linear":
            from torch.optim.lr_scheduler import LinearLR

            return LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=end_factor,
                total_iters=total_iters,
            )
        if name == "constant_with_warmup":
            from torch.optim.lr_scheduler import LambdaLR

            return LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(
                    float(step) / float(warmup_steps),
                    1.0,
                ),
            )

        from torch.optim.lr_scheduler import ConstantLR

        return ConstantLR(optimizer, factor=factor)

    return factory


def model_patch_size(model: nn.Module) -> tuple[int, ...]:
    """Patch size a patch-embedding DiT was configured with."""
    return tuple(
        int(value) for value in getattr(getattr(model, "config", None), "patch_size", (1, 2, 2))
    )


def patched_token_count(targets: torch.Tensor, patch_size: tuple[int, ...]) -> int:
    """Count post-patch tokens behind a ``[B, C, F, H, W]`` latent target."""
    if targets.ndim != 5 or len(patch_size) != 3:
        raise ValueError(
            "token accounting requires targets shaped [B,C,F,H,W] "
            f"and a 3D patch size; got {tuple(targets.shape)} and {patch_size}"
        )
    samples, _, frames, height, width = targets.shape
    patch_t, patch_h, patch_w = patch_size
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(
            "latent dimensions must be divisible by patch size for token "
            f"accounting; got {(frames, height, width)} and {patch_size}"
        )
    return int(samples * (frames // patch_t) * (height // patch_h) * (width // patch_w))


__all__ = [
    "build_optimizer_factory",
    "build_scheduler_factory",
    "model_patch_size",
    "patched_token_count",
    "resolve_dtype",
]
