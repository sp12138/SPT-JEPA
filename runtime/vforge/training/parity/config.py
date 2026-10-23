"""Configuration for opt-in numerical parity behavior."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

# What this knob may name: the sequence parallel degree, under the current
# spelling or the one before it. The value arrives from integration YAML in
# downstream repositories, so the old spelling is accepted rather than rejected,
# and the one given is kept verbatim -- a caller reading its own config back, or
# the diagnostics probe that reports it, should see what it wrote.
# resolve_grad_scale() treats both alike. sp_size is not accepted: nothing sets
# it outside one downstream test, and sp is the word Megatron uses for a
# different mechanism, which is what this rename was for.
_GRAD_SCALE_DIMENSION = "usp_size"
_GRAD_SCALE_ALIASES = ("ulysses_size",)
GRAD_SCALE_NAMES = (_GRAD_SCALE_DIMENSION, *_GRAD_SCALE_ALIASES)


@dataclass(frozen=True)
class ParityStepConfig:
    """Optional numerical controls that replace the native optimizer step."""

    optimizer_step_mode: str = "torch"
    scale_grads_before_optimizer_step: str | float | None = None
    quantize_after_optimizer_step: str | None = None
    emit_legacy_probe_aliases: bool = False

    def __post_init__(self) -> None:
        if self.optimizer_step_mode not in {"torch", "manual_adamw_bf16"}:
            raise ValueError("optimizer_step_mode must be 'torch' or 'manual_adamw_bf16'")
        if self.scale_grads_before_optimizer_step in _GRAD_SCALE_ALIASES:
            warnings.warn(
                f"scale_grads_before_optimizer_step="
                f"{self.scale_grads_before_optimizer_step!r} is deprecated; use "
                f"{_GRAD_SCALE_DIMENSION!r}.",
                FutureWarning,
                stacklevel=3,
            )
        grad_scale = self.scale_grads_before_optimizer_step
        if isinstance(grad_scale, bool) or (
            grad_scale is not None and not isinstance(grad_scale, (str, int, float))
        ):
            raise ValueError(
                "scale_grads_before_optimizer_step must be None, 'usp_size', or a positive number"
            )
        if isinstance(grad_scale, str) and grad_scale not in GRAD_SCALE_NAMES:
            raise ValueError(
                "scale_grads_before_optimizer_step must be None, 'usp_size', or a positive number"
            )
        if isinstance(grad_scale, (int, float)) and float(grad_scale) <= 0:
            raise ValueError("scale_grads_before_optimizer_step must be positive")
        if self.quantize_after_optimizer_step not in {None, "bf16"}:
            raise ValueError("quantize_after_optimizer_step must be None or 'bf16'")
        if not isinstance(self.emit_legacy_probe_aliases, bool):
            raise ValueError("emit_legacy_probe_aliases must be a boolean")


__all__ = ["ParityStepConfig"]
