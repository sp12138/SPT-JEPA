"""Per-call attention routing policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from torch.nn.attention import SDPBackend


@dataclass(frozen=True)
class AttentionCallPolicy:
    """Override the process backend for one attention call.

    The default keeps the immutable process policy. ``sdpa`` is intended for
    call sites whose semantics cannot be represented by the configured flash
    backend, such as rectangular cross-attention with an additive mask.
    """

    backend: Literal["configured", "sdpa"] = "configured"
    sdpa_priority: tuple[SDPBackend, ...] | None = None
    sdpa_contiguous: bool = True
    # Model-selected dtype boundary; disabled for existing callers.
    cast_qk_to_v: bool = False

    def __post_init__(self) -> None:
        if self.backend != "sdpa" and (self.sdpa_priority is not None or not self.sdpa_contiguous):
            raise ValueError("SDPA options require backend='sdpa'")
        if self.sdpa_priority == ():
            raise ValueError("sdpa_priority must contain at least one backend")

    @classmethod
    def sdpa(
        cls,
        priority: tuple[SDPBackend, ...] | None = None,
        *,
        contiguous: bool = True,
    ) -> AttentionCallPolicy:
        return cls(
            backend="sdpa",
            sdpa_priority=priority,
            sdpa_contiguous=contiguous,
        )


__all__ = ["AttentionCallPolicy"]
