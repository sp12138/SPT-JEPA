"""Ops policy configuration.

Lives here rather than beside the Hydra schema so the layer that consumes the
policy also owns it. Everything that needs the type -- ``vforge.ops`` itself,
``distributed.runtime``, the Taichu compat mapping, and ``Trainer`` -- sits
below or beside the entry schema and could not import it from there without a
cycle, which is why each of them used to reach for a deferred or
``TYPE_CHECKING`` import instead.

This module deliberately depends on nothing but the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpsConfig:
    """Process-level numerical ops policy.

    Dense/varlen and distributed-attention selection are call-site/topology
    choices; this object only owns the immutable local backend policy for one
    process. Determinism applies to PyTorch, cuDNN, and the selected FA4 backend.
    """

    backend: str = "auto"
    allow_fallback: bool = True
    deterministic: bool = False
    debug: bool = False
    blas_backend: str = "default"

    def __post_init__(self) -> None:
        if self.backend not in {
            "auto",
            "sdpa",
            "sdpa_math",
            "flash_attention_3",
            "flash_attention_4",
        }:
            raise ValueError(f"unsupported attention backend: {self.backend!r}")
        if self.backend != "auto" and self.allow_fallback:
            raise ValueError("allow_fallback=true is only valid with attention.backend=auto")
        if type(self.deterministic) is not bool:
            raise TypeError("runtime.deterministic must be a boolean")
        if self.deterministic and self.backend == "flash_attention_3":
            raise ValueError(
                "runtime.deterministic=true is unsupported by attention.backend=flash_attention_3"
            )
        if self.blas_backend not in {"default", "cublas"}:
            raise ValueError(
                f"unsupported attention.blas_backend={self.blas_backend!r}; "
                "expected 'default' or 'cublas'"
            )


__all__ = ["OpsConfig"]
