"""Backend-neutral context-parallel operations.

Models call this module without knowing which training engine implements
context parallelism. An engine installs exactly one backend before model
execution; unsupported engines reject ``cp_size > 1`` at construction instead
of leaving the model to fail inside an attention kernel.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

import torch


class ContextParallelBackend(Protocol):
    """Operations an engine must provide for one context-parallel layout."""

    name: str

    def shard_sequence(
        self,
        tensor: torch.Tensor,
        *,
        dim: int = 1,
        group: Any = None,
    ) -> torch.Tensor: ...

    def gather_sequence(
        self,
        tensor: torch.Tensor,
        *,
        dim: int = 1,
        group: Any = None,
    ) -> torch.Tensor: ...

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor: ...

    def close(self) -> None: ...


ContextParallelBackendName = Literal["mcore"]
_CONTEXT_PARALLEL_BACKEND: ContextParallelBackend | None = None


def initialize_context_parallel(
    backend: ContextParallelBackendName,
) -> ContextParallelBackend:
    """Install the engine-owned CP backend for this process.

    Re-applying the same backend is idempotent. Switching implementations in a
    live distributed runtime is rejected because cached kernel modules and
    activation layouts belong to the original process groups.
    """
    global _CONTEXT_PARALLEL_BACKEND
    current = _CONTEXT_PARALLEL_BACKEND
    if current is not None:
        if current.name != backend:
            raise RuntimeError(
                "context parallel backend is already initialized as "
                f"{current.name!r}; cannot replace it with {backend!r}"
            )
        return current

    if backend == "mcore":
        from vforge.distributed.context_parallel_mcore import MCoreContextParallelBackend

        selected: ContextParallelBackend = MCoreContextParallelBackend()
    else:  # pragma: no cover - Literal guards typed callers; runtime guards config input.
        raise ValueError(f"unsupported context parallel backend {backend!r}")
    _CONTEXT_PARALLEL_BACKEND = selected
    return selected


def get_context_parallel_backend() -> ContextParallelBackend:
    """Return the installed backend, failing before silently running locally."""
    if _CONTEXT_PARALLEL_BACKEND is None:
        raise RuntimeError(
            "context parallelism is enabled but no backend is initialized; "
            "the selected training engine must install one before model execution"
        )
    return _CONTEXT_PARALLEL_BACKEND


def shard_sequence(
    tensor: torch.Tensor,
    *,
    dim: int = 1,
    group: Any = None,
) -> torch.Tensor:
    """Shard a replicated sequence using the active backend's layout."""
    return get_context_parallel_backend().shard_sequence(tensor, dim=dim, group=group)


def gather_sequence(
    tensor: torch.Tensor,
    *,
    dim: int = 1,
    group: Any = None,
) -> torch.Tensor:
    """Restore a sequence sharded with the active backend's layout."""
    return get_context_parallel_backend().gather_sequence(tensor, dim=dim, group=group)


def context_parallel_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Run attention using the active engine-owned CP implementation."""
    return get_context_parallel_backend().attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=is_causal,
    )


def reset_context_parallel() -> None:
    """Release backend-owned caches during process runtime teardown."""
    global _CONTEXT_PARALLEL_BACKEND
    backend = _CONTEXT_PARALLEL_BACKEND
    _CONTEXT_PARALLEL_BACKEND = None
    if backend is not None:
        backend.close()


__all__ = [
    "ContextParallelBackend",
    "ContextParallelBackendName",
    "context_parallel_attention",
    "gather_sequence",
    "get_context_parallel_backend",
    "initialize_context_parallel",
    "reset_context_parallel",
    "shard_sequence",
]
