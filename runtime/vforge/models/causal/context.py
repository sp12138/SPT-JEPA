"""Per-call execution context for incremental KV self-attention."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from vforge.models.causal.cache import CausalKVCache, CausalKVCacheLayer

_ACTIVE_INCREMENTAL_KV_CACHE: ContextVar[CausalKVCache | None] = ContextVar(
    "active_incremental_kv_cache",
    default=None,
)


@contextmanager
def use_incremental_kv_cache(cache: CausalKVCache) -> Iterator[None]:
    """Expose a mutable cache without passing it through FSDP's input tree."""
    token = _ACTIVE_INCREMENTAL_KV_CACHE.set(cache)
    try:
        yield
    finally:
        _ACTIVE_INCREMENTAL_KV_CACHE.reset(token)


def get_incremental_kv_cache() -> CausalKVCache:
    """Return the cache bound around the current incremental model call."""
    cache = _ACTIVE_INCREMENTAL_KV_CACHE.get()
    if cache is None:
        raise RuntimeError("incremental KV forward has no active cache context")
    return cache


@dataclass(frozen=True)
class IncrementalKVAttentionContext:
    """Bind one cache layer and write operation to an attention call.

    ``history_tokens`` pins how much committed history this call may attend.
    Leaving it unset reads the cache's live extent, which keeps single-pass
    inference simple but is unsafe under activation recompute: later blocks
    commit more history before ``backward`` runs, so a recomputed forward would
    attend frames that did not exist when it first ran.
    """

    cache_layer: CausalKVCacheLayer | None = None
    layer_index: int | None = None
    write_start: int = 0
    history_tokens: int | None = None
    commit: bool = False
    frame_tokens: int = 1
    local_window_frames: int = -1
    sink_frames: int = 0
    usp_size: int = 1
    usp_group: Any = None
    original_heads: int | None = None

    def __post_init__(self) -> None:
        if (self.cache_layer is None) == (self.layer_index is None):
            raise ValueError("set exactly one of cache_layer or layer_index")
        if self.layer_index is not None and self.layer_index < 0:
            raise ValueError(f"layer_index must be non-negative, got {self.layer_index}")
        if self.write_start < 0:
            raise ValueError(f"write_start must be non-negative, got {self.write_start}")
        if self.history_tokens is not None and self.history_tokens < self.write_start:
            raise ValueError(
                "history_tokens cannot exclude already-written slots; "
                f"history_tokens={self.history_tokens}, write_start={self.write_start}"
            )
        if self.frame_tokens < 1:
            raise ValueError(f"frame_tokens must be positive, got {self.frame_tokens}")
        if self.local_window_frames == 0 or self.local_window_frames < -1:
            raise ValueError(
                f"local_window_frames must be -1 or positive; got {self.local_window_frames}"
            )
        if self.sink_frames < 0:
            raise ValueError(f"sink_frames must be non-negative, got {self.sink_frames}")
        if self.usp_size < 1:
            raise ValueError(f"usp_size must be positive, got {self.usp_size}")

    def resolve_cache_layer(self) -> CausalKVCacheLayer:
        """Resolve mutable storage after crossing any FSDP block boundary."""
        if self.cache_layer is not None:
            return self.cache_layer
        assert self.layer_index is not None
        return get_incremental_kv_cache()[self.layer_index]


__all__ = [
    "IncrementalKVAttentionContext",
    "get_incremental_kv_cache",
    "use_incremental_kv_cache",
]
