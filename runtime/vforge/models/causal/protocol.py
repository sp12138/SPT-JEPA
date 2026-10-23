"""Model boundary required by an incremental block-causal rollout."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from vforge.models.causal.cache import CausalKVCache


@runtime_checkable
class IncrementalKVTransformer(Protocol):
    """Minimal transformer interface used by an incremental causal executor."""

    def init_kv_cache(
        self,
        *,
        batch_size: int,
        num_cache_tokens: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> CausalKVCache: ...

    def full_clip_rotary_emb(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    def forward_kv_block(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        kv_cache: CausalKVCache,
        full_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        kv_write_start: int,
        commit_kv: bool,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor: ...


__all__ = ["IncrementalKVTransformer"]
