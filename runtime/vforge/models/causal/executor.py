"""Reusable execution adapter for incremental block-causal transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn

from vforge.distributed.parallel_state import get_parallel_state
from vforge.models.causal.cache import CausalKVCache, allocate_kv_cache
from vforge.models.causal.protocol import IncrementalKVTransformer


class IncrementalKVBlockForward(Protocol):
    """Model-owned single-block operation injected into the neutral executor."""

    def __call__(
        self,
        transformer: nn.Module,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        kv_cache: CausalKVCache,
        full_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        kv_write_start: int,
        commit_kv: bool,
        usp_size: int = 1,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor: ...


class IncrementalKVExecutor:
    """Expose the neutral incremental-KV protocol for a compatible transformer.

    The wrapped transformer supplies conventional ``config.patch_size``,
    ``blocks[*].attn1`` and ``rope`` attributes. The only model-specific
    operation is ``block_forward``, which owns patching, conditioning, trunk
    execution and output projection.

    This adapter is intentionally not an ``nn.Module``: it references an
    engine-owned transformer without re-registering its parameters.
    """

    def __init__(
        self,
        transformer: nn.Module,
        *,
        block_forward: IncrementalKVBlockForward,
        usp_size: int | None = None,
    ) -> None:
        self.transformer = transformer
        self.block_forward = block_forward
        self._configured_usp_size = usp_size
        self._require_supported()

    def _usp_size(self) -> int:
        if self._configured_usp_size is not None:
            return int(self._configured_usp_size)
        return int(get_parallel_state().config.usp_size)

    def _require_supported(self) -> None:
        patch_size = tuple(int(value) for value in self.transformer.config.patch_size)
        if patch_size[0] != 1:
            raise NotImplementedError(
                f"incremental_kv requires temporal patch size p_t=1; got patch_size={patch_size}"
            )
        usp_size = self._usp_size()
        if usp_size < 1:
            raise ValueError(f"incremental_kv requires usp_size >= 1; got {usp_size}")

    def init_kv_cache(
        self,
        *,
        batch_size: int,
        num_cache_tokens: int,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> CausalKVCache:
        """Allocate fixed-capacity absolute-slot storage for every layer."""
        self._require_supported()
        blocks = self.transformer.blocks
        if not blocks:
            raise ValueError("incremental_kv requires at least one transformer block")
        attention = blocks[0].attn1
        num_heads = int(attention.heads)
        usp_size = self._usp_size()
        if num_heads % usp_size:
            raise ValueError(
                f"incremental_kv requires num_heads={num_heads} to divide usp_size={usp_size}"
            )
        return allocate_kv_cache(
            num_layers=len(blocks),
            batch_size=batch_size,
            capacity=num_cache_tokens,
            num_heads=num_heads // usp_size,
            head_dim=int(attention.inner_dim) // num_heads,
            device=device,
            dtype=dtype,
        )

    def full_clip_rotary_emb(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build absolute rotary tables once for the complete clip geometry."""
        self._require_supported()
        return self.transformer.rope(hidden_states)

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
    ) -> torch.Tensor:
        """Run one model-specific block forward through the neutral protocol."""
        del return_dict
        self._require_supported()
        return self.block_forward(
            self.transformer,
            hidden_states,
            timestep,
            encoder_hidden_states,
            kv_cache=kv_cache,
            full_rotary_emb=full_rotary_emb,
            kv_write_start=kv_write_start,
            commit_kv=commit_kv,
            # The same size the cache was allocated with. ``init_kv_cache``
            # stores ``num_heads // usp_size`` heads per rank, so a forward that
            # defaulted to 1 here would build full-width keys against a
            # head-sharded cache.
            usp_size=self._usp_size(),
            attention_kwargs=attention_kwargs,
        )


@dataclass(frozen=True)
class IncrementalKVGuidanceCaches:
    """Conditional and optional unconditional caches for one rollout."""

    conditional: CausalKVCache
    unconditional: CausalKVCache | None = None

    def __post_init__(self) -> None:
        if self.unconditional is self.conditional:
            raise RuntimeError(
                "CFG conditional and unconditional branches must use separate caches"
            )


def allocate_incremental_kv_guidance_caches(
    executor: IncrementalKVTransformer,
    *,
    batch_size: int,
    num_cache_tokens: int,
    device: torch.device | str,
    dtype: torch.dtype,
    use_cfg: bool,
) -> IncrementalKVGuidanceCaches:
    """Allocate isolated cache lifetimes for conditional and CFG branches."""
    conditional = executor.init_kv_cache(
        batch_size=batch_size,
        num_cache_tokens=num_cache_tokens,
        device=device,
        dtype=dtype,
    )
    unconditional = (
        executor.init_kv_cache(
            batch_size=batch_size,
            num_cache_tokens=num_cache_tokens,
            device=device,
            dtype=dtype,
        )
        if use_cfg
        else None
    )
    return IncrementalKVGuidanceCaches(
        conditional=conditional,
        unconditional=unconditional,
    )


__all__ = [
    "IncrementalKVBlockForward",
    "IncrementalKVExecutor",
    "IncrementalKVGuidanceCaches",
    "allocate_incremental_kv_guidance_caches",
]
