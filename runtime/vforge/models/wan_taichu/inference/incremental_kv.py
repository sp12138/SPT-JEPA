"""Incremental KV execution for Wan Taichu causal teacher models."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from diffusers.utils import logging

from vforge.distributed.parallel_state import get_parallel_state
from vforge.distributed.ulysses import gather_sequence
from vforge.models.causal import (
    CausalKVCache,
    IncrementalKVAttentionContext,
    IncrementalKVExecutor,
    IncrementalKVTransformer,
    use_incremental_kv_cache,
)
from vforge.models.wan_taichu.transformer import (
    apply_lora_scale,
    project_output,
    run_transformer_blocks,
    slice_ulysses_sequence_locally,
    unpatchify,
)

logger = logging.get_logger(__name__)


def forward_incremental_kv_block(
    transformer: nn.Module,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    *,
    kv_cache: CausalKVCache,
    full_rotary_emb: tuple[torch.Tensor, torch.Tensor],
    kv_write_start: int,
    commit_kv: bool,
    local_attn_size: int = -1,
    sink_size: int = 0,
    usp_size: int = 1,
    attention_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Run one block against explicitly supplied committed KV history."""
    apply_lora_scale(transformer, attention_kwargs, logger=logger)
    if len(kv_cache) != len(transformer.blocks):
        raise ValueError(
            f"KV cache has {len(kv_cache)} layers, transformer has {len(transformer.blocks)}"
        )

    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = (int(value) for value in transformer.config.patch_size)
    if num_frames % p_t or height % p_h or width % p_w:
        raise ValueError(
            f"block shape {tuple(hidden_states.shape[-3:])} must divide by "
            f"patch_size={tuple(transformer.config.patch_size)}"
        )
    patch_grid = (num_frames // p_t, height // p_h, width // p_w)
    block_tokens = patch_grid[0] * patch_grid[1] * patch_grid[2]
    if timestep.ndim != 2 or tuple(timestep.shape) != (batch_size, block_tokens):
        raise ValueError(
            "incremental_kv needs a per-token timestep matching the current block; "
            f"expected {(batch_size, block_tokens)}, got {tuple(timestep.shape)}"
        )

    recompute_blocks = bool(getattr(transformer, "gradient_checkpointing", False))
    if recompute_blocks and torch.is_grad_enabled() and int(local_attn_size) != -1:
        raise NotImplementedError(
            "incremental_kv cannot combine activation recompute with a sliding KV "
            f"window (local_attn_size={local_attn_size}); the window compacts cache "
            "slots in place, so recomputed blocks would attend different history"
        )

    rotary_start = int(kv_write_start)
    rotary_end = rotary_start + block_tokens
    full_cos, full_sin = full_rotary_emb
    if full_cos.shape[1] < rotary_end or full_sin.shape[1] < rotary_end:
        raise ValueError(
            f"full clip RoPE has {full_cos.shape[1]} tokens but block needs "
            f"[{rotary_start}, {rotary_end})"
        )
    rotary_emb = (
        full_cos[:, rotary_start:rotary_end],
        full_sin[:, rotary_start:rotary_end],
    )

    tokens = transformer.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
    condition = transformer.condition_embedder(
        timestep.flatten(),
        encoder_hidden_states,
        timestep_seq_len=block_tokens,
    )
    temb, timestep_proj, encoder_hidden_states, *image_conditions = condition
    if any(value is not None for value in image_conditions):
        raise ValueError("incremental_kv does not accept CLIP or DINO image conditions")
    timestep_proj = timestep_proj.unflatten(2, (6, -1))
    parallel_state = get_parallel_state()
    usp_size = int(usp_size)
    usp_group = parallel_state.usp_group if usp_size > 1 else None
    if usp_size > 1:
        if block_tokens % usp_size:
            raise ValueError(
                f"incremental KV block has {block_tokens} tokens, which must divide "
                f"parallel.usp_size={usp_size}"
            )
        # Match the full-sequence path: backpropagate each rank's local token
        # contribution and let FSDP merge parameter gradients. Gathering these
        # activation gradients first changes BF16 timestep-embedding accumulation.
        tokens = slice_ulysses_sequence_locally(tokens, dim=1, group=usp_group)
        timestep_proj = slice_ulysses_sequence_locally(timestep_proj, dim=1, group=usp_group)
        temb = slice_ulysses_sequence_locally(temb, dim=1, group=usp_group)
        rotary_emb = (
            slice_ulysses_sequence_locally(rotary_emb[0], dim=1, group=usp_group),
            slice_ulysses_sequence_locally(rotary_emb[1], dim=1, group=usp_group),
        )

    # Pin the history extent now: later blocks commit more frames before
    # backward runs, and a recomputed forward must not attend them.
    history_tokens = kv_cache.committed_tokens

    def attention_context(layer_index: int) -> dict[str, Any]:
        return {
            "self_attention_mask": IncrementalKVAttentionContext(
                layer_index=layer_index,
                write_start=rotary_start,
                history_tokens=history_tokens,
                commit=bool(commit_kv),
                frame_tokens=patch_grid[1] * patch_grid[2],
                local_window_frames=int(local_attn_size),
                sink_frames=int(sink_size),
                usp_size=usp_size,
                usp_group=usp_group,
                original_heads=int(transformer.blocks[layer_index].attn1.heads),
            )
        }

    tokens = run_transformer_blocks(
        transformer,
        tokens,
        encoder_hidden_states,
        timestep_proj,
        rotary_emb,
        block_kwargs=attention_context,
        # Recompute happens inside backward, where the caller's cache context
        # has already exited, so each checkpointed block has to rebind it.
        block_context=lambda: use_incremental_kv_cache(kv_cache),
    )
    tokens = project_output(
        tokens,
        temb,
        scale_shift_table=transformer.scale_shift_table,
        norm_out=transformer.norm_out,
        proj_out=transformer.proj_out,
    )
    if usp_size > 1:
        tokens = gather_sequence(
            tokens,
            dim=1,
            group=usp_group,
            backward_mode=getattr(transformer, "ulysses_gather_backward", None),
        )
    return unpatchify(
        tokens,
        batch_size=batch_size,
        patch_grid=patch_grid,
        patch_size=(p_t, p_h, p_w),
    )


def _forward_incremental_kv_block_through_module(
    transformer: nn.Module,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """Enter through ``Module.__call__`` so FSDP root hooks execute."""
    kv_cache = kwargs.pop("kv_cache")
    with use_incremental_kv_cache(kv_cache):
        return transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            incremental_kv=True,
            **kwargs,
        )


def build_incremental_kv_executor(
    transformer: nn.Module,
    *,
    usp_size: int | None = None,
) -> IncrementalKVTransformer:
    """Build the protocol implementation without changing the model object."""
    return IncrementalKVExecutor(
        transformer,
        block_forward=_forward_incremental_kv_block_through_module,
        usp_size=usp_size,
    )


__all__ = [
    "build_incremental_kv_executor",
    "forward_incremental_kv_block",
]
