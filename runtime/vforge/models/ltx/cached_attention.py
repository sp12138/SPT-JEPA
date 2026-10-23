"""LTX cached self-attention using the existing framework attention backend."""

from __future__ import annotations

import torch
import torch.distributed as dist

from vforge.distributed.ulysses import get_usp_group, pad_to_multiple
from vforge.models.ltx.kv_cache import CacheMode, KVCache, Pair
from vforge.models.ltx.rope import LTXRopeType, apply_rotary_emb


def cached_attention(
    attn,
    x,
    context,
    pe,
    k_pe,
    cache: KVCache,
    mode: CacheMode,
    cond_pe: Pair | None = None,
    perturbation_mask=None,
    all_perturbed=False,
    sequence_length: int | None = None,
):
    context = x if context is None else context
    if (all_perturbed or perturbation_mask is not None) and mode != CacheMode.DENOISE:
        raise ValueError("perturbations are only supported during denoising")
    value = attn.to_v(context)
    if all_perturbed:
        output = value
    else:
        query = attn.q_norm(attn.to_q(x))
        raw_key = attn.k_norm(attn.to_k(context))
        key_pe = pe if k_pe is None else k_pe
        query = apply_rotary_emb(query, pe, attn.rope_type) if pe is not None else query
        key = apply_rotary_emb(raw_key, key_pe, attn.rope_type) if key_pe is not None else raw_key
        sharded = sequence_length is not None
        local_value = value
        if sharded:
            query = attn._sequence_to_head(query, sequence_length).flatten(2)
            key = attn._sequence_to_head(key, sequence_length).flatten(2)
            value = attn._sequence_to_head(value, sequence_length).flatten(2)
            if mode == CacheMode.PREFILL_REF:
                raw_key = attn._sequence_to_head(raw_key, sequence_length).flatten(2)
            if cond_pe is not None:
                cond_pe = _reference_head_pe(cond_pe, attn)
        pairs = []
        if mode == CacheMode.PREFILL_REF:
            if cond_pe is None:
                raise ValueError("reference prefill requires dynamic reference RoPE")
            cache.write(raw_key, value, mode)
        elif mode == CacheMode.PREFILL_CLEAN:
            cache.write(key, value, mode)
            pairs.extend(cache.clean)
        elif mode == CacheMode.DENOISE:
            if cache.reference is None or cond_pe is None:
                raise ValueError("denoising requires reference KV and RoPE")
            ref_key, ref_value = cache.reference
            ref_pe = tuple(part.to(ref_key.dtype) for part in cond_pe)
            pairs.append((apply_rotary_emb(ref_key, ref_pe, attn.rope_type), ref_value))
            pairs.extend(cache.clean)
        else:
            raise ValueError(f"unsupported cache mode: {mode}")
        if mode != CacheMode.PREFILL_CLEAN:
            pairs.append((key, value))
        keys = torch.cat([pair[0] for pair in pairs], dim=1)
        values = torch.cat([pair[1] for pair in pairs], dim=1)
        if sharded:
            # History already has full valid sequence / local heads. Exchange
            # only the current block; do not all-to-all the cached prefix again.
            output = attn.attention_function(
                query.unflatten(-1, (-1, attn.dim_head)),
                keys.unflatten(-1, (-1, attn.dim_head)),
                values.unflatten(-1, (-1, attn.dim_head)),
                attn_mask=None,
                is_causal=False,
                sparsity=None,
            )
            output = attn._head_to_sequence(output)
        else:
            output = attn._run_attention(
                attn.attention_function,
                query,
                keys,
                values,
                mask=None,
                sparsity=None,
            )
        if perturbation_mask is not None:
            output = output * perturbation_mask + local_value * (1 - perturbation_mask)
    if attn.to_gate_logits is not None:
        output = attn.gated_attention_function(x, output, attn)
    return attn.to_out(output)


def _reference_head_pe(pe, attn):
    """Select the same head shard as cached raw reference keys (no communication)."""
    group = get_usp_group()
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    result = []
    for part in pe:
        if attn.rope_type == LTXRopeType.SPLIT:
            part, _ = pad_to_multiple(part, size, dim=1)
            result.append(part.chunk(size, dim=1)[rank])
        else:
            part = part.unflatten(-1, (attn.heads, -1))
            part, _ = pad_to_multiple(part, size, dim=2)
            result.append(part.chunk(size, dim=2)[rank].flatten(2))
    return tuple(result)


def compute_pe_for_positions(model, positions, modality="video"):
    prep = model.video_args_preprocessor if modality == "video" else model.audio_args_preprocessor
    simple = getattr(prep, "simple_preprocessor", prep)
    return simple._prepare_positional_embeddings(
        positions=positions,
        inner_dim=simple.inner_dim,
        max_pos=simple.max_pos,
        use_middle_indices_grid=simple.use_middle_indices_grid,
        num_attention_heads=simple.num_attention_heads,
        x_dtype=positions.dtype,
    )
