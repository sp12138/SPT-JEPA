"""Model-neutral rectangular segmented attention sparsity.

Callers assign each physical query and key/value token a segment id, then list
the visible ``(query_segment, key_segment)`` pairs. ``INVALID_SEGMENT`` marks
tail padding. Valid queries never see invalid keys; invalid queries see one
valid key solely to keep dense and fused attention numerically safe, and their
outputs must be discarded by the caller.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch

INVALID_SEGMENT = -1


@dataclass(frozen=True)
class SegmentedSparsity:
    """Hashable logical/physical contract for rectangular sparse attention."""

    q_segment_ids: tuple[int, ...]
    kv_segment_ids: tuple[int, ...]
    allowed_pairs: tuple[tuple[int, int], ...]
    discarded_query_segments: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.q_segment_ids or not self.kv_segment_ids:
            raise ValueError("q_segment_ids and kv_segment_ids must be non-empty")
        valid_q = {value for value in self.q_segment_ids if value != INVALID_SEGMENT}
        valid_kv = {value for value in self.kv_segment_ids if value != INVALID_SEGMENT}
        if not valid_q or not valid_kv:
            raise ValueError("segmented sparsity requires valid query and key/value tokens")
        if len(set(self.allowed_pairs)) != len(self.allowed_pairs):
            raise ValueError("allowed_pairs must not contain duplicates")
        if len(set(self.discarded_query_segments)) != len(self.discarded_query_segments):
            raise ValueError("discarded_query_segments must not contain duplicates")
        for query_segment, key_segment in self.allowed_pairs:
            if query_segment not in valid_q or key_segment not in valid_kv:
                raise ValueError(
                    "allowed_pairs must reference present, non-padding query and key segments"
                )
        allowed_query_segments = {query_segment for query_segment, _ in self.allowed_pairs}
        for query_segment in self.discarded_query_segments:
            if query_segment not in valid_q:
                raise ValueError(
                    "discarded_query_segments must reference present, non-padding query segments"
                )
            if query_segment in allowed_query_segments:
                raise ValueError("discarded_query_segments must not appear in allowed_pairs")
        _validate_contiguous_segments(self.q_segment_ids, name="q_segment_ids")
        _validate_contiguous_segments(self.kv_segment_ids, name="kv_segment_ids")

    @property
    def seqlen_q(self) -> int:
        return len(self.q_segment_ids)

    @property
    def seqlen_kv(self) -> int:
        return len(self.kv_segment_ids)

    @property
    def logical_seqlen_q(self) -> int:
        return sum(value != INVALID_SEGMENT for value in self.q_segment_ids)

    @property
    def logical_seqlen_kv(self) -> int:
        return sum(value != INVALID_SEGMENT for value in self.kv_segment_ids)


def _validate_contiguous_segments(segment_ids: tuple[int, ...], *, name: str) -> None:
    seen: set[int] = set()
    previous: int | None = None
    for segment in segment_ids:
        if segment != previous:
            if segment in seen:
                raise ValueError(f"{name} segment {segment} appears in multiple runs")
            seen.add(segment)
            previous = segment
    if INVALID_SEGMENT in segment_ids:
        padding_start = segment_ids.index(INVALID_SEGMENT)
        if any(value != INVALID_SEGMENT for value in segment_ids[padding_start:]):
            raise ValueError(f"{name} padding must be one contiguous tail")


def _segment_ranges(segment_ids: tuple[int, ...]) -> dict[int, tuple[int, int]]:
    ranges: dict[int, tuple[int, int]] = {}
    start = 0
    while start < len(segment_ids):
        segment = segment_ids[start]
        end = start + 1
        while end < len(segment_ids) and segment_ids[end] == segment:
            end += 1
        ranges[segment] = (start, end)
        start = end
    return ranges


def _safe_pairs(spec: SegmentedSparsity) -> tuple[tuple[int, int], ...]:
    """Add numerically safe visibility for query rows discarded by the caller."""
    safe_key_segment = next(value for value in spec.kv_segment_ids if value != INVALID_SEGMENT)
    safe_query_segments = spec.discarded_query_segments
    if INVALID_SEGMENT in spec.q_segment_ids:
        safe_query_segments = (*safe_query_segments, INVALID_SEGMENT)
    return (*spec.allowed_pairs, *((segment, safe_key_segment) for segment in safe_query_segments))


def segmented_dense_mask(
    spec: SegmentedSparsity,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compile the segment graph to a ``[1,1,Q,K]`` boolean reference mask."""
    query_ids = torch.tensor(spec.q_segment_ids, device=device, dtype=torch.int64)
    key_ids = torch.tensor(spec.kv_segment_ids, device=device, dtype=torch.int64)
    visible = torch.zeros(
        (spec.seqlen_q, spec.seqlen_kv),
        device=device,
        dtype=torch.bool,
    )
    for query_segment, key_segment in _safe_pairs(spec):
        visible |= (query_ids[:, None] == query_segment) & (key_ids[None, :] == key_segment)
    return visible.view(1, 1, spec.seqlen_q, spec.seqlen_kv)


def segmented_tile_matrices(
    spec: SegmentedSparsity,
    *,
    tile: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return exact ``(partial, full)`` FA4 tile classification on CPU."""
    tile_m, tile_n = tile
    if tile_m < 1 or tile_n < 1:
        raise ValueError("tile dimensions must be positive")
    query_values = tuple(sorted(set(spec.q_segment_ids)))
    key_values = tuple(sorted(set(spec.kv_segment_ids)))
    query_category = {value: index for index, value in enumerate(query_values)}
    key_category = {value: index for index, value in enumerate(key_values)}
    query_ids = torch.tensor(
        [query_category[value] for value in spec.q_segment_ids], dtype=torch.int64
    )
    key_ids = torch.tensor(
        [key_category[value] for value in spec.kv_segment_ids], dtype=torch.int64
    )

    allow = torch.zeros((len(query_values), len(key_values)), dtype=torch.bool)
    for query_segment, key_segment in _safe_pairs(spec):
        allow[query_category[query_segment], key_category[key_segment]] = True

    def presence(
        ids: torch.Tensor,
        *,
        tile_size: int,
        categories: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tile_count = (ids.numel() + tile_size - 1) // tile_size
        present = torch.zeros((tile_count, categories), dtype=torch.bool)
        whole = torch.zeros(tile_count, dtype=torch.bool)
        for tile_index in range(tile_count):
            chunk = ids[tile_index * tile_size : (tile_index + 1) * tile_size]
            present[tile_index, chunk.unique()] = True
            whole[tile_index] = chunk.numel() == tile_size
        return present, whole

    present_q, whole_q = presence(
        query_ids,
        tile_size=tile_m,
        categories=len(query_values),
    )
    present_k, whole_k = presence(
        key_ids,
        tile_size=tile_n,
        categories=len(key_values),
    )
    q_int = present_q.to(torch.int32)
    k_int = present_k.to(torch.int32)
    any_allowed = (q_int @ allow.to(torch.int32) @ k_int.T) > 0
    all_allowed = (q_int @ (~allow).to(torch.int32) @ k_int.T) == 0
    full = any_allowed & all_allowed & whole_q[:, None] & whole_k[None, :]
    partial = any_allowed & ~full
    return partial, full


def _mask_mod_flex(spec: SegmentedSparsity):
    query_ranges = _segment_ranges(spec.q_segment_ids)
    key_ranges = _segment_ranges(spec.kv_segment_ids)
    pairs = _safe_pairs(spec)

    def mask_mod(batch, head, query_index, key_index):  # noqa: ARG001
        visible = query_index < 0
        for query_segment, key_segment in pairs:
            query_start, query_end = query_ranges[query_segment]
            key_start, key_end = key_ranges[key_segment]
            visible = visible | (
                (query_index >= query_start)
                & (query_index < query_end)
                & (key_index >= key_start)
                & (key_index < key_end)
            )
        return visible

    return mask_mod


def _uniform_state_causal_layout(spec: SegmentedSparsity) -> tuple[int, int] | None:
    """Recognize a complete state prefix followed by a contiguous query suffix."""
    keys, queries = spec.kv_segment_ids, spec.q_segment_ids
    if spec.discarded_query_segments or INVALID_SEGMENT in keys or INVALID_SEGMENT in queries:
        return None
    block = next((i for i, value in enumerate(keys) if value != keys[0]), len(keys))
    if block < 1 or len(keys) % block or len(queries) % block:
        return None
    past = len(keys) - len(queries)
    if past < 0 or past % block or queries != keys[past:]:
        return None
    blocks = len(keys) // block
    if any(value != index // block for index, value in enumerate(keys)):
        return None
    expected = tuple(
        (query, key)
        for query in range(past // block, blocks)
        for key in range(query + 1)
    )
    return (block, past) if spec.allowed_pairs == expected else None


@functools.cache
def segmented_mask_mod_cute(spec: SegmentedSparsity):
    """Build the FA4 in-register predicate for one contiguous segment graph."""
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils

    state_layout = _uniform_state_causal_layout(spec)
    if state_layout is not None:
        block, past = state_layout

        @cute.jit
        def state_causal_mask_mod(
            batch: cute.TensorSSA,
            head: cute.TensorSSA,
            query_index: cute.TensorSSA,
            key_index: cute.TensorSSA,
            seqlen_info,
            aux_tensors,
        ) -> cute.TensorSSA:
            zero = utils.scalar_to_ssa(0, cutlass.Int32)
            block_size = utils.scalar_to_ssa(block, cutlass.Int32)
            past_size = utils.scalar_to_ssa(past, cutlass.Int32)
            return (query_index < zero) | (
                (query_index + past_size) // block_size >= key_index // block_size
            )

        return state_causal_mask_mod

    query_ranges = _segment_ranges(spec.q_segment_ids)
    key_ranges = _segment_ranges(spec.kv_segment_ids)
    pairs = _safe_pairs(spec)
    static_ranges = tuple(
        (*query_ranges[query_segment], *key_ranges[key_segment])
        for query_segment, key_segment in pairs
    )

    @cute.jit
    def mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        query_index: cute.TensorSSA,
        key_index: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        visible = query_index < utils.scalar_to_ssa(0, cutlass.Int32)
        for query_start, query_end, key_start, key_end in static_ranges:
            q_start = utils.scalar_to_ssa(query_start, cutlass.Int32)
            q_end = utils.scalar_to_ssa(query_end, cutlass.Int32)
            k_start = utils.scalar_to_ssa(key_start, cutlass.Int32)
            k_end = utils.scalar_to_ssa(key_end, cutlass.Int32)
            visible = visible | (
                (query_index >= q_start)
                & (query_index < q_end)
                & (key_index >= k_start)
                & (key_index < k_end)
            )
        return visible

    return mask_mod


_SEGMENTED_METADATA_CACHE: dict[Any, tuple[Any, Any]] = {}


def segmented_available() -> tuple[bool, str]:
    """Whether the current process can execute segmented FA4 attention."""
    from vforge.ops._attention.teacher_forcing import teacher_forcing_available

    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9:
        try:
            from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch  # noqa: F401
            from flash_attn.cute.flash_bwd_sm90 import FlashAttentionBackwardSm90  # noqa: F401
            from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90  # noqa: F401
            from torch.nn.attention.flex_attention import BlockMask  # noqa: F401
        except (ImportError, ModuleNotFoundError) as exc:
            return False, f"Hopper FA4 segmented kernels are unavailable: {exc}"
        return True, ""
    return teacher_forcing_available()


def build_segmented_metadata(
    spec: SegmentedSparsity,
    *,
    device: Any,
    batch_size: int = 1,
    num_heads: int = 1,
    need_bwd: bool = True,
    deterministic: bool = False,
    tile: tuple[int, int] | None = None,
):
    """Build exact rectangular FA4 forward/backward tile metadata."""
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from torch.nn.attention.flex_attention import BlockMask

    from vforge.ops._attention.teacher_forcing import (
        _FA4_DETERMINISTIC_SPT,
        _dense_to_ordered,
        fa4_tile_size,
        teacher_forcing_deterministic_available,
    )

    ok, reason = segmented_available()
    if not ok:
        raise RuntimeError(f"cannot build segmented FA4 metadata: {reason}")
    if deterministic and need_bwd:
        deterministic_ok, deterministic_reason = teacher_forcing_deterministic_available()
        if not deterministic_ok:
            raise RuntimeError(
                f"cannot build deterministic segmented FA4 metadata: {deterministic_reason}"
            )
    if tile is None:
        tile = fa4_tile_size(spec.seqlen_q)
    tile_m, tile_n = tile
    key = (spec, tile, str(device), batch_size, num_heads, need_bwd, deterministic)
    cached = _SEGMENTED_METADATA_CACHE.get(key)
    if cached is not None:
        return cached

    partial_tiles, full_tiles = segmented_tile_matrices(spec, tile=tile)
    kv_mask_cnt, kv_mask_idx = _dense_to_ordered(partial_tiles, device, batch_size, num_heads)
    full_kv_cnt, full_kv_idx = _dense_to_ordered(full_tiles, device, batch_size, num_heads)
    block_mask = BlockMask.from_kv_blocks(
        kv_mask_cnt,
        kv_mask_idx,
        full_kv_cnt,
        full_kv_idx,
        BLOCK_SIZE=(tile_m, tile_n),
        mask_mod=_mask_mod_flex(spec),
        seq_lengths=(spec.seqlen_q, spec.seqlen_kv),
    )
    (
        _seq_q,
        _seq_k,
        _kv_cnt,
        _kv_idx,
        _full_kv_cnt,
        _full_kv_idx,
        q_mask_cnt,
        q_mask_idx,
        full_q_cnt,
        full_q_idx,
        *_rest,
    ) = block_mask.as_tuple()
    forward = BlockSparseTensorsTorch(
        mask_block_cnt=kv_mask_cnt,
        mask_block_idx=kv_mask_idx,
        full_block_cnt=full_kv_cnt,
        full_block_idx=full_kv_idx,
        block_size=(tile_m, tile_n),
    )
    # SM100 backward uses two 128-token Q subtiles even when a short
    # forward uses just one. Reclassify the padded Q tail for backward;
    # relabelling block_size alone would incorrectly retain full tiles.
    backward_tile = (max(tile_m, 256), tile_n)
    if need_bwd and backward_tile != tile:
        _, backward = build_segmented_metadata(
            spec,
            device=device,
            batch_size=batch_size,
            num_heads=num_heads,
            need_bwd=True,
            deterministic=deterministic,
            tile=backward_tile,
        )
        _SEGMENTED_METADATA_CACHE[key] = (forward, backward)
        return forward, backward
    dq_write_order = None
    dq_write_order_full = None
    if need_bwd and deterministic:
        from flash_attn.cute.block_sparsity import compute_dq_write_order_from_block_mask

        dq_write_order, dq_write_order_full = compute_dq_write_order_from_block_mask(
            block_mask,
            spt=_FA4_DETERMINISTIC_SPT,
        )
    backward = (
        BlockSparseTensorsTorch(
            mask_block_cnt=q_mask_cnt,
            mask_block_idx=q_mask_idx,
            full_block_cnt=full_q_cnt,
            full_block_idx=full_q_idx,
            block_size=(tile_m, tile_n),
            dq_write_order=dq_write_order,
            dq_write_order_full=dq_write_order_full,
            spt=_FA4_DETERMINISTIC_SPT if deterministic else None,
        )
        if need_bwd
        else None
    )
    _SEGMENTED_METADATA_CACHE[key] = (forward, backward)
    return forward, backward


__all__ = [
    "INVALID_SEGMENT",
    "SegmentedSparsity",
    "build_segmented_metadata",
    "segmented_dense_mask",
    "segmented_available",
    "segmented_mask_mod_cute",
    "segmented_tile_matrices",
]
