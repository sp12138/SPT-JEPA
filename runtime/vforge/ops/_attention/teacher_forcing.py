"""Block-causal teacher-forcing attention.

Teacher-forcing models feed the transformer a doubled ``[clean | noisy]``
sequence and forbid most of the resulting ``2S x 2S`` attention. Writing that
predicate as a dense boolean mask costs 2.77 GiB at the production Wan2.2
shape (31 latent frames x 880 tokens/frame -> S=27280), and flash-attn cannot
consume an arbitrary mask anyway, so the dense form also forces the whole
attention onto SDPA at full ``O((2S)^2)`` cost.

Everything here serves that one predicate. The vehicle is FlashAttention-4's
block-sparse machinery -- ``mask_mod`` plus ``BlockSparseTensorsTorch`` -- but
this is not a general block-sparse attention facility, and nothing outside
teacher forcing should reach for it. The FA4 spellings that survive below are
upstream API names, not ours.

The predicate reaches the kernel as two FA4 inputs:

* ``mask_mod`` -- a ``@cute.jit`` predicate the kernel evaluates in-register,
  for tiles that are only partially masked.
* tile metadata -- per q-tile, which kv-tiles are fully unmasked and which are
  partially masked. Tiles in neither list are never loaded, which is where the
  speedup comes from.

Predicate
---------
Token layout is ``[clean (S tokens) | noisy (S tokens)]``. For token ``i``,
``is_clean(i) = i < S``, and its causal block comes from its latent frame
``f = (i % S) // tokens_per_frame``:

* ``first_frame_own_block=True``  -> ``blk(f) = ceildiv(f, frame_per_block)``
  (frame 0 alone in block 0, then ``frame_per_block`` frames per block);
* ``first_frame_own_block=False`` -> ``blk(f) = f // frame_per_block``.

``ceildiv(f, fpb) == 1 + (f - 1) // fpb`` for ``f >= 1`` and ``0`` for
``f == 0``, so the two-branch layout collapses to one branch-free expression.
That matters: the kernel body cannot branch on runtime indices.

Attend iff any of:

* ``clean_q & clean_k & blk_k <= blk_q`` -- clean history is block-causal;
* ``noisy_q & noisy_k & blk_k == blk_q`` -- a noisy token sees its own block;
* ``noisy_q & clean_k & blk_k <  blk_q`` -- and strictly-earlier clean context;
* ``i == j`` -- the diagonal, redundant given the three rules but kept so this
  is literally the same boolean function as the dense reference.

A noisy block therefore never sees its own clean answer, nor any future block.
When ``local_attn_size`` is finite, the same predicate is additionally limited
to whole causal blocks: the first ``sink_size`` frames remain globally visible,
and the remaining window follows the query block. Both sizes are expressed in
latent frames and must align to ``frame_per_block``. This mirrors Taichu's
teacher-forcing SWA and its inference-time KV eviction boundary.

Requirements
------------
Needs FA4 (``flash_attn.cute``) and an SM100-class Blackwell device.
:func:`teacher_forcing_available` reports whether this process qualifies;
callers should fall back to the dense mask on SDPA when it does not.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

# Largest dense mask we will materialise for the SDPA fallback. 8192 tokens is
# ~64 MiB of bool, which keeps tests and CPU debugging comfortable while
# refusing anything near the production shape, where the dense form is the
# whole problem this module exists to avoid.
_MAX_DENSE_MASK_SEQLEN = 8192

# FA4 on SM100 runs a 128x128 tile and pairs q tiles when the query is long
# enough (``flash_fwd_sm100``: ``q_stage = 2 if seqlen_q > 128 else 1``), so the
# effective q tile is ``q_stage * 128``. Metadata granularity has to match, or
# ``normalize_block_sparse_config`` either rejects it or mis-indexes tiles.
_FA4_TILE_M = 128
_FA4_TILE_N = 128

# FA4 derives each dQ semaphore lock value from a tile's position in the
# backward scheduler's traversal, and reads the direction back off
# ``BlockSparseTensorsTorch.spt``. The write order we precompute must use the
# same flag the kernel schedules with or the semaphores deadlock. ``True``
# mirrors what FA4 picks for a dense causal mask (``spt = (causal or local) and
# deterministic``): the block-causal mask has the same lopsided per-tile
# workload, so longest-processing-time-first is the right traversal.
_FA4_DETERMINISTIC_SPT = True


def _ceildiv(a: int, b: int) -> int:
    return -(-a // b)


def _ceildiv_tensor(x: torch.Tensor, d: int) -> torch.Tensor:
    return (x + (d - 1)) // d


@dataclass(frozen=True)
class TeacherForcingSparsity:
    """Everything the block-causal teacher-forcing mask is a function of.

    Frozen and hashable so it can key the compiled predicate and the metadata
    caches. Holds plain integers only, so model code can build one without
    reaching for a kernel package.
    """

    num_latent_frames: int
    tokens_per_frame: int
    frame_per_block: int
    first_frame_own_block: bool = True
    local_attn_size: int = -1
    sink_size: int = 0

    def __post_init__(self) -> None:
        if self.num_latent_frames < 1:
            raise ValueError(f"num_latent_frames must be >= 1, got {self.num_latent_frames}")
        if self.tokens_per_frame < 1:
            raise ValueError(f"tokens_per_frame must be >= 1, got {self.tokens_per_frame}")
        if self.frame_per_block < 1:
            raise ValueError(f"frame_per_block must be >= 1, got {self.frame_per_block}")
        if self.local_attn_size == 0 or self.local_attn_size < -1:
            raise ValueError(
                "local_attn_size must be -1 (global) or a positive frame count, "
                f"got {self.local_attn_size}"
            )
        if self.sink_size < 0:
            raise ValueError(f"sink_size must be >= 0, got {self.sink_size}")
        if self.local_attn_size == -1:
            if self.sink_size:
                raise ValueError("sink_size requires a finite local_attn_size")
            return
        recent_size = self.local_attn_size - self.sink_size
        if recent_size < self.frame_per_block or recent_size % self.frame_per_block:
            raise ValueError(
                "local_attn_size - sink_size must be a positive multiple of "
                f"frame_per_block; got local={self.local_attn_size}, "
                f"sink={self.sink_size}, frame_per_block={self.frame_per_block}"
            )
        sink_offset = 1 if self.first_frame_own_block and self.sink_size else 0
        if (self.sink_size - sink_offset) % self.frame_per_block:
            raise ValueError(
                "sink_size must end on a causal block boundary; got "
                f"sink_size={self.sink_size}, frame_per_block={self.frame_per_block}, "
                f"first_frame_own_block={self.first_frame_own_block}"
            )
        if self.sink_size >= self.local_attn_size:
            raise ValueError(
                f"sink_size ({self.sink_size}) must be < local_attn_size ({self.local_attn_size})"
            )

    @property
    def sink_blocks(self) -> int:
        if not self.sink_size:
            return 0
        if self.first_frame_own_block:
            return 1 + (self.sink_size - 1) // self.frame_per_block
        return self.sink_size // self.frame_per_block

    @property
    def recent_blocks(self) -> int:
        if self.local_attn_size <= 0:
            return 0
        return (self.local_attn_size - self.sink_size) // self.frame_per_block

    @property
    def tokens_per_half(self) -> int:
        """``S`` -- tokens in the clean half, and equally in the noisy half."""
        return self.num_latent_frames * self.tokens_per_frame

    @property
    def seqlen(self) -> int:
        """``2S`` -- the doubled sequence length the transformer actually sees."""
        return 2 * self.tokens_per_half

    @property
    def num_blocks(self) -> int:
        if self.first_frame_own_block:
            return 1 + _ceildiv(self.num_latent_frames - 1, self.frame_per_block)
        return _ceildiv(self.num_latent_frames, self.frame_per_block)


def frame_block_ids(spec: TeacherForcingSparsity, device: Any = None) -> torch.Tensor:
    """``[T]`` causal block id per latent frame."""
    frames = torch.arange(spec.num_latent_frames, device=device)
    if spec.first_frame_own_block:
        return _ceildiv_tensor(frames, spec.frame_per_block)
    return frames // spec.frame_per_block


def _allowed(
    spec: TeacherForcingSparsity,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    """The predicate on broadcastable integer index tensors."""
    s = spec.tokens_per_half
    q_frame = (q_idx % s) // spec.tokens_per_frame
    k_frame = (kv_idx % s) // spec.tokens_per_frame
    if spec.first_frame_own_block:
        q_blk = _ceildiv_tensor(q_frame, spec.frame_per_block)
        k_blk = _ceildiv_tensor(k_frame, spec.frame_per_block)
    else:
        q_blk = q_frame // spec.frame_per_block
        k_blk = k_frame // spec.frame_per_block

    q_clean = q_idx < s
    k_clean = kv_idx < s

    allowed = (
        (q_clean & k_clean & (k_blk <= q_blk))
        | ((~q_clean) & (~k_clean) & (k_blk == q_blk))
        | ((~q_clean) & k_clean & (k_blk < q_blk))
    )
    if spec.local_attn_size > 0:
        allowed = allowed & ((k_blk < spec.sink_blocks) | (k_blk >= q_blk - spec.recent_blocks + 1))
    return (q_idx == kv_idx) | allowed


def teacher_forcing_dense_mask(
    spec: TeacherForcingSparsity,
    device: Any = None,
) -> torch.Tensor:
    """``[1, 1, 2S, 2S]`` bool mask, for the SDPA fallback and for tests."""
    n = spec.seqlen
    if n > _MAX_DENSE_MASK_SEQLEN:
        raise ValueError(
            f"refusing to materialise a {n}x{n} dense teacher-forcing mask "
            f"({n * n / 2**30:.2f} GiB of bool); the tile-sparse FA4 path "
            "exists precisely to avoid this. Run on an SM100 device, or shrink "
            "the clip for debugging."
        )
    idx = torch.arange(n, device=device)
    return _allowed(spec, idx.unsqueeze(1), idx.unsqueeze(0)).view(1, 1, n, n)


def _mask_mod_flex(spec: TeacherForcingSparsity) -> Callable:
    """FlexAttention-style ``mask_mod(b, h, q_idx, kv_idx)``.

    Only used to label the ``BlockMask`` we borrow for transposing metadata;
    the tile classification never calls it.
    """

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001 - signature fixed by flex
        return _allowed(spec, q_idx, kv_idx)

    return mask_mod


@functools.cache
def mask_mod_cute(spec: TeacherForcingSparsity) -> Callable:
    """The FA4 ``mask_mod`` for this predicate.

    Cached on the spec: this is read once per transformer block per forward,
    and a fresh ``@cute.jit`` object each time would both redo the wrapper
    construction and defeat FA4's own kernel cache, which keys on
    ``utils.hash_callable(mask_mod)``.

    Deliberately not decorated with ``block_sparsity.fast_sampling``. That opts
    into 5-point (four corners plus centre) tile classification, which is only
    sound while both tile dims stay below the smallest causal block --
    ``tokens_per_frame`` tokens, for block 0 -- and fails silently below that.
    :func:`_tile_matrices` classifies exactly instead, so sampling never enters.
    """
    import cutlass
    import cutlass.cute as cute
    from flash_attn.cute import utils

    s = int(spec.tokens_per_half)
    tpf = int(spec.tokens_per_frame)
    fpb = int(spec.frame_per_block)
    fpb_bias = fpb - 1 if spec.first_frame_own_block else 0
    sink_blocks = int(spec.sink_blocks)
    recent_blocks = int(spec.recent_blocks)
    has_window = spec.local_attn_size > 0

    @cute.jit
    def _mask_mod(
        batch: cute.TensorSSA,
        head: cute.TensorSSA,
        m_idx: cute.TensorSSA,
        n_idx: cute.TensorSSA,
        seqlen_info,
        aux_tensors,
    ) -> cute.TensorSSA:
        s_ssa = utils.scalar_to_ssa(s, cutlass.Int32)
        tpf_ssa = utils.scalar_to_ssa(tpf, cutlass.Int32)
        fpb_ssa = utils.scalar_to_ssa(fpb, cutlass.Int32)
        bias_ssa = utils.scalar_to_ssa(fpb_bias, cutlass.Int32)

        # blk(i) = ceildiv(frame(i), fpb) with frame(i) = (i % S) // tpf.
        # bias == fpb - 1 gives the ceil (first_frame_own_block=True);
        # bias == 0 gives the plain floor division (uniform blocks).
        q_blk = (((m_idx % s_ssa) // tpf_ssa) + bias_ssa) // fpb_ssa
        k_blk = (((n_idx % s_ssa) // tpf_ssa) + bias_ssa) // fpb_ssa

        q_clean = m_idx < s_ssa
        k_clean = n_idx < s_ssa
        q_noisy = m_idx >= s_ssa
        k_noisy = n_idx >= s_ssa

        allowed = (
            (q_clean & k_clean & (k_blk <= q_blk))
            | (q_noisy & k_noisy & (k_blk == q_blk))
            | (q_noisy & k_clean & (k_blk < q_blk))
        )
        if has_window:
            sink_ssa = utils.scalar_to_ssa(sink_blocks, cutlass.Int32)
            recent_ssa = utils.scalar_to_ssa(recent_blocks, cutlass.Int32)
            allowed = allowed & ((k_blk < sink_ssa) | (k_blk >= q_blk - recent_ssa + 1))
        return (m_idx == n_idx) | allowed

    return _mask_mod


def fa4_tile_size(seqlen_q: int) -> tuple[int, int]:
    """``(tile_m, tile_n)`` that FA4 will use on SM100 for this query length."""
    q_stage = 2 if seqlen_q > _FA4_TILE_M else 1
    return q_stage * _FA4_TILE_M, _FA4_TILE_N


def _tile_matrices(
    spec: TeacherForcingSparsity,
    tile: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(partial, full)`` ``[n_m, n_n]`` bool tile matrices, on CPU.

    Classifies every ``(q-tile, kv-tile)`` pair as skipped (in neither),
    fully unmasked, or partially masked, without materialising the predicate.

    The reduction is exact, not sampled. The predicate depends only on the
    category pair ``(is_clean, block_id)``, and every category present in a
    tile is realised by at least one token there, so "some position in this
    tile is allowed" and "every position is allowed" both fall out of two
    matmuls over the ``2 * num_blocks`` categories. At the production shape
    that is a ~214 x 427 tile grid rather than ~3e9 token pairs.

    The ``q_idx == kv_idx`` term is omitted here: a disallowed category pair
    has ``cat_q != cat_k`` and so contributes only off-diagonal positions,
    meaning the diagonal can neither make a skipped tile non-empty nor complete
    a partial one.

    A tile running past ``seqlen`` is never marked full, matching how the flex
    builder zero-pads the trailing tile before classifying it.
    """
    seqlen = spec.seqlen
    tile_m, tile_n = tile
    n_m = _ceildiv(seqlen, tile_m)
    n_n = _ceildiv(seqlen, tile_n)

    # Category of a token = (is_clean, block id), in [0, 2 * num_blocks).
    idx = torch.arange(seqlen)
    s = spec.tokens_per_half
    frame = (idx % s) // spec.tokens_per_frame
    blk = (
        _ceildiv_tensor(frame, spec.frame_per_block)
        if spec.first_frame_own_block
        else frame // spec.frame_per_block
    )
    n_blocks = spec.num_blocks
    cat = torch.where(idx < s, blk, blk + n_blocks)
    n_cat = 2 * n_blocks

    def present_in_tiles(n_tiles: int, tile_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(present[t, c], whole[t])`` -- category c occurs in tile t; t is full-width."""
        present = torch.zeros(n_tiles, n_cat, dtype=torch.bool)
        whole = torch.zeros(n_tiles, dtype=torch.bool)
        for t in range(n_tiles):
            chunk = cat[t * tile_size : (t + 1) * tile_size]
            present[t, chunk.unique()] = True
            whole[t] = chunk.numel() == tile_size
        return present, whole

    present_m, whole_m = present_in_tiles(n_m, tile_m)
    present_n, whole_n = present_in_tiles(n_n, tile_n)

    # allow[cq, ck] at category granularity.
    cq = torch.arange(n_cat).unsqueeze(1)
    ck = torch.arange(n_cat).unsqueeze(0)
    q_clean_c, k_clean_c = cq < n_blocks, ck < n_blocks
    bq, bk = cq % n_blocks, ck % n_blocks
    allow = (
        (q_clean_c & k_clean_c & (bk <= bq))
        | ((~q_clean_c) & (~k_clean_c) & (bk == bq))
        | ((~q_clean_c) & k_clean_c & (bk < bq))
    )
    if spec.local_attn_size > 0:
        allow = allow & ((bk < spec.sink_blocks) | (bk >= bq - spec.recent_blocks + 1))

    pm = present_m.to(torch.int32)
    pn = present_n.to(torch.int32)
    any_allowed = (pm @ allow.to(torch.int32) @ pn.T) > 0
    all_allowed = (pm @ (~allow).to(torch.int32) @ pn.T) == 0

    full = any_allowed & all_allowed & whole_m.unsqueeze(1) & whole_n.unsqueeze(0)
    partial = any_allowed & ~full
    return partial, full


def _dense_to_ordered(
    dense: torch.Tensor,
    device: Any,
    batch_size: int,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[n_rows, n_cols]`` bool tile matrix -> ``(cnt, idx)`` in FA4 layout.

    Returns ``([B, H, n_rows] int32, [B, H, n_rows, n_cols] int32)``. Per row
    the set-column indices come first in increasing order and the rest follow;
    only the first ``cnt`` entries are read. This is the same packing
    ``flex_attention`` performs when converting a dense block mask, so the
    metadata stays interchangeable with what its builder emits.
    """
    d = dense.to(torch.int32)
    cnt = d.sum(dim=-1).to(torch.int32)
    idx = torch.sort(d, dim=-1, stable=True, descending=True).indices.to(torch.int32)
    cnt = cnt.expand(batch_size, num_heads, *cnt.shape).contiguous()
    idx = idx.expand(batch_size, num_heads, *idx.shape).contiguous()
    return cnt.to(device), idx.to(device)


def teacher_forcing_available() -> tuple[bool, str]:
    """Whether this process can run the teacher-forcing attention on FA4.

    Returns ``(ok, reason)``; ``reason`` is empty when ``ok``.
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    major = torch.cuda.get_device_capability()[0]
    if major != 10:
        return False, (
            "FA4 mask_mod/block-sparsity needs an SM100-class (Blackwell) GPU; "
            f"got compute capability major={major}"
        )
    try:
        import flash_attn.cute.interface  # noqa: F401
        from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        return False, f"flash_attn.cute (FA4) is not importable: {exc}"
    try:
        from torch.nn.attention.flex_attention import BlockMask  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        return False, f"torch flex_attention is not importable: {exc}"
    return True, ""


def teacher_forcing_deterministic_available() -> tuple[bool, str]:
    """Whether the teacher-forcing backward can run deterministically.

    FA4's only architecture exclusion for the ordered backward is SM 12.0,
    which :func:`teacher_forcing_available` already rules out by requiring
    SM100, so what remains is whether this build carries the write-order
    helpers.
    """
    ok, reason = teacher_forcing_available()
    if not ok:
        return False, reason
    try:
        from flash_attn.cute.block_sparsity import (  # noqa: F401
            compute_dq_write_order_from_block_mask,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        return False, f"this flash_attn build has no deterministic sparse backward: {exc}"
    return True, ""


# (spec, tile, device, batch, heads, need_bwd, deterministic) -> (fwd, bwd)
_METADATA_CACHE: dict[Any, tuple[Any, Any]] = {}


def build_metadata(
    spec: TeacherForcingSparsity,
    *,
    device: Any,
    batch_size: int = 1,
    num_heads: int = 1,
    need_bwd: bool = True,
    deterministic: bool = False,
    tile: tuple[int, int] | None = None,
):
    """Compute the FA4 tile metadata for this predicate.

    Returns ``(fwd, bwd)`` as ``BlockSparseTensorsTorch``, with ``bwd`` set to
    ``None`` when ``need_bwd=False``. ``fwd`` is the KV-direction metadata (for
    each q-tile, which kv-tiles to visit); ``bwd`` is the transposed
    Q-direction metadata the backward kernel needs.

    Classification is exact, via :func:`_tile_matrices`, and the result is
    cached. Deliberately not routed through ``flex_attention.create_block_mask``
    -- that helper materialises the dense predicate plus several same-sized
    temporaries just to reach the tile grid, a multi-GiB spike that OOMs on top
    of live activations at the production shape.

    ``batch_size=1`` / ``num_heads=1`` are the useful defaults: the mask
    depends on neither and FA4 broadcasts singleton dims, so metadata stays
    tiny.
    """
    ok, reason = teacher_forcing_available()
    if not ok:
        raise RuntimeError(f"cannot build teacher-forcing FA4 metadata: {reason}")
    if deterministic and need_bwd:
        det_ok, det_reason = teacher_forcing_deterministic_available()
        if not det_ok:
            raise RuntimeError(f"cannot build deterministic FA4 metadata: {det_reason}")

    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    from torch.nn.attention.flex_attention import BlockMask

    seqlen = spec.seqlen
    if tile is None:
        tile = fa4_tile_size(seqlen)
    tile_m, tile_n = tile

    key = (spec, tile, str(device), batch_size, num_heads, need_bwd, deterministic)
    cached = _METADATA_CACHE.get(key)
    if cached is not None:
        return cached

    partial_tiles, full_tiles = _tile_matrices(spec, tile)
    kv_mask_cnt, kv_mask_idx = _dense_to_ordered(partial_tiles, device, batch_size, num_heads)
    full_kv_cnt, full_kv_idx = _dense_to_ordered(full_tiles, device, batch_size, num_heads)

    # The q-direction metadata is the transpose of the kv-direction one, which
    # ``from_kv_blocks`` derives at tile granularity. Going through a real
    # ``BlockMask`` rather than packing the transpose by hand also produces the
    # object ``compute_dq_write_order_from_block_mask`` expects below.
    block_mask = BlockMask.from_kv_blocks(
        kv_mask_cnt,
        kv_mask_idx,
        full_kv_cnt,
        full_kv_idx,
        BLOCK_SIZE=(tile_m, tile_n),
        mask_mod=_mask_mod_flex(spec),
        seq_lengths=(seqlen, seqlen),
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

    fwd = BlockSparseTensorsTorch(
        mask_block_cnt=kv_mask_cnt,
        mask_block_idx=kv_mask_idx,
        full_block_cnt=full_kv_cnt,
        full_block_idx=full_kv_idx,
        block_size=(tile_m, tile_n),
    )

    # Rank of each contributing kv-tile in the ordered list of tiles writing the
    # same dQ tile. The backward turns those ranks into semaphore lock values,
    # which is what replaces the atomicAdd accumulation.
    dq_write_order = None
    dq_write_order_full = None
    if need_bwd and deterministic:
        from flash_attn.cute.block_sparsity import compute_dq_write_order_from_block_mask

        dq_write_order, dq_write_order_full = compute_dq_write_order_from_block_mask(
            block_mask,
            spt=_FA4_DETERMINISTIC_SPT,
        )

    bwd = (
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

    _METADATA_CACHE[key] = (fwd, bwd)
    return fwd, bwd


def teacher_forcing_sparsity_stats(
    spec: TeacherForcingSparsity,
    tile: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Tile-level accounting: how many tile pairs the kernel gets to skip.

    Pure CPU work at tile granularity. Reads the same classification
    :func:`build_metadata` hands to FA4, so it doubles as the cheap way to
    sanity-check that metadata.
    """
    seqlen = spec.seqlen
    if tile is None:
        tile = fa4_tile_size(seqlen)
    tile_m, tile_n = tile
    n_m = _ceildiv(seqlen, tile_m)
    n_n = _ceildiv(seqlen, tile_n)

    partial_tiles, full_tiles = _tile_matrices(spec, tile)

    total = n_m * n_n
    full = int(full_tiles.sum())
    partial = int(partial_tiles.sum())
    visited = full + partial
    return {
        "tile": (tile_m, tile_n),
        "num_q_tiles": n_m,
        "num_kv_tiles": n_n,
        "total_tile_pairs": total,
        "visited_tile_pairs": visited,
        "full_tile_pairs": full,
        "partial_tile_pairs": partial,
        "skipped_tile_pairs": total - visited,
        "density": visited / total,
    }
