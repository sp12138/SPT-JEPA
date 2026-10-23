"""RepWorld visibility graphs for the shared FA4 segmented-attention backend.

These describe state blocks, not token-causal raster ordering. SDPA keeps its
existing dense masks; no parameters, RoPE positions or token order change.
"""

from functools import lru_cache

from vforge.ops import SegmentedSparsity, get_resolved_attention_backend


def use_segmented_attention():
    return get_resolved_attention_backend() == "flash_attention_4"


@lru_cache(maxsize=128)
def state_causal_sparsity(length, tokens_per_state, past_tokens=0):
    if length <= 0 or tokens_per_state <= 0 or past_tokens < 0:
        raise ValueError("State-causal dimensions must be positive, with nonnegative history")
    query = tuple(i // tokens_per_state for i in range(past_tokens, past_tokens + length))
    keys = tuple(i // tokens_per_state for i in range(past_tokens + length))
    pairs = tuple((q, k) for q in sorted(set(query)) for k in sorted(set(keys)) if k <= q)
    return SegmentedSparsity(query, keys, pairs)


@lru_cache(maxsize=128)
def prefix_sparsity(length, prefix_tokens):
    if not 0 < prefix_tokens < length:
        raise ValueError("Prefix must leave at least one target token")
    segments = (0,) * prefix_tokens + (1,) * (length - prefix_tokens)
    return SegmentedSparsity(segments, segments, ((0, 0), (1, 0), (1, 1)))


@lru_cache(maxsize=128)
def teacher_forcing_sparsity(length, tokens_per_state):
    if tokens_per_state <= 0 or length <= 0 or length % (2 * tokens_per_state):
        raise ValueError("Teacher forcing requires complete clean/query state pairs")
    blocks = length // tokens_per_state
    segments = tuple(i // tokens_per_state for i in range(length))
    pairs = tuple(
        (q, k)
        for q in range(blocks)
        for k in range(blocks)
        if (k % 2 == 0 and k <= q) or (q % 2 == 1 and k == q)
    )
    return SegmentedSparsity(segments, segments, pairs)


@lru_cache(maxsize=128)
def slice_query_sparsity(spec, start, length):
    """Slice only queries for replicated-KV Ulysses inference."""
    if length <= 0 or start < 0 or start + length > spec.seqlen_q:
        raise ValueError("Query slice must be nonempty and inside the global sequence")
    query = spec.q_segment_ids[start : start + length]
    visible = set(query)
    return SegmentedSparsity(
        query,
        spec.kv_segment_ids,
        tuple(pair for pair in spec.allowed_pairs if pair[0] in visible),
        tuple(segment for segment in spec.discarded_query_segments if segment in visible),
    )
