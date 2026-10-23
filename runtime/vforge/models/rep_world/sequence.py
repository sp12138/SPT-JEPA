"""RepWorld token layout using Native Ulysses collectives.

Embedding and per-token modulation are computed from replicated inputs. Their
slice backward gathers contributions and divides by USP; the final gather
backward sums replicated losses. FSDP's average then produces the unsharded
model gradient, including parameters on either side of the sequence boundary.
"""

from __future__ import annotations

import torch.distributed as dist

from vforge.distributed.parallel_state import get_parallel_state
from vforge.distributed.ulysses import gather_sequence, slice_sequence


def usp_state(enabled=True):
    state = get_parallel_state()
    return state if enabled and state.usp_enabled else None


def token_range(length, enabled=True):
    state = usp_state(enabled)
    if state is None:
        return 0, length
    size = state.config.usp_size
    if length % size:
        raise ValueError(f"RepWorld token count {length} must be divisible by USP size {size}")
    width = length // size
    return state.usp_rank * width, width


def shard_tokens(tensor, *, dim=1, enabled=True):
    state = usp_state(enabled)
    if state is None or tensor is None:
        return tensor
    token_range(tensor.shape[dim], enabled)
    return slice_sequence(tensor, dim=dim, group=state.usp_group, scale_grad=True, gather_grad=True)


def gather_tokens(tensor, *, dim=1, enabled=True):
    state = usp_state(enabled)
    if state is None:
        return tensor
    return gather_sequence(tensor, dim=dim, group=state.usp_group, backward_mode="reduce_scatter")


def share_random(tensor, *, enabled=True):
    """A sequence group shares one sample, including FM noise/time/dropout."""
    state = usp_state(enabled)
    if state is not None:
        # Packed codec states can make randn_like preserve noncontiguous strides.
        # NCCL needs contiguous transport; keep the original tensor layout.
        transport = tensor.contiguous()
        dist.broadcast(
            transport, src=dist.get_global_rank(state.usp_group, 0), group=state.usp_group
        )
        if transport is not tensor:
            tensor.copy_(transport)
    return tensor
