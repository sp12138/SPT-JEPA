"""What tensor parallelism changes about Wan specifically.

The sharded projections themselves are generic and come from
``vforge.distributed.tensor_parallel``. What belongs here is everything that
depends on how *this* model is put together: that a diffusers ``FeedForward``
keeps its two linears at ``net[0].proj`` and ``net[2]``, and that Wan reshapes a
q/k/v projection into heads and so has to know how many of them a rank owns.
"""

from __future__ import annotations

import torch.nn as nn
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from vforge.distributed.parallel_state import get_parallel_state
from vforge.distributed.tensor_parallel import (
    ShardedRMSNorm,
    build_linear,
    tensor_parallel_enabled,
)


def local_heads(heads: int) -> int:
    """How many attention heads this rank owns.

    Column-sharded q/k/v hand each rank a contiguous slice of the projection,
    which is a whole number of heads, so the reshape that follows must use this
    rather than the configured count.
    """
    state = get_parallel_state()
    if not state.tp_enabled:
        return heads
    tp_size = state.config.tp_size
    if heads % tp_size != 0:
        raise ValueError(f"attention heads {heads} must be divisible by tp_size {tp_size}")
    return heads // tp_size


def local_rms_norm(
    local_features: int,
    *,
    eps: float | None = None,
    elementwise_affine: bool = True,
) -> nn.Module:
    """Wan's q/k norm, which under tp spans ranks.

    Wan normalizes q and k after projecting them and before splitting out
    heads, so the root mean square is over every head at once -- that is what
    ``qk_norm="rms_norm_across_heads"`` names. Column-sharding q and k leaves
    each rank with a slice of exactly the axis being normalized over, so the
    norm stops being computable locally and needs the group.

    At ``tp_size == 1`` this is upstream's ``nn.RMSNorm`` and nothing else.
    """
    if not tensor_parallel_enabled():
        return nn.RMSNorm(local_features, eps=eps, elementwise_affine=elementwise_affine)
    return ShardedRMSNorm(
        local_features,
        eps=eps,
        elementwise_affine=elementwise_affine,
    )


def declare_feed_forward_split(ffn: nn.Module) -> nn.Module:
    """Declare how a diffusers ``FeedForward`` splits, in place.

    Swapping the two projections inside the module diffusers built, rather than
    writing our own, is what keeps the parameters named ``net.0.proj.weight``
    and ``net.2.weight``. Those names are in every pretrained checkpoint and in
    the state-dict mapper, and a private feed-forward would rename them for no
    gain: the surrounding activation and dropout are not what tp changes.

    Unconditional, like every other declaration: what goes in is an ordinary
    ``nn.Linear`` of the same shape whatever ``tp_size`` is, only now carrying its
    style. Gating this on tp being enabled would leave these two the single pair
    of projections in the model whose declaration depends on how many ranks
    happen to be up -- so a check that walks the tree and expects to find every
    split would pass or fail by topology.
    """
    up = ffn.net[0].proj
    down = ffn.net[2]
    if not isinstance(up, nn.Linear) or not isinstance(down, nn.Linear):
        raise TypeError(
            "expected a diffusers FeedForward with Linear at net[0].proj and net[2]; "
            f"got {type(up).__name__} and {type(down).__name__}"
        )
    ffn.net[0].proj = build_linear(
        up.in_features, up.out_features, bias=up.bias is not None, style=ColwiseParallel()
    )
    ffn.net[2] = build_linear(
        down.in_features, down.out_features, bias=down.bias is not None, style=RowwiseParallel()
    )
    return ffn


__all__ = ["declare_feed_forward_split", "local_heads", "local_rms_norm"]
