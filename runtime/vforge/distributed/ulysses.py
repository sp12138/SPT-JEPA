"""Ulysses collectives: shard a sequence across ranks with all-to-all.

Named for the mechanism rather than for "sequence parallelism", because that
phrase means something else in Megatron, which vforge now vendors. Megatron's
sequence parallelism shards activations along the sequence *inside* the tensor
parallel group; it consumes no ranks of its own and has no communicator of its
own. Ulysses hands whole heads between ranks and does consume ranks. Anything
of Megatron's is spelled ``sequence_parallel`` in full, matching its own field
name; ``sp`` is not used as an abbreviation for either.

These functions are low-level primitives. Model wrapping and enablement should
be orchestrated from ``vforge.distributed.parallelize``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

from vforge.distributed.parallel_state import get_parallel_state

_GATHER_BACKWARD_ENV = "VFORGE_ULYSSES_GATHER_BACKWARD"


def get_usp_group():
    """Return the active Ulysses group, or ``None`` when Ulysses is disabled."""
    return get_parallel_state().usp_group


def sequence_all_to_all(
    tensor: torch.Tensor,
    *,
    scatter_dim: int,
    gather_dim: int,
    group=None,
) -> torch.Tensor:
    """All-to-all a tensor between sequence and head dimensions."""
    return _AllToAll.apply(tensor, scatter_dim, gather_dim, group or get_usp_group())


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, scatter_dim: int, gather_dim: int, group):
        if group is None:
            ctx.disabled = True
            return tensor
        ctx.disabled = False
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group
        return _all_to_all_forward(
            tensor,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
            group=group,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        if ctx.disabled:
            return grad_output, None, None, None
        grad = _all_to_all_forward(
            grad_output,
            scatter_dim=ctx.gather_dim,
            gather_dim=ctx.scatter_dim,
            group=ctx.group,
        )
        return grad, None, None, None


def _all_to_all_forward(
    tensor: torch.Tensor,
    *,
    scatter_dim: int,
    gather_dim: int,
    group,
) -> torch.Tensor:
    world_size = dist.get_world_size(group)
    if tensor.shape[scatter_dim] % world_size != 0:
        raise ValueError(
            f"scatter dimension {tensor.shape[scatter_dim]} must divide sp world size {world_size}"
        )
    chunks = [chunk.contiguous() for chunk in torch.tensor_split(tensor, world_size, scatter_dim)]
    # PyTorch 2.9 Gloo implements the tensor collective, but not list all-to-all.
    if dist.get_backend(group) == dist.Backend.GLOO:
        packed = torch.stack(chunks)
        received = torch.empty_like(packed)
        dist.all_to_all_single(received, packed, group=group)
        return torch.cat(received.unbind(), dim=gather_dim).contiguous()
    output = [torch.empty_like(chunks[0]) for _ in range(world_size)]
    dist.all_to_all(output, chunks, group=group)
    return torch.cat(output, dim=gather_dim).contiguous()


class _SliceSequence(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, tensor: torch.Tensor, dim: int, group, scale_grad: bool, gather_grad: bool
    ) -> torch.Tensor:
        if group is None:
            ctx.disabled = True
            return tensor
        ctx.disabled = False
        ctx.group = group
        ctx.dim = dim
        ctx.scale_grad = scale_grad
        ctx.gather_grad = gather_grad
        ctx.input_shape = tensor.shape
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        dim_size = tensor.shape[dim]
        if dim_size % ctx.world_size != 0:
            raise ValueError(
                f"sequence dimension {dim_size} must divide sp world size {ctx.world_size}"
            )
        return tensor.chunk(ctx.world_size, dim=dim)[ctx.rank].contiguous()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        if ctx.disabled:
            return grad_output, None, None, None, None
        if ctx.gather_grad:
            gathered = _all_gather_same_shape(grad_output, ctx.group)
            grad = torch.cat(gathered, dim=ctx.dim)
        else:
            grad = grad_output.new_zeros(ctx.input_shape)
            length = grad_output.shape[ctx.dim]
            grad.narrow(ctx.dim, ctx.rank * length, length).copy_(grad_output)
        if ctx.scale_grad:
            grad = grad / ctx.world_size
        return grad, None, None, None, None


class _GatherSequence(torch.autograd.Function):
    """Historical vforge gather with all-reduce followed by a local slice."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, dim: int, group, scale_grad: bool) -> torch.Tensor:
        if group is None:
            ctx.disabled = True
            return tensor
        ctx.disabled = False
        ctx.group = group
        ctx.dim = dim
        ctx.scale_grad = scale_grad
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        gathered = _all_gather_same_shape(tensor.contiguous(), group)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        if ctx.disabled:
            return grad_output, None, None, None
        if ctx.scale_grad:
            grad_output = grad_output * ctx.world_size
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        grad = grad_output.chunk(ctx.world_size, dim=ctx.dim)[ctx.rank].contiguous()
        return grad, None, None, None


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, scale: int) -> torch.Tensor:
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        return grad_output * ctx.scale, None


def slice_sequence(
    tensor: torch.Tensor,
    *,
    dim: int,
    group=None,
    scale_grad: bool = False,
    gather_grad: bool = True,
) -> torch.Tensor:
    """Slice a sequence dimension, gathering its gradients by default.

    ``gather_grad=False`` retains only the current rank's slice gradient, like
    a local ``narrow``. It adds no backward collective. Optional ``scale_grad``
    divides the gradient by the group size in either mode.
    """
    return _SliceSequence.apply(tensor, dim, group or get_usp_group(), scale_grad, gather_grad)


def gather_sequence(
    tensor: torch.Tensor,
    *,
    dim: int,
    group=None,
    scale_grad: bool = False,
    backward_mode: str | None = None,
) -> torch.Tensor:
    """Gather a sliced sequence and select its backward collective.

    An explicit ``backward_mode`` overrides ``VFORGE_ULYSSES_GATHER_BACKWARD``.
    Otherwise the environment setting defaults to ``all_reduce``, preserving
    the historical all-reduce-then-slice implementation. ``reduce_scatter``
    uses PyTorch's differentiable all-gather, whose backward is a reduce-scatter
    SUM on NCCL. Every rank in the Ulysses group must use the same value.
    """
    resolved_group = group or get_usp_group()
    if resolved_group is None:
        return tensor
    backward_mode = (
        (
            backward_mode
            if backward_mode is not None
            else os.environ.get(_GATHER_BACKWARD_ENV, "all_reduce")
        )
        .strip()
        .lower()
    )
    if backward_mode == "all_reduce":
        return _GatherSequence.apply(tensor, dim, resolved_group, scale_grad)
    if backward_mode != "reduce_scatter":
        raise ValueError(
            f"gather backward mode ({_GATHER_BACKWARD_ENV}) "
            "must be 'all_reduce' or 'reduce_scatter'; "
            f"got {backward_mode!r}"
        )
    gathered = dist_nn.all_gather(tensor.contiguous(), group=resolved_group)
    output = torch.cat(list(gathered), dim=dim).contiguous()
    if scale_grad:
        output = _ScaleGradient.apply(output, dist.get_world_size(resolved_group))
    return output


def _all_gather_same_shape(tensor: torch.Tensor, group) -> list[torch.Tensor]:
    world_size = dist.get_world_size(group)
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return gathered


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int,
    *,
    dim: int = 1,
) -> tuple[torch.Tensor, int]:
    """Zero-pad ``dim`` to a multiple and return the padding length."""
    size = tensor.shape[dim]
    pad_len = (-size) % multiple
    if pad_len == 0:
        return tensor, 0
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_len
    return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=dim), pad_len


def sequence_to_head(
    tensor: torch.Tensor,
    group,
    world_size: int,
) -> torch.Tensor:
    """Transform ``(B, S/N, H, D)`` to ``(B, S, ceil(H/N), D)``.

    Zero heads are appended when ``H`` is not divisible by the Ulysses degree.
    Callers must trim them after :func:`head_to_sequence`.
    """
    tensor, _ = pad_to_multiple(tensor, world_size, dim=2)
    batch, seq_local, heads, head_dim = tensor.shape
    heads_local = heads // world_size
    tensor = tensor.reshape(
        batch,
        seq_local,
        world_size,
        heads_local,
        head_dim,
    )
    tensor = tensor.permute(2, 0, 1, 3, 4).contiguous()
    tensor = sequence_all_to_all(
        tensor,
        scatter_dim=0,
        gather_dim=0,
        group=group,
    )
    return (
        tensor.permute(1, 0, 2, 3, 4)
        .contiguous()
        .reshape(
            batch,
            world_size * seq_local,
            heads_local,
            head_dim,
        )
    )


def head_to_sequence(
    tensor: torch.Tensor,
    group,
    world_size: int,
    *,
    original_heads: int | None = None,
) -> torch.Tensor:
    """Transform ``(B, S, H/N, D)`` to ``(B, S/N, H, D)`` and trim padding."""
    batch, seq, heads_local, head_dim = tensor.shape
    if seq % world_size != 0:
        raise ValueError(f"sequence length {seq} must divide world_size={world_size}")
    seq_local = seq // world_size
    tensor = tensor.reshape(
        batch,
        world_size,
        seq_local,
        heads_local,
        head_dim,
    )
    tensor = tensor.permute(1, 0, 2, 3, 4).contiguous()
    tensor = sequence_all_to_all(
        tensor,
        scatter_dim=0,
        gather_dim=0,
        group=group,
    )
    output = (
        tensor.permute(1, 2, 0, 3, 4)
        .contiguous()
        .reshape(
            batch,
            seq_local,
            world_size * heads_local,
            head_dim,
        )
    )
    if original_heads is None or original_heads == output.shape[2]:
        return output
    if original_heads < 1 or original_heads > output.shape[2]:
        raise ValueError(
            f"original_heads={original_heads} must be in [1, padded_heads={output.shape[2]}]"
        )
    return output.narrow(2, 0, original_heads)


def _pad_attention_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    world_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad Q/K/V while preserving the local grouped-query head ratio."""
    query_heads = q.shape[2]
    key_heads = k.shape[2]
    value_heads = v.shape[2]
    if key_heads != value_heads:
        raise ValueError(
            f"Ulysses requires matching key/value heads, got key={key_heads}, value={value_heads}"
        )
    if query_heads % key_heads:
        raise ValueError(
            "Ulysses grouped-query attention requires query heads to be a "
            f"multiple of key/value heads, got query={query_heads}, key/value={key_heads}"
        )

    query_groups = query_heads // key_heads
    padded_kv_heads = ((key_heads + world_size - 1) // world_size) * world_size
    padded_query_heads = padded_kv_heads * query_groups

    def pad_to_heads(tensor: torch.Tensor, target_heads: int) -> torch.Tensor:
        pad_heads = target_heads - tensor.shape[2]
        if pad_heads == 0:
            return tensor
        pad_shape = list(tensor.shape)
        pad_shape[2] = pad_heads
        return torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=2)

    return (
        pad_to_heads(q, padded_query_heads),
        pad_to_heads(k, padded_kv_heads),
        pad_to_heads(v, padded_kv_heads),
    )


def ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    group,
    world_size: int,
    attention_fn: Callable[..., torch.Tensor],
    varlen_attention_fn: Callable[..., torch.Tensor] | None = None,
    attn_mask: torch.Tensor | None = None,
    seq_real_len: int | None = None,
    is_causal: bool = False,
    sparsity: object | None = None,
) -> torch.Tensor:
    """Run an injected attention kernel between Ulysses layout transforms.

    ``sparsity`` passes straight through: the all-to-all restores the full
    sequence on every rank before the kernel runs, so a mask spec written
    against the global sequence stays valid unsharded.
    """
    if sparsity is not None and (attn_mask is not None or seq_real_len is not None):
        raise ValueError("pass either sparsity or attn_mask/seq_real_len, not both")

    original_query_heads = q.shape[2]
    q, k, v = _pad_attention_heads(q, k, v, world_size)
    q = sequence_to_head(q, group, world_size)
    k = sequence_to_head(k, group, world_size)
    v = sequence_to_head(v, group, world_size)

    if seq_real_len is not None and seq_real_len < q.shape[1]:
        if attn_mask is not None:
            raise ValueError("pass either attn_mask or seq_real_len, not both")
        if varlen_attention_fn is None:
            raise ValueError("varlen_attention_fn is required when seq_real_len is used")
        out = varlen_attention_fn(
            q,
            k,
            v,
            seq_real_len,
            is_causal=is_causal,
        )
    elif sparsity is not None:
        out = attention_fn(q, k, v, sparsity=sparsity)
    else:
        out = attention_fn(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )
    return head_to_sequence(
        out,
        group,
        world_size,
        original_heads=original_query_heads,
    )


__all__ = [
    "gather_sequence",
    "get_usp_group",
    "head_to_sequence",
    "pad_to_multiple",
    "sequence_all_to_all",
    "sequence_to_head",
    "slice_sequence",
    "ulysses_attention",
]
