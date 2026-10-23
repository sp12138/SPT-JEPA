"""Megatron-Core context-parallel sequence layout and attention backend.

Wan tensors use ``[B, S, H, D]`` while MCore uses ``[S, B, H, D]``, and MCore
P2P CP expects each rank to hold two opposite chunks of the global sequence.
Keeping both conversions here prevents models and the public CP facade from
depending on Transformer Engine or MCore types.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from vforge.distributed.parallel_state import get_parallel_state


def _autocast_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Make TE's CP kernel observe the surrounding torch autocast policy.

    PyTorch attention operators are autocast-aware themselves. TE's module is
    called through a Python adapter and sees its inputs before such a cast, so
    FP32 norm weights can promote q/k even inside a BF16 autocast region. TE has
    no FP32 CP backend; applying the active autocast dtype here gives this
    backend the same operator-boundary semantics as torch SDPA.
    """
    device_type = q.device.type
    if not torch.is_autocast_enabled(device_type):
        return q, k, v
    dtype = torch.get_autocast_dtype(device_type)
    return q.to(dtype=dtype), k.to(dtype=dtype), v.to(dtype=dtype)


def context_parallel_indices(
    sequence_length: int,
    *,
    world_size: int,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    """Return MCore's two-chunk zigzag sequence indices for one CP rank."""
    chunks = 2 * int(world_size)
    if sequence_length % chunks != 0:
        raise ValueError(
            "context parallel sequence length must be divisible by 2 * cp_size: "
            f"{sequence_length} % {chunks} != 0"
        )
    if rank < 0 or rank >= world_size:
        raise ValueError(f"context parallel rank must be in [0, {world_size}), got {rank}")
    chunk_length = sequence_length // chunks
    positions = torch.arange(sequence_length, device=device).view(chunks, chunk_length)
    selected = torch.tensor([rank, chunks - rank - 1], device=device)
    return positions.index_select(0, selected).reshape(-1)


class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, dim: int, group: Any) -> torch.Tensor:
        ctx.dim = dim
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        if tensor.shape[dim] % 2 != 0:
            raise ValueError(
                "an MCore context-parallel local sequence must contain two equal chunks; "
                f"got local length {tensor.shape[dim]}"
            )
        gathered = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=group)
        chunks: list[torch.Tensor | None] = [None] * (2 * ctx.world_size)
        for rank, local in enumerate(gathered):
            first, second = local.chunk(2, dim=dim)
            chunks[rank] = first
            chunks[2 * ctx.world_size - rank - 1] = second
        return torch.cat([chunk for chunk in chunks if chunk is not None], dim=dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        indices = context_parallel_indices(
            grad_output.shape[ctx.dim],
            world_size=ctx.world_size,
            rank=ctx.rank,
            device=grad_output.device,
        )
        return grad_output.index_select(ctx.dim, indices).contiguous(), None, None


class MCoreContextParallelBackend:
    """MCore/Transformer Engine P2P context parallelism."""

    name = "mcore"

    def __init__(self) -> None:
        self._attention_modules: dict[tuple[Any, ...], torch.nn.Module] = {}

    def shard_sequence(
        self,
        tensor: torch.Tensor,
        *,
        dim: int = 1,
        group: Any = None,
    ) -> torch.Tensor:
        """Select this CP rank's two zigzag chunks from a replicated sequence."""
        selected_group = group or get_parallel_state().cp_group
        if selected_group is None:
            return tensor
        indices = context_parallel_indices(
            tensor.shape[dim],
            world_size=dist.get_world_size(selected_group),
            rank=dist.get_rank(selected_group),
            device=tensor.device,
        )
        return tensor.index_select(dim, indices).contiguous()

    def gather_sequence(
        self,
        tensor: torch.Tensor,
        *,
        dim: int = 1,
        group: Any = None,
    ) -> torch.Tensor:
        """Gather zigzag CP shards and restore the original sequence order."""
        selected_group = group or get_parallel_state().cp_group
        if selected_group is None:
            return tensor
        return _GatherSequence.apply(tensor, dim, selected_group)

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Run MCore/Transformer Engine CP on local ``[B,S,H,D]`` tensors."""
        if attn_mask is not None:
            raise NotImplementedError("MCore context parallel attention does not support masks yet")
        if is_causal:
            raise NotImplementedError("causal MCore context parallel attention is not enabled yet")
        if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
            raise ValueError(
                "MCore context parallel attention requires equal rank-4 q/k/v shapes, got "
                f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
            )
        q, k, v = _autocast_attention_inputs(q, k, v)
        if q.dtype not in {torch.bfloat16, torch.float16}:
            raise TypeError(
                "MCore context parallel attention requires BF16 or FP16 inputs; "
                f"got {q.dtype}. Enable torch autocast or provide low-precision q/k/v"
            )

        state = get_parallel_state()
        if not state.cp_enabled:
            raise RuntimeError("MCore context parallel attention requires cp_size > 1")
        cp_group = state.cp_group
        tp_group = state.attention_tp_group
        if cp_group is None or tp_group is None:
            raise RuntimeError(
                "MCore context parallel attention requires initialized CP and TP groups"
            )

        batch, sequence, local_heads, head_dim = q.shape
        module = self._attention_module(
            local_heads=local_heads,
            head_dim=head_dim,
            dtype=q.dtype,
            device=q.device,
            cp_group=cp_group,
            tp_group=tp_group,
            cp_size=state.config.cp_size,
            tp_size=state.config.tp_size,
        )
        from vforge.mcore.transformer.enums import AttnMaskType

        output = module(
            q.transpose(0, 1).contiguous(),
            k.transpose(0, 1).contiguous(),
            v.transpose(0, 1).contiguous(),
            attention_mask=None,
            attn_mask_type=AttnMaskType.no_mask,
        )
        return output.view(sequence, batch, local_heads, head_dim).transpose(0, 1).contiguous()

    def close(self) -> None:
        """Drop cached parameterless TE modules."""
        self._attention_modules.clear()

    def _attention_module(
        self,
        *,
        local_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        cp_group: Any,
        tp_group: Any,
        cp_size: int,
        tp_size: int,
    ) -> torch.nn.Module:
        key = (
            local_heads,
            head_dim,
            dtype,
            device.type,
            device.index,
            id(cp_group),
            id(tp_group),
            cp_size,
            tp_size,
        )
        cached = self._attention_modules.get(key)
        if cached is not None:
            return cached

        try:
            import transformer_engine
            import transformer_engine.pytorch
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "MCore context parallelism requires a loadable Transformer Engine PyTorch extension"
            ) from exc

        del transformer_engine
        from vforge.mcore.extensions.transformer_engine import TEDotProductAttention
        from vforge.mcore.process_groups_config import ProcessGroupCollection
        from vforge.mcore.transformer.enums import AttnMaskType
        from vforge.mcore.transformer.transformer_config import TransformerConfig

        global_heads = local_heads * tp_size
        config = TransformerConfig(
            num_layers=1,
            hidden_size=global_heads * head_dim,
            num_attention_heads=global_heads,
            num_query_groups=global_heads,
            kv_channels=head_dim,
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            bf16=dtype == torch.bfloat16,
            fp16=dtype == torch.float16,
            params_dtype=dtype,
            deterministic_mode=False,
        )
        module = TEDotProductAttention(
            config=config,
            layer_number=1,
            attn_mask_type=AttnMaskType.no_mask,
            attention_type="self",
            attention_dropout=0.0,
            cp_comm_type="p2p",
            pg_collection=ProcessGroupCollection(tp=tp_group, cp=cp_group),
        ).to(device)
        module.train()
        self._attention_modules[key] = module
        return module


__all__ = ["MCoreContextParallelBackend", "context_parallel_indices"]
