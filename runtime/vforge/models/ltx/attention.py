"""Checkpoint-compatible LTX attention projections.

This module owns only model structure and pre/post processing. Attention
kernel selection and execution belong to :mod:`vforge.ops`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Protocol

import torch
import torch.distributed as dist

import vforge.ops as vforge_ops
from vforge.distributed.ulysses import (
    get_usp_group,
    head_to_sequence,
    pad_to_multiple,
    sequence_to_head,
)
from vforge.models.ltx.ops import (
    GatedAttentionCallable,
    LTXLinear,
    LTXRMSNorm,
    PreAttentionCallable,
    PytorchGatedAttention,
    PytorchPreAttention,
    attention_in_value_dtype,
)
from vforge.models.ltx.rope import LTXRopeType
from vforge.ops._attention.segmented import SegmentedSparsity


class AttentionKernel(Protocol):
    """Framework attention callable using the canonical BTHD layout."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        sparsity: SegmentedSparsity | None = None,
        query_length: int | None = None,
        context_length: int | None = None,
    ) -> torch.Tensor: ...


def _configured_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    sparsity: SegmentedSparsity | None = None,
) -> torch.Tensor:
    return vforge_ops.attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=is_causal,
        sparsity=sparsity,
    )


@dataclass(frozen=True)
class AttentionOps:
    """Framework kernel adapters and model-local tensor transforms."""

    # Standalone DMD preprocessing retains its original native BF16 norms.
    norm_dtype: torch.dtype | None = torch.float32
    attention_function: AttentionKernel = field(default=_configured_attention)
    masked_attention_function: AttentionKernel = field(default=vforge_ops.sdpa_attention)
    preattention_function: PreAttentionCallable = field(default_factory=PytorchPreAttention)
    gated_attention_function: GatedAttentionCallable = field(default_factory=PytorchGatedAttention)


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        ops: AttentionOps | None = None,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        ops = AttentionOps() if ops is None else ops
        self.rope_type = rope_type
        self.attention_function = partial(attention_in_value_dtype, ops.attention_function)
        self.masked_attention_function = partial(
            attention_in_value_dtype, ops.masked_attention_function
        )
        self.preattention_function = ops.preattention_function
        self.gated_attention_function = ops.gated_attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = LTXRMSNorm(inner_dim, eps=norm_eps, compute_dtype=ops.norm_dtype)
        self.k_norm = LTXRMSNorm(inner_dim, eps=norm_eps, compute_dtype=ops.norm_dtype)

        self.to_q = LTXLinear(query_dim, inner_dim, bias=True)
        self.to_k = LTXLinear(context_dim, inner_dim, bias=True)
        self.to_v = LTXLinear(context_dim, inner_dim, bias=True)

        self.to_gate_logits = (
            LTXLinear(query_dim, heads, bias=True) if apply_gated_attention else None
        )
        self.to_out = torch.nn.Sequential(
            LTXLinear(inner_dim, query_dim, bias=True),
            torch.nn.Identity(),
        )

    def _sequence_to_head(self, tensor, length):
        group = get_usp_group()
        if group is None:
            raise ValueError("sharded LTX attention requires an active Ulysses group")
        value = tensor.unflatten(-1, (self.heads, self.dim_head))
        value = sequence_to_head(value, group, dist.get_world_size(group))
        return value[:, :length]

    def _head_to_sequence(self, tensor):
        group = get_usp_group()
        size = dist.get_world_size(group)
        tensor, _ = pad_to_multiple(tensor, size, dim=1)
        return head_to_sequence(tensor, group, size, original_heads=self.heads).flatten(2)

    def _run_attention(
        self,
        kernel: AttentionKernel,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        mask: torch.Tensor | None,
        sparsity: SegmentedSparsity | None,
        query_length: int | None = None,
        context_length: int | None = None,
    ) -> torch.Tensor:
        if query_length is not None:
            if context_length is None or mask is not None or sparsity is not None:
                raise ValueError("sharded LTX A/V attention requires valid Q/K lengths and no mask")
            q = self._sequence_to_head(q, query_length)
            k = self._sequence_to_head(k, context_length)
            v = self._sequence_to_head(v, context_length)
            out = kernel(q, k, v, attn_mask=None, is_causal=False, sparsity=None)
            return self._head_to_sequence(out)
        batch = q.shape[0]
        q = q.view(batch, -1, self.heads, self.dim_head)
        k = k.view(batch, -1, self.heads, self.dim_head)
        v = v.view(batch, -1, self.heads, self.dim_head)
        out = kernel(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=False,
            sparsity=sparsity,
        )
        if out.shape != q.shape:
            raise ValueError(
                "framework attention must return the query BTHD shape; "
                f"expected {tuple(q.shape)}, got {tuple(out.shape)}"
            )
        return out.reshape(batch, -1, self.heads * self.dim_head)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        sparsity: SegmentedSparsity | None = None,
        query_length: int | None = None,
        context_length: int | None = None,
    ) -> torch.Tensor:
        """Apply LTX projections around a framework-owned attention kernel."""
        if mask is not None and sparsity is not None:
            raise ValueError("mask and sparsity are mutually exclusive")

        context = x if context is None else context
        v = self.to_v(context)

        if all_perturbed:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)
            q, k = self.preattention_function(q, k, self, mask, pe, k_pe)
            kernel = self.masked_attention_function if mask is not None else self.attention_function
            out = self._run_attention(
                kernel,
                q,
                k,
                v,
                mask=mask,
                sparsity=sparsity,
                query_length=query_length,
                context_length=context_length,
            )
            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        if self.to_gate_logits is not None:
            out = self.gated_attention_function(x, out, self)

        return self.to_out(out)


__all__ = ["Attention", "AttentionKernel", "AttentionOps"]
