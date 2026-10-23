"""Model projections only; attention dispatch is owned by vforge.ops."""

from __future__ import annotations

import torch
from torch import nn

from vforge.models.rep_world.layers import RMSNorm
from vforge.models.rep_world.sequence import gather_tokens, token_range
from vforge.models.rep_world.sparsity import slice_query_sparsity
from vforge.ops import AttentionCallPolicy, SegmentedSparsity, attention

_POLICY = AttentionCallPolicy(cast_qk_to_v=True)


class SelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, qk_norm: bool):
        super().__init__()
        self.num_heads, self.head_dim = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(width, width)

    def forward(
        self, x, *, rope, mask=None, past_kv=None, return_kv=False, sequence_parallel=False
    ):
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, length, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k = rope(self.q_norm(q)), rope(self.k_norm(k))
        # The public inference cache stays in global token order. Each rank
        # computes its local queries against this shared cache; normal training
        # uses the Native Ulysses sequence/head all-to-all.
        cached_parallel = sequence_parallel and (return_kv or past_kv is not None)
        if cached_parallel:
            k = gather_tokens(k, dim=2)
            v = gather_tokens(v, dim=2)
            if mask is not None:
                global_length = (
                    mask.seqlen_q if isinstance(mask, SegmentedSparsity) else mask.shape[-2]
                )
                start, local_length = token_range(global_length)
                mask = (
                    slice_query_sparsity(mask, start, local_length)
                    if isinstance(mask, SegmentedSparsity)
                    else mask.narrow(-2, start, local_length)
                )
        current = (k, v)
        if past_kv is not None:
            k = torch.cat((past_kv[0].to(k), k), dim=-2)
            v = torch.cat((past_kv[1].to(v), v), dim=-2)
        result = attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=None if isinstance(mask, SegmentedSparsity) else mask,
            sparsity=mask if isinstance(mask, SegmentedSparsity) else None,
            call_policy=_POLICY,
            distributed=sequence_parallel and not cached_parallel,
        )
        result = self.proj(result.reshape(batch, length, width).to(x.dtype))
        return (result, current) if return_kv else result


class CrossAttention(nn.Module):
    def __init__(self, width: int, heads: int, qk_norm: bool):
        super().__init__()
        self.num_heads, self.head_dim = heads, width // heads
        self.to_q = nn.Linear(width, width)
        self.to_kv = nn.Linear(width, 2 * width)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(width, width)

    def forward(
        self, x, context, *, spatial_size=None, context_positions=None, q_rope=None, k_rope=None
    ):
        batch, length, width = x.shape
        q = self.to_q(x).reshape(batch, length, self.num_heads, self.head_dim)
        k, v = self.to_kv(context).reshape(batch, -1, 2, self.num_heads, self.head_dim).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        if (q_rope is None) != (k_rope is None):
            raise ValueError("Cross-attention RoPE requires both query and key coordinates")
        if q_rope is not None:
            q = q_rope(q.transpose(1, 2)).transpose(1, 2)
            k = k_rope(k.transpose(1, 2)).transpose(1, 2)
        if spatial_size is None:
            result = attention(q, k, v, call_policy=_POLICY)
            return self.proj(result.reshape(batch, length, width))
        return self._same_spatial(q, k, v, spatial_size, context_positions)

    def _same_spatial(self, q, k, v, spatial, positions):
        """Batch independent (h,w) groups through the configured Native backend.

        Original token order is [plane,h,w]. Sparse positions are original state
        indices, never compressed sequence indices. No padded key participates in
        softmax. An empty group has a zero update including projection bias.
        """
        batch, length, heads, dim = q.shape
        if length % spatial:
            raise ValueError("Query length must contain complete spatial planes")
        planes = length // spatial
        grouped_q = q.reshape(batch, planes, spatial, heads, dim).transpose(1, 2)
        grouped_q = grouped_q.reshape(batch * spatial, planes, heads, dim)

        def attend(query, key, value):
            # Dense FVD batches can contain >65535 independent spatial groups.
            # Bound launch batch size without changing any softmax domain.
            return torch.cat(
                [
                    attention(
                        query[i : i + 32768].contiguous(),
                        key[i : i + 32768].contiguous(),
                        value[i : i + 32768].contiguous(),
                        call_policy=_POLICY,
                    )
                    for i in range(0, query.shape[0], 32768)
                ],
                dim=0,
            )

        if positions is None:
            if k.shape[1] != length:
                raise ValueError("Dense same-spatial context must match the query grid")
            grouped_k = k.reshape(batch, planes, spatial, heads, dim).transpose(1, 2)
            grouped_v = v.reshape(batch, planes, spatial, heads, dim).transpose(1, 2)
            result = attend(
                grouped_q, grouped_k.reshape_as(grouped_q), grouped_v.reshape_as(grouped_q)
            )
            valid = None
        else:
            # Store index+1 so zero denotes an absent token. Sort each spatial
            # group's temporal candidates, then batch by visible key count.
            count = k.shape[1]
            slots = positions.new_zeros(batch, length).scatter(
                1,
                positions,
                torch.arange(1, count + 1, device=q.device).expand(batch, -1),
            )
            slots = slots.reshape(batch, planes, spatial).transpose(1, 2)
            slots = slots.reshape(batch * spatial, planes).sort(dim=1, descending=True).values
            counts = (slots > 0).sum(dim=1)
            result = grouped_q.to(v.dtype) * 0
            flat_k, flat_v = k.flatten(0, 1), v.flatten(0, 1)
            for visible in range(1, planes + 1):
                groups = (counts == visible).nonzero(as_tuple=True)[0]
                if groups.numel() == 0:
                    continue
                indices = slots[groups, :visible] - 1 + (groups // spatial)[:, None] * count
                values = attend(grouped_q[groups], flat_k[indices], flat_v[indices])
                result = result.index_copy(0, groups, values)
            valid = (counts > 0).reshape(batch, spatial)
            valid = valid[:, None, :].expand(-1, planes, -1).reshape(batch, length, 1)
        result = result.reshape(batch, spatial, planes, heads, dim).transpose(1, 2)
        output = self.proj(result.reshape(batch, length, heads * dim))
        return output if valid is None else output * valid.to(output.dtype)
