"""Parameter-compatible attention primitives shared by Wan transformers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention import AttentionModuleMixin

from vforge.ops import attention as vforge_attention


def get_qkv_projections(
    attn: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project self- or cross-attention inputs, honoring fused projections."""
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    if attn.fused_projections:
        if not attn.is_cross_attention:
            return attn.to_qkv(hidden_states).chunk(3, dim=-1)
        query = attn.to_q(hidden_states)
        key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
        return query, key, value
    return (
        attn.to_q(hidden_states),
        attn.to_k(encoder_hidden_states),
        attn.to_v(encoder_hidden_states),
    )


def get_added_kv_projections(
    attn: Any,
    encoder_hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if attn.fused_projections:
        return attn.to_added_kv(encoder_hidden_states).chunk(2, dim=-1)
    return attn.add_k_proj(encoder_hidden_states), attn.add_v_proj(encoder_hidden_states)


def apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply Wan's interleaved real-valued rotary embedding."""
    first, second = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = first * cos - second * sin
    output[..., 1::2] = first * sin + second * cos
    return output.type_as(hidden_states)


def narrow_qk_to_value(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bring q/k to v's dtype right before an attention kernel runs.

    The kernels need one dtype across the three, and autocast runs the QK norms
    in fp32 while ``value`` skips them and stays in the compute dtype. ``value``
    is the only one of the three that never passes through a norm, which makes
    it the compute-dtype reference.

    Call this at the kernel boundary rather than right after the norms: rotary
    then still runs at the norms' precision, which is where the reference
    implementation applies it. Callers must not rebind ``query`` from this --
    the attention output is cast back with ``type_as(query)``, and narrowing
    that reference would store the output one width short of every other path.
    """
    return query.to(value.dtype), key.to(value.dtype)


class WanVforgeSelfAttentionProcessor:
    """Wan self-attention routed through vforge sequence-parallel ops."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self) -> None:
        self._pad_attn_mask = None
        self._seq_real_len = None

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise AssertionError("Wan sequence-parallel processor only supports self-attention")
        if attention_mask is None:
            attention_mask = self._pad_attn_mask

        query, key, value = get_qkv_projections(attn, hidden_states, None)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        attend_query, attend_key = narrow_qk_to_value(query, key, value)
        output = vforge_attention(
            attend_query,
            attend_key,
            value,
            attn_mask=attention_mask,
            is_causal=False,
            distributed=True,
            seq_real_len=self._seq_real_len,
        )
        output = output.flatten(2, 3).type_as(query)
        output = attn.to_out[0](output)
        return attn.to_out[1](output)


class WanAttention(nn.Module, AttentionModuleMixin):
    """Wan QKV module, including the optional I2V added-KV branch."""

    _default_processor_cls = None
    _available_processors: list[type] = []

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: int | None = None,
        cross_attention_dim_head: int | None = None,
        processor=None,
        is_cross_attention: bool | None = None,
    ) -> None:
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = (
            self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads
        )

        self.to_q = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = nn.ModuleList(
            [
                nn.Linear(self.inner_dim, dim, bias=True),
                nn.Dropout(dropout),
            ]
        )
        self.norm_q = nn.RMSNorm(
            self.inner_dim,
            eps=eps,
            elementwise_affine=True,
        )
        self.norm_k = nn.RMSNorm(
            self.kv_inner_dim,
            eps=eps,
            elementwise_affine=True,
        )

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = nn.RMSNorm(self.inner_dim, eps=eps)

        self.is_cross_attention = (
            is_cross_attention
            if is_cross_attention is not None
            else cross_attention_dim_head is not None
        )
        self.set_processor(processor)

    def fuse_projections(self) -> None:
        if getattr(self, "fused_projections", False):
            return
        if not self.is_cross_attention:
            weights = torch.cat(
                [self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data]
            )
            bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            self.to_qkv = _meta_linear_from_tensors(weights, bias)
        else:
            weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            self.to_kv = _meta_linear_from_tensors(weights, bias)

        if self.added_kv_proj_dim is not None:
            weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            self.to_added_kv = _meta_linear_from_tensors(weights, bias)
        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self) -> None:
        if not getattr(self, "fused_projections", False):
            return
        for name in ("to_qkv", "to_kv", "to_added_kv"):
            if hasattr(self, name):
                delattr(self, name)
        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb,
            **kwargs,
        )


def _meta_linear_from_tensors(weights: torch.Tensor, bias: torch.Tensor) -> nn.Linear:
    out_features, in_features = weights.shape
    with torch.device("meta"):
        linear = nn.Linear(in_features, out_features, bias=True)
    linear.load_state_dict(
        {"weight": weights, "bias": bias},
        strict=True,
        assign=True,
    )
    return linear


__all__ = [
    "WanAttention",
    "WanVforgeSelfAttentionProcessor",
    "apply_rotary_emb",
    "get_added_kv_projections",
    "get_qkv_projections",
    "narrow_qk_to_value",
]
