from typing import Protocol

import torch
from torch import nn

from vforge.models.ltx.rope import apply_rotary_emb


class LTXLinear(nn.Linear):
    """Match the gathered weight dtype at GEMM without rounding upstream tensors."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value.to(self.weight.dtype), self.weight, self.bias)


def rms_norm(value, normalized_shape, weight=None, eps=None, *, compute_dtype=torch.float32):
    """Keep norm output and its following modulation in the requested precision."""
    if compute_dtype is not None:
        value = value.to(compute_dtype)
        weight = None if weight is None else weight.to(compute_dtype)
    return torch.nn.functional.rms_norm(value, normalized_shape, weight, eps)


class LTXRMSNorm(nn.RMSNorm):
    """FP32 norm arithmetic over FSDP's gathered parameters; state keys are unchanged."""

    def __init__(self, normalized_shape, eps=1e-6, *, compute_dtype=torch.float32):
        super().__init__(normalized_shape, eps=eps)
        self.compute_dtype = compute_dtype

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return rms_norm(
            value, self.normalized_shape, self.weight, self.eps, compute_dtype=self.compute_dtype
        )


def attention_in_value_dtype(kernel, q, k, v, *, attn_mask=None, **kwargs):
    """Round Q/K at the kernel boundary, preserving RoPE and stored KV precision."""
    if attn_mask is not None and attn_mask.is_floating_point():
        attn_mask = attn_mask.to(v.dtype)
    return kernel(q.to(v.dtype), k.to(v.dtype), v, attn_mask=attn_mask, **kwargs)


class PreAttentionCallable(Protocol):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_module: nn.Module,
        mask: torch.Tensor | None,
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class PytorchPreAttention(PreAttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_module: nn.Module,
        mask: torch.Tensor | None,  # noqa: ARG002
        pe: torch.Tensor | None,
        k_pe: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = attn_module.q_norm(q)
        k = attn_module.k_norm(k)
        if pe is not None:
            q = apply_rotary_emb(q, pe, attn_module.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, attn_module.rope_type)
        return q, k


class AdaZeroCallable(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor: ...


class PytorchAdaZeroFunction(AdaZeroCallable):
    def __init__(self, norm_dtype=torch.float32):
        self.norm_dtype = norm_dtype

    def __call__(
        self,
        x: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        return (
            rms_norm(x, (x.shape[-1],), eps=eps, compute_dtype=self.norm_dtype) * (1 + scale)
            + shift
        )


class PostSACallable(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        norm_weights: torch.Tensor | None,
        eps: float,
        gate: torch.Tensor,
    ) -> list[torch.Tensor]: ...


class PytorchPostSAFunction(PostSACallable):
    def __init__(self, norm_dtype=torch.float32):
        self.norm_dtype = norm_dtype

    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        norm_weights: torch.Tensor | None,
        eps: float,
        gate: torch.Tensor,
    ) -> list[torch.Tensor]:
        x_fma = x + y * gate
        return x_fma, rms_norm(
            x_fma,
            (x_fma.shape[-1],),
            weight=norm_weights,
            eps=eps,
            compute_dtype=self.norm_dtype,
        )


class GatedAttentionCallable(Protocol):
    def __call__(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        attn_module: nn.Module,
    ) -> torch.Tensor: ...


class PytorchGatedAttention(GatedAttentionCallable):
    def __call__(
        self,
        x: torch.Tensor,
        attn_out: torch.Tensor,
        attn_module: nn.Module,
    ) -> torch.Tensor:
        gate_logits = attn_module.to_gate_logits(x)  # (B, T, H)
        b, t, _ = attn_out.shape
        out = attn_out.view(b, t, attn_module.heads, attn_module.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)  # (B, T, H)
        out = out * gates.unsqueeze(-1)  # (B, T, H, D) * (B, T, H, 1)
        return out.view(b, t, attn_module.heads * attn_module.dim_head)


def dmd_transformer_ops(*, norm_dtype=torch.float32):
    """Local kernels; the LTX attention modules own sequence/head exchange."""
    from dataclasses import replace
    from functools import partial

    from torch.nn.attention import SDPBackend

    import vforge.ops as vforge_ops
    from vforge.models.ltx.transformer import TransformerOpsConfig

    backend = vforge_ops.get_resolved_attention_backend()
    if backend == "sdpa_math":
        priority = (SDPBackend.MATH,)
    else:
        priority = (
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        )
    masked_attention = partial(
        vforge_ops.sdpa_attention,
        priority=priority,
        contiguous=False,
    )
    config = TransformerOpsConfig.from_functions(
        attention=partial(vforge_ops.attention, ulysses=False),
        masked_attention=masked_attention,
        connector_attention=vforge_ops.attention,
        connector_masked_attention=vforge_ops.sdpa_attention,
    )

    return replace(
        config,
        attention_ops=replace(config.attention_ops, norm_dtype=norm_dtype),
        connector_attention_ops=replace(config.connector_attention_ops, norm_dtype=None),
        ada_zero_function=PytorchAdaZeroFunction(norm_dtype),
        post_sa_function=PytorchPostSAFunction(norm_dtype),
    )
