"""Shared RoPE, block, Ulysses, and trunk helpers for Wan transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.models.normalization import FP32LayerNorm
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph

from vforge.distributed.parallelize import FSDPPrecisionContract
from vforge.models.block_kwargs import (
    TransformerBlockKwargsProvider,
    resolve_transformer_block_kwargs,
)
from vforge.models.wan.attention import (
    WanAttention,
    WanVforgeSelfAttentionProcessor,
)
from vforge.models.wan.tensor_parallel import declare_feed_forward_split


def wan_fsdp_precision_contract() -> FSDPPrecisionContract:
    """Wan's forward-input boundaries for FSDP mixed precision."""
    return FSDPPrecisionContract(
        preserve_forward_input_paths=(".", "blocks.*", "norm_out"),
    )


class WanRotaryPosEmbed(nn.Module):
    """Wan's factorized temporal/height/width rotary tables."""

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        self.t_dim = attention_head_dim - h_dim - w_dim
        self.h_dim = h_dim
        self.w_dim = w_dim
        self.freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        freqs_cos, freqs_sin = self._build_freqs()
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _build_freqs(self) -> tuple[torch.Tensor, torch.Tensor]:
        freqs_cos = []
        freqs_sin = []
        for dim in (self.t_dim, self.h_dim, self.w_dim):
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                self.max_seq_len,
                self.theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=self.freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)
        return torch.cat(freqs_cos, dim=1), torch.cat(freqs_sin, dim=1)

    def reset_non_persistent_buffers(self) -> None:
        """Rebuild RoPE tables discarded by meta ``to_empty`` materialization."""
        with torch.device("cpu"):
            freqs_cos, freqs_sin = self._build_freqs()
        device = self.freqs_cos.device
        dtype = self.freqs_cos.dtype
        self.freqs_cos = freqs_cos.to(device=device, dtype=dtype)
        self.freqs_sin = freqs_sin.to(device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
        split_sizes = [self.t_dim, self.h_dim, self.w_dim]
        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        cos = torch.cat([cos_f, cos_h, cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        sin = torch.cat([sin_f, sin_h, sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        return cos, sin

    def build_ref(
        self,
        ref_hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build time-neutral RoPE for in-context reference tokens.

        Reference tokens reuse the video's height/width coordinates. Both
        temporal components are zero, rather than an identity rotation, so the
        temporal Q/K subspace contributes nothing to attention scores.
        """
        _, _, num_frames, height, width = ref_hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
        split_sizes = [self.t_dim, self.h_dim, self.w_dim]
        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        cos_f = freqs_cos[0].new_zeros(ppf, pph, ppw, self.t_dim)
        sin_f = freqs_sin[0].new_zeros(ppf, pph, ppw, self.t_dim)
        cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        cos = torch.cat([cos_f, cos_h, cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        sin = torch.cat([sin_f, sin_h, sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        return cos, sin


@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    """Parameter-identical Wan block with injected cross-attention semantics."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
        *,
        attention_type: type[WanAttention] = WanAttention,
        cross_attention_processor: Any,
    ) -> None:
        super().__init__()
        del qk_norm  # Kept in the public shape for Diffusers config compatibility.
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = attention_type(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanVforgeSelfAttentionProcessor(),
        )
        self.attn2 = attention_type(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=cross_attention_processor,
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )
        self.ffn = declare_feed_forward_split(
            FeedForward(
                dim,
                inner_dim=ffn_dim,
                activation_fn="gelu-approximate",
            )
        )
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        self_attention_mask: Any = None,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            modulation = (self.scale_shift_table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                value.squeeze(2) for value in modulation
            )
        else:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                c_shift_msa,
                c_scale_msa,
                c_gate_msa,
            ) = (self.scale_shift_table + temb.float()).chunk(6, dim=1)

        norm_hidden_states = (
            self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa
        ).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            None,
            self_attention_mask,
            rotary_emb,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        hidden_states = hidden_states + self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            None,
            None,
        )
        norm_hidden_states = (
            self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
        ).type_as(hidden_states)
        ff_output = self.ffn(norm_hidden_states)
        return (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)


def apply_lora_scale(
    model: nn.Module,
    attention_kwargs: dict[str, Any] | None,
    *,
    logger: Any,
) -> None:
    """Apply the manual Diffusers PEFT scale convention used by Wan forks."""
    if attention_kwargs is not None:
        copied = attention_kwargs.copy()
        lora_scale = copied.pop("scale", 1.0)
    else:
        copied = None
        lora_scale = 1.0
    if USE_PEFT_BACKEND:
        scale_lora_layers(model, lora_scale)
    elif copied is not None and copied.get("scale") is not None:
        logger.warning(
            "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
        )


@dataclass(frozen=True)
class WanUlyssesContext:
    size: int = 1
    group: Any = None
    full_seq_len: int | None = None
    backend: str = "none"

    @property
    def enabled(self) -> bool:
        return self.size > 1


def prepare_ulysses_inputs(
    hidden_states: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    *,
    blocks: nn.ModuleList,
    parallel_state: Any,
    sequence_tensors: tuple[torch.Tensor, ...] = (),
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, ...],
    WanUlyssesContext,
]:
    """Pad and shard the standard Wan Ulysses sequence inputs."""
    usp_size = int(parallel_state.config.usp_size)
    usp_group = parallel_state.usp_group
    cp_size = int(getattr(parallel_state.config, "cp_size", 1))
    if usp_size <= 1 and cp_size <= 1:
        context = WanUlyssesContext()
        return hidden_states, rotary_emb, sequence_tensors, context

    if cp_size > 1:
        from vforge.distributed.context_parallel import shard_sequence

        full_seq_len = hidden_states.shape[1]
        hidden_states = shard_sequence(
            hidden_states,
            dim=1,
            group=parallel_state.cp_group,
        )
        rotary_emb = tuple(
            shard_sequence(tensor, dim=1, group=parallel_state.cp_group) for tensor in rotary_emb
        )
        sharded_tensors = tuple(
            shard_sequence(tensor, dim=1, group=parallel_state.cp_group)
            for tensor in sequence_tensors
        )
        return (
            hidden_states,
            rotary_emb,
            sharded_tensors,
            WanUlyssesContext(
                size=cp_size,
                group=parallel_state.cp_group,
                full_seq_len=full_seq_len,
                backend="context",
            ),
        )

    context = WanUlyssesContext(size=usp_size, group=usp_group, backend="ulysses")

    from vforge.distributed.ulysses import pad_to_multiple, slice_sequence

    full_seq_len = hidden_states.shape[1]
    hidden_states, pad_len = pad_to_multiple(hidden_states, usp_size, dim=1)
    padded_seq_len = hidden_states.shape[1]
    pad_attn_mask = None
    if pad_len > 0:
        keep = torch.ones(
            padded_seq_len,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        keep[full_seq_len:] = False
        pad_attn_mask = keep.view(1, 1, 1, padded_seq_len)

    cos_full, _ = pad_to_multiple(rotary_emb[0], usp_size, dim=1)
    sin_full, _ = pad_to_multiple(rotary_emb[1], usp_size, dim=1)
    hidden_states = slice_sequence(
        hidden_states,
        dim=1,
        group=usp_group,
        scale_grad=True,
    )
    rotary_emb = (
        slice_sequence(cos_full, dim=1, group=usp_group),
        slice_sequence(sin_full, dim=1, group=usp_group),
    )
    sharded_tensors = []
    for tensor in sequence_tensors:
        tensor, _ = pad_to_multiple(tensor, usp_size, dim=1)
        sharded_tensors.append(
            slice_sequence(
                tensor,
                dim=1,
                group=usp_group,
                scale_grad=True,
            )
        )
    for block in blocks:
        block.attn1.processor._pad_attn_mask = pad_attn_mask
    return (
        hidden_states,
        rotary_emb,
        tuple(sharded_tensors),
        WanUlyssesContext(
            size=usp_size,
            group=usp_group,
            full_seq_len=full_seq_len,
            backend="ulysses",
        ),
    )


def run_transformer_blocks(
    model: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep_proj: torch.Tensor,
    rotary_emb: tuple[torch.Tensor, torch.Tensor],
    *,
    block_kwargs: TransformerBlockKwargsProvider | None = None,
) -> torch.Tensor:
    """Run the shared block trunk with the model's checkpoint policy.

    ``block_kwargs`` is either one mapping shared by every layer or a callable
    that receives the layer index. Resolved non-tensor inputs are captured in a
    fresh closure so activation-checkpoint recomputation cannot observe mutated
    processor state.
    """

    def bind_forward(block: nn.Module, kwargs: dict[str, Any]):
        if not kwargs:
            return block

        def forward(
            states: torch.Tensor,
            context: torch.Tensor,
            modulation: torch.Tensor,
            rope: tuple[torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:
            return block(states, context, modulation, rope, **kwargs)

        return forward

    if torch.is_grad_enabled() and model.gradient_checkpointing:
        for index, block in enumerate(model.blocks):
            kwargs = resolve_transformer_block_kwargs(block_kwargs, index)
            hidden_states = model._gradient_checkpointing_func(
                bind_forward(block, kwargs),
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
            )
    else:
        for index, block in enumerate(model.blocks):
            kwargs = resolve_transformer_block_kwargs(block_kwargs, index)
            hidden_states = bind_forward(block, kwargs)(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
            )
    return hidden_states


def gather_ulysses_output(
    hidden_states: torch.Tensor,
    context: WanUlyssesContext,
) -> torch.Tensor:
    if not context.enabled:
        return hidden_states
    if context.backend == "context":
        from vforge.distributed.context_parallel import gather_sequence

        return gather_sequence(hidden_states, dim=1, group=context.group)
    from vforge.distributed.ulysses import gather_sequence

    assert context.full_seq_len is not None
    return gather_sequence(
        hidden_states,
        dim=1,
        group=context.group,
    ).narrow(1, 0, context.full_seq_len)


def project_output(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    *,
    scale_shift_table: torch.Tensor,
    norm_out: nn.Module,
    proj_out: nn.Module,
) -> torch.Tensor:
    """Apply scalar- or token-shaped output AdaLN and projection."""
    if temb.ndim == 3:
        shift, scale = (scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(
            2, dim=2
        )
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        shift, scale = (scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)
    hidden_states = (norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    return proj_out(hidden_states)


def unpatchify(
    hidden_states: torch.Tensor,
    *,
    batch_size: int,
    patch_grid: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    frames, height, width = patch_grid
    p_t, p_h, p_w = patch_size
    hidden_states = hidden_states.reshape(
        batch_size,
        frames,
        height,
        width,
        p_t,
        p_h,
        p_w,
        -1,
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


__all__ = [
    "WanRotaryPosEmbed",
    "WanUlyssesContext",
    "WanTransformerBlock",
    "apply_lora_scale",
    "gather_ulysses_output",
    "prepare_ulysses_inputs",
    "project_output",
    "run_transformer_blocks",
    "unpatchify",
    "wan_fsdp_precision_contract",
]
