# Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Standalone MiniMax-H3 transformer using vforge attention."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vforge.distributed.parallel_state import get_parallel_state
from vforge.distributed.parallelize import FSDPPrecisionContract
from vforge.ops import attention as vforge_attention

MINIMAX_H3_MODALITY_NUM = 3


@torch.no_grad()
def _trace_finite(name: str, tensor: torch.Tensor) -> None:
    if os.environ.get("VFORGE_H3_FINITE_TRACE") != "1":
        return
    if int(os.environ.get("RANK", "0")) != 0:
        return
    local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
    values = local.detach()
    finite = torch.isfinite(values)
    finite_count = int(finite.sum().item())
    finite_values = values[finite].float()
    max_abs = float(finite_values.abs().max().item()) if finite_count else float("nan")
    value_sum = (
        float(finite_values.sum(dtype=torch.float64).item()) if finite_count else float("nan")
    )
    square_sum = (
        float(finite_values.square().sum(dtype=torch.float64).item())
        if finite_count
        else float("nan")
    )
    print(
        "VFORGE_CKPT H3_NATIVE_FINITE "
        + json.dumps(
            {
                "finite": finite_count,
                "max_abs": max_abs,
                "name": name,
                "numel": values.numel(),
                "square_sum": square_sum,
                "sum": value_sum,
            },
            sort_keys=True,
        ),
        flush=True,
    )


class _Config(dict):
    """Dictionary config with Diffusers-compatible attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass
class MiniMaxH3TransformerOutput:
    sample: torch.Tensor
    audio_sample: torch.Tensor


def _apply_rotary_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    rotary_dim = cos.shape[-1]
    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]
    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    x1, x2 = hidden_states_rotary.chunk(2, dim=-1)
    hidden_states_rotated = torch.cat((-x2, x1), dim=-1)
    hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
    return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()


class SwiGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * self.activation(gate)


class FeedForward(nn.Module):
    """The checkpoint-compatible Diffusers SwiGLU feed-forward layout."""

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "swiglu",
        final_dropout: bool = False,
        inner_dim: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        if activation_fn != "swiglu":
            raise ValueError(f"MiniMax-H3 only supports swiglu, got {activation_fn!r}")
        inner_dim = int(dim * mult) if inner_dim is None else inner_dim
        dim_out = dim if dim_out is None else dim_out
        self.net = nn.ModuleList(
            [
                SwiGLU(dim, inner_dim, bias=bias),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim_out, bias=bias),
            ]
        )
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


def _get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        0, half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    emb = scale * emb
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
    if embedding_dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return _get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int, out_dim: int | None = None):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, True)
        self.cond_proj = None
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, out_dim or time_embed_dim, True)
        self.post_act = None

    def forward(
        self,
        sample: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if condition is not None:
            if self.cond_proj is None:
                raise ValueError("condition was provided without a conditioning projection")
            sample = sample + self.cond_proj(condition)
        sample = self.linear_2(self.act(self.linear_1(sample)))
        return sample


class MiniMaxH3RotaryPosEmbed(nn.Module):
    def __init__(self, rope_freq_dim: int = 16, rope_theta: float = 10000.0):
        super().__init__()
        self.rope_freq_dim = rope_freq_dim
        self.rope_theta = rope_theta
        self.register_buffer("inv_freq", self._make_inv_freq(), persistent=False)

    def _make_inv_freq(self) -> torch.Tensor:
        return 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, 2 * self.rope_freq_dim, 2, dtype=torch.float32)
                / (2 * self.rope_freq_dim)
            )
        )

    def reset_non_persistent_buffers(self) -> None:
        self.inv_freq = self._make_inv_freq().to(
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = position_ids.to(torch.float32).unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        freqs = torch.cat(freqs.unbind(dim=1), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    def __init__(self, time_embed_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(
            time_embed_dim,
            6 * hidden_size * MINIMAX_H3_MODALITY_NUM,
            bias=True,
        )

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        temb = self.linear(nn.functional.silu(temb).to(self.linear.weight.dtype))
        return temb.view(-1, 6 * self.hidden_size).chunk(6, dim=-1)


class MiniMaxH3AdaLayerNormOut(nn.Module):
    def __init__(self, hidden_size: int, time_embed_dim: int, eps: float):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.linear(nn.functional.silu(temb).to(self.linear.weight.dtype)).chunk(
            2, dim=-1
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )


class MiniMaxH3VforgeAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self) -> None:
        self._pad_attn_mask: torch.Tensor | None = None
        self._seq_real_len: int | None = None
        self._sequence_parallel = False

    def _resolve_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if self._seq_real_len is not None:
            return None
        return attention_mask if attention_mask is not None else self._pad_attn_mask

    def __call__(
        self,
        attn: MiniMaxH3Attention,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)
        if getattr(self, "_debug_name", "") == "block_0":
            _trace_finite("block_0.query", query)
            _trace_finite("block_0.key", key)
            _trace_finite("block_0.value", value)
        hidden_states = vforge_attention(
            query,
            key,
            value,
            attn_mask=self._resolve_attention_mask(attention_mask),
            is_causal=False,
            ulysses=self._sequence_parallel,
            seq_real_len=self._seq_real_len,
        )
        if getattr(self, "_debug_name", "") == "block_0":
            _trace_finite("block_0.attention_output", hidden_states)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


class MiniMaxH3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dim_head: int,
        qk_norm_eps: float = 1e-5,
        processor: Any = None,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = heads * dim_head
        self.use_bias = False
        self.fused_projections = False
        self.to_q = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, hidden_size, bias=False), nn.Dropout(0.0)]
        )
        self.set_processor(processor or MiniMaxH3VforgeAttnProcessor())

    def set_processor(self, processor: Any) -> None:
        self.processor = processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, rotary_emb, attention_mask)


class MiniMaxH3TokenRefinerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size, num_attention_heads, attention_head_dim, qk_norm_eps
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = FeedForward(hidden_size, inner_dim=ffn_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.ff(self.norm2(hidden_states))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
    ):
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    ffn_dim,
                    norm_eps,
                    qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(hidden_size, eps=final_norm_eps)
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func: Callable[..., torch.Tensor] | None = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.refiner_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                assert self._gradient_checkpointing_func is not None
                hidden_states = self._gradient_checkpointing_func(block, hidden_states)
            else:
                hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size, num_attention_heads, attention_head_dim, qk_norm_eps
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.ff = FeedForward(hidden_size, inner_dim=ffn_dim, bias=False)
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(time_embed_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
        residual = hidden_states
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_msa.index_select(0, adaln_indices)
        ) + shift_msa.index_select(0, adaln_indices)
        hidden_states = residual + gate_msa.index_select(0, adaln_indices) * self.attn(
            norm_hidden_states, rotary_emb, attention_mask
        )
        residual = hidden_states
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (
            1.0 + scale_mlp.index_select(0, adaln_indices)
        ) + shift_mlp.index_select(0, adaln_indices)
        return residual + gate_mlp.index_select(0, adaln_indices) * self.ff(norm_hidden_states)


class MiniMaxH3Transformer3DModel(nn.Module):
    config_name = "config.json"
    _supports_gradient_checkpointing = True
    _no_split_modules = [
        "MiniMaxH3TransformerBlock",
        "MiniMaxH3TokenRefinerBlock",
        "MiniMaxH3AdaLayerNormOut",
    ]
    _repeated_blocks = ["MiniMaxH3TransformerBlock", "MiniMaxH3TokenRefinerBlock"]
    _skip_layerwise_casting_patterns = ["norm"]
    _keep_in_fp32_modules = [
        "proj_in",
        "audio_proj_in",
        "time_embedder",
        "proj_out",
        "audio_proj_out",
        "rope",
    ]

    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        config_values = {
            name: value for name, value in locals().items() if name not in {"self", "__class__"}
        }
        self.config = _Config(config_values)
        self._gradient_checkpointing_func: Callable[..., torch.Tensor] | None = None
        video_patch_dim = in_channels * math.prod(patch_size)
        self.proj_in = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_proj_in = nn.Linear(audio_in_channels, hidden_size, bias=True)
        self.context_embedder = nn.Linear(text_dim, hidden_size, bias=True)
        self.time_proj = Timesteps(freq_dim, True, 0)
        self.time_embedder = TimestepEmbedding(
            freq_dim, time_embed_hidden_dim, out_dim=time_embed_dim
        )
        self.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim, rope_theta)
        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size,
            num_attention_heads,
            attention_head_dim,
            ffn_dim,
            num_refiner_layers,
            norm_eps,
            qk_norm_eps,
            final_norm_eps,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3TransformerBlock(
                    hidden_size,
                    num_attention_heads,
                    attention_head_dim,
                    ffn_dim,
                    time_embed_dim,
                    norm_eps,
                    qk_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        for block_index, block in enumerate(self.transformer_blocks):
            block.attn.processor._debug_name = f"block_{block_index}"
        self.norm_out = MiniMaxH3AdaLayerNormOut(hidden_size, time_embed_dim, final_norm_eps)
        self.proj_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_proj_out = nn.Linear(hidden_size, audio_in_channels, bias=True)
        self.gradient_checkpointing = False

    def get_fsdp_precision_contract(self) -> FSDPPrecisionContract:
        """Declare H3's auxiliary-input precision boundaries.

        The preserve modules come from ``_keep_in_fp32_modules``, which already
        names them for storage dtype.
        """
        return FSDPPrecisionContract(
            preserve_forward_input_paths=(
                ".",
                "norm_out",
                *(f"transformer_blocks.{index}" for index in range(len(self.transformer_blocks))),
            ),
        )

    @classmethod
    def load_config(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        subfolder: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        path = Path(pretrained_model_name_or_path)
        if path.is_dir():
            candidate = path / subfolder / cls.config_name if subfolder else path / cls.config_name
            if not candidate.is_file() and subfolder:
                candidate = path / cls.config_name
        elif path.is_file():
            candidate = path
        else:
            from huggingface_hub import hf_hub_download

            candidate = Path(
                hf_hub_download(
                    str(pretrained_model_name_or_path),
                    filename=cls.config_name,
                    subfolder=subfolder,
                    **kwargs,
                )
            )
        if not candidate.is_file():
            raise OSError(f"no {cls.config_name} found under {path}")
        with candidate.open(encoding="utf-8") as handle:
            return json.load(handle)

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enable
                module._gradient_checkpointing_func = gradient_checkpointing_func

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> None:
        checkpoint_kwargs = (
            {"use_reentrant": False}
            if gradient_checkpointing_kwargs is None
            else dict(gradient_checkpointing_kwargs)
        )
        gradient_checkpointing_func = partial(
            torch.utils.checkpoint.checkpoint,
            **checkpoint_kwargs,
        )
        self._set_gradient_checkpointing(True, gradient_checkpointing_func)

    def disable_gradient_checkpointing(self) -> None:
        self._set_gradient_checkpointing(False, None)

    @property
    def is_gradient_checkpointing(self) -> bool:
        return any(getattr(module, "gradient_checkpointing", False) for module in self.modules())

    def cache_context(self, name: str):
        del name
        from contextlib import nullcontext

        return nullcontext()

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor]:
        del attention_kwargs
        if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
            raise ValueError(
                f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}."
            )
        sequence_length = position_ids.shape[0]
        if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
            raise ValueError(
                "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors "
                f"matching `position_ids`, got {list(token_tags.shape)} and "
                f"{list(timestep_indices.shape)} for seq_len={sequence_length}."
            )
        rotary_emb = self.rope(position_ids)
        _trace_finite("rope.inv_freq", self.rope.inv_freq)
        _trace_finite("rotary.cos", rotary_emb[0])
        _trace_finite("rotary.sin", rotary_emb[1])
        video_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
        text_embeds = self.context_embedder(
            encoder_hidden_states.to(self.context_embedder.weight.dtype)
        )
        text_embeds = self.token_refiner(text_embeds)
        _trace_finite("proj_out.weight", self.proj_out.weight)
        _trace_finite("audio_proj_out.weight", self.audio_proj_out.weight)
        packed = text_embeds.new_zeros(
            (text_embeds.shape[0], sequence_length, text_embeds.shape[-1])
        )
        packed = packed.index_copy(1, text_indices, text_embeds)
        packed = packed.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        packed = packed.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))
        _trace_finite("packed.before_blocks", packed)
        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))
        _trace_finite("temb", temb)
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags

        state = get_parallel_state()
        sp_size = int(state.config.usp_size)
        sp_group = state.usp_group
        full_seq_len = sequence_length
        if sp_size > 1:
            from vforge.distributed.ulysses import (
                gather_sequence,
                pad_to_multiple,
                slice_sequence,
            )

            packed, _ = pad_to_multiple(packed, sp_size, dim=1)
            cos, sin = rotary_emb
            cos, _ = pad_to_multiple(cos, sp_size, dim=0)
            sin, _ = pad_to_multiple(sin, sp_size, dim=0)
            adaln_indices, _ = pad_to_multiple(adaln_indices, sp_size, dim=0)
            timestep_indices, _ = pad_to_multiple(timestep_indices, sp_size, dim=0)
            packed = slice_sequence(packed, dim=1, group=sp_group, scale_grad=True)
            rotary_emb = (
                slice_sequence(cos, dim=0, group=sp_group),
                slice_sequence(sin, dim=0, group=sp_group),
            )
            adaln_indices = slice_sequence(adaln_indices, dim=0, group=sp_group)
            timestep_indices = slice_sequence(timestep_indices, dim=0, group=sp_group)
            for block in self.transformer_blocks:
                block.attn.processor._seq_real_len = full_seq_len
                block.attn.processor._sequence_parallel = True
        else:
            gather_sequence = None

        for block_index, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                assert self._gradient_checkpointing_func is not None
                packed = self._gradient_checkpointing_func(
                    block, packed, temb, adaln_indices, rotary_emb
                )
            else:
                packed = block(packed, temb, adaln_indices, rotary_emb)
            _trace_finite(f"packed.block_{block_index}", packed)
        packed = self.norm_out(packed, temb, timestep_indices).to(self.proj_out.weight.dtype)
        video_output = self.proj_out(packed)
        audio_output = self.audio_proj_out(packed)
        if sp_size > 1:
            assert gather_sequence is not None
            video_output = gather_sequence(video_output, dim=1, group=sp_group)
            audio_output = gather_sequence(audio_output, dim=1, group=sp_group)
            video_output = video_output.narrow(1, 0, full_seq_len)
            audio_output = audio_output.narrow(1, 0, full_seq_len)
        video_output = video_output.index_select(1, video_indices)
        audio_output = audio_output.index_select(1, audio_indices)
        _trace_finite("video_output", video_output)
        _trace_finite("audio_output", audio_output)
        if not return_dict:
            return video_output, audio_output
        return MiniMaxH3TransformerOutput(
            sample=video_output,
            audio_sample=audio_output,
        )


__all__ = [
    "MiniMaxH3Transformer3DModel",
    "MiniMaxH3TransformerOutput",
]
