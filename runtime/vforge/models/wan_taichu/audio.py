"""Wan Taichu audio tokens and speak/listen cross-attention.

The parameter names in this module intentionally match Taichu's Wan-S2V
branch.  In particular, the owning transformer registers
``casual_audio_encoder`` (including the historical typo) and
``audio_injector.injector_{s,l}`` so existing Application checkpoints can be
loaded without a model-specific key rewrite.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from vforge.ops import attention as vforge_attention


class CausalConv1d(nn.Module):
    """One-dimensional convolution padded only on the temporal left edge."""

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "replicate",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.pad_mode = pad_mode
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(
            hidden_states,
            self.time_causal_padding,
            mode=self.pad_mode,
        )
        return self.conv(hidden_states)


class MotionEncoder(nn.Module):
    """Compress frame-rate audio features into per-latent-frame token groups."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_heads: int,
        *,
        need_global: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"dtype": dtype, "device": device}
        self.num_heads = int(num_heads)
        self.need_global = bool(need_global)
        self.conv1_local = CausalConv1d(
            in_dim,
            hidden_dim // 4 * self.num_heads,
            3,
            stride=1,
        )
        if self.need_global:
            self.conv1_global = CausalConv1d(in_dim, hidden_dim // 4, 3, stride=1)
            self.final_linear = nn.Linear(hidden_dim, hidden_dim, **factory_kwargs)
        self.norm1 = nn.LayerNorm(
            hidden_dim // 4,
            elementwise_affine=False,
            eps=1e-6,
            **factory_kwargs,
        )
        self.norm2 = nn.LayerNorm(
            hidden_dim // 2,
            elementwise_affine=False,
            eps=1e-6,
            **factory_kwargs,
        )
        self.norm3 = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.conv2 = CausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = CausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)
        self.padding_tokens = nn.Parameter(
            torch.zeros(1, 1, 1, hidden_dim, dtype=dtype, device=device)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B,T,C] -> [B,C,T]
        hidden_states = hidden_states.permute(0, 2, 1)
        original = hidden_states.clone()
        batch_size = int(hidden_states.shape[0])

        hidden_states = self.conv1_local(hidden_states)
        _, _, frames = hidden_states.shape
        hidden_states = (
            hidden_states.view(batch_size, self.num_heads, -1, frames)
            .permute(0, 1, 3, 2)
            .reshape(batch_size * self.num_heads, frames, -1)
        )
        hidden_states = self.act(self.norm1(hidden_states))
        hidden_states = self.conv2(hidden_states.permute(0, 2, 1))
        hidden_states = self.act(self.norm2(hidden_states.permute(0, 2, 1)))
        hidden_states = self.conv3(hidden_states.permute(0, 2, 1))
        hidden_states = self.act(self.norm3(hidden_states.permute(0, 2, 1)))
        hidden_states = hidden_states.view(batch_size, self.num_heads, hidden_states.shape[1], -1)
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        padding = self.padding_tokens.expand(batch_size, hidden_states.shape[1], -1, -1)
        local = torch.cat([hidden_states, padding], dim=2)
        if not self.need_global:
            return local

        global_states = self.conv1_global(original)
        global_states = self.act(self.norm1(global_states.permute(0, 2, 1)))
        global_states = self.conv2(global_states.permute(0, 2, 1))
        global_states = self.act(self.norm2(global_states.permute(0, 2, 1)))
        global_states = self.conv3(global_states.permute(0, 2, 1))
        global_states = self.act(self.norm3(global_states.permute(0, 2, 1)))
        global_states = self.final_linear(global_states)
        return global_states, local


class CausalAudioEncoder(nn.Module):
    """Weighted Wav2Vec layer fusion followed by the causal motion encoder."""

    def __init__(
        self,
        dim: int = 1024,
        num_layers: int = 25,
        out_dim: int = 5120,
        num_token: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = MotionEncoder(
            in_dim=dim,
            hidden_dim=out_dim,
            num_heads=num_token,
            need_global=False,
        )
        self.weights = nn.Parameter(torch.full((1, num_layers, 1, 1), 0.01))
        self.act = nn.SiLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = self.act(self.weights)
        weighted = ((features * weights) / weights.sum(dim=1, keepdim=True)).sum(dim=1)
        return self.encoder(weighted.permute(0, 2, 1))


class AudioCrossAttention(nn.Module):
    """Checkpoint-compatible q/k/v/o audio cross-attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"audio dim={dim} must divide num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.o = nn.Linear(dim, dim, bias=True)
        if qk_norm:
            self.norm_q = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
            self.norm_k = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del context_lens  # Taichu supplies equal context lengths, hence no mask.
        batch, query_length, _ = hidden_states.shape
        context_length = int(context.shape[1])
        query = self.norm_q(self.q(hidden_states)).view(
            batch, query_length, self.num_heads, self.head_dim
        )
        key = self.norm_k(self.k(context)).view(
            batch, context_length, self.num_heads, self.head_dim
        )
        value = self.v(context).view(batch, context_length, self.num_heads, self.head_dim)
        output = vforge_attention(query, key, value, is_causal=False)
        return self.o(output.reshape(batch, query_length, self.dim))


class AudioInjectorWAN(nn.Module):
    """Own the independent speak and listen injector lists."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        inject_layers_s: Sequence[int],
        inject_layers_l: Sequence[int],
    ) -> None:
        super().__init__()
        self.block_id_s = {int(block): index for index, block in enumerate(inject_layers_s)}
        self.block_id_l = {int(block): index for index, block in enumerate(inject_layers_l)}
        self.injector_s = (
            nn.ModuleList(
                [AudioCrossAttention(dim, num_heads) for _ in range(len(self.block_id_s))]
            )
            if self.block_id_s
            else None
        )
        self.injector_l = (
            nn.ModuleList(
                [AudioCrossAttention(dim, num_heads) for _ in range(len(self.block_id_l))]
            )
            if self.block_id_l
            else None
        )

    def for_block(
        self,
        block_index: int,
    ) -> tuple[AudioCrossAttention | None, AudioCrossAttention | None]:
        speak = None
        listen = None
        speak_index = self.block_id_s.get(block_index)
        if speak_index is not None and self.injector_s is not None:
            speak = self.injector_s[speak_index]
        listen_index = self.block_id_l.get(block_index)
        if listen_index is not None and self.injector_l is not None:
            listen = self.injector_l[listen_index]
        return speak, listen

    @torch.no_grad()
    def zero_output_projections(self) -> None:
        for injectors in (self.injector_s, self.injector_l):
            if injectors is None:
                continue
            for injector in injectors:
                injector.o.weight.zero_()
                if injector.o.bias is not None:
                    injector.o.bias.zero_()


def build_local_audio_context(audio_emb: torch.Tensor, window: int) -> torch.Tensor:
    """Convert ``[B,T,N,C]`` audio tokens to per-frame local contexts."""
    batch, frames, tokens, channels = audio_emb.shape
    if window is None or window < 0:
        context = audio_emb.reshape(batch, frames * tokens, channels)
        return (
            context.unsqueeze(1)
            .expand(batch, frames, frames * tokens, channels)
            .reshape(batch * frames, frames * tokens, channels)
        )
    if window == 0:
        return audio_emb.reshape(batch * frames, tokens, channels)

    pad_left = audio_emb[:, :1].expand(batch, window, tokens, channels)
    pad_right = audio_emb[:, -1:].expand(batch, window, tokens, channels)
    padded = torch.cat([pad_left, audio_emb, pad_right], dim=1)
    window_length = 2 * window + 1
    windows = padded.unfold(1, window_length, 1).permute(0, 1, 4, 2, 3).contiguous()
    return windows.reshape(batch * frames, window_length * tokens, channels)


def audio_cross_attention_residual(
    hidden_states: torch.Tensor,
    *,
    audio_ctx_s: torch.Tensor | None,
    audio_ctx_l: torch.Tensor | None,
    audio_attn_s: AudioCrossAttention | None,
    audio_attn_l: AudioCrossAttention | None,
    num_latent_frames: int | None,
    spatial_tokens: int | None,
    usp_group: Any = None,
    usp_size: int = 1,
) -> torch.Tensor | None:
    """Apply speak/listen residuals to video tokens, including Ulysses."""
    if audio_attn_s is None and audio_attn_l is None:
        return None
    if audio_ctx_s is None and audio_ctx_l is None:
        return None
    if num_latent_frames is None or spatial_tokens is None:
        raise ValueError("audio injection requires latent frame and spatial token counts")

    if usp_group is not None and usp_size > 1:
        residual = None
        for attention, context in (
            (audio_attn_s, audio_ctx_s),
            (audio_attn_l, audio_ctx_l),
        ):
            if attention is None or context is None:
                continue
            value = _audio_attention_ulysses(
                attention,
                hidden_states,
                context,
                group=usp_group,
                world_size=usp_size,
                num_latent_frames=num_latent_frames,
                spatial_tokens=spatial_tokens,
            )
            residual = value if residual is None else residual + value
        return None if residual is None else residual.to(hidden_states.dtype)

    batch, _, channels = hidden_states.shape
    video = hidden_states.reshape(batch, num_latent_frames, spatial_tokens, channels)
    video = video.reshape(batch * num_latent_frames, spatial_tokens, channels)
    residual = None
    for attention, context in (
        (audio_attn_s, audio_ctx_s),
        (audio_attn_l, audio_ctx_l),
    ):
        if attention is None or context is None:
            continue
        value = attention(video, context)
        residual = value if residual is None else residual + value
    if residual is None:
        return None
    return residual.reshape(batch, num_latent_frames * spatial_tokens, channels).to(
        hidden_states.dtype
    )


def _audio_attention_ulysses(
    attention: AudioCrossAttention,
    hidden_states: torch.Tensor,
    context: torch.Tensor,
    *,
    group: Any,
    world_size: int,
    num_latent_frames: int,
    spatial_tokens: int,
) -> torch.Tensor:
    from vforge.distributed.ulysses import (
        head_to_sequence,
        pad_to_multiple,
        sequence_to_head,
    )

    batch, local_sequence, channels = hidden_states.shape
    heads = attention.num_heads
    head_dim = attention.head_dim
    full_sequence = local_sequence * world_size
    expected_sequence = num_latent_frames * spatial_tokens
    if full_sequence != expected_sequence:
        raise ValueError(
            "Wan Taichu audio with Ulysses does not permit sequence padding: "
            f"sharded sequence reconstructs {full_sequence} tokens, expected "
            f"T*S={num_latent_frames}*{spatial_tokens}={expected_sequence}"
        )

    query = attention.norm_q(attention.q(hidden_states)).view(
        batch, local_sequence, heads, head_dim
    )
    query = sequence_to_head(query, group, world_size)
    local_heads = query.shape[2]
    query = query.reshape(batch, num_latent_frames, spatial_tokens, local_heads, head_dim)
    query = query.reshape(batch * num_latent_frames, spatial_tokens, local_heads, head_dim)

    context_batch, context_length, _ = context.shape
    if context_batch != batch * num_latent_frames:
        raise ValueError(
            f"audio context batch={context_batch} must equal B*T={batch * num_latent_frames}"
        )
    key = attention.norm_k(attention.k(context)).view(
        context_batch, context_length, heads, head_dim
    )
    value = attention.v(context).view(context_batch, context_length, heads, head_dim)
    key, _ = pad_to_multiple(key, world_size, dim=2)
    value, _ = pad_to_multiple(value, world_size, dim=2)
    rank = dist.get_rank(group)
    head_slice = slice(rank * local_heads, (rank + 1) * local_heads)
    output = vforge_attention(
        query,
        key[:, :, head_slice, :].contiguous(),
        value[:, :, head_slice, :].contiguous(),
        is_causal=False,
    )
    output = output.reshape(batch, expected_sequence, local_heads, head_dim)
    output = head_to_sequence(
        output,
        group,
        world_size,
        original_heads=heads,
    ).reshape(batch, local_sequence, channels)
    return attention.o(output)


__all__ = [
    "AudioCrossAttention",
    "AudioInjectorWAN",
    "CausalAudioEncoder",
    "CausalConv1d",
    "MotionEncoder",
    "audio_cross_attention_residual",
    "build_local_audio_context",
]
