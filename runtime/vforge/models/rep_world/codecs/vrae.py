"""V-RAE pooling and chunk-causal reconstruction, adapted for Native ops.

Architecture/weight names follow V-RAE (see THIRD_PARTY.md). No upstream
registry, checkpoint manager, attention dispatcher, or training code is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from vforge.models.rep_world.conditioner import rotary
from vforge.ops import attention


class TemporalAttentionPool(nn.Module):
    """Pool only along time at each spatial location, preserving checkpoint keys."""

    def __init__(self, dim, group_size, num_heads=16, use_time_bias=True):
        super().__init__()
        if group_size <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("Invalid temporal attention pool geometry")
        self.dim, self.group_size, self.num_heads = dim, group_size, num_heads
        self.head_dim = dim // num_heads
        self.norm_k = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))
        self.key, self.value, self.proj = (nn.Linear(dim, dim) for _ in range(3))
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False)
        self.time_bias = nn.Parameter(torch.zeros(num_heads, group_size)) if use_time_bias else None

    def forward(self, x):
        if x.ndim != 5 or x.shape[2] != self.dim or x.shape[1] % self.group_size:
            raise ValueError("Pool expects [B,T,C,H,W] with complete temporal groups")
        batch, time, channels, height, width = x.shape
        length = time // self.group_size
        grouped = x.reshape(batch, length, self.group_size, channels, height, width)
        grouped = grouped.permute(0, 1, 4, 5, 2, 3).reshape(-1, self.group_size, channels)
        shape = (-1, self.group_size, self.num_heads, self.head_dim)
        key = self.key(self.norm_k(grouped)).reshape(shape).permute(0, 2, 1, 3)
        value = self.value(grouped).reshape(shape).permute(0, 2, 1, 3)
        # This is a learned single-query weighted reduction, not a second
        # attention backend. Preserve the official elementwise reduction order.
        logits = (self.query * key).sum(-1) * self.head_dim**-0.5
        if self.time_bias is not None:
            logits = logits + self.time_bias.to(logits)[None]
        pooled = (logits.softmax(-1).unsqueeze(-1) * value).sum(2).reshape(-1, channels)
        pooled = self.norm_out(self.proj(pooled))
        return pooled.reshape(batch, length, height, width, channels).permute(0, 1, 4, 2, 3)


@dataclass(frozen=True)
class VRAEDecoderConfig:
    input_dim: int = 768
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    patch_size: int = 16
    tubelet_size: int = 4
    image_size: tuple[int, int] = (256, 256)
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    attention_mode: str = "chunk_causal"

    def __post_init__(self):
        if self.attention_mode not in {"chunk_causal", "full"}:
            raise ValueError("Unsupported V-RAE decoder attention mode")
        if min(self.input_dim, self.hidden_size, self.depth, self.num_heads) <= 0:
            raise ValueError("V-RAE decoder dimensions must be positive")
        if self.hidden_size % self.num_heads or self.hidden_size // self.num_heads < 6:
            raise ValueError("V-RAE decoder width/head geometry is invalid")
        if self.tubelet_size != 4 or self.patch_size <= 0:
            raise ValueError("Released V-RAE decoder requires temporal compression 4")
        if len(self.image_size) != 2 or any(s % self.patch_size for s in self.image_size):
            raise ValueError("V-RAE resolution must be divisible by spatial patch")


class DecoderAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, *, positions, tokens_per_state, past=None):
        batch, tokens, width = x.shape
        heads, head_dim = self.config.num_heads, width // self.config.num_heads
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
        axis_dim = 2 * ((head_dim // 3) // 2)

        def rotate(value):
            parts = [
                rotary(value[..., i * axis_dim : (i + 1) * axis_dim], p)
                for i, p in enumerate(positions)
            ]
            return torch.cat((*parts, value[..., 3 * axis_dim :]), -1).transpose(1, 2)

        q, k, v = rotate(q), rotate(k), v.transpose(1, 2)
        offset = 0
        if past is not None:
            offset = past[0].shape[1]
            k, v = torch.cat((past[0], k), 1), torch.cat((past[1], v), 1)
        # Per-chunk prefix calls match the released decoder and avoid a large
        # dense causal mask. Each chunk still has full within-state attention.
        if self.config.attention_mode == "chunk_causal":
            result = torch.cat(
                [
                    attention(
                        q[:, start : start + tokens_per_state],
                        k[:, : offset + start + tokens_per_state],
                        v[:, : offset + start + tokens_per_state],
                    )
                    for start in range(0, tokens, tokens_per_state)
                ],
                1,
            )
        else:
            if past is not None:
                raise ValueError("Full-attention V-RAE decoder cannot stream a causal prefix")
            result = attention(q, k, v)
        return self.out(result.reshape(batch, tokens, width)), (k, v)


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = DecoderAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, int(config.hidden_size * config.mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(config.hidden_size * config.mlp_ratio), config.hidden_size),
        )

    def forward(self, x, **kwargs):
        output, cache = self.attention(self.norm1(x), **kwargs)
        x = x + output
        return x + self.mlp(self.norm2(x)), cache


class VRAEDecoder(nn.Module):
    def __init__(self, config: VRAEDecoderConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.hidden_size)
        self.nominal_grid = tuple(s // config.patch_size for s in config.image_size)
        self.decoder_spatial_pos_embed = nn.Parameter(
            torch.empty(1, self.nominal_grid[0] * self.nominal_grid[1], config.hidden_size)
        )
        self.blocks = nn.ModuleList(DecoderBlock(config) for _ in range(config.depth))
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.prediction = nn.Linear(
            config.hidden_size, config.tubelet_size * config.patch_size**2 * config.num_channels
        )

    def _position(self, height, width, reference):
        position = self.decoder_spatial_pos_embed
        if (height, width) != self.nominal_grid:
            position = position.reshape(1, *self.nominal_grid, -1).permute(0, 3, 1, 2)
            position = F.interpolate(
                position.float(), size=(height, width), mode="bicubic", align_corners=False
            ).to(self.decoder_spatial_pos_embed.dtype)
            position = position.permute(0, 2, 3, 1).reshape(1, height * width, -1)
        return position.to(reference)

    def forward(self, latents, *, past_key_values=None, return_key_values=False):
        if latents.ndim != 5 or latents.shape[2] != self.config.input_dim:
            raise ValueError("V-RAE decoder expects [B,states,C,H,W]")
        batch, states, channels, height, width = latents.shape
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("V-RAE cache depth mismatch")
        previous = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        if previous % (height * width):
            raise ValueError("V-RAE cache must contain complete states")
        hidden = latents.permute(0, 1, 3, 4, 2).reshape(batch, states * height * width, channels)
        hidden = self.input_projection(hidden).reshape(batch, states, height * width, -1)
        hidden = (hidden + self._position(height, width, hidden)[:, None]).flatten(1, 2)
        positions = tuple(
            t.flatten()
            for t in torch.meshgrid(
                torch.arange(
                    previous // (height * width),
                    previous // (height * width) + states,
                    device=hidden.device,
                ),
                torch.arange(height, device=hidden.device),
                torch.arange(width, device=hidden.device),
                indexing="ij",
            )
        )
        caches = []
        for index, block in enumerate(self.blocks):
            hidden, cache = block(
                hidden,
                positions=positions,
                tokens_per_state=height * width,
                past=None if past_key_values is None else past_key_values[index],
            )
            caches.append(cache)
        cfg = self.config
        predicted = self.prediction(self.norm(hidden))
        pixels = predicted.reshape(
            batch,
            states,
            height,
            width,
            cfg.tubelet_size,
            cfg.patch_size,
            cfg.patch_size,
            cfg.num_channels,
        )
        pixels = pixels.permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(
            batch,
            states * cfg.tubelet_size,
            cfg.num_channels,
            height * cfg.patch_size,
            width * cfg.patch_size,
        )
        return (pixels, caches) if return_key_values else pixels


class VRAE(nn.Module):
    def __init__(self, encoder, temporal_pool, decoder):
        super().__init__()
        self.encoder, self.temporal_pool, self.decoder = encoder, temporal_pool, decoder
        self.eval().requires_grad_(False)

    def encode(self, video):
        return self.temporal_pool(self.encoder(video))

    def decode(self, latents, **kwargs):
        return self.decoder(latents, **kwargs)
