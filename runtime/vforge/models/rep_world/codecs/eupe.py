# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from EUPE for frozen V-RAE inference using Native attention.
# FAIR Noncommercial Research License; see licenses/EUPE-LICENSE.md.
"""Released EUPE-B/16 encoder only, without hub, pretraining or projector code."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from vforge.models.rep_world.conditioner import ViTMLP
from vforge.ops import attention


class MaskedKeyBiasLinear(nn.Linear):
    def __init__(self, dim):
        super().__init__(dim, dim * 3)
        mask = torch.ones(dim * 3)
        mask[dim : 2 * dim] = 0
        self.register_buffer("bias_mask", mask)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias * self.bias_mask.to(self.bias.dtype))


class EUPERope(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.register_buffer(
            "periods",
            100.0 ** (2 * torch.arange(head_dim // 4, dtype=torch.float32) / (head_dim // 2)),
        )

    def forward(self, height, width):
        # Official eval uses fp32 coordinates even after model.to(bfloat16).
        h = torch.arange(0.5, height, dtype=torch.float32, device=self.periods.device) / height
        w = torch.arange(0.5, width, dtype=torch.float32, device=self.periods.device) / width
        coords = torch.stack(torch.meshgrid(h, w, indexing="ij"), -1).flatten(0, 1)
        coords = 2 * coords - 1
        angles = (2 * math.pi * coords[:, :, None] / self.periods[None, None]).flatten(1, 2).tile(2)
        return angles.sin(), angles.cos()


class EUPEAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = MaskedKeyBiasLinear(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope):
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.heads, dim // self.heads).unbind(2)
        sine, cosine = (t[:, None] for t in rope)
        prefix = tokens - sine.shape[0]

        def rotate(value):
            original = value.dtype
            value = value.to(sine.dtype)
            patch = value[:, prefix:]
            first, second = patch.chunk(2, -1)
            patch = patch * cosine + torch.cat((-second, first), -1) * sine
            return torch.cat((value[:, :prefix], patch), 1).to(original)

        return self.proj(attention(rotate(q), rotate(k), v).reshape(batch, tokens, dim))


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), 1e-5))

    def forward(self, x):
        return x * self.gamma


class EUPEBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-5), nn.LayerNorm(dim, eps=1e-5)
        self.attn = EUPEAttention(dim, heads)
        self.ls1, self.ls2 = LayerScale(dim), LayerScale(dim)
        self.mlp = ViTMLP(dim, 4)

    def forward(self, x, rope):
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


class EUPEBackbone(nn.Module):
    def __init__(self, *, width=768, depth=12, heads=12, patch=16, layers=(5, 6, 7, 8, 9, 10, 11)):
        super().__init__()
        if width % (4 * heads) or not layers or min(layers) < 0 or max(layers) >= depth:
            raise ValueError("Invalid EUPE feature geometry")
        self.layers = layers
        self.cls_token = nn.Parameter(torch.empty(1, 1, width))
        self.storage_tokens = nn.Parameter(torch.empty(1, 4, width))
        self.mask_token = nn.Parameter(torch.empty(1, width))
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, width, patch, stride=patch)
        self.rope_embed = EUPERope(width // heads)
        self.blocks = nn.ModuleList(EUPEBlock(width, heads) for _ in range(depth))
        # V-RAE intentionally discards the pretrained affine final norm.
        self.norm = nn.LayerNorm(width, eps=1e-5, elementwise_affine=False)

    def forward(self, frames):
        patches = self.patch_embed.proj(frames)
        height, width = patches.shape[-2:]
        patches = patches.flatten(2).transpose(1, 2)
        hidden = torch.cat(
            (
                self.cls_token.expand(frames.shape[0], -1, -1),
                self.storage_tokens.expand(frames.shape[0], -1, -1),
                patches,
            ),
            1,
        )
        rope = self.rope_embed(height, width)
        outputs = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, rope)
            if index in self.layers:
                outputs.append(self.norm(hidden)[:, 5:])
        return torch.stack(outputs).sum(0)
