"""Two-level direct RGB DiT: global spatiotemporal blocks plus PiT blocks.

The coarse backbone and first-block CA belong to WorldDiT. Each PiT block
compresses the 256 local pixel tokens of a patch, attends across the 16x16
patch grid, then expands the result back to per-pixel features. The final
12-channel velocity is four ordered RGB frames, without neural decoding.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from vforge.models.rep_world.attention import SelfAttention
from vforge.models.rep_world.layers import RMSNorm, modulate


def _local_position(device, dtype, width):
    coordinates = torch.arange(16, device=device, dtype=torch.float32)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    frequencies = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * torch.arange(width // 4, device=device) / max(1, width // 4))
    angles_y = y.reshape(-1, 1) * frequencies
    angles_x = x.reshape(-1, 1) * frequencies
    return torch.cat((angles_y.sin(), angles_y.cos(), angles_x.sin(), angles_x.cos()), dim=-1).to(dtype)


class PixelPiTBlock(nn.Module):
    def __init__(self, pixel_width, patch_width, heads, qk_norm):
        super().__init__()
        local = 256 * pixel_width
        self.pixel_width = pixel_width
        self.condition = nn.Linear(patch_width, 6 * local)
        self.norm1 = RMSNorm(pixel_width)
        self.compress = nn.Linear(local, patch_width)
        self.attn = SelfAttention(patch_width, heads, qk_norm)
        self.expand = nn.Linear(patch_width, local)
        self.norm2 = RMSNorm(pixel_width)
        self.mlp = nn.Sequential(
            nn.Linear(pixel_width, 4 * pixel_width),
            nn.SiLU(),
            nn.Linear(4 * pixel_width, pixel_width),
        )

    def forward(self, pixels, semantic, rope):
        # pixels [B*patches,256,C], semantic [B*patches,D]
        batch_patches, count, width = pixels.shape
        if count != 256 or width != self.pixel_width or batch_patches % 256:
            raise ValueError("PixelPiT requires 16x16 patches of 16x16 pixels")
        shift1, scale1, gate1, shift2, scale2, gate2 = self.condition(semantic).reshape(
            batch_patches, count, 6, width
        ).unbind(2)
        compressed = self.compress(modulate(self.norm1(pixels), shift1, scale1).reshape(batch_patches, -1))
        coarse = compressed.reshape(batch_patches // 256, 256, -1)
        attended = self.attn(coarse, rope=rope).reshape(batch_patches, -1)
        pixels = pixels + gate1.tanh() * self.expand(attended).reshape(batch_patches, count, width)
        pixels = pixels + gate2.tanh() * self.mlp(modulate(self.norm2(pixels), shift2, scale2))
        return pixels


class PixelRefiner(nn.Module):
    def __init__(self, patch_width, heads, qk_norm, pixel_width, depth):
        super().__init__()
        self.pixel_width = pixel_width
        self.embed = nn.Linear(12, pixel_width)
        self.blocks = nn.ModuleList(
            PixelPiTBlock(pixel_width, patch_width, heads, qk_norm) for _ in range(depth)
        )
        self.norm = RMSNorm(pixel_width)
        self.output = nn.Linear(pixel_width, 12)

    def initialize_weights(self):
        for block in self.blocks:
            nn.init.zeros_(block.condition.weight)
            nn.init.zeros_(block.condition.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, packed, semantic, emb, rope, *, gradient_checkpointing=False):
        if (packed.ndim != 5 or packed.shape[1] != 3072
            or packed.shape[-2:] != (16, 16)
            or semantic.shape[1:] != (packed.shape[2] * 256, emb.shape[-1])):
            raise ValueError("Direct RGB PixelDiT needs 256 semantic tokens per 4-frame plane")
        batch, planes = packed.shape[0], packed.shape[2]
        noisy = packed.permute(0, 2, 3, 4, 1).reshape(batch * planes, 256, 4, 3, 16, 16)
        noisy = noisy.permute(0, 1, 4, 5, 2, 3).reshape(batch * planes * 256, 256, 12)
        pixels = self.embed(noisy)
        pixels = pixels + _local_position(pixels.device, pixels.dtype, self.pixel_width)[None]
        condition = F.silu(semantic + emb).reshape(batch * planes * 256, -1)
        for block in self.blocks:
            if gradient_checkpointing and self.training and torch.is_grad_enabled():
                pixels = checkpoint(block, pixels, condition, rope, use_reentrant=False)
            else:
                pixels = block(pixels, condition, rope)
        velocity = self.output(self.norm(pixels))
        velocity = velocity.reshape(batch, planes, 256, 16, 16, 4, 3)
        velocity = velocity.permute(0, 1, 2, 5, 6, 3, 4)
        return velocity.reshape(batch, planes, 16, 16, 3072).permute(0, 4, 1, 2, 3)
