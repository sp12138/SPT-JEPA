"""Frozen V-RAE feature fusion, separate from the trainable causal conditioner."""

from __future__ import annotations

import torch
from torch import nn

from vforge.models.rep_world.conditioner import CausalAttention, CausalViTBlock
from vforge.models.rep_world.config import ConditionerConfig
from vforge.models.rep_world.layers import PatchEmbed3D


class FrozenVJEPA21Attention(CausalAttention):
    def rotate_axis(self, value, position):
        # The frozen official encoder differs from rep_miniworld's trainable
        # causal adapter: frequencies are calculated in Q's dtype and the
        # sin/cos multiplication promotes to position dtype. Keep both rules.
        width = value.shape[-1]
        omega = torch.arange(width // 2, device=value.device, dtype=value.dtype)
        omega /= width / 2.0
        omega = 1.0 / 10000**omega
        frequency = torch.einsum("...,f->...f", position, omega)
        sine, cosine = (
            frequency.sin().repeat_interleave(2, -1),
            frequency.cos().repeat_interleave(2, -1),
        )
        first, second = value.unflatten(-1, (-1, 2)).unbind(-1)
        return value * cosine + torch.stack((-second, first), -1).flatten(-2) * sine


class VJEPA21Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        config = ConditionerConfig(width=1024, depth=24, heads=16, feature_layers=(23,))
        self.patch_embed = PatchEmbed3D((4, 256, 256), (2, 16, 16), 3, 1024)
        self.video_mod_embed = nn.Parameter(torch.empty(1, 1, 1024))
        self.blocks = nn.ModuleList(CausalViTBlock(config) for _ in range(config.depth))
        for block in self.blocks:
            block.attn = FrozenVJEPA21Attention(config.width, config.heads)
        self.norms_block = nn.ModuleList(nn.LayerNorm(1024, eps=1e-6) for _ in range(4))

    def forward(self, video):
        batch, _, time, height, width = video.shape
        time, height, width = time // 2, height // 16, width // 16
        hidden = self.patch_embed(video) + self.video_mod_embed
        positions = tuple(
            p.flatten().float()
            for p in torch.meshgrid(
                torch.arange(time, device=video.device),
                torch.arange(height, device=video.device),
                torch.arange(width, device=video.device),
                indexing="ij",
            )
        )
        # V-JEPA2.1 interpolates from a 16x16 reference grid, not 24x24.
        positions = (
            positions[0],
            positions[1] * (15 / max(height - 1, 1)),
            positions[2] * (15 / max(width - 1, 1)),
        )
        selected = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, mask=None, positions=positions)
            if index in (11, 13, 15, 17, 19, 21, 23):
                selected.append(self.norms_block[-1](hidden))
        fused = torch.stack(selected).mean(0).reshape(batch, time, height * width, 1024)
        final_mean = selected[-1].reshape_as(fused).mean(2, keepdim=True)
        return (fused + final_mean).reshape(batch, time, height, width, 1024).permute(0, 1, 4, 2, 3)


class FrozenFeatureEncoder(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        if variant == "eupe":
            from vforge.models.rep_world.codecs.eupe import EUPEBackbone

            self.backbone = EUPEBackbone()
        elif variant == "vjepa2_1":
            self.backbone = VJEPA21Backbone()
        else:
            raise ValueError(f"Unsupported V-RAE encoder: {variant}")

    def forward(self, video):
        if video.ndim != 5 or video.shape[2] != 3 or video.shape[1] % 4:
            raise ValueError("V-RAE encoder expects [B,4*K,3,H,W] unit-range RGB")
        dtype = next(self.parameters()).dtype
        # Match official adapter normalization after its model-wide dtype cast.
        pixels = video.float()
        mean = pixels.new_tensor((0.485, 0.456, 0.406), dtype=dtype).view(1, 1, 3, 1, 1)
        std = pixels.new_tensor((0.229, 0.224, 0.225), dtype=dtype).view(1, 1, 3, 1, 1)
        pixels = ((pixels - mean) / std).to(dtype)
        if self.variant == "vjepa2_1":
            return self.backbone(pixels.permute(0, 2, 1, 3, 4).contiguous())
        batch, time, _, height, width = pixels.shape
        features = self.backbone(pixels.flatten(0, 1))
        return features.reshape(batch, time, height // 16, width // 16, -1).permute(0, 1, 4, 2, 3)
