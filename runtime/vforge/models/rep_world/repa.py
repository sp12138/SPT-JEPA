"""Student-side temporal REPA projection and frozen V-JEPA 2.1 video teacher."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from vforge.models.rep_world.codecs.encoders import FrozenVJEPA21Attention
from vforge.models.rep_world.conditioner import CausalViTBlock, preprocess_rgb
from vforge.models.rep_world.config import ConditionerConfig
from vforge.models.rep_world.layers import PatchEmbed3D


class TemporalREPAProjector(nn.Module):
    """Map each four-frame student token to two ordered teacher tubelets."""

    def __init__(self, student_width: int, teacher_width: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            student_width, teacher_width, kernel_size=(2, 1, 1), stride=(2, 1, 1)
        )
        self.local = nn.Conv3d(
            teacher_width, teacher_width, kernel_size=3, padding=1,
            groups=teacher_width,
        )

    def forward(self, hidden: torch.Tensor, planes: int, grid: int = 16):
        if hidden.ndim != 3 or hidden.shape[1] != planes * grid * grid:
            raise ValueError("REPA hidden tokens must match the target spatiotemporal grid")
        batch, _, width = hidden.shape
        volume = hidden.transpose(1, 2).reshape(batch, width, planes, grid, grid)
        up = self.up(volume)
        return up + self.local(F.silu(up))


class VJEPA21VideoTeacher(nn.Module):
    """Official ViT-B EMA video encoder with final normalized patch tokens."""

    def __init__(self):
        super().__init__()
        cfg = ConditionerConfig(width=768, depth=12, heads=12, feature_layers=(11,))
        self.patch_embed = PatchEmbed3D((16, 256, 256), (2, 16, 16), 3, 768)
        self.video_mod_embed = nn.Parameter(torch.empty(1, 1, 768))
        self.blocks = nn.ModuleList(CausalViTBlock(cfg) for _ in range(12))
        for block in self.blocks:
            block.attn = FrozenVJEPA21Attention(768, 12)
        self.norm = nn.LayerNorm(768, eps=1e-6)

    @classmethod
    def from_checkpoint(cls, path: str, device, dtype=torch.bfloat16):
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        source = payload.get("ema_encoder") if isinstance(payload, Mapping) else None
        if not isinstance(source, Mapping):
            raise ValueError("V-JEPA 2.1 checkpoint lacks ema_encoder")
        with torch.device("meta"):
            model = cls()
        prefix = "module.backbone."
        state = {}
        for key, value in source.items():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            if name == "video_mod_embed" or name.startswith("patch_embed.") or name.startswith("blocks."):
                state[name] = value
            elif name.startswith("norms_block.3."):
                state["norm." + name.split(".")[-1]] = value
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    def forward(self, video: torch.Tensor):
        if (video.ndim != 5 or video.shape[1] != 3 or video.shape[2] % 2
            or video.shape[-2:] != (256, 256)):
            raise ValueError("V-JEPA teacher expects [B,3,2*T,256,256]")
        batch, _, frames, height, width = video.shape
        time, grid_h, grid_w = frames // 2, height // 16, width // 16
        pixels = preprocess_rgb(video, 256, self.patch_embed.proj.weight.dtype)
        hidden = self.patch_embed(pixels) + self.video_mod_embed
        positions = tuple(
            axis.flatten().float()
            for axis in torch.meshgrid(
                torch.arange(time, device=video.device),
                torch.arange(grid_h, device=video.device),
                torch.arange(grid_w, device=video.device),
                indexing="ij",
            )
        )
        positions = (
            positions[0],
            positions[1] * (15 / max(grid_h - 1, 1)),
            positions[2] * (15 / max(grid_w - 1, 1)),
        )
        for block in self.blocks:
            hidden = block(hidden, mask=None, positions=positions)
        hidden = self.norm(hidden)
        return hidden.transpose(1, 2).reshape(batch, 768, time, grid_h, grid_w)

    @torch.no_grad()
    def encode_states(self, rgb: torch.Tensor, frames_per_state: int):
        if rgb.ndim != 5 or rgb.shape[2] % frames_per_state:
            raise ValueError("Teacher input must contain complete RGB states")
        states = rgb.shape[2] // frames_per_state
        return torch.stack(
            [self(rgb[:, :, i * frames_per_state:(i + 1) * frames_per_state])
             for i in range(states)], dim=1
        )
