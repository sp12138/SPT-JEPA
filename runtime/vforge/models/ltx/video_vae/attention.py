"""Convolution video VAE from LTX reference 95db86f (Lightricks).

Only encoding/decoding used by segment handover; no tiled or diffusion decoder.
"""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from vforge.ops import sdpa_attention


class _RMSNorm2D(nn.Module):
    """Channel-first RMSNorm for 4D tensors with a learnable per-channel gain.
    Args:
        channels: The number of channels in the input.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = channels**0.5
        self.gamma = nn.Parameter(torch.ones(channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * (self.scale * self.gamma)


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels: int, attention: Callable = sdpa_attention) -> None:
        super().__init__()
        self._attention = attention
        self.norm = _RMSNorm2D(in_channels)
        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor, causal: bool = True) -> torch.Tensor:
        identity = hidden_states
        b = hidden_states.shape[0]
        x = rearrange(hidden_states, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        qkv = rearrange(self.to_qkv(x), "bt c h w -> bt (h w) c")
        (q, k, v) = qkv.chunk(3, dim=-1)
        x = self._attention(q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2)).squeeze(2)
        x = rearrange(x, "bt (h w) c -> bt c h w", h=hidden_states.shape[3])
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=b)
        return identity + x
