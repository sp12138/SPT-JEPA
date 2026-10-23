"""Convolution video VAE from LTX reference 95db86f (Lightricks).

Only encoding/decoding used by segment handover; no tiled or diffusion decoder.
"""

from typing import NamedTuple

import torch
from einops import rearrange
from torch import nn


def patchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")
    return x


def unpatchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )
    return x


class PerChannelStatistics(nn.Module):
    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.register_buffer("std-of-means", torch.ones(latent_channels))
        self.register_buffer("mean-of-means", torch.zeros(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x) + self.get_buffer(
            "mean-of-means"
        ).view(1, -1, 1, 1, 1).to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)) / self.get_buffer(
            "std-of-means"
        ).view(1, -1, 1, 1, 1).to(x)


class PixelNorm(nn.Module):
    """
    Per-pixel (per-location) RMS normalization layer.
    For each element along the chosen dimension, this layer normalizes the tensor
    by the root-mean-square of its values across that dimension:
        y = x / sqrt(mean(x^2, dim=dim, keepdim=True) + eps)
    """

    def __init__(self, dim: int = 1, eps: float = 1e-08) -> None:
        """
        Args:
            dim: Dimension along which to compute the RMS (typically channels).
            eps: Small constant added for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization along the configured dimension.
        """
        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        return x / rms


class SpatioTemporalScaleFactors(NamedTuple):
    """
    Describes the spatiotemporal downscaling between decoded video space and
    the corresponding VAE latent grid.
    Field order matches the (frame/time, height, width) axis layout used by
    latent tensors and meshgrid coordinates elsewhere in the codebase.
    """

    time: int
    height: int
    width: int

    @classmethod
    def default(cls) -> "SpatioTemporalScaleFactors":
        return cls(time=8, height=32, width=32)

    @classmethod
    def from_blocks(cls, blocks: list, patch_size: int) -> "SpatioTemporalScaleFactors":
        spatial_steps = 0
        temporal_steps = 0
        for block_name, _ in blocks:
            if block_name.startswith(("compress_space", "compress_all")):
                spatial_steps += 1
            if block_name.startswith(("compress_time", "compress_all")):
                temporal_steps += 1
        spatial = patch_size * 2**spatial_steps
        return cls(time=2**temporal_steps, height=spatial, width=spatial)
