"""Lossless RGB state packing for direct pixel-space flow matching.

Each state is four ordered RGB frames.  A 16x16 spatial patch becomes one
3072-channel state token; this is a layout transform, not a learned codec.
"""

from __future__ import annotations

import torch

from vforge.models.rep_world.config import CodecGeometry


class PixelRGB4Codec:
    def __init__(self, geometry: CodecGeometry):
        if geometry.family != "pixel" or (
            geometry.channels, geometry.grid, geometry.rgb_frames // geometry.planes
        ) != (3072, 16, 4):
            raise ValueError("PixelRGB4Codec requires four RGB frames per temporal plane")
        self.geometry = geometry

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        if (rgb.ndim != 5 or rgb.shape[1] != 3
            or rgb.shape[2] % self.geometry.rgb_frames
            or rgb.shape[-2:] != (256, 256)):
            raise ValueError(f"Expected RGB [B,3,{self.geometry.rgb_frames}*S,256,256]")
        b, _, t, _, _ = rgb.shape
        planes = t // 4
        packed = (
            rgb.reshape(b, 3, planes, 4, 16, 16, 16, 16)
            .permute(0, 3, 1, 5, 7, 2, 4, 6)
            .reshape(b, 3072, planes, 16, 16)
        )
        states = planes // self.geometry.planes
        return packed.reshape(
            b, 3072, states, self.geometry.planes, 16, 16
        ).permute(0, 3, 1, 2, 4, 5).reshape(
            b, self.geometry.packed_channels, states, 16, 16
        )

    def decode(self, packed: torch.Tensor) -> torch.Tensor:
        if (packed.ndim != 5 or packed.shape[1] != self.geometry.packed_channels
            or packed.shape[-2:] != (16, 16)):
            raise ValueError("Expected packed RGB [B,3072*N,S,16,16]")
        b, _, states, _, _ = packed.shape
        planes = states * self.geometry.planes
        packed = packed.reshape(
            b, self.geometry.planes, 3072, states, 16, 16
        ).permute(0, 2, 3, 1, 4, 5).reshape(b, 3072, planes, 16, 16)
        return (
            packed.reshape(b, 4, 3, 16, 16, planes, 16, 16)
            .permute(0, 2, 5, 1, 6, 3, 7, 4)
            .reshape(b, 3, 4 * planes, 256, 256)
        )

    def stream(self):
        return _PixelRGB4Stream(self)


class _PixelRGB4Stream:
    def __init__(self, codec: PixelRGB4Codec):
        self.codec = codec

    def prime(self, *_args, **_kwargs):
        pass

    def step(self, packed: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(packed)
