"""Wan2.2 TI2V-5B VAE adapter with exact four-RGB-frame state boundaries.

Wan2.2 encodes one special initial frame followed by groups of four. For a
40-frame RepWorld clip we prepend a duplicate of its first frame. We keep
only the ten subsequent 48x16x16 latents as trainable World states and retain
the initial latent solely as frozen decoder context.
"""

from __future__ import annotations

import torch

from vforge.models.rep_world.config import CodecGeometry


class Wan22RGB4Codec:
    def __init__(self, geometry: CodecGeometry, checkpoint: str, device, dtype, *, vae=None):
        if (geometry.family, geometry.variant) != ("wan22", "ti2v5b") or (
            geometry.rgb_frames, geometry.channels, geometry.planes, geometry.grid
        ) != (4, 48, 1, 16):
            raise ValueError("Wan22RGB4Codec requires Wan2.2 TI2V-5B 4x16x16 geometry")
        if not checkpoint and vae is None:
            raise ValueError("Wan2.2 VAE checkpoint is required")
        self.device = torch.device(device)
        self.dtype = dtype
        if vae is None:
            from vforge.models.rep_world.vendor.wan22.vae2_2 import Wan2_2_VAE

            vae = Wan2_2_VAE(vae_pth=checkpoint, dtype=dtype, device=self.device)
        self.vae = vae
        self._anchor = None

    @torch.no_grad()
    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 5 or rgb.shape[1] != 3 or rgb.shape[2] % 4 or rgb.shape[-2:] != (256, 256):
            raise ValueError("Expected RGB [B,3,4*S,256,256]")
        padded = torch.cat((rgb[:, :, :1], rgb), dim=2).to(device=self.device)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            encoded = self.vae.model.encode(padded, self.vae.scale)
        expected = rgb.shape[2] // 4 + 1
        if encoded.shape != (rgb.shape[0], 48, expected, 16, 16):
            raise ValueError(f"Wan2.2 VAE produced unexpected latent shape {tuple(encoded.shape)}")
        self._anchor = encoded[:, :, :1].detach().float()
        return encoded[:, :, 1:].detach().float()

    @torch.no_grad()
    def decode(self, packed: torch.Tensor, *, anchor: torch.Tensor | None = None) -> torch.Tensor:
        if packed.ndim != 5 or packed.shape[1] != 48 or packed.shape[-2:] != (16, 16):
            raise ValueError("Expected Wan latent [B,48,S,16,16]")
        initial = self._anchor if anchor is None else anchor
        if initial is None or initial.shape != (packed.shape[0], 48, 1, 16, 16):
            raise ValueError("Wan2.2 decoding requires the encoded initial-frame latent")
        sequence = torch.cat((initial.to(packed), packed), dim=2)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32):
            decoded = self.vae.model.decode(sequence, self.vae.scale)
        if decoded.shape != (packed.shape[0], 3, 1 + 4 * packed.shape[2], 256, 256):
            raise ValueError(f"Wan2.2 VAE produced unexpected RGB shape {tuple(decoded.shape)}")
        return decoded[:, :, 1:].float().clamp(-1, 1)

    def stream(self):
        return _Wan22RGB4Stream(self)


class _Wan22RGB4Stream:
    def __init__(self, codec: Wan22RGB4Codec):
        self.codec = codec
        self.anchor = None
        self.states = None

    def prime(self, observed: torch.Tensor):
        anchor = self.codec._anchor
        if anchor is None or anchor.shape[0] != observed.shape[0]:
            raise ValueError("Wan2.2 stream requires an encoded observed RGB prefix")
        self.anchor = anchor.clone()
        self.states = observed.clone()

    def step(self, packed: torch.Tensor) -> torch.Tensor:
        if self.states is None or packed.shape[2] != 1:
            raise ValueError("Wan2.2 stream requires prime() and one new state")
        self.states = torch.cat((self.states, packed), dim=2)
        return self.codec.decode(self.states, anchor=self.anchor)[:, :, -4:]
