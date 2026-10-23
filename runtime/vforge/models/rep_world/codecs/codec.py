"""Frozen RGB/latent boundary and per-rollout decoder state."""

from __future__ import annotations

import torch
from torch import nn

from vforge.models.rep_world.config import CodecGeometry
from vforge.models.rep_world.model import pack_states, unpack_states


class FrozenCodec:
    """Canonical packed latents [B,P*C,S,H,W], RGB [B,3,S*F,256,256].

    V-RAE encoding is state-independent. Its decoder may be chunk-causal or
    full-attention, as specified by the released weights. VideoRAE decodes
    each state independently. This wrapper never changes those semantics.
    """

    def __init__(
        self,
        model: nn.Module,
        geometry: CodecGeometry,
        *,
        encode_batch_size=1,
        decode_batch_size=1,
        mean=None,
        std=None,
        independent_state_decode=False,
    ):
        if min(encode_batch_size, decode_batch_size) <= 0:
            raise ValueError("Codec microbatches must be positive")
        if (mean is None) != (std is None):
            raise ValueError("Both latent mean and std are required")
        if mean is not None:
            mean, std = mean.float().reshape(-1), std.float().reshape(-1)
            if mean.numel() != geometry.channels or std.numel() != geometry.channels:
                raise ValueError("Latent statistics channel count does not match codec")
            if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError("Invalid latent statistics")
        self.model = model.eval().requires_grad_(False)
        self.geometry = geometry
        self.encode_batch_size, self.decode_batch_size = encode_batch_size, decode_batch_size
        self.mean, self.std = mean, std
        self.independent_state_decode = independent_state_decode

    def _statistics(self, volumes, *, inverse=False):
        if self.mean is None:
            return volumes
        mean, std = (x.to(volumes).view(1, 1, -1, 1, 1, 1) for x in (self.mean, self.std))
        return volumes * std + mean if inverse else (volumes - mean) / std

    @torch.no_grad()
    def encode(self, rgb):
        geo = self.geometry
        if (
            rgb.ndim != 5
            or rgb.shape[1] != 3
            or rgb.shape[-2:] != (256, 256)
            or rgb.shape[2] < geo.rgb_frames
            or rgb.shape[2] % geo.rgb_frames
        ):
            raise ValueError("Codec RGB must contain complete native states at 256x256")
        batch, _, frames, height, width = rgb.shape
        states = frames // geo.rgb_frames
        clips = rgb.reshape(batch, 3, states, geo.rgb_frames, height, width)
        clips = clips.permute(0, 2, 1, 3, 4, 5).flatten(0, 1)
        clips = ((clips + 1) * 0.5).clamp(0, 1)
        volumes = []
        for chunk in clips.split(self.encode_batch_size):
            if geo.family == "vrae":
                encoded = self.model.encode(chunk.permute(0, 2, 1, 3, 4))
                expected = (1, geo.channels, geo.grid, geo.grid)
                if encoded.shape[1:] != expected:
                    raise ValueError("Unexpected V-RAE latent layout")
                encoded = encoded.permute(0, 2, 1, 3, 4)
            else:
                encoded = self.model.encode(chunk)
                if encoded.shape[1:] != (geo.tokens_per_state, geo.channels):
                    raise ValueError("Unexpected VideoRAE token layout")
                encoded = encoded.reshape(-1, geo.planes, geo.grid, geo.grid, geo.channels)
                encoded = encoded.permute(0, 4, 1, 2, 3)
            volumes.append(encoded)
        volumes = torch.cat(volumes).reshape(
            batch, states, geo.channels, geo.planes, geo.grid, geo.grid
        )
        return pack_states(self._statistics(volumes)).contiguous().float()

    @torch.no_grad()
    def decode(self, packed):
        geo = self.geometry
        # Preserve the source wrapper's state-major contiguous storage before
        # projection. A channel-major view is numerically equivalent in fp32,
        # but selects a different GEMM path under bf16 and changes long decodes.
        volumes = self._statistics(unpack_states(packed, geo).contiguous(), inverse=True)
        batch, states = volumes.shape[:2]
        if geo.family == "vrae" and not self.independent_state_decode:
            value = volumes.squeeze(3)
            decoded = torch.cat([self.model.decode(x) for x in value.split(self.decode_batch_size)])
            decoded = decoded.permute(0, 2, 1, 3, 4)
        else:
            flat = volumes.flatten(0, 1)
            outputs = []
            for chunk in flat.split(self.decode_batch_size):
                if geo.family == "vrae":
                    outputs.append(
                        self.model.decode(chunk.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
                    )
                else:
                    value = (
                        chunk
                        if geo.spatially_ordered
                        else chunk.flatten(2).transpose(1, 2).contiguous()
                    )
                    outputs.append(self.model.decode(value))
            decoded = torch.cat(outputs).reshape(batch, states, 3, geo.rgb_frames, 256, 256)
            decoded = decoded.permute(0, 2, 1, 3, 4, 5).flatten(2, 3)
        if decoded.shape != (batch, 3, states * geo.rgb_frames, 256, 256):
            raise ValueError("Codec decoder returned incorrect RGB geometry")
        return decoded.clamp(0, 1).mul(2).sub(1).contiguous().float()

    def stream(self):
        return CodecStream(self)


class CodecStream:
    """One stream per rollout; no model-global caches shared across samples."""

    def __init__(self, codec):
        self.codec = codec
        self.cache = None
        self.history = None

    @torch.no_grad()
    def step(self, packed):
        codec, geo = self.codec, self.codec.geometry
        if geo.family != "vrae" or codec.independent_state_decode:
            return codec.decode(packed)
        outputs = []
        for state in packed.split(1, 2):
            if codec.model.decoder.config.attention_mode == "chunk_causal":
                volumes = codec._statistics(
                    unpack_states(state, geo).contiguous(), inverse=True
                ).squeeze(3)
                decoded, self.cache = codec.model.decode(
                    volumes, past_key_values=self.cache, return_key_values=True
                )
                outputs.append(decoded.permute(0, 2, 1, 3, 4).clamp(0, 1).mul(2).sub(1).float())
            else:
                # The V-JEPA2.1 release has a full-attention decoder. Recompute
                # the prefix and emit ONLY the newest state, exactly as before.
                self.history = (
                    state if self.history is None else torch.cat((self.history, state), 2)
                )
                outputs.append(codec.decode(self.history)[:, :, -geo.rgb_frames :])
        return torch.cat(outputs, 2)

    def prime(self, observed):
        if self.codec.geometry.family == "vrae" and not self.codec.independent_state_decode:
            self.step(observed)
