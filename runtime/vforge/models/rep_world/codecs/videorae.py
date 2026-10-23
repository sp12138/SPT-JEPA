"""Released VideoRAE 3D/1D inference components, without upstream runtime imports.

Adapted from VideoRAE (MIT); attribution in THIRD_PARTY.md and licenses/.
The 1D release has 512 learned latent slots, not a physical 2x16x16 grid.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from vforge.models.rep_world.codecs.encoders import FrozenVJEPA21Attention
from vforge.models.rep_world.conditioner import CausalAttention, CausalViTBlock
from vforge.models.rep_world.config import ConditionerConfig
from vforge.models.rep_world.layers import PatchEmbed3D
from vforge.ops import AttentionCallPolicy, attention

_POLICY = AttentionCallPolicy(cast_qk_to_v=True)


def video_frequencies(grid, prefix=0, head_dim=64):
    """Original complex128, increasing-frequency, interleaved THW rotary table.

    Build on CPU even inside a meta construction context. This is derived
    non-parameter data, cached lazily by VideoRotary, never a meta buffer.
    """
    if head_dim < 6 or head_dim % 2:
        raise ValueError("VideoRAE rotary needs an even head width >= 6")
    with torch.device("cpu"):
        axes = [2 * ((head_dim // 3) // 2)] * 3
        axes[0] += head_dim - sum(axes)
        positions = torch.stack(
            torch.meshgrid(*(torch.arange(size) for size in grid), indexing="ij"), -1
        ).reshape(-1, 3)
        positions = torch.cat((torch.arange(prefix)[:, None].expand(-1, 3), positions + prefix))
        frequencies = []
        for axis, width in enumerate(axes):
            omega = 10000.0 ** torch.linspace(0, 1, width // 2, dtype=torch.float64)
            angle = (omega * math.pi / 2) * positions[:, axis, None]
            frequencies.append(torch.polar(torch.ones_like(angle), angle))
        # Upstream sorts by axis width then interleaves available frequencies;
        # for head64 this is T,H,W repeated10 times, then the remaining2 T.
        frequencies.sort(key=lambda value: value.shape[-1], reverse=True)
        return torch.stack(
            [
                value[:, index]
                for index in range(max(x.shape[-1] for x in frequencies))
                for value in frequencies
                if index < value.shape[-1]
            ],
            -1,
        )


class VideoRotary(nn.Module):
    def __init__(self, grid, prefix=0, head_dim=64):
        super().__init__()
        self.grid, self.prefix, self.head_dim = tuple(grid), prefix, head_dim
        self._tables = {}

    def forward(self, value):
        if value.device not in self._tables:
            self._tables[value.device] = video_frequencies(
                self.grid, self.prefix, self.head_dim
            ).to(value.device)
        frequency = self._tables[value.device].unsqueeze(-2)
        if value.shape[1] != frequency.shape[0] or value.shape[-1] != self.head_dim:
            raise ValueError("VideoRAE rotary token/head geometry mismatch")
        # Do not downcast the table with model.to(bf16): the published codec
        # rotates float32 inputs with float64 sine/cosine before casting back.
        with torch.autocast(value.device.type, enabled=False):
            real, imag = value.float().unflatten(-1, (-1, 2)).unbind(-1)
            result = torch.stack(
                (
                    real * frequency.real - imag * frequency.imag,
                    real * frequency.imag + imag * frequency.real,
                ),
                -1,
            ).flatten(-2)
        return result.to(value.dtype)


class GatedVideoAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.to_qkv = nn.Linear(width, 4 * width, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.out_proj = nn.Linear(width, width, bias=False)

    def forward(self, hidden, rotary):
        q, k, v, gate = self.to_qkv(hidden).chunk(4, -1)
        q, k, v = (value.unflatten(-1, (self.heads, self.head_dim)) for value in (q, k, v))
        q, k = self.q_norm(q.contiguous()).to(q), self.k_norm(k.contiguous()).to(k)
        result = attention(rotary(q), rotary(k), v, call_policy=_POLICY)
        result = result.flatten(-2).contiguous() * torch.sigmoid(gate)
        return self.out_proj(result)


class GEGLU(nn.Module):
    def forward(self, hidden):
        value, gate = hidden.chunk(2, -1)
        return F.gelu(gate) * value


class VideoTransformer(nn.Module):
    """Keep published attn_layer/ffd_layer names and depth-dependent scaling."""

    def __init__(self, width, heads, depth):
        super().__init__()
        inner = 32 * math.ceil(int(4 * (2 / 3) * width) / 32)
        self.attn_layer = nn.ModuleList(GatedVideoAttention(width, heads) for _ in range(depth))
        self.ffd_layer = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, inner * 2, bias=False),
                GEGLU(),
                nn.Linear(inner, width, bias=False),
            )
            for _ in range(depth)
        )

    def forward(self, hidden, rotary):
        for index, (attn, ffd) in enumerate(zip(self.attn_layer, self.ffd_layer, strict=True)):
            hidden = hidden + attn(hidden.contiguous(), rotary)
            hidden = hidden + ffd(hidden.contiguous())
            hidden = hidden * (1 / math.sqrt(index + 1))
        return hidden


class FrozenVJEPA2Attention(CausalAttention):
    def rotate_axis(self, value, position):
        # The original V-JEPA2 release repeats the entire frequency vector,
        # rather than each element. Preserve this documented upstream quirk;
        # correcting it would change the meaning of the published weights.
        width = value.shape[-1]
        omega = torch.arange(width // 2, device=value.device, dtype=value.dtype)
        omega /= width / 2.0
        omega = 1.0 / 10000**omega
        frequency = torch.einsum("...,f->...f", position, omega)
        sine, cosine = frequency.sin().repeat(1, 1, 1, 2), frequency.cos().repeat(1, 1, 1, 2)
        first, second = value.unflatten(-1, (-1, 2)).unbind(-1)
        return value * cosine + torch.stack((-second, first), -1).flatten(-2) * sine


class VideoFeatureTeacher(nn.Module):
    """3D: four hierarchical ViT-B norms; 1D: nine shared-norm ViT-L layers."""

    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        width, depth, heads = (768, 12, 12) if variant == "3d" else (1024, 24, 16)
        config = ConditionerConfig(width=width, depth=depth, heads=heads)
        self.patch_embed = PatchEmbed3D((16, 256, 256), (2, 16, 16), 3, width)
        self.blocks = nn.ModuleList(CausalViTBlock(config) for _ in range(depth))
        attention_class = FrozenVJEPA21Attention if variant == "3d" else FrozenVJEPA2Attention
        for block in self.blocks:
            block.attn = attention_class(width, heads)
        if variant == "3d":
            self.video_mod_embed = nn.Parameter(torch.empty(1, 1, width))
            self.norms_block = nn.ModuleList(nn.LayerNorm(width, eps=1e-6) for _ in range(4))
            self.feature_layers = (2, 5, 8, 11)
        else:
            self.norm = nn.LayerNorm(width, eps=1e-6)
            self.feature_layers = (6, 8, 10, 12, 14, 16, 18, 20, 22)

    def forward(self, video):
        hidden = self.patch_embed(video)
        positions = tuple(
            value.flatten()
            for value in torch.meshgrid(
                torch.arange(8, device=video.device),
                torch.arange(16, device=video.device),
                torch.arange(16, device=video.device),
                indexing="ij",
            )
        )
        if self.variant == "3d":
            hidden = hidden + self.video_mod_embed
            positions = tuple(value.float() for value in positions)
        outputs = []
        for index, block in enumerate(self.blocks):
            hidden = block(hidden, mask=None, positions=positions)
            if index in self.feature_layers:
                norm = (
                    self.norms_block[self.feature_layers.index(index)]
                    if self.variant == "3d"
                    else self.norm
                )
                outputs.append(norm(hidden))
        # Python sum preserves the source's sequential addition (not stack.sum).
        return sum(outputs).float()


class Compressor3D(nn.Module):
    def __init__(
        self, *, teacher_width=768, width=512, channels=64, depth=8, heads=8, grid=(4, 16, 16)
    ):
        super().__init__()
        self.grid = tuple(grid)
        self.temporal_down = nn.Conv3d(teacher_width, width, (2, 1, 1), stride=(2, 1, 1))
        self.model_layers = VideoTransformer(width, heads, depth)
        self.proj_out = nn.Conv3d(width, channels, 1)
        self.rotary = VideoRotary(grid, head_dim=width // heads)

    def forward(self, features):
        batch, _, channels = features.shape
        frames, height, width = self.grid
        value = features.reshape(batch, frames * 2, height, width, channels).permute(0, 4, 1, 2, 3)
        value = self.temporal_down(value).flatten(2).transpose(1, 2)
        value = self.model_layers(value, self.rotary)
        value = value.transpose(1, 2).reshape(batch, -1, *self.grid)
        return self.proj_out(value)


class Tokenizer1D(nn.Module):
    def __init__(
        self, *, teacher_width=1024, width=512, depth=8, heads=8, tokens=512, grid=(8, 16, 16)
    ):
        super().__init__()
        self.tokens, self.width = tokens, width
        self.proj_in = nn.Linear(teacher_width, width)
        self.mask_token = nn.Parameter(torch.empty(1, 1, 1))
        self.model_layers = VideoTransformer(width, heads, depth)
        self.proj_out = nn.Linear(width, width)
        self.rotary = VideoRotary(grid, prefix=tokens, head_dim=width // heads)

    def forward(self, features):
        hidden = self.proj_in(features)
        prefix = self.mask_token.expand(hidden.shape[0], self.tokens, self.width)
        hidden = self.model_layers(torch.cat((prefix, hidden), 1), self.rotary)
        return self.proj_out(hidden[:, : self.tokens])


class VideoDecoder(nn.Module):
    def __init__(
        self,
        variant,
        *,
        channels=64,
        width=768,
        depth=12,
        heads=12,
        grid=(4, 16, 16),
        patch=(4, 16, 16),
        tokens=512,
    ):
        super().__init__()
        self.variant, self.grid, self.tokens = variant, tuple(grid), tokens
        self.proj_in = (
            nn.Conv3d(channels, width, 1) if variant == "3d" else nn.Linear(channels, width)
        )
        if variant == "1d":
            self.mask_token = nn.Parameter(torch.empty(1, 1, 1))
        self.model_layers = VideoTransformer(width, heads, depth)
        self.proj_out = nn.ConvTranspose3d(width, 3, patch, stride=patch)
        self.rotary = VideoRotary(
            grid, prefix=0 if variant == "3d" else tokens, head_dim=width // heads
        )

    def forward(self, latent):
        hidden = self.proj_in(latent)
        if self.variant == "3d":
            hidden = hidden.flatten(2).transpose(1, 2)
        else:
            mask = self.mask_token.expand(hidden.shape[0], math.prod(self.grid), hidden.shape[-1])
            hidden = torch.cat((hidden, mask), 1)
        hidden = self.model_layers(hidden, self.rotary)
        if self.variant == "1d":
            hidden = hidden[:, self.tokens :]
        hidden = hidden.transpose(1, 2).reshape(hidden.shape[0], -1, *self.grid)
        return self.proj_out(hidden)


class VideoRAE(nn.Module):
    def __init__(self, variant):
        super().__init__()
        if variant not in {"3d", "1d"}:
            raise ValueError("VideoRAE variant must be 3d or 1d")
        self.variant = variant
        self.teacher_model = VideoFeatureTeacher(variant)
        if variant == "3d":
            self.latent_compressor = Compressor3D()
            self.decoder = VideoDecoder(variant)
        else:
            self.tokenizer_encoder = Tokenizer1D()
            self.enc_proj, self.dec_proj = nn.Linear(512, 32), nn.Linear(32, 512)
            self.decoder = VideoDecoder(variant, channels=512)
        self.eval().requires_grad_(False)

    def encode(self, video):
        if video.ndim != 5 or video.shape[1:] != (3, 16, 256, 256):
            raise ValueError("VideoRAE requires [B,3,16,256,256] unit-range RGB")
        # Unlike V-RAE, these constants are created in input dtype, not rounded
        # with model-wide bf16 buffers before normalization.
        mean = video.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
        std = video.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
        features = self.teacher_model(
            ((video - mean) / std).to(next(self.teacher_model.parameters()))
        )
        if self.variant == "3d":
            return self.latent_compressor(features).flatten(2).transpose(1, 2)
        return self.enc_proj(self.tokenizer_encoder(features))

    def decode(self, latent):
        if self.variant == "3d":
            if latent.ndim != 5 or latent.shape[1:] != (64, 4, 16, 16):
                raise ValueError("VideoRAE3D decoder requires [B,64,4,16,16]")
            return self.decoder(latent)
        if latent.ndim != 3 or latent.shape[1:] != (512, 32):
            raise ValueError("VideoRAE1D decoder requires [B,512,32]")
        return self.decoder(self.dec_proj(latent))
