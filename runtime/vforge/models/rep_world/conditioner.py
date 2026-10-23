"""State-causal V-JEPA-compatible ViT with RGB or codec-latent input.

Only the encoder is instantiated: no JEPA predictor, image-only branch,
teacher, or auxiliary objective. Features retain their original norm/sum rule.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig
from vforge.models.rep_world.layers import PatchEmbed3D
from vforge.models.rep_world.sequence import gather_tokens, shard_tokens, token_range, usp_state
from vforge.models.rep_world.sparsity import state_causal_sparsity, use_segmented_attention
from vforge.ops import AttentionCallPolicy, SegmentedSparsity, attention

_POLICY = AttentionCallPolicy(cast_qk_to_v=True)


def rotary(x, positions):
    if x.shape[-1] == 0:
        return x
    omega = torch.arange(x.shape[-1] // 2, dtype=torch.float32, device=x.device)
    omega = 1.0 / (10000.0 ** (omega / (x.shape[-1] / 2.0)))
    frequency = positions.float()[..., None] * omega
    sine = frequency.sin().repeat_interleave(2, -1).to(x.dtype)
    cosine = frequency.cos().repeat_interleave(2, -1).to(x.dtype)
    first, second = x.unflatten(-1, (-1, 2)).unbind(-1)
    return x * cosine + torch.stack((-second, first), -1).flatten(-2) * sine


def sample_context_indices(batch, states, planes, spatial, ratio, *, device):
    """DuET spatial sampling: independent masks per sample/state/latent plane.

    Return sorted original global token indices; never renumber rotary positions.
    At least one spatial token survives on every plane.
    """
    keep = max(1, int(spatial * (1.0 - ratio) + 1e-8))
    scores = torch.rand(batch, states, planes * spatial, device=device)
    selected = scores.reshape(batch, states, planes, spatial).argsort(dim=-1)[..., :keep]
    offsets = torch.arange(states * planes, device=device).reshape(1, states, planes, 1)
    selected = selected + offsets * spatial
    return selected.reshape(batch, states * planes * keep).sort(dim=-1).values


class CausalAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.num_heads, self.head_dim = heads, width // heads
        self.axis_dim = 2 * ((self.head_dim // 3) // 2)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def rotate_axis(self, value, position):
        return rotary(value, position)

    def _rotate(self, value, positions):
        pieces = [
            self.rotate_axis(
                value[..., i * self.axis_dim : (i + 1) * self.axis_dim], pos
            )
            for i, pos in enumerate(positions)
        ]
        pieces.append(value[..., 3 * self.axis_dim :])
        return torch.cat(pieces, -1).transpose(1, 2)

    def forward(self, x, *, mask, positions, sequence_parallel=False):
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, length, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )

        out = attention(
            self._rotate(q, positions),
            self._rotate(k, positions),
            v.transpose(1, 2),
            attn_mask=None if isinstance(mask, SegmentedSparsity) else mask,
            sparsity=mask if isinstance(mask, SegmentedSparsity) else None,
            call_policy=_POLICY,
            distributed=sequence_parallel,
        )
        return self.proj(out.reshape(batch, length, width))

    def forward_cached(self, x, *, positions, past_kv=None):
        """Attend from one complete state to cached earlier states and itself."""
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, length, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q = self._rotate(q, positions)
        k = self._rotate(k, positions)
        v = v.transpose(1, 2)
        if past_kv is not None:
            k = torch.cat((past_kv[0], k), dim=1)
            v = torch.cat((past_kv[1], v), dim=1)
        out = attention(q, k, v, call_policy=_POLICY)
        return self.proj(out.reshape(batch, length, width)), (k, v)


class ViTMLP(nn.Module):
    def __init__(self, width, ratio):
        super().__init__()
        self.fc1, self.fc2 = (
            nn.Linear(width, int(width * ratio)),
            nn.Linear(int(width * ratio), width),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class CausalViTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width, eps=1e-6)
        self.attn = CausalAttention(config.width, config.heads)
        self.norm2 = nn.LayerNorm(config.width, eps=1e-6)
        self.mlp = ViTMLP(config.width, config.mlp_ratio)

    def forward(self, x, *, mask, positions, sequence_parallel=False):
        x = x + self.attn(
            self.norm1(x), mask=mask, positions=positions, sequence_parallel=sequence_parallel
        )
        return x + self.mlp(self.norm2(x))

    def forward_cached(self, x, *, positions, past_kv=None):
        update, kv = self.attn.forward_cached(
            self.norm1(x), positions=positions, past_kv=past_kv
        )
        x = x + update
        return x + self.mlp(self.norm2(x)), kv


def preprocess_rgb(video, size, dtype):
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError("Expected RGB [B,3,F,H,W] in [-1,1]")
    batch, channels, frames, height, width = video.shape
    pixels = ((video.float() + 1) * 0.5).clamp(0, 1)
    if (height, width) != (size, size):
        pixels = F.interpolate(
            pixels.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        pixels = pixels.reshape(batch, frames, channels, size, size).permute(0, 2, 1, 3, 4)
    mean = pixels.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1, 1)
    std = pixels.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1, 1)
    return ((pixels - mean) / std).to(dtype).contiguous()


class CausalViT(nn.Module):
    """Return [B,S,N,D] features; training masks reduce N to visible tokens."""

    def __init__(self, config: ConditionerConfig, geometry: CodecGeometry):
        super().__init__()
        self.config, self.geometry = config, geometry
        self.gradient_checkpointing = False
        self.sequence_parallel = True
        self.tokens_per_state = config.tokens_per_state(geometry)
        self.output_dim = config.width
        self.hierarchical_layers = {
            12: (2, 5, 8, 11),
            24: (5, 11, 17, 23),
            40: (9, 19, 29, 39),
            48: (11, 23, 37, 47),
        }.get(config.depth, tuple(range(config.depth)))
        if any(layer not in self.hierarchical_layers for layer in config.feature_layers):
            raise ValueError(
                f"No feature norm for requested layers; supported {self.hierarchical_layers}"
            )
        # A container preserves original encoder.* checkpoint names without
        # constructing unused upstream predictor/image branches.
        self.encoder = nn.Module()
        if config.input == "rgb":
            self.encoder.patch_embed = PatchEmbed3D(
                (geometry.rgb_frames, config.input_size, config.input_size),
                (config.temporal_patch, config.spatial_patch, config.spatial_patch),
                3,
                config.width,
            )
            self.encoder.video_mod_embed = nn.Parameter(torch.zeros(1, 1, config.width))
            self.latent_patch_embed = None
        else:
            self.encoder.patch_embed = nn.Identity()
            self.latent_patch_embed = nn.Linear(geometry.channels, config.width)
        self.encoder.blocks = nn.ModuleList(CausalViTBlock(config) for _ in range(config.depth))
        self.encoder.norms_block = nn.ModuleList(
            nn.LayerNorm(config.width, eps=1e-6) for _ in self.hierarchical_layers
        )
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.encoder.modules():
            if isinstance(module, (nn.Linear, nn.Conv3d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            for index, block in enumerate(self.encoder.blocks, 1):
                block.attn.proj.weight.div_(math.sqrt(2.0 * index))
                block.mlp.fc2.weight.div_(math.sqrt(2.0 * index))
        if self.latent_patch_embed is None:
            nn.init.normal_(self.encoder.video_mod_embed, std=1e-6)
        else:
            # The original latent adapter creates a fresh default nn.Linear,
            # not a V-JEPA trunc-normal input projection.
            self.latent_patch_embed.reset_parameters()

    def preprocess_rgb(self, video):
        return preprocess_rgb(
            video, self.config.input_size, self.encoder.patch_embed.proj.weight.dtype
        )

    def forward(self, source, *, return_indices=False):
        cfg, geo = self.config, self.geometry
        if cfg.input == "rgb":
            if (
                source.ndim != 5
                or source.shape[2] < geo.rgb_frames
                or source.shape[2] % geo.rgb_frames
            ):
                raise ValueError("RGB must contain complete codec states")
            states, grid, planes = (
                source.shape[2] // geo.rgb_frames,
                cfg.input_size // cfg.spatial_patch,
                geo.rgb_frames // cfg.temporal_patch,
            )
            hidden = (
                self.encoder.patch_embed(self.preprocess_rgb(source)) + self.encoder.video_mod_embed
            )
        else:
            if (
                source.ndim != 5
                or source.shape[1] != geo.packed_channels
                or source.shape[-2:] != (geo.grid, geo.grid)
            ):
                raise ValueError("Expected packed latent states [B,planes*channels,S,H,W]")
            states, grid, planes = source.shape[2], geo.grid, geo.planes
            batch = source.shape[0]
            tokens = source.reshape(batch, planes, geo.channels, states, grid, grid).permute(
                0, 3, 1, 4, 5, 2
            )
            hidden = self.latent_patch_embed(
                tokens.reshape(batch, states * self.tokens_per_state, geo.channels).to(
                    self.latent_patch_embed.weight.dtype
                )
            )
        if states < 1:
            raise ValueError("Conditioner requires a nonempty history")
        length = states * self.tokens_per_state
        visible_per_state = self.tokens_per_state
        selected = None
        if self.training and cfg.context_mask_ratio:
            if usp_state(self.sequence_parallel) is not None:
                raise ValueError("Context token masking currently requires USP=1")
            selected = sample_context_indices(
                source.shape[0],
                states,
                planes,
                grid**2,
                cfg.context_mask_ratio,
                device=hidden.device,
            )
            hidden = hidden.gather(1, selected[..., None].expand(-1, -1, cfg.width))
            length = selected.shape[1]
            visible_per_state = length // states
        if use_segmented_attention():
            mask = state_causal_sparsity(length, visible_per_state)
        else:
            group = torch.arange(length, device=hidden.device) // visible_per_state
            mask = group[:, None] >= group[None, :]
        scale = (cfg.pretrained_grid - 1) / (grid - 1) if grid > 1 else 1.0
        temporal = (
            torch.arange(states * planes, device=hidden.device).float().repeat_interleave(grid**2)
        )
        row = (
            (torch.arange(grid, device=hidden.device).float() * scale)
            .repeat_interleave(grid)
            .repeat(states * planes)
        )
        column = (
            (torch.arange(grid, device=hidden.device).float() * scale)
            .repeat(grid)
            .repeat(states * planes)
        )
        parallel = usp_state(self.sequence_parallel) is not None
        start, local_length = token_range(length, self.sequence_parallel)
        hidden = shard_tokens(hidden, enabled=self.sequence_parallel)
        local_positions = tuple(
            p[selected].unsqueeze(1) if selected is not None else p.narrow(0, start, local_length)
            for p in (temporal, row, column)
        )
        features = []
        for index, block in enumerate(self.encoder.blocks):
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                hidden = self._gradient_checkpointing_func(
                    block, hidden, mask=mask, positions=local_positions, sequence_parallel=parallel
                )
            else:
                hidden = block(
                    hidden, mask=mask, positions=local_positions, sequence_parallel=parallel
                )
            if index in cfg.feature_layers:
                features.append(
                    self.encoder.norms_block[self.hierarchical_layers.index(index)](hidden)
                )
        result = features[0]
        for feature in features[1:]:
            result = result + feature
        result = gather_tokens(result, enabled=self.sequence_parallel)
        result = result.reshape(source.shape[0], states, visible_per_state, cfg.width)
        if return_indices:
            state_indices = None
            if selected is not None:
                state_indices = selected.reshape(source.shape[0], states, visible_per_state)
                state_indices = state_indices - (
                    torch.arange(states, device=selected.device)[None, :, None]
                    * self.tokens_per_state
                )
            return result, state_indices
        return result

    def training_context(self, source, *, return_indices=False):
        # Preserve the original two batching paths, including RGB's discarded
        # final group; clean target latent never enters its own condition.
        if self.config.input == "rgb":
            if return_indices:
                raise ValueError("Spatial index output currently requires latent input")
            return self(source)[:, :-1]
        return self(source[:, :, :-1].detach(), return_indices=return_indices)

    def next_context(self, source):
        return self(source)[:, -1]
