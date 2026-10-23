"""Independent latent-state representation (ViT-E), then state-causal ViT-C."""

from __future__ import annotations

import math
from dataclasses import replace

import torch
from torch import nn

from vforge.models.rep_world.conditioner import (
    CausalViTBlock,
    PatchEmbed3D,
    preprocess_rgb,
    sample_context_indices,
)
from vforge.models.rep_world.sequence import usp_state
from vforge.models.rep_world.sparsity import state_causal_sparsity, use_segmented_attention


def _require_no_usp():
    if usp_state() is not None:
        raise ValueError("Split ViT-E/ViT-C currently requires USP=1; use DDP or FSDP")


def _positions(states, geometry, config, device):
    """Retain the original V-JEPA spatial interpolation and latent-plane units."""
    grid, planes = geometry.grid, geometry.planes
    scale = (config.pretrained_grid - 1) / (grid - 1) if grid > 1 else 1.0
    time = torch.arange(states * planes, device=device).float().repeat_interleave(grid**2)
    row = (torch.arange(grid, device=device).float() * scale).repeat_interleave(grid)
    col = (torch.arange(grid, device=device).float() * scale).repeat(grid)
    return time, row.repeat(states * planes), col.repeat(states * planes)


def _initialize(module, blocks):
    for child in module.modules():
        if isinstance(child, (nn.Linear, nn.Conv3d)):
            nn.init.trunc_normal_(child.weight, std=0.02)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.LayerNorm):
            nn.init.ones_(child.weight)
            nn.init.zeros_(child.bias)
    with torch.no_grad():
        for index, block in enumerate(blocks, 1):
            block.attn.proj.weight.div_(math.sqrt(2.0 * index))
            block.mlp.fc2.weight.div_(math.sqrt(2.0 * index))


def _run(block, hidden, *, mask, positions, checkpoint_fn):
    kwargs = dict(mask=mask, positions=positions, sequence_parallel=False)
    return (
        checkpoint_fn(block, hidden, **kwargs)
        if checkpoint_fn is not None
        else block(hidden, **kwargs)
    )


class StateViTEncoder(nn.Module):
    """ViT-E: independent states, optionally masking before its attention blocks."""

    def __init__(self, config, geometry):
        super().__init__()
        self.config, self.geometry = config, geometry
        if config.input == "rgb":
            self.patch_embed = PatchEmbed3D(
                (geometry.rgb_frames, config.input_size, config.input_size),
                (config.temporal_patch, config.spatial_patch, config.spatial_patch),
                3,
                config.width,
            )
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, config.width))
        else:
            self.patch_embed = nn.Linear(geometry.channels, config.width)
        self.blocks = nn.ModuleList(CausalViTBlock(config) for _ in range(config.vit_e_depth))
        self.norm = nn.LayerNorm(config.width, eps=1e-6)
        self.initialize_weights()

    def initialize_weights(self):
        _initialize(self, self.blocks)
        if self.config.input == "rgb":
            nn.init.normal_(self.video_mod_embed, std=1e-6)
        else:
            # Match the existing latent-input projection's initialization family.
            self.patch_embed.reset_parameters()

    def forward(self, source, *, return_indices=False, checkpoint_fn=None):
        _require_no_usp()
        cfg, geo = self.config, self.geometry
        if cfg.input == "rgb":
            if (
                source.ndim != 5
                or source.shape[1] != 3
                or source.shape[2] < geo.rgb_frames
                or source.shape[2] % geo.rgb_frames
                or source.shape[0] < 1
            ):
                raise ValueError("RGB ViT-E expects complete states [B,3,S*F,H,W]")
            batch, states = source.shape[0], source.shape[2] // geo.rgb_frames
            pixels = source.reshape(batch, 3, states, geo.rgb_frames, *source.shape[-2:])
            pixels = pixels.permute(0, 2, 1, 3, 4, 5).reshape(
                batch * states, 3, geo.rgb_frames, *source.shape[-2:]
            )
            hidden = self.patch_embed(
                preprocess_rgb(pixels, cfg.input_size, self.patch_embed.proj.weight.dtype)
            ) + self.video_mod_embed
        else:
            if (
                source.ndim != 5
                or source.shape[1] != geo.packed_channels
                or source.shape[-2:] != (geo.grid, geo.grid)
                or min(source.shape[0], source.shape[2]) < 1
            ):
                raise ValueError("ViT-E expects nonempty packed latent states [B,P*C,S,H,W]")
            batch, states = source.shape[0], source.shape[2]
            tokens = source.reshape(batch, geo.planes, geo.channels, states, geo.grid, geo.grid)
            tokens = tokens.permute(0, 3, 1, 4, 5, 2).reshape(
                batch * states, geo.tokens_per_state, geo.channels
            )
            hidden = None
        selected = None
        if self.training and cfg.mask_stage == "before_e" and cfg.context_mask_ratio:
            selected = sample_context_indices(
                batch, states, geo.planes, geo.grid**2, cfg.context_mask_ratio,
                device=source.device,
            )
            selected = selected.reshape(batch, states, -1) - (
                torch.arange(states, device=source.device)[None, :, None]
                * geo.tokens_per_state
            )
            local = selected.reshape(batch * states, -1)
            if cfg.input == "rgb":
                # RGB patchification still runs densely; E's expensive blocks do not.
                hidden = hidden.gather(1, local[..., None].expand(-1, -1, cfg.width))
            else:
                # For latent input, skip even the patch projection on masked tokens.
                tokens = tokens.gather(1, local[..., None].expand(-1, -1, geo.channels))
        if cfg.input == "latent":
            hidden = self.patch_embed(tokens.to(self.patch_embed.weight.dtype))
        positions = _positions(1, geo, cfg, hidden.device)
        if selected is not None:
            positions = tuple(p[local].unsqueeze(1) for p in positions)
        for block in self.blocks:
            hidden = _run(
                block, hidden, mask=None, positions=positions, checkpoint_fn=checkpoint_fn
            )
        result = self.norm(hidden).reshape(batch, states, hidden.shape[1], cfg.width)
        return (result, selected) if return_indices else result


class TokenCausalViT(nn.Module):
    """ViT-C: causal prefill using the surviving tokens' original coordinates."""

    def __init__(self, config, geometry):
        super().__init__()
        self.config, self.geometry = config, geometry
        block_config = (
            replace(config, mlp_ratio=config.vit_c_mlp_ratio)
            if config.vit_c_mlp_ratio is not None else config
        )
        self.blocks = nn.ModuleList(CausalViTBlock(block_config) for _ in range(config.vit_c_depth))
        # Public feature indices still refer to the combined E+C depth.
        self.feature_layers = tuple(i - config.vit_e_depth for i in config.feature_layers)
        self.norms = nn.ModuleList(
            nn.LayerNorm(config.width, eps=1e-6) for _ in self.feature_layers
        )
        self.initialize_weights()

    def initialize_weights(self):
        _initialize(self, self.blocks)

    def forward(self, encoded, *, return_indices=False, selected_indices=None, checkpoint_fn=None):
        _require_no_usp()
        cfg, geo = self.config, self.geometry
        if (
            encoded.ndim != 4
            or encoded.shape[-1] != cfg.width
            or min(encoded.shape[0], encoded.shape[1]) < 1
        ):
            raise ValueError("ViT-C expects encoded states [B,S,V,D]")
        batch, states, tokens, _ = encoded.shape
        if selected_indices is None and tokens != geo.tokens_per_state:
            raise ValueError("Sparse ViT-E output requires its original token indices")
        if selected_indices is not None and (
            not self.training
            or cfg.mask_stage != "before_e"
            or selected_indices.shape != (batch, states, tokens)
            or selected_indices.dtype != torch.long
            or selected_indices.device != encoded.device
        ):
            raise ValueError("ViT-C requires valid training-time ViT-E token indices")
        hidden = encoded.reshape(batch, states * tokens, cfg.width)
        selected = None
        visible = tokens
        if selected_indices is not None:
            selected = (
                selected_indices
                + torch.arange(states, device=hidden.device)[None, :, None]
                * geo.tokens_per_state
            ).reshape(batch, states * tokens)
        elif self.training and cfg.context_mask_ratio and cfg.mask_stage == "after_e":
            selected = sample_context_indices(
                batch,
                states,
                geo.planes,
                geo.grid**2,
                cfg.context_mask_ratio,
                device=hidden.device,
            )
            hidden = hidden.gather(1, selected[..., None].expand(-1, -1, cfg.width))
            visible = selected.shape[1] // states
        length = states * visible
        if use_segmented_attention():
            mask = state_causal_sparsity(length, visible)
        else:
            group = torch.arange(length, device=hidden.device) // visible
            mask = group[:, None] >= group[None, :]
        positions = _positions(states, geo, cfg, hidden.device)
        if selected is not None:
            positions = tuple(p[selected].unsqueeze(1) for p in positions)
        features = []
        for index, block in enumerate(self.blocks):
            hidden = _run(
                block, hidden, mask=mask, positions=positions, checkpoint_fn=checkpoint_fn
            )
            if index in self.feature_layers:
                features.append(self.norms[self.feature_layers.index(index)](hidden))
        result = features[0]
        for feature in features[1:]:
            result = result + feature
        result = result.reshape(batch, states, visible, cfg.width)
        if return_indices:
            indices = None
            if selected is not None:
                indices = selected.reshape(batch, states, visible) - (
                    torch.arange(states, device=hidden.device)[None, :, None]
                    * geo.tokens_per_state
                )
            return result, indices
        return result

    def prefill_cached(self, encoded, *, cache=None, start_state=0):
        """Incrementally prefill complete states; cache stores per-layer rotated K/V.

        Causal visibility is by state, so all tokens in the incoming state may
        attend to every cached state and to each other. This path is inference
        only; training keeps the original full-sequence attention and masking.
        """
        _require_no_usp()
        if self.training or torch.is_grad_enabled():
            raise ValueError("ViT-C KV cache is inference-only")
        cfg, geo = self.config, self.geometry
        if (
            encoded.ndim != 4
            or encoded.shape[-2:] != (geo.tokens_per_state, cfg.width)
            or min(encoded.shape[0], encoded.shape[1]) < 1
        ):
            raise ValueError("ViT-C expects encoded states [B,S,P*H*W,D]")
        if cache is not None and len(cache) != len(self.blocks):
            raise ValueError("ViT-C KV cache layer count does not match the model")
        if start_state < 0:
            raise ValueError("ViT-C start_state must be nonnegative")
        tokens = geo.tokens_per_state
        for offset in range(encoded.shape[1]):
            state_index = start_state + offset
            hidden = encoded[:, offset]
            positions = tuple(
                p[state_index * tokens : (state_index + 1) * tokens]
                for p in _positions(state_index + 1, geo, cfg, hidden.device)
            )
            next_cache = []
            features = []
            for index, block in enumerate(self.blocks):
                past = None if cache is None else cache[index]
                hidden, kv = block.forward_cached(hidden, positions=positions, past_kv=past)
                next_cache.append(kv)
                if index in self.feature_layers:
                    features.append(self.norms[self.feature_layers.index(index)](hidden))
            cache = next_cache
        result = features[0]
        for feature in features[1:]:
            result = result + feature
        return result, cache


class SplitViTConditioner(nn.Module):
    """End-to-end E/C with a configurable mask boundary and original coordinates."""

    def __init__(self, config, geometry):
        super().__init__()
        if config.architecture != "split_ec":
            raise ValueError("Split ViT requires architecture=split_ec")
        if config.input == "rgb" and (
            geometry.rgb_frames // config.temporal_patch != geometry.planes
            or config.input_size // config.spatial_patch != geometry.grid
            or geometry.rgb_frames % config.temporal_patch
        ):
            raise ValueError("RGB split ViT must match the codec's P/H/W token grid")
        self.config, self.geometry = config, geometry
        self.tokens_per_state = geometry.tokens_per_state
        self.output_dim = config.width
        self.gradient_checkpointing = False
        self.sequence_parallel = False
        self.vit_e = StateViTEncoder(config, geometry)
        self.vit_c = TokenCausalViT(config, geometry)

    def initialize_weights(self):
        self.vit_e.initialize_weights()
        self.vit_c.initialize_weights()

    def _checkpoint_fn(self):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return self._gradient_checkpointing_func
        return None

    def encode_states(self, source, *, return_indices=False):
        return self.vit_e(
            source, return_indices=return_indices, checkpoint_fn=self._checkpoint_fn()
        )

    def prefill(self, encoded, *, return_indices=False, selected_indices=None):
        return self.vit_c(
            encoded, return_indices=return_indices, selected_indices=selected_indices,
            checkpoint_fn=self._checkpoint_fn()
        )

    def prefill_cached(self, encoded, *, cache=None, start_state=0):
        return self.vit_c.prefill_cached(
            encoded, cache=cache, start_state=start_state
        )

    def forward(self, source, *, return_indices=False):
        if self.training and self.config.mask_stage == "before_e" and self.config.context_mask_ratio:
            encoded, indices = self.encode_states(source, return_indices=True)
            return self.prefill(
                encoded, selected_indices=indices, return_indices=return_indices
            )
        return self.prefill(self.encode_states(source), return_indices=return_indices)

    def training_context(self, source, *, return_indices=False):
        tail = self.geometry.rgb_frames if self.config.input == "rgb" else 1
        return self(source[:, :, :-tail].detach(), return_indices=return_indices)

    def next_context(self, source):
        return self(source)[:, -1]
