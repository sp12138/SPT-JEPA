"""Representation-space DiT, without camera, predictor, or trainer branches."""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from vforge.models.rep_world.attention import CrossAttention, SelfAttention
from vforge.models.rep_world.config import WorldConfig
from vforge.models.rep_world.pixel_dit import PixelRefiner
from vforge.models.rep_world.horizon import target_gap_embedding, validate_target_offsets
from vforge.models.rep_world.layers import (
    BlockModulation,
    FinalLayer,
    PatchEmbed3D,
    RMSNorm,
    SwiGLUFFN,
    TimestepEmbedder,
    VisionRotaryEmbeddingFast3D,
    _build_full_history_teacher_forcing_attn_mask,
    _build_prefix_block_causal_attn_mask,
    modulate,
)
from vforge.models.rep_world.sequence import (
    gather_tokens,
    shard_tokens,
    share_random,
    token_range,
    usp_state,
)
from vforge.models.rep_world.sparsity import (
    prefix_sparsity,
    state_causal_sparsity,
    teacher_forcing_sparsity,
    use_segmented_attention,
)


class WorldBlock(nn.Module):
    def __init__(self, config: WorldConfig, *, cross_attention: bool):
        super().__init__()
        width = config.width
        self.cross_attention_position = config.cross_attention_position
        self.cross_attention_spatial_size = (
            config.grid**2 if config.cross_attention_scope == "same_spatial" else None
        )
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.attn = SelfAttention(width, config.heads, config.qk_norm)
        self.norm_cross = RMSNorm(width) if cross_attention else None
        self.cross_attn = (
            CrossAttention(width, config.heads, config.qk_norm) if cross_attention else None
        )
        self.mlp = SwiGLUFFN(width, int(2 / 3 * int(width * config.mlp_ratio)))
        self.modulation = BlockModulation(width, config.adaln_mode, lora_dim=config.adaln_lora_dim)

    def forward(
        self,
        x,
        emb,
        shared,
        *,
        rope,
        mask=None,
        context=None,
        context_positions=None,
        cross_rope=None,
        prefix_tokens=0,
        past_kv=None,
        return_kv=False,
        sequence_parallel=False,
    ):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(emb, shared, None).chunk(
            6, -1
        )
        if self.cross_attention_position == "before_self":
            x = self._apply_cross_attention(
                x, context, prefix_tokens, context_positions, cross_rope=cross_rope
            )
        result = self.attn(
            modulate(self.norm1(x), shift1, scale1),
            rope=rope,
            mask=mask,
            past_kv=past_kv,
            return_kv=return_kv,
            sequence_parallel=sequence_parallel,
        )
        output, kv = result if return_kv else (result, None)
        x = x + gate1 * output
        if self.cross_attention_position == "after_self":
            x = self._apply_cross_attention(
                x, context, prefix_tokens, context_positions, cross_rope=cross_rope
            )
        x = x + gate2 * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return (x, kv) if return_kv else x

    def _apply_cross_attention(
        self, x, context, prefix_tokens, context_positions=None, *, cross_rope=None
    ):
        if self.cross_attn is None:
            return x
        if context is None:
            raise ValueError("Cross-attention requires condition tokens")
        target = x[:, prefix_tokens:]
        rope_options = (
            {} if cross_rope is None else dict(q_rope=cross_rope[0], k_rope=cross_rope[1])
        )
        if self.cross_attention_spatial_size is None:
            update = self.cross_attn(self.norm_cross(target), context, **rope_options)
        else:
            update = self.cross_attn(
                self.norm_cross(target),
                context,
                spatial_size=self.cross_attention_spatial_size,
                context_positions=context_positions,
                **rope_options,
            )
        target = target + update
        return torch.cat((x[:, :prefix_tokens], target), dim=1) if prefix_tokens else target


class WorldDiT(nn.Module):
    """Predict codec-space velocity; clean history visibility is explicit.

    ``prefix_frames`` is the zprev two-block path. ``teacher_forcing_planes``
    is the pure-TF interleaved clean/query path. They are mutually exclusive.
    Module names deliberately match the corresponding original ``net`` keys.
    """

    def __init__(self, config: WorldConfig):
        super().__init__()
        self.gradient_checkpointing = False
        self.sequence_parallel = True
        self.config = config
        cfg = config
        self.x_embedder = PatchEmbed3D(
            (cfg.state_planes, cfg.grid, cfg.grid),
            (1, cfg.patch_size, cfg.patch_size),
            cfg.channels,
            cfg.width,
        )
        self.semantic_entry_fusion = None
        if cfg.condition_mode == "entry_concat":
            self.x_embedder.proj = nn.Identity()
            self.x_embedder.embed_dim = cfg.channels
            self.semantic_entry_fusion = nn.Sequential(
                nn.Linear(cfg.channels + cfg.width, cfg.width),
                nn.SiLU(),
                nn.Linear(cfg.width, cfg.width),
            )
        self.t_embedder = TimestepEmbedder(cfg.width)
        self.emb_norm = RMSNorm(cfg.width)
        self.shared_mod = nn.Sequential(nn.SiLU(), nn.Linear(cfg.width, 6 * cfg.width))
        self.semantic_raw_projector = (
            nn.Sequential(
                nn.LayerNorm(cfg.condition_width), nn.Linear(cfg.condition_width, cfg.width)
            )
            if cfg.condition_width
            else None
        )
        self.null_semantic_context = (
            nn.Parameter(torch.zeros(1, 1, cfg.width)) if cfg.condition_width else None
        )
        self.feat_rope = VisionRotaryEmbeddingFast3D(
            cfg.width // cfg.heads,
            cfg.state_planes,
            cfg.grid // cfg.patch_size,
            cfg.grid // cfg.patch_size,
        )
        self.blocks = nn.ModuleList(
            WorldBlock(
                cfg,
                cross_attention=bool(cfg.condition_width)
                and cfg.condition_mode == "cross_attention"
                and index % cfg.cross_attention_every == 0,
            )
            for index in range(cfg.depth)
        )
        self.final_layer = (
            None if cfg.pixel_depth else FinalLayer(cfg.width, cfg.patch_size, cfg.channels)
        )
        self.pixel_refiner = (
            PixelRefiner(cfg.width, cfg.heads, cfg.qk_norm, cfg.pixel_width, cfg.pixel_depth)
            if cfg.pixel_depth
            else None
        )
        self.initialize_weights()

    def initialize_weights(self):
        """Also used by the task after Native meta materialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if self.semantic_entry_fusion is None:
            nn.init.xavier_uniform_(self.x_embedder.proj.weight.flatten(1))
            nn.init.zeros_(self.x_embedder.proj.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.shared_mod[-1].weight)
        nn.init.zeros_(self.shared_mod[-1].bias)
        for block in self.blocks:
            if self.config.adaln_mode == "adaln_lora":
                nn.init.zeros_(block.modulation.lora[-1].weight)
            elif self.config.adaln_mode == "per_block":
                nn.init.zeros_(block.modulation.full[-1].weight)
                nn.init.zeros_(block.modulation.full[-1].bias)
        if self.null_semantic_context is not None:
            nn.init.zeros_(self.null_semantic_context)
        if self.final_layer is not None:
            for module in (self.final_layer.adaLN_modulation[-1], self.final_layer.linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
        if self.pixel_refiner is not None:
            self.pixel_refiner.initialize_weights()
        # These buffers are persistent in original checkpoints. For scratch
        # materialization, rebuild instead of reading uninitialized to_empty data.
        rope = self.feat_rope
        with torch.no_grad():
            for axis in ("t", "h", "w"):
                buffer = getattr(rope, f"base_freqs_{axis}")
                dim = getattr(rope, f"dim_{axis}")
                buffer.copy_(
                    1 / (10000 ** (torch.arange(0, dim, 2, device=buffer.device).float() / dim))
                )
            cosine, sine = rope._build_freqs(
                rope._num_frames,
                rope.base_freqs_t,
                rope.base_freqs_h,
                rope.base_freqs_w,
                rope.frame_height,
                rope.frame_width,
                rope.dim,
            )
            rope.freqs_cos.copy_(cosine)
            rope.freqs_sin.copy_(sine)

    def _condition(self, tokens, batch, dtype, device, drop):
        if self.semantic_raw_projector is None:
            if tokens is not None:
                raise ValueError("Pure TF has no semantic conditioning")
            return None
        null = self.null_semantic_context.to(device=device, dtype=dtype)
        if tokens is None:
            return null.expand(batch, -1, -1)
        if (
            tokens.ndim != 3
            or tokens.shape[0] != batch
            or tokens.shape[-1] != self.config.condition_width
        ):
            raise ValueError("Expected condition [B,N,condition_width]")
        projected = self.semantic_raw_projector(
            tokens.to(self.semantic_raw_projector[1].weight.dtype)
        ).to(dtype)
        if drop is None and self.training and self.config.condition_dropout > 0:
            drop = share_random(
                torch.rand(batch, device=device) < self.config.condition_dropout,
                enabled=self.sequence_parallel,
            )
        if drop is not None:
            if drop.shape != (batch,):
                raise ValueError("Condition dropout must have shape [B]")
            projected = torch.where(drop.bool().view(batch, 1, 1), null, projected)
        return projected

    def forward(
        self,
        latent,
        timestep,
        *,
        condition=None,
        condition_drop=None,
        condition_positions=None,
        target_offsets=None,
        max_target_offset=2,
        prefix_frames=0,
        teacher_forcing_planes=0,
        frame_offset=0,
        past_key_values=None,
        return_key_values=False,
        causal_state_planes=0,
        repa_layer=None,
    ):
        cfg = self.config
        if (
            latent.ndim != 5
            or latent.shape[1] != cfg.channels
            or latent.shape[-2:] != (cfg.grid, cfg.grid)
        ):
            raise ValueError("World input must be [B,C,T,grid,grid]")
        if sum(bool(v) for v in (prefix_frames, teacher_forcing_planes, causal_state_planes)) > 1:
            raise ValueError("Choose one self-attention visibility mode")
        batch, _, frames, _, _ = latent.shape
        if repa_layer is not None and (
            not self.training or self.pixel_refiner is None
            or not 1 <= repa_layer <= len(self.blocks) or return_key_values
        ):
            raise ValueError("REPA requires a training-time PixelDiT patch layer")
        if not 0 <= prefix_frames < frames:
            raise ValueError("Prefix must leave at least one target plane")
        if cfg.condition_mode == "entry_concat" and (
            prefix_frames
            or teacher_forcing_planes
            or causal_state_planes
            or frame_offset
            or past_key_values is not None
            or frames != cfg.state_planes
        ):
            raise ValueError("Entry concat requires a single target-only no-zprev state")
        if target_offsets is not None:
            if (
                prefix_frames
                or teacher_forcing_planes
                or causal_state_planes
                or frame_offset
                or past_key_values is not None
                or frames != cfg.state_planes
                or not cfg.condition_width
            ):
                raise ValueError("Target offsets require one conditioned target-only state")
            validate_target_offsets(
                target_offsets,
                batch,
                latent.device,
                tokens=frames * (cfg.grid // cfg.patch_size) ** 2,
                max_offset=max_target_offset,
            )
        hidden = self.x_embedder(latent)
        per_plane = (cfg.grid // cfg.patch_size) ** 2
        length = frames * per_plane
        mask, positions = None, None
        segmented = use_segmented_attention()
        if prefix_frames:
            mask = (
                prefix_sparsity(length, prefix_frames * per_plane)
                if segmented
                else _build_prefix_block_causal_attn_mask(
                    seq_len=length,
                    tokens_per_frame=per_plane,
                    context_frames=prefix_frames,
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
            )
        elif teacher_forcing_planes:
            if cfg.condition_width or frame_offset or frames % (2 * teacher_forcing_planes):
                raise ValueError(
                    "Pure TF requires complete interleaved pairs, no condition or offset"
                )
            mask = (
                teacher_forcing_sparsity(length, teacher_forcing_planes * per_plane)
                if segmented
                else _build_full_history_teacher_forcing_attn_mask(
                    seq_len=length,
                    tokens_per_frame=per_plane,
                    state_frames=teacher_forcing_planes,
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
            )
            pairs = frames // (2 * teacher_forcing_planes)
            positions = (
                torch.arange(pairs, device=hidden.device)[:, None, None] * teacher_forcing_planes
                + torch.tensor([0, teacher_forcing_planes], device=hidden.device)[None, :, None]
                + torch.arange(teacher_forcing_planes, device=hidden.device)[None, None, :]
            ).flatten()
        elif causal_state_planes:
            if frames % causal_state_planes:
                raise ValueError("Causal sequence must contain complete states")
            past = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
            if segmented:
                mask = state_causal_sparsity(length, causal_state_planes * per_plane, past)
            else:
                q = torch.arange(past, past + length, device=hidden.device) // (
                    causal_state_planes * per_plane
                )
                k = torch.arange(past + length, device=hidden.device) // (
                    causal_state_planes * per_plane
                )
                mask = k[None, :] <= q[:, None]
            # The source pure-TF cache is sized for the full training window.
            # Every valid rollout prefix is shorter, so it computes RoPE under
            # the active autocast context, including at offset zero. Reusing
            # our one-state fp32 table would change that first KV prefill.
            positions = torch.arange(frame_offset, frame_offset + frames, device=hidden.device)
        if past_key_values is not None and (prefix_frames or teacher_forcing_planes):
            raise ValueError("KV prefill is only for causal pure-TF inference")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("One KV pair is required for each World block")
        parallel = usp_state(self.sequence_parallel) is not None
        if (cfg.cross_attention_scope == "same_spatial" or cfg.cross_attention_3d_rope) and (
            parallel
            or prefix_frames
            or teacher_forcing_planes
            or causal_state_planes
            or frame_offset
            or past_key_values is not None
            or frames != cfg.state_planes
        ):
            raise ValueError("Position-aware CA requires USP=1 and one target-only no-zprev state")
        start, local_length = token_range(length, self.sequence_parallel)
        rope = partial(
            self.feat_rope,
            num_frames_override=frames,
            start_frame=0 if positions is not None else frame_offset,
            frame_position_ids=positions,
            token_slice=(start, local_length) if parallel else None,
        )
        tokenwise_offsets = target_offsets is not None and target_offsets.ndim == 2
        target_positions = None
        if tokenwise_offsets:
            # A mosaic contains different times within one state. Both SA Q and K
            # must see those times. SA uses the ordinary target as t=0: offset*P+plane.
            target_positions = (
                target_offsets * length + torch.arange(length, device=hidden.device)[None]
            )
            rope = partial(
                self.feat_rope,
                num_frames_override=4 * cfg.state_planes,
                token_position_ids=target_positions[:, start : start + local_length],
            )
        if timestep.ndim == 1:
            timestep = timestep[:, None]
        if timestep.shape == (batch, 1):
            timestep = timestep.expand(batch, frames)
        if (
            timestep.ndim != 2
            or timestep.shape[0] != batch
            or timestep.shape[1] not in (frames, length)
        ):
            raise ValueError("Timestep must have one value per sample, plane, or token")
        emb = self.emb_norm(self.t_embedder(timestep))
        if timestep.shape[1] != length:
            emb = emb[:, torch.arange(length, device=hidden.device) // per_plane]
        shared = self.shared_mod(emb) if cfg.adaln_mode != "per_block" else None
        context = self._condition(condition, batch, hidden.dtype, hidden.device, condition_drop)
        if self.semantic_entry_fusion is not None:
            if context is None:
                raise ValueError("Entry concat requires projected condition tokens")
            if condition_positions is not None:
                if (
                    condition_positions.shape != context.shape[:2]
                    or condition_positions.dtype != torch.long
                ):
                    raise ValueError("Condition positions must be int64 [B,N]")
                if bool(((condition_positions < 0) | (condition_positions >= length)).any()):
                    raise ValueError("Condition positions are outside the target spatial grid")
                if condition_positions.shape[1] > 1 and bool(
                    (condition_positions.sort(dim=1).values.diff(dim=1) == 0).any()
                ):
                    raise ValueError("Condition positions must be unique within each state")
                # Only visible ViT outputs return to their original coordinates.
                # Masked positions are zero after projection; no clean shortcut.
                aligned = context.new_zeros(batch, length, cfg.width).scatter(
                    1, condition_positions[..., None].expand(-1, -1, cfg.width), context
                )
            elif context.shape[1] == length:
                aligned = context
            elif condition is None and context.shape[1] == 1:
                aligned = context.expand(-1, length, -1)
            else:
                raise ValueError("Entry concat requires dense tokens or explicit spatial positions")
            hidden = self.semantic_entry_fusion(torch.cat((hidden, aligned), dim=-1))
        elif cfg.cross_attention_scope == "same_spatial" or cfg.cross_attention_3d_rope:
            if context is None:
                raise ValueError("Same-spatial CA requires condition tokens")
            if condition_positions is not None:
                if (
                    condition_positions.shape != context.shape[:2]
                    or condition_positions.dtype != torch.long
                    or condition_positions.device != context.device
                ):
                    raise ValueError(
                        "Condition positions must be int64 [B,N] on the context device"
                    )
                if bool(((condition_positions < 0) | (condition_positions >= length)).any()):
                    raise ValueError("Condition positions are outside the target spatial grid")
                if condition_positions.shape[1] > 1 and bool(
                    (condition_positions.sort(dim=1).values.diff(dim=1) == 0).any()
                ):
                    raise ValueError("Condition positions must be unique within each state")
            elif condition is None and context.shape[1] == 1:
                context = context.expand(-1, length, -1)
            elif context.shape[1] != length:
                raise ValueError(
                    "Same-spatial CA requires dense tokens or explicit spatial positions"
                )
        elif condition_positions is not None:
            raise ValueError("Cross-attention accepts visible tokens without spatial scatter")
        if target_offsets is not None:
            gap = target_gap_embedding(target_offsets, cfg.width, hidden.dtype)
            hidden = hidden + (gap if tokenwise_offsets else gap[:, None])
        cross_rope = None
        if cfg.cross_attention_3d_rope:
            # Shared latent-plane units: previous state [0,P), next state [P,2P).
            # Rebase both by the same history offset; Q/K relative coordinates are unchanged.
            q_rope = partial(
                self.feat_rope, num_frames_override=frames, start_frame=cfg.state_planes
            )
            if tokenwise_offsets:
                q_rope = partial(
                    self.feat_rope,
                    num_frames_override=5 * cfg.state_planes,
                    # CA keeps condition as t=0, one state earlier than the SA origin.
                    token_position_ids=target_positions + length,
                )
            elif target_offsets is not None:
                # Zero-based offset: Q at (offset+1)*P + [0,P), K at [0,P).
                query_positions = (target_offsets[:, None] + 1) * length + torch.arange(
                    length, device=hidden.device
                )[None]
                q_rope = partial(
                    self.feat_rope,
                    num_frames_override=5 * cfg.state_planes,
                    token_position_ids=query_positions,
                )
            cross_rope = (
                q_rope,
                partial(
                    self.feat_rope,
                    num_frames_override=cfg.state_planes,
                    token_position_ids=condition_positions,
                ),
            )
        hidden = shard_tokens(hidden, enabled=self.sequence_parallel)
        emb = shard_tokens(emb, enabled=self.sequence_parallel)
        shared = shard_tokens(shared, enabled=self.sequence_parallel)
        local_prefix = min(max(prefix_frames * per_plane - start, 0), local_length)
        new_cache = []
        repa_hidden = None
        for index, block in enumerate(self.blocks):
            run_block = block
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                if return_key_values or past_key_values is not None:
                    raise ValueError(
                        "Training activation checkpointing cannot retain inference KV state"
                    )
                run_block = partial(self._gradient_checkpointing_func, block)
            result = run_block(
                hidden,
                emb,
                shared,
                rope=rope,
                mask=mask,
                context=context,
                context_positions=condition_positions,
                cross_rope=cross_rope,
                prefix_tokens=local_prefix,
                past_kv=None if past_key_values is None else past_key_values[index],
                return_kv=return_key_values,
                sequence_parallel=parallel,
            )
            if return_key_values:
                hidden, current = result
                new_cache.append(current)
            else:
                hidden = result
            if repa_layer == index + 1:
                repa_hidden = hidden
        if self.pixel_refiner is not None:
            if parallel or return_key_values or past_key_values is not None or frames != cfg.state_planes:
                raise ValueError("PixelDiT supports one complete RGB state without USP or World KV cache")
            velocity = self.pixel_refiner(
                latent, hidden, emb,
                partial(self.feat_rope, num_frames_override=1),
                gradient_checkpointing=self.gradient_checkpointing,
            )
            return (velocity, repa_hidden) if repa_layer is not None else velocity
        output = gather_tokens(self.final_layer(hidden, emb), enabled=self.sequence_parallel)
        side, patch = cfg.grid // cfg.patch_size, cfg.patch_size
        output = output.reshape(batch, frames, side, side, 1, patch, patch, cfg.channels)
        output = torch.einsum("nthwpqrc->nctphqwr", output).reshape(
            batch, cfg.channels, frames, cfg.grid, cfg.grid
        )
        return (output, new_cache) if return_key_values else output
