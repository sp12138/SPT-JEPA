"""Joint causal conditioner and World-DiT with the original latent FM objective."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache, partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from vforge.models.rep_world.conditioner import CausalViT
from vforge.models.rep_world.config import CodecGeometry, ConditionerConfig, RepaConfig, WorldConfig
from vforge.models.rep_world.horizon import (
    half_future_targets,
    quadrant_future_targets,
    sample_available_target_offsets,
    sample_half_offsets,
    sample_quadrant_offsets,
    sample_target_offsets,
    sample_token_offsets,
    token_future_targets,
    validate_quadrant_geometry,
    validate_token_mosaic_geometry,
)
from vforge.models.rep_world.sequence import share_random
from vforge.models.rep_world.repa import TemporalREPAProjector
from vforge.models.rep_world.split_conditioner import SplitViTConditioner
from vforge.models.rep_world.world import WorldDiT


@dataclass(frozen=True)
class FlowConfig:
    mode: str = "no_zprev"
    mean: float = 0.8
    std: float = 0.8
    shift: float = -1.0
    base_shift: float = 2.667
    distant_target_prob: float = 0.0
    target_layout: str = "single_state"
    future_target_choices: int = 3
    target_origin_reserve: int | None = None
    future_target_policy: str = "legacy"
    df_train_time_bins: int = 50

    def __post_init__(self):
        if self.target_layout not in {
            "single_state",
            "quadrant_mosaic",
            "left_right_mosaic",
            "random_token_mosaic",
        }:
            raise ValueError(
                "Unknown target_layout; choose single_state/quadrant_mosaic/"
                "left_right_mosaic/random_token_mosaic"
            )
        if self.future_target_policy not in {"legacy", "valid_uniform"}:
            raise ValueError("future_target_policy must be legacy or valid_uniform")
        valid_uniform = self.future_target_policy == "valid_uniform"
        supported = {1, 2, 3, 4} if valid_uniform else {2, 3}
        if (
            type(self.future_target_choices) is not int
            or self.future_target_choices not in supported
        ):
            raise ValueError(f"future_target_choices must be one of {sorted(supported)}")
        if valid_uniform and (
            self.target_layout != "single_state"
            or self.mode != "no_zprev"
            or self.distant_target_prob != 0
            or self.target_origin_reserve != 1
        ):
            raise ValueError(
                "valid_uniform requires single_state, no_zprev, distant_target_prob=0, "
                "and target_origin_reserve=1"
            )
        if self.target_origin_reserve is not None:
            if type(self.target_origin_reserve) is not int or self.target_origin_reserve < (
                1 if valid_uniform else self.required_future_states
            ):
                raise ValueError("target_origin_reserve must cover every possible future target")
            if self.mode != "no_zprev":
                raise ValueError("Matched target origins require flow.mode=no_zprev")
        if self.target_layout in {"left_right_mosaic", "random_token_mosaic"}:
            if self.mode != "no_zprev" or self.distant_target_prob != 0:
                raise ValueError("Two-horizon mosaics require no_zprev and distant_target_prob=0")
        if self.target_layout == "quadrant_mosaic":
            if self.mode != "no_zprev":
                raise ValueError("Quadrant targets require flow.mode=no_zprev")
            if self.distant_target_prob != 0:
                raise ValueError(
                    "Quadrant targets use all four horizons; distant_target_prob must be 0"
                )
        if isinstance(self.distant_target_prob, bool) or not 0 <= self.distant_target_prob <= 1:
            raise ValueError("distant_target_prob must be in [0,1]")
        if self.distant_target_prob and self.mode != "no_zprev":
            raise ValueError("Distant targets require flow.mode=no_zprev")
        if self.mode not in {"no_zprev", "zprev", "pure_tf", "pure_df"}:
            raise ValueError("mode must be no_zprev, zprev, pure_tf, or pure_df")
        if type(self.df_train_time_bins) is not int or self.df_train_time_bins < 2:
            raise ValueError("df_train_time_bins must be an integer >= 2")
        if not all(math.isfinite(x) for x in (self.mean, self.std, self.shift, self.base_shift)):
            raise ValueError("Flow parameters must be finite")
        if self.base_shift <= 0 or self.shift == 0:
            raise ValueError("Flow shift must be positive, or negative for automatic")

    @property
    def is_mosaic(self):
        return self.target_layout in {"quadrant_mosaic", "left_right_mosaic", "random_token_mosaic"}

    @property
    def required_future_states(self):
        if self.target_layout == "quadrant_mosaic":
            return 4
        if self.target_layout in {"left_right_mosaic", "random_token_mosaic"}:
            return 2
        return (
            self.future_target_choices
            if self.future_target_policy == "valid_uniform" or self.distant_target_prob
            else 1
        )

    @property
    def uses_future_targets(self):
        return (
            self.is_mosaic
            or bool(self.distant_target_prob)
            or (self.future_target_policy == "valid_uniform" and self.future_target_choices > 1)
        )

    @property
    def uses_target_selection(self):
        return self.uses_future_targets or self.target_origin_reserve is not None

    def target_count(self, states):
        reserve = self.target_origin_reserve or self.required_future_states
        count = states - reserve
        if count < 1:
            raise ValueError(f"Targets need at least {reserve + 1} states")
        return count


def shift_timestep(timestep, shift):
    return timestep if shift == 1 else shift * timestep / (1 + (shift - 1) * timestep)


@lru_cache(maxsize=16)
def _incremental_timestep_weights(states, bins):
    """Combinatorial weights of the source DF monotone discrete-time sampler."""
    before = torch.empty(bins, states, dtype=torch.float64)
    after = torch.empty(bins, states, dtype=torch.float64)
    for time in range(bins):
        for state in range(states):
            before[time, state] = math.comb(time + state, state)
            after[time, state] = math.comb(bins - 1 - time + states - 1 - state, states - 1 - state)
    return before, after


def sample_incremental_timesteps(batch, states, bins, device):
    """Sample nondecreasing DF noise bins using the torch CPU RNG (checkpointable)."""
    before, after = _incremental_timestep_weights(states, bins)
    result = torch.empty(batch, states, dtype=torch.long)
    anchors = torch.randint(states, (batch,))
    anchor_times = torch.randint(bins, (batch,))
    for sample in range(batch):
        anchor = int(anchors[sample])
        result[sample, anchor] = anchor_times[sample]
        for state in range(anchor - 1, -1, -1):
            upper = int(result[sample, state + 1])
            result[sample, state] = torch.multinomial(before[: upper + 1, state], 1)[0]
        for state in range(anchor + 1, states):
            lower = int(result[sample, state - 1])
            result[sample, state] = lower + torch.multinomial(after[lower:, state], 1)[0]
    return result.to(device=device)


def unpack_states(packed, geometry):
    if (
        packed.ndim != 5
        or packed.shape[1] != geometry.packed_channels
        or packed.shape[-2:] != (geometry.grid, geometry.grid)
    ):
        raise ValueError("Packed latent shape must match codec geometry")
    batch, _, states, height, width = packed.shape
    return packed.reshape(batch, geometry.planes, geometry.channels, states, height, width).permute(
        0, 3, 2, 1, 4, 5
    )


def pack_states(states):
    if states.ndim != 6:
        raise ValueError("Expected [B,S,C,P,H,W] explicit states")
    batch, length, channels, planes, height, width = states.shape
    return states.permute(0, 3, 2, 1, 4, 5).reshape(batch, planes * channels, length, height, width)


class RepWorld(nn.Module):
    """The single Native execution root owns every trainable parameter.

    Frozen codecs are separate inference components, not members of this
    optimizer/checkpoint graph. ``operation`` lets the eval runner use its
    injected engine_forward for both conditioning and denoising.
    """

    _no_split_modules = ["WorldBlock", "CausalViTBlock"]

    def __init__(
        self,
        geometry: CodecGeometry,
        world: WorldConfig,
        conditioner: ConditionerConfig | None,
        flow: FlowConfig,
        repa: RepaConfig | None = None,
    ):
        super().__init__()
        self.geometry, self.flow = geometry, flow
        self.repa = repa or RepaConfig()
        if self.repa.enabled and (
            geometry.family != "pixel" or not world.pixel_depth
            or not flow.uses_target_selection or flow.target_layout != "single_state"
            or self.repa.align_layer > world.depth
        ):
            raise ValueError("REPA requires direct RGB PixelDiT with single-state future targets")
        self.sequence_parallel = True
        if (world.channels, world.state_planes, world.grid) != (
            geometry.channels,
            geometry.planes,
            geometry.grid,
        ):
            raise ValueError("World dimensions must agree with the selected codec")
        if flow.mode in {"pure_tf", "pure_df"}:
            if conditioner is not None or world.condition_width:
                raise ValueError("Pure TF/DF must not instantiate an unused conditioner")
        elif conditioner is None or conditioner.width != world.condition_width:
            raise ValueError("Conditioner width must match World condition width")
        if conditioner is not None and conditioner.input == "latent" and flow.mode != "no_zprev":
            raise ValueError("Existing latent-input baseline supports no_zprev only")
        if world.condition_mode == "entry_concat" and (
            flow.mode != "no_zprev"
            or conditioner is None
            or conditioner.input != "latent"
            or not geometry.spatially_ordered
        ):
            raise ValueError("Entry concat requires spatially ordered latent input and no_zprev")
        positioned_conditioner = conditioner is not None and (
            conditioner.input == "latent"
            or (conditioner.input == "rgb" and conditioner.architecture == "split_ec")
        )
        if (world.cross_attention_scope == "same_spatial" or world.cross_attention_3d_rope) and (
            flow.mode != "no_zprev" or not positioned_conditioner or not geometry.spatially_ordered
        ):
            raise ValueError(
                "Position-aware CA requires spatially ordered latent or grid-aligned split RGB "
                "input and no_zprev"
            )
        if flow.target_layout == "random_token_mosaic":
            validate_token_mosaic_geometry(geometry, world.patch_size)
        elif flow.is_mosaic:
            validate_quadrant_geometry(geometry, world.patch_size)
        # Preserve original denoiser and encoder parameter prefixes for import.
        self.net = WorldDiT(world)
        self.repa_projector = (
            TemporalREPAProjector(world.width, self.repa.teacher_width)
            if self.repa.enabled else None
        )
        conditioner_type = (
            SplitViTConditioner
            if conditioner is not None and conditioner.architecture == "split_ec"
            else CausalViT
        )
        self.causal_semantic_encoder = (
            conditioner_type(conditioner, geometry) if conditioner else None
        )
        # Original df_chunk_size=1 determines the automatic schedule shift,
        # even for codecs with multiple latent planes per state.
        self.timestep_shift = (
            flow.shift
            if flow.shift > 0
            else flow.base_shift * ((world.grid // world.patch_size) ** 2 / 600) ** 0.5
        )

    def initialize_weights(self):
        self.net.initialize_weights()
        if self.repa_projector is not None:
            for module in self.repa_projector.modules():
                if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                    nn.init.xavier_uniform_(module.weight.flatten(1))
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        if self.causal_semantic_encoder is not None:
            self.causal_semantic_encoder.initialize_weights()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        options = {"use_reentrant": False, **(gradient_checkpointing_kwargs or {})}
        if options["use_reentrant"]:
            raise ValueError("RepWorld activation checkpointing requires use_reentrant=false")
        for module in (self.net, self.causal_semantic_encoder):
            if module is not None:
                module.gradient_checkpointing = True
                module._gradient_checkpointing_func = partial(checkpoint, **options)

    def gradient_checkpointing_disable(self):
        for module in (self.net, self.causal_semantic_encoder):
            if module is not None:
                module.gradient_checkpointing = False

    def draw_timesteps(self, shape, reference, uniform=None):
        if uniform is None:
            uniform = share_random(
                torch.rand(shape, device=reference.device, dtype=reference.dtype).clamp_(
                    1e-5, 1 - 1e-5
                ),
                enabled=self.sequence_parallel,
            )
        elif tuple(uniform.shape) != tuple(shape):
            raise ValueError("Uniform timestep samples have wrong shape")
        t = uniform
        if self.flow.std > 0:
            t = torch.sigmoid(self.flow.mean + self.flow.std * torch.special.ndtri(t.double())).to(
                uniform.dtype
            )
        return shift_timestep(t, self.timestep_shift)

    def forward(
        self,
        latents=None,
        rgb=None,
        teacher_features=None,
        *,
        operation="loss",
        source=None,
        cache=None,
        cache_start_state=0,
        noisy=None,
        timestep=None,
        condition=None,
        previous=None,
        condition_drop=None,
        condition_positions=None,
        noise=None,
        uniform=None,
        target_uniform=None,
        quadrant_offsets=None,
        half_offsets=None,
        token_assignments=None,
        next_state_only=False,
        target_offsets=None,
        return_metrics=False,
        **world_kwargs,
    ):
        if operation in {"encode_states", "prefill_condition", "prefill_condition_cached"}:
            encoder = self.causal_semantic_encoder
            if not isinstance(encoder, SplitViTConditioner):
                raise ValueError("Separate encode/prefill operations require architecture=split_ec")
            if operation == "prefill_condition_cached":
                return encoder.prefill_cached(
                    source,
                    cache=cache,
                    start_state=cache_start_state,
                )
            return (
                encoder.encode_states(source)
                if operation == "encode_states"
                else encoder.prefill(source)[:, -1]
            )
        if operation == "condition":
            if self.causal_semantic_encoder is None:
                raise ValueError("Pure TF has no conditioner")
            return self.causal_semantic_encoder.next_context(source)
        if operation == "world":
            return self.net(
                noisy,
                timestep,
                condition=condition,
                condition_drop=condition_drop,
                target_offsets=target_offsets,
                max_target_offset=3
                if self.flow.future_target_policy == "valid_uniform"
                and self.flow.future_target_choices == 4
                else 2,
                **world_kwargs,
            )
        if operation == "velocity":
            return self.velocity(
                noisy,
                timestep,
                condition,
                previous,
                condition_drop,
                condition_positions,
                target_offsets,
            )
        if operation != "loss":
            raise ValueError(f"Unknown model operation {operation}")
        if latents is None:
            raise ValueError("Training requires frozen codec latents")
        states = unpack_states(latents.detach(), self.geometry)
        if states.shape[1] < 2:
            raise ValueError("Training needs at least two states")
        if target_offsets is not None:
            raise ValueError(
                "Loss samples offsets; use target_uniform or quadrant_offsets for controlled draws"
            )
        if quadrant_offsets is not None and self.flow.target_layout != "quadrant_mosaic":
            raise ValueError("quadrant_offsets requires target_layout=quadrant_mosaic")
        if type(next_state_only) is not bool:
            raise ValueError("next_state_only must be a boolean")
        if half_offsets is not None and self.flow.target_layout != "left_right_mosaic":
            raise ValueError("half_offsets requires target_layout=left_right_mosaic")
        if token_assignments is not None and self.flow.target_layout != "random_token_mosaic":
            raise ValueError("token_assignments requires target_layout=random_token_mosaic")
        if next_state_only and any(
            item is not None
            for item in (target_uniform, quadrant_offsets, half_offsets, token_assignments)
        ):
            raise ValueError("Next-state tuning does not accept future target assignments")
        if self.flow.uses_target_selection:
            return self._future_target_loss(
                latents,
                states,
                rgb,
                teacher_features,
                noise,
                uniform,
                target_uniform,
                quadrant_offsets,
                half_offsets,
                token_assignments,
                next_state_only,
                condition_drop,
                return_metrics,
            )
        if target_uniform is not None:
            raise ValueError("target_uniform requires distant_target_prob > 0")
        if self.flow.mode in {"pure_tf", "pure_df"}:
            loss = (
                self._full_history_loss(states, noise, uniform)
                if self.flow.mode == "pure_tf"
                else self._diffusion_forcing_loss(states, noise, uniform)
            )
            return {"loss": loss, "metrics": {}} if return_metrics else loss
        encoder = self.causal_semantic_encoder
        positions = None
        need_positions = (
            self.net.config.condition_mode == "entry_concat"
            or self.net.config.cross_attention_scope == "same_spatial"
            or self.net.config.cross_attention_3d_rope
        )
        if encoder.config.input == "rgb":
            if rgb is None or rgb.shape[:3] != (
                states.shape[0],
                3,
                states.shape[1] * self.geometry.rgb_frames,
            ):
                raise ValueError("RGB clip and latent states must align")
            source = rgb
        else:
            source = latents
        encoded = encoder.training_context(source, return_indices=need_positions)
        if need_positions:
            contexts, positions = encoded
            if positions is not None:
                positions = positions.reshape(-1, positions.shape[-1])
        else:
            contexts = encoded
        contexts = contexts.reshape(-1, contexts.shape[-2], contexts.shape[-1])
        batch, length, channels, planes, height, width = states.shape
        target = states[:, 1:].reshape(batch * (length - 1), channels, planes, height, width)
        t = self.draw_timesteps((target.shape[0], 1), target, uniform)
        if noise is None:
            noise = share_random(torch.randn_like(target), enabled=self.sequence_parallel)
        if noise.shape != target.shape:
            raise ValueError("Noise must match flattened target states")
        t_view = t.reshape(-1, 1, 1, 1, 1)
        noisy = (1 - t_view) * target + t_view * noise
        previous = states[:, :-1].reshape_as(target) if self.flow.mode == "zprev" else None
        prediction = self.velocity(noisy, t, contexts, previous, condition_drop, positions)
        loss = (target - noise - prediction).square().mean()
        return {"loss": loss, "metrics": {}} if return_metrics else loss

    def _future_target_loss(
        self,
        latents,
        states,
        rgb,
        teacher_features,
        noise,
        uniform,
        target_uniform,
        quadrant_offsets,
        half_offsets,
        token_assignments,
        next_state_only,
        condition_drop,
        return_metrics,
    ):
        batch, length, channels, planes, height, width = states.shape
        count = self.flow.target_count(length)
        mosaic = self.flow.is_mosaic and not next_state_only
        if mosaic and self.flow.target_layout == "quadrant_mosaic":
            if target_uniform is not None:
                raise ValueError("Quadrant targets use quadrant_offsets, not target_uniform")
            assignment = sample_quadrant_offsets(
                (batch, count),
                states.device,
                offsets=quadrant_offsets,
                sequence_parallel=self.sequence_parallel,
            )
            target, token_offsets, pixel_offsets = quadrant_future_targets(
                states[:, : count + 4], assignment, self.net.config.patch_size
            )
        elif mosaic and self.flow.target_layout == "random_token_mosaic":
            if target_uniform is not None:
                raise ValueError("Random token targets use token_assignments, not target_uniform")
            patch = self.net.config.patch_size
            tokens = planes * (height // patch) * (width // patch)
            assignment = sample_token_offsets(
                (batch, count),
                tokens,
                states.device,
                offsets=token_assignments,
                sequence_parallel=self.sequence_parallel,
            )
            target, token_offsets, pixel_offsets = token_future_targets(states, assignment, patch)
        elif mosaic:
            if target_uniform is not None:
                raise ValueError("Half targets use half_offsets, not target_uniform")
            assignment = sample_half_offsets(
                (batch, count),
                states.device,
                offsets=half_offsets,
                sequence_parallel=self.sequence_parallel,
            )
            target, token_offsets, pixel_offsets = half_future_targets(
                states, assignment, self.net.config.patch_size
            )
        else:
            if self.flow.future_target_policy == "valid_uniform":
                offsets = sample_available_target_offsets(
                    (batch, count),
                    length,
                    1 if next_state_only else self.flow.future_target_choices,
                    states.device,
                    uniform=target_uniform,
                    sequence_parallel=self.sequence_parallel,
                )
            else:
                offsets = sample_target_offsets(
                    (batch, count),
                    0.0 if next_state_only else self.flow.distant_target_prob,
                    states.device,
                    choices=self.flow.future_target_choices,
                    uniform=target_uniform,
                    sequence_parallel=self.sequence_parallel,
                )
            origins = torch.arange(count, device=states.device).expand(batch, -1)
            indices = origins + 1 + offsets
            target = states[torch.arange(batch, device=states.device)[:, None], indices]
            token_offsets = offsets.flatten()
        # Encode only origins; the state-causal ViT limits prefix i to states <= i.
        # Skipped future states never enter their origin's condition.
        encoder = self.causal_semantic_encoder
        positions = None
        need_positions = (
            self.net.config.condition_mode == "entry_concat"
            or self.net.config.cross_attention_scope == "same_spatial"
            or self.net.config.cross_attention_3d_rope
        )
        if encoder.config.input == "rgb":
            if rgb is None or rgb.shape[:3] != (batch, 3, length * self.geometry.rgb_frames):
                raise ValueError("RGB clip and latent states must align")
            source = rgb[:, :, : count * self.geometry.rgb_frames]
        else:
            source = latents[:, :, :count].detach()
        encoded = encoder(source, return_indices=need_positions)
        if need_positions:
            contexts, positions = encoded
            if positions is not None:
                positions = positions.reshape(-1, positions.shape[-1])
        else:
            contexts = encoded
        contexts = contexts.reshape(batch * count, contexts.shape[-2], contexts.shape[-1])
        target = target.reshape(batch * count, channels, planes, height, width)
        t = self.draw_timesteps((batch * count, 1), target, uniform)
        if noise is None:
            noise = share_random(torch.randn_like(target), enabled=self.sequence_parallel)
        if noise.shape != target.shape:
            raise ValueError("Noise must match flattened future targets [B*origins,C,P,H,W]")
        t_view = t.reshape(-1, 1, 1, 1, 1)
        noisy = (1 - t_view) * target + t_view * noise
        prediction = self.velocity(
            noisy,
            t,
            contexts,
            condition_drop=condition_drop,
            condition_positions=positions,
            target_offsets=token_offsets,
            repa_layer=self.repa.align_layer if self.repa.enabled else None,
        )
        repa_loss = None
        if self.repa.enabled:
            if (teacher_features is None or teacher_features.shape[:2] != (batch, length)
                or teacher_features.shape[2:] != (
                    self.repa.teacher_width, 2 * planes, height, width
                )):
                raise ValueError("REPA teacher features must cover the clean RGB states")
            prediction, repa_hidden = prediction
            selected_teacher = teacher_features[
                torch.arange(batch, device=states.device)[:, None], indices
            ].reshape(batch * count, self.repa.teacher_width, 2 * planes, height, width)
            aligned = self.repa_projector(repa_hidden, planes, height)
            repa_loss = (
                1 - F.cosine_similarity(aligned.float(), selected_teacher.float(), dim=1)
            ).mean()
        squared = (target - noise - prediction).square()
        fm_loss = squared.mean()
        loss = fm_loss + self.repa.weight * repa_loss if repa_loss is not None else fm_loss
        if not return_metrics:
            return loss
        if mosaic and self.flow.target_layout == "random_token_mosaic":
            per_target = squared.detach().mean(dim=1)
        else:
            per_target = (
                squared.detach().mean(dim=(1, 2)) if mosaic else squared.detach().flatten(1).mean(1)
            )
        metric_offsets = (pixel_offsets if mosaic else offsets.flatten()) + 1
        metrics = {"loss_fm": fm_loss.detach()}
        if repa_loss is not None:
            metrics["loss_repa"] = repa_loss.detach()
        for gap in range(1, self.flow.required_future_states + 1):
            selected = (metric_offsets == gap).to(per_target.dtype)
            metrics[f"target_horizon_{gap}_fraction"] = selected.mean()
            # Contributions sum to loss_fm, including batches with no selected gap.
            metrics[f"loss_fm_horizon_{gap}_contribution"] = (per_target * selected).mean()
        return {"loss": loss, "metrics": metrics}

    def velocity(
        self,
        noisy,
        timestep,
        condition,
        previous=None,
        condition_drop=None,
        condition_positions=None,
        target_offsets=None,
        repa_layer=None,
    ):
        if target_offsets is not None and self.flow.mode != "no_zprev":
            raise ValueError("Target offsets require flow.mode=no_zprev")
        if self.flow.mode in {"pure_tf", "pure_df"}:
            raise ValueError("Pure TF/DF sampling uses the explicit World KV operation")
        if self.flow.mode == "zprev":
            if previous is None or previous.shape != noisy.shape:
                raise ValueError("zprev requires a matching clean previous state")
            frames = noisy.shape[2]
            time = timestep.reshape(noisy.shape[0], -1).expand(-1, frames)
            output = self.net(
                torch.cat((previous, noisy), 2),
                torch.cat((torch.zeros_like(time), time), 1),
                condition=condition,
                condition_drop=condition_drop,
                prefix_frames=frames,
            )
            return output[:, :, frames:]
        if previous is not None:
            raise ValueError("no_zprev must not receive clean previous-state tokens")
        return self.net(
            noisy,
            timestep,
            condition=condition,
            condition_drop=condition_drop,
            condition_positions=condition_positions,
            target_offsets=target_offsets,
            repa_layer=repa_layer,
            max_target_offset=3
            if self.flow.future_target_policy == "valid_uniform"
            and self.flow.future_target_choices == 4
            else 2,
        )

    def _full_history_loss(self, states, noise, uniform):
        batch, length, channels, planes, height, width = states.shape
        target_count = length - 1
        clean = states[:, :-1].permute(0, 2, 1, 3, 4, 5).contiguous()
        targets = states[:, 1:].permute(0, 2, 1, 3, 4, 5).contiguous()
        t = self.draw_timesteps((batch, target_count), targets, uniform)
        if noise is None:
            noise = share_random(torch.randn_like(targets), enabled=self.sequence_parallel)
        if noise.shape != targets.shape:
            raise ValueError("Pure TF noise must match [B,C,S-1,P,H,W]")
        t_view = t.reshape(batch, 1, target_count, 1, 1, 1)
        noisy = (1 - t_view) * targets + t_view * noise
        model_input = torch.stack((clean, noisy), 3).reshape(
            batch, channels, 2 * target_count * planes, height, width
        )
        t_frames = t.unsqueeze(-1).expand(batch, target_count, planes)
        model_time = torch.stack((torch.zeros_like(t_frames), t_frames), 2).reshape(
            batch, 2 * target_count * planes
        )
        output = self.net(model_input, model_time, teacher_forcing_planes=planes)
        prediction = output.reshape(batch, channels, target_count, 2, planes, height, width)[
            :, :, :, 1
        ]
        return (targets - noise - prediction).square().mean()

    def _diffusion_forcing_loss(self, states, noise, uniform):
        batch, length, channels, planes, height, width = states.shape
        targets = states[:, 1:].permute(0, 2, 1, 3, 4, 5).contiguous()
        if uniform is None:
            bins = sample_incremental_timesteps(
                batch, length - 1, self.flow.df_train_time_bins, states.device
            )
            uniform = bins.to(dtype=states.dtype) / (self.flow.df_train_time_bins - 1)
        elif uniform.shape != (batch, length - 1):
            raise ValueError("DF uniform timesteps must be [B,S-1]")
        elif bool((uniform[:, 1:] < uniform[:, :-1]).any()):
            raise ValueError("DF timesteps must be nondecreasing across target states")
        times = self.draw_timesteps((batch, length - 1), targets, uniform)
        if noise is None:
            noise = share_random(torch.randn_like(targets), enabled=self.sequence_parallel)
        if noise.shape != targets.shape:
            raise ValueError("Pure DF noise must match [B,C,S-1,P,H,W]")
        noisy = (1 - times[:, None, :, None, None, None]) * targets + (
            times[:, None, :, None, None, None] * noise
        )
        model_input = torch.cat((states[:, 0].unsqueeze(2), noisy), dim=2).reshape(
            batch, channels, length * planes, height, width
        )
        model_time = torch.cat((torch.zeros_like(times[:, :1]), times), dim=1)
        model_time = model_time.repeat_interleave(planes, dim=1)
        output = self.net(model_input, model_time, causal_state_planes=planes)
        prediction = output.reshape(batch, channels, length, planes, height, width)[:, :, 1:]
        return (targets - noise - prediction).square().mean()
