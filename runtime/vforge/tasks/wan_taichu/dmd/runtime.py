"""Taichu DMD random draws, score noising and incremental-KV rollout."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from vforge.algorithms.dmd.model import (
    DistributionMatchingData,
    DmdNoiseLevel,
    LatentBundle,
    PreparedDmdBatch,
)
from vforge.algorithms.dmd.objectives import remap_dmd_timesteps
from vforge.algorithms.dmd.runtime import (
    BlockReplayState,
    DistributionMatchingRuntime,
    DmdRolloutSchedule,
    build_paired_data,
    detached_bundle,
    level_tensor,
    matching_bundle_keys,
)
from vforge.utils.rng import RandomStreams


@dataclass(frozen=True)
class WanDistributionMatchingData(DistributionMatchingData):
    random_context: Any = None


def build_wan_dmd_data(family, *, config, seed, stream_seed_offsets, random_seed):
    runtime = WanDistributionMatchingRuntime(
        family=family, config=config, seed=seed, random_seed=random_seed
    )
    data = build_paired_data(family, runtime=runtime, stream_seed_offsets=stream_seed_offsets)
    return WanDistributionMatchingData(**vars(data), random_context=runtime.random_context)


class WanDistributionMatchingRuntime(DistributionMatchingRuntime):
    """Preserve Taichu's global RNG and rounding without changing other families."""

    def __init__(self, *, random_seed=None, **kwargs):
        super().__init__(**kwargs)
        self._rollout_exit = None
        self.random_seed = self.seed if random_seed is None else int(random_seed)

    @contextmanager
    def random_context(self, streams: RandomStreams, device: torch.device):
        """Own Taichu's phase-shared RNG without reseeding CT/TF or the data loader.

        Taichu seeds zero-worker training from its data seed and creates two
        DataLoader iterators before the first phase. Their base-seed draws are
        part of the CPU decision stream. Named states are checkpointed normally;
        model construction and lazy frozen-encoder loading cannot perturb them.
        """
        from vforge.distributed.parallel_state import get_parallel_state

        state = get_parallel_state()
        seed = (self.random_seed + int(state.dp_rank) * 1_000_003) & 0xFFFFFFFF
        name = "wan_dmd.cpu"
        fresh = name not in streams.state_dict()["streams"]
        cpu = streams.generator(name, seed=seed, device=torch.device("cpu"))
        if fresh:
            for _ in ("generator", "critic"):
                torch.empty((), dtype=torch.int64).random_(generator=cpu)
        cuda = (
            streams.generator("wan_dmd.cuda", seed=seed, device=device)
            if device.type == "cuda"
            else None
        )
        with torch.random.fork_rng(devices=[device] if cuda is not None else []):
            torch.set_rng_state(cpu.get_state())
            if cuda is not None:
                torch.cuda.set_rng_state(cuda.get_state(), device)
            try:
                yield
            finally:
                cpu.set_state(torch.get_rng_state())
                if cuda is not None:
                    cuda.set_state(torch.cuda.get_rng_state(device))

    def build_scheduler(self):
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.config.score.num_train_timesteps,
            shift=self.config.score.timestep_shift,
        )

        # Match the self-forcing grid, excluding the trailing zero sigma.
        steps = int(self.config.score.num_train_timesteps)
        scheduler.set_timesteps(steps)
        scheduler.sigmas = scheduler.sigmas[:steps].detach().clone()
        scheduler.timesteps = scheduler.timesteps[:steps].detach().clone()
        return scheduler

    def build_rollout_schedule(self):
        config = self.config
        return DmdRolloutSchedule(
            noise_levels=tuple(
                step / config.score.num_train_timesteps for step in config.rollout.denoising_steps
            ),
            terminal_is_boundary=config.rollout.terminal_zero_is_boundary,
            context_noise_level=config.rollout.context_noise / config.score.num_train_timesteps,
            # The sampled exit is the only rung that carries gradients.
            last_step_only=True,
        )

    def rollout(
        self, generator_model, adapter, prepared, random_streams, phase, seed_offset, *, trace=None
    ):
        # RandomStreams checkpoints the process RNG too. Keep Taichu draws on
        # that stream, in their original order, rather than splitting by phase.
        del random_streams, seed_offset
        initial_noise = {key: _noise_like(prepared.clean[key]) for key in sorted(prepared.clean)}
        schedule = self.rollout_schedule
        if phase not in {"generator", "critic"}:
            raise ValueError(f"unsupported DMD rollout phase {phase!r}")
        matching_bundle_keys(initial_noise, prepared.clean)
        forward_levels = schedule.forward_noise_levels
        if not forward_levels:
            raise ValueError("DMD rollout has no generator-forward noise levels")

        outputs: list[tuple[Any, LatentBundle]] = []
        reference = next(iter(initial_noise.values()))
        with adapter.open_rollout_context(generator_model, prepared) as context:
            blocks = adapter.block_layout(prepared, initial_noise)
            if not blocks:
                raise ValueError("DMD generator adapter returned no rollout blocks")
            self._rollout_exit = self._rollout_exit_index(reference.device, len(blocks))
            exit_index = self._rollout_exit
            for block in blocks:
                matching_bundle_keys(block.initial_value, initial_noise)
                if block.is_anchor:
                    accepted = detached_bundle(block.initial_value)
                else:
                    noisy_block = block.initial_value
                    accepted: LatentBundle | None = None
                    for index, normalized in enumerate(forward_levels[: exit_index + 1]):
                        level = self.resolve_rollout_level(
                            level_tensor(normalized, reference),
                            prepared,
                        )
                        enable_grad = phase == "generator" and index == exit_index
                        with torch.set_grad_enabled(enable_grad and torch.is_grad_enabled()):
                            accepted = adapter.denoise_block(
                                generator_model,
                                noisy_block,
                                block.selection,
                                level,
                                prepared,
                                context,
                            )
                        matching_bundle_keys(accepted, initial_noise)
                        if index < exit_index:
                            next_level = self.resolve_rollout_level(
                                level_tensor(forward_levels[index + 1], reference),
                                prepared,
                            )
                            with torch.no_grad():
                                noisy_block = _add_noise_bundle(
                                    accepted,
                                    next_level.sigmas,
                                )
                    assert accepted is not None
                outputs.append((block.selection, accepted))

                # What the loss scores is the sample the rollout exited on, but what
                # the KV cache carries forward should look like inference does. When
                # the exit was early, finish the remaining rungs without gradient so
                # the next block reads a fully denoised context instead of a coarse
                # one that would compound block by block.
                committed = accepted
                if self.config.rollout.context_full_trajectory and not block.is_anchor:
                    with torch.no_grad():
                        for normalized in forward_levels[exit_index + 1 :]:
                            level = self.resolve_rollout_level(
                                level_tensor(normalized, reference),
                                prepared,
                            )
                            committed = adapter.denoise_block(
                                generator_model,
                                _add_noise_bundle(
                                    committed,
                                    level.sigmas,
                                ),
                                block.selection,
                                level,
                                prepared,
                                context,
                            )

                context_level = self.resolve_rollout_level(
                    level_tensor(schedule.context_noise_level, reference),
                    prepared,
                )
                if schedule.context_noise_level <= 0.0:
                    context_level = replace(
                        context_level,
                        normalized=torch.zeros_like(context_level.normalized),
                        sigmas={
                            key: torch.zeros_like(value)
                            for key, value in context_level.sigmas.items()
                        },
                        model_timestep=torch.zeros_like(context_level.model_timestep),
                    )
                if trace is not None:
                    trace.append(
                        BlockReplayState(
                            block.selection,
                            detached_bundle(
                                block.initial_value if block.is_anchor else noisy_block
                            ),
                            context_level if block.is_anchor else level,
                            detached_bundle(accepted),
                            context_level,
                            block.is_anchor,
                        )
                    )
                with torch.no_grad():
                    # Clean history must not draw throwaway noise: it changes subsequent RNG.
                    context_value = (
                        detached_bundle(committed)
                        if schedule.context_noise_level <= 0.0
                        else _add_noise_bundle(
                            committed,
                            context_level.sigmas,
                        )
                    )
                    adapter.commit_block(
                        generator_model,
                        block.selection,
                        context_value,
                        context_level,
                        prepared,
                        context,
                    )
            return dict(adapter.concat_blocks(outputs, prepared))

    def sample_score_noise(
        self,
        clean: LatentBundle,
        prepared: PreparedDmdBatch[Any],
        random_streams: RandomStreams,
        seed_offset: int,
    ) -> tuple[DmdNoiseLevel[Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        reference = next(iter(clean.values()))
        del random_streams, seed_offset
        timestep = self.sample_score_timestep(prepared, reference, None)
        level = self.family.resolve_noise_level(
            self.scheduler,
            timestep.double() / self.config.score.num_train_timesteps,
            prepared,
        )
        noise = {
            key: _broadcast_sp(
                torch.randn(
                    clean[key].shape,
                    device=clean[key].device,
                    dtype=clean[key].dtype,
                )
            )
            for key in sorted(clean)
        }
        noisy = {
            key: _mix_noise(
                value,
                level.sigmas[key],
                noise[key],
            )
            for key, value in clean.items()
        }
        if self.family.postprocess_score_noise is not None:
            noisy, level = self.family.postprocess_score_noise(noisy, clean, level)
        level = replace(
            level,
            sigmas=self.family.resolve_noise_level(
                self.scheduler,
                level.model_timestep.double() / self.config.score.num_train_timesteps,
                prepared,
            ).sigmas,
        )
        return level, dict(noisy), noise

    def _rollout_exit_index(self, device: torch.device, num_blocks: int) -> int:
        """Draw one exit on rank zero so every shard runs the same expert."""

        count = len(self.rollout_schedule.forward_noise_levels)
        if self.config.rollout.last_step_only:
            return count - 1
        indices = torch.empty(num_blocks, device=device, dtype=torch.long)
        ready = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not ready or torch.distributed.get_rank() == 0:
            indices = torch.randint(count, (num_blocks,), device=device)
        if ready:
            torch.distributed.broadcast(indices, src=0)
        return int(indices[0].item())

    def _score_draw_bounds(self, device: torch.device) -> tuple[int, int]:
        """Choose an expert collectively, then draw locally within its noise band."""

        low = int(self.config.score.sampled_timestep_min)
        high = int(self.config.score.num_train_timesteps) + 1
        if self.config.score.timestep_floor_at_rollout_exit:
            low = max(low, self._rollout_exit_floor())
        split = self._score_expert_split()
        if split is None or high <= low or high <= split or low >= split:
            return low, high
        probability_high = (high - split) / (high - low)
        pick_high = torch.empty(1, device=device, dtype=torch.long)
        ready = torch.distributed.is_available() and torch.distributed.is_initialized()
        if not ready or torch.distributed.get_rank() == 0:
            pick_high.fill_(int(float(torch.rand(1).item()) < probability_high))
        if ready:
            torch.distributed.broadcast(pick_high, src=0)
        return (split, high) if bool(pick_high.item()) else (low, split)

    def _rollout_exit_floor(self) -> int:
        """Invert the next rollout boundary through the tabulated timestep grid."""

        if self._rollout_exit is None:
            raise RuntimeError("score noise must be sampled from a rollout's own exit rung")
        levels = self.rollout_schedule.noise_levels
        index = self._rollout_exit + 1
        steps = int(self.config.score.num_train_timesteps)
        if index >= len(levels):
            return 0
        target = float(levels[index]) * steps
        if target <= 0:
            return 0
        table = self.scheduler.timesteps.detach().to("cpu", torch.float64)
        return steps - int(torch.argmin((table - target).abs()).item())

    def _score_expert_split(self) -> int | None:
        """Find the raw draw where the score model switches experts."""

        ratio = getattr(self.family.models["fake_score"], "boundary_ratio", None)
        boundary = None if ratio is None else ratio * self.config.score.num_train_timesteps
        if boundary is None:
            return None
        steps = int(self.config.score.num_train_timesteps)
        table = self.scheduler.timesteps.detach().to("cpu", torch.float64)
        for draw in range(1, steps + 1):
            index = steps - draw
            if 0 <= index < table.numel() and float(table[index]) >= boundary:
                return draw
        # No tabulated level reaches the boundary, so every draw is low-noise.
        return steps + 1

    def sample_score_timestep(self, prepared, reference, random):
        low, high = self._score_draw_bounds(reference.device)
        timestep = torch.randint(
            low,
            high,
            self.family.score_noise_shape(prepared),
            device=reference.device,
            generator=random,
        )
        timestep = _broadcast_sp(timestep)
        return remap_dmd_timesteps(
            timestep,
            self.scheduler.timesteps,
            timestep_shift=self.config.score.timestep_shift,
            min_timestep=self.config.score.effective_timestep_clamp[0],
            max_timestep=self.config.score.effective_timestep_clamp[1],
            num_train_timesteps=self.config.score.num_train_timesteps,
        )


def _add_noise_bundle(clean, sigmas):
    matching_bundle_keys(clean, sigmas)
    return {key: _mix_noise(value, sigmas[key], _noise_like(value)) for key, value in clean.items()}


def _mix_noise(clean, sigma, noise):
    # Retain sigma precision and round only the interpolated result.
    sigma = sigma.to(device=clean.device)
    return ((1.0 - sigma) * clean + sigma * noise).to(noise.dtype)


def _noise_like(value):
    return _broadcast_sp(torch.randn(value.shape, device=value.device, dtype=value.dtype))


def _broadcast_sp(value):
    """Every rank draws, then SP peers use their leader's sample, as in Taichu."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        from vforge.distributed.parallel_state import get_parallel_state

        state = get_parallel_state()
        if state.config.usp_size > 1:
            group = state.usp_group
            torch.distributed.broadcast(
                value, src=torch.distributed.get_global_rank(group, 0), group=group
            )
    return value
