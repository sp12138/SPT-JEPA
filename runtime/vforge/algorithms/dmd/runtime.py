"""Model-independent runtime wiring for distribution matching."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from vforge.algorithms.dmd.config import DistributionMatchingConfig
from vforge.algorithms.dmd.model import (
    DistributionMatchingData,
    DmdFamilyComponents,
    DmdGeneratorAdapter,
    DmdNoiseLevel,
    LatentBundle,
    PreparedDmdBatch,
)
from vforge.algorithms.dmd.objectives import remap_dmd_timesteps
from vforge.data.multi_stream_loader import MultiStreamStatefulDataLoader
from vforge.utils.rng import RandomStreams


def build_distribution_matching_data(
    family: DmdFamilyComponents,
    *,
    config: DistributionMatchingConfig,
    seed: int,
    stream_seed_offsets: Mapping[str, int],
    runtime_type: type[DistributionMatchingRuntime] | None = None,
) -> DistributionMatchingData:
    """Combine model-family hooks with the generic paired DMD runtime."""

    runtime = (runtime_type or DistributionMatchingRuntime)(
        family=family,
        config=config,
        seed=seed,
    )
    return build_paired_data(family, runtime=runtime, stream_seed_offsets=stream_seed_offsets)


def build_paired_data(family, *, runtime, stream_seed_offsets):
    """Share resumable streams and model hooks across DMD algorithms."""
    if set(stream_seed_offsets) != {"generator", "critic"}:
        raise ValueError("distribution matching requires generator and critic streams")
    streams = {
        name: family.build_train_stream(int(offset)) for name, offset in stream_seed_offsets.items()
    }
    generator_loader, generator_contract = streams["generator"]
    critic_loader, critic_contract = streams["critic"]
    loader = MultiStreamStatefulDataLoader({"generator": generator_loader, "critic": critic_loader})
    return DistributionMatchingData(
        dataloader=loader,
        prepare_batch=family.prepare_batch,
        rollout=runtime.rollout,
        rollout_trace=runtime.rollout_trace,
        sample_score_noise=runtime.sample_score_noise,
        valid_masks=family.valid_masks,
        batch_selector=_select_phase_batch,
        workload=family.workload,
        metrics=family.metrics,
        loss_partitions=family.loss_partitions,
        dataloader_resume_contract={
            "type": "paired_dmd",
            "generator": generator_contract,
            "critic": critic_contract,
            "critic_sampler_seed_offset": int(stream_seed_offsets["critic"]),
        },
    )


class DistributionMatchingRuntime:
    """Shared rollout and score-noising workflow with explicit strategy hooks."""

    noise_dtype: torch.dtype | None = None
    clean_context = False

    def __init__(
        self,
        *,
        family: DmdFamilyComponents,
        config: DistributionMatchingConfig,
        seed: int,
    ) -> None:
        self.family = family
        self.config = config
        self.seed = int(seed)
        self.scheduler = self.build_scheduler()
        self.rollout_schedule = self.build_rollout_schedule()

    def build_scheduler(self):
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.config.score.num_train_timesteps,
            shift=self.config.score.timestep_shift,
        )

    def build_rollout_schedule(self):
        config = self.config
        return DmdRolloutSchedule(
            noise_levels=tuple(
                step / config.score.num_train_timesteps for step in config.rollout.denoising_steps
            ),
            terminal_is_boundary=config.rollout.terminal_zero_is_boundary,
            context_noise_level=config.rollout.context_noise / config.score.num_train_timesteps,
            last_step_only=config.rollout.last_step_only,
        )

    def resolve_rollout_level(self, level, batch):
        return self.family.resolve_noise_level(self.scheduler, level, batch)

    def select_exit_steps(self, block_count, forward_count, reference, random):
        """Standard DMD traverses all forward levels without sampling an exit."""
        return [forward_count - 1] * block_count

    def rollout(
        self,
        generator_model: nn.Module,
        adapter,
        prepared: PreparedDmdBatch[Any],
        random_streams: RandomStreams,
        phase: Literal["generator", "critic"],
        seed_offset: int,
        *,
        trace: list[BlockReplayState] | None = None,
    ) -> dict[str, torch.Tensor]:
        reference = next(iter(prepared.clean.values()))
        random = self._generator(random_streams, reference.device, seed_offset)

        def noise_like(tensor: torch.Tensor) -> torch.Tensor:
            return torch.randn(
                tensor.shape,
                device=tensor.device,
                dtype=tensor.dtype,
                generator=random,
            )

        initial_noise = {key: noise_like(value) for key, value in prepared.clean.items()}
        return rollout_generator(
            generator_model,
            adapter,
            prepared,
            initial_noise,
            resolve_noise_level=self.resolve_rollout_level,
            schedule=self.rollout_schedule,
            phase=phase,
            noise_fn=noise_like,
            select_exit_steps=lambda count, levels: self.select_exit_steps(
                count, levels, reference, random
            ),
            noise_dtype=self.noise_dtype,
            clean_context=self.clean_context,
            trace=trace,
        )

    def rollout_trace(self, *args):
        trace = []
        with torch.no_grad():
            generated = self.rollout(*args, trace=trace)
        return generated, trace

    def sample_score_noise(
        self,
        clean: LatentBundle,
        prepared: PreparedDmdBatch[Any],
        random_streams: RandomStreams,
        seed_offset: int,
    ) -> tuple[DmdNoiseLevel[Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        reference = next(iter(clean.values()))
        random = self._generator(random_streams, reference.device, seed_offset)
        timestep = self.sample_score_timestep(prepared, reference, random)
        level = self.family.resolve_noise_level(
            self.scheduler,
            timestep / self.config.score.num_train_timesteps,
            prepared,
        )
        noise = {
            key: torch.randn(
                value.shape,
                device=value.device,
                dtype=value.dtype,
                generator=random,
            )
            for key, value in clean.items()
        }
        noisy = {
            key: mix_noise(value, level.sigmas[key], noise[key], compute_dtype=self.noise_dtype)
            for key, value in clean.items()
        }
        if self.family.postprocess_score_noise is not None:
            noisy, level = self.family.postprocess_score_noise(noisy, clean, level)
        return level, dict(noisy), noise

    def sample_score_timestep(self, prepared, reference, random):
        timestep = torch.randint(
            self.config.score.sampled_timestep_min,
            self.config.score.num_train_timesteps + 1,
            self.family.score_noise_shape(prepared),
            device=reference.device,
            generator=random,
        )
        return remap_dmd_timesteps(
            timestep,
            self.scheduler.timesteps,
            timestep_shift=self.config.score.timestep_shift,
            min_timestep=self.config.score.effective_timestep_clamp[0],
            max_timestep=self.config.score.effective_timestep_clamp[1],
            num_train_timesteps=self.config.score.num_train_timesteps,
        )

    def _generator(
        self,
        random_streams: RandomStreams,
        device: torch.device,
        seed_offset: int,
    ) -> torch.Generator:
        return random_streams.generator(
            f"dmd.{seed_offset}", seed=self.seed + seed_offset, device=device
        )


def _select_phase_batch(batch: Any, phase: str) -> Any:
    if hasattr(batch, phase):
        return getattr(batch, phase)
    if isinstance(batch, dict):
        return batch[phase]
    raise TypeError(f"DMD paired batch must expose phase {phase!r}")


RolloutPhase = Literal["generator", "critic"]
NoiseFn = Callable[[torch.Tensor], torch.Tensor]
NoiseLevelResolver = Callable[
    [torch.Tensor, PreparedDmdBatch[Any]],
    DmdNoiseLevel[Any],
]


@dataclass(frozen=True)
class DmdRolloutSchedule:
    """Normalized denoising boundaries and context commit policy."""

    noise_levels: tuple[float, ...]
    terminal_is_boundary: bool = True
    context_noise_level: float = 0.0
    last_step_only: bool = True

    def __post_init__(self) -> None:
        if len(self.noise_levels) < 2:
            raise ValueError("DMD rollout requires at least two noise boundaries")
        if any(not 0.0 <= level <= 1.0 for level in self.noise_levels):
            raise ValueError("DMD noise levels must lie in [0, 1]")
        if any(
            current <= following
            for current, following in zip(
                self.noise_levels[:-1],
                self.noise_levels[1:],
                strict=True,
            )
        ):
            raise ValueError("DMD noise levels must be strictly descending")
        if not 0.0 <= self.context_noise_level <= 1.0:
            raise ValueError("DMD context_noise_level must lie in [0, 1]")

    @property
    def forward_noise_levels(self) -> tuple[float, ...]:
        """Noise levels that require generator forwards."""

        if self.terminal_is_boundary:
            return self.noise_levels[:-1]
        return self.noise_levels


@dataclass(frozen=True)
class BlockReplayState:
    selection: Any
    noisy: LatentBundle
    level: DmdNoiseLevel
    accepted: LatentBundle
    context_level: DmdNoiseLevel
    is_anchor: bool = False


def rollout_generator(
    generator: nn.Module,
    adapter: DmdGeneratorAdapter,
    batch: PreparedDmdBatch[Any],
    initial_noise: LatentBundle,
    *,
    resolve_noise_level: NoiseLevelResolver,
    schedule: DmdRolloutSchedule,
    phase: RolloutPhase,
    noise_fn: NoiseFn = torch.randn_like,
    select_exit_steps: Callable[[int, int], list[int]] | None = None,
    noise_dtype: torch.dtype | None = None,
    clean_context: bool = False,
    trace: list[BlockReplayState] | None = None,
    initial_block_noise_fn: NoiseFn | None = None,
) -> LatentBundle:
    """Generate all ordered blocks and commit each accepted block once."""

    if phase not in {"generator", "critic"}:
        raise ValueError(f"unsupported DMD rollout phase {phase!r}")
    matching_bundle_keys(initial_noise, batch.clean)
    forward_levels = schedule.forward_noise_levels
    if not forward_levels:
        raise ValueError("DMD rollout has no generator-forward noise levels")

    outputs: list[tuple[Any, LatentBundle]] = []
    reference = next(iter(initial_noise.values()))
    with adapter.open_rollout_context(generator, batch) as context:
        blocks = adapter.block_layout(batch, initial_noise)
        if not blocks:
            raise ValueError("DMD generator adapter returned no rollout blocks")
        exits = (
            [len(forward_levels) - 1] * len(blocks)
            if select_exit_steps is None
            else select_exit_steps(len(blocks), len(forward_levels))
        )
        if len(exits) != len(blocks) or any(
            not 0 <= index < len(forward_levels) for index in exits
        ):
            raise ValueError("rollout must select one valid exit step per block")
        for block, exit_index in zip(blocks, exits, strict=True):
            matching_bundle_keys(block.initial_value, initial_noise)
            if block.is_anchor:
                accepted = detached_bundle(block.initial_value)
            else:
                noisy_block = (
                    block.initial_value
                    if initial_block_noise_fn is None
                    else {
                        key: initial_block_noise_fn(value)
                        for key, value in block.initial_value.items()
                    }
                )
                accepted: LatentBundle | None = None
                for index, normalized in enumerate(forward_levels[: exit_index + 1]):
                    level = resolve_noise_level(
                        level_tensor(normalized, reference),
                        batch,
                    )
                    enable_grad = phase == "generator" and (
                        not schedule.last_step_only or index == exit_index
                    )
                    with torch.set_grad_enabled(enable_grad and torch.is_grad_enabled()):
                        accepted = adapter.denoise_block(
                            generator,
                            noisy_block,
                            block.selection,
                            level,
                            batch,
                            context,
                        )
                    matching_bundle_keys(accepted, initial_noise)
                    if index < exit_index:
                        next_level = resolve_noise_level(
                            level_tensor(forward_levels[index + 1], reference),
                            batch,
                        )
                        with torch.no_grad():
                            noisy_block = add_noise_bundle(
                                accepted,
                                next_level.sigmas,
                                noise_fn=noise_fn,
                                compute_dtype=noise_dtype,
                            )
                assert accepted is not None
            outputs.append((block.selection, accepted))

            context_level = resolve_noise_level(
                level_tensor(schedule.context_noise_level, reference),
                batch,
            )
            if trace is not None:
                trace.append(
                    BlockReplayState(
                        block.selection,
                        detached_bundle(block.initial_value if block.is_anchor else noisy_block),
                        context_level if block.is_anchor else level,
                        detached_bundle(accepted),
                        context_level,
                        block.is_anchor,
                    )
                )
            with torch.no_grad():
                # Clean history must not draw throwaway noise: it changes subsequent RNG.
                context_value = (
                    detached_bundle(accepted)
                    if clean_context
                    else add_noise_bundle(
                        accepted,
                        context_level.sigmas,
                        noise_fn=noise_fn,
                        compute_dtype=noise_dtype,
                    )
                )
                adapter.commit_block(
                    generator,
                    block.selection,
                    context_value,
                    context_level,
                    batch,
                    context,
                )
        return dict(adapter.concat_blocks(outputs, batch))


def add_noise_bundle(
    clean: LatentBundle,
    sigmas: LatentBundle,
    *,
    noise_fn: NoiseFn,
    compute_dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    """Apply flow noising independently to every latent component."""

    keys = matching_bundle_keys(clean, sigmas)
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        sample = clean[key]
        sigma = sigmas[key]
        try:
            torch.broadcast_shapes(sample.shape, sigma.shape)
        except RuntimeError as error:
            raise ValueError(
                f"sigma component {key!r} with shape {tuple(sigma.shape)} cannot "
                f"broadcast to latent shape {tuple(sample.shape)}"
            ) from error
        noise = noise_fn(sample)
        if noise.shape != sample.shape:
            raise ValueError(f"noise_fn changed shape for component {key!r}")
        result[key] = mix_noise(sample, sigma, noise, compute_dtype=compute_dtype)
    return result


def mix_noise(clean, sigma, noise, *, compute_dtype=None):
    """Preserve native promotion, or explicitly compute and cast back for alignment."""
    sigma = sigma.to(device=clean.device, dtype=compute_dtype or clean.dtype)
    mixed = (1.0 - sigma) * clean + sigma * noise
    return mixed if compute_dtype is None else mixed.to(clean.dtype)


def matching_bundle_keys(
    *bundles: Mapping[str, torch.Tensor],
) -> tuple[str, ...]:
    if not bundles or not bundles[0]:
        raise ValueError("latent bundles must contain at least one component")
    keys = tuple(bundles[0])
    expected = set(keys)
    for bundle in bundles[1:]:
        if set(bundle) != expected:
            raise ValueError("latent bundle component keys must match")
    return keys


def detached_bundle(bundle: LatentBundle) -> dict[str, torch.Tensor]:
    return {key: value.detach() for key, value in bundle.items()}


def level_tensor(value: float, reference: torch.Tensor) -> torch.Tensor:
    return torch.tensor(value, device=reference.device, dtype=torch.float32)


__all__ = [
    "DmdRolloutSchedule",
    "DistributionMatchingRuntime",
    "NoiseFn",
    "NoiseLevelResolver",
    "RolloutPhase",
    "add_noise_bundle",
    "build_distribution_matching_data",
    "rollout_generator",
]
