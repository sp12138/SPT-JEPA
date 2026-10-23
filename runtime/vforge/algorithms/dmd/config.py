"""Validated configuration for the standard distribution-matching algorithm."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DmdRolloutConfig:
    denoising_steps: tuple[int, ...]
    terminal_zero_is_boundary: bool
    last_step_only: bool
    context_noise: float

    def __post_init__(self) -> None:
        if len(self.denoising_steps) < 2:
            raise ValueError("dmd.rollout.denoising_steps requires at least two boundaries")
        if any(step < 0 for step in self.denoising_steps):
            raise ValueError("dmd.rollout.denoising_steps must be non-negative")
        if any(
            current <= following
            for current, following in zip(
                self.denoising_steps[:-1],
                self.denoising_steps[1:],
                strict=True,
            )
        ):
            raise ValueError("dmd.rollout.denoising_steps must be strictly descending")
        if self.terminal_zero_is_boundary and self.denoising_steps[-1] != 0:
            raise ValueError(
                "terminal_zero_is_boundary=true requires denoising_steps to end at zero"
            )
        if self.context_noise < 0:
            raise ValueError("dmd.rollout.context_noise must be non-negative")

    @property
    def effective_forward_timesteps(self) -> tuple[int, ...]:
        if self.terminal_zero_is_boundary:
            return self.denoising_steps[:-1]
        return self.denoising_steps


@dataclass(frozen=True)
class DmdScoreConfig:
    num_train_timesteps: int
    timestep_shift: float
    sampled_timestep_min: int
    effective_timestep_clamp: tuple[int, int]
    real_guidance_scale: float
    fake_guidance_scale: float
    denoising_loss: str
    mse_weight: float

    def __post_init__(self) -> None:
        if self.num_train_timesteps < 2:
            raise ValueError("dmd.score.num_train_timesteps must be at least two")
        if self.timestep_shift <= 0:
            raise ValueError("dmd.score.timestep_shift must be positive")
        if not 0 <= self.sampled_timestep_min < self.num_train_timesteps:
            raise ValueError("dmd.score.sampled_timestep_min is outside the schedule")
        lower, upper = self.effective_timestep_clamp
        if not 0 <= lower <= upper < self.num_train_timesteps:
            raise ValueError("dmd.score.effective_timestep_clamp is outside the schedule")
        if self.denoising_loss != "flow":
            raise ValueError("distribution matching currently supports denoising_loss='flow' only")
        if self.fake_guidance_scale != 0.0:
            raise ValueError("distribution matching currently requires fake_guidance_scale=0")
        if self.mse_weight < 0:
            raise ValueError("dmd.score.mse_weight must be non-negative")


@dataclass(frozen=True)
class DmdScheduleConfig:
    generator_every_n_steps: int

    def __post_init__(self) -> None:
        if self.generator_every_n_steps < 1:
            raise ValueError("dmd.schedule.generator_every_n_steps must be positive")


@dataclass(frozen=True)
class DmdEmaConfig:
    enabled: bool
    decay: float
    start_step: int
    storage: str

    def __post_init__(self) -> None:
        if not 0 <= self.decay < 1:
            raise ValueError("dmd.ema.decay must be in [0, 1)")
        if self.start_step < 0:
            raise ValueError("dmd.ema.start_step must be non-negative")
        if self.storage != "sharded_fp32":
            raise ValueError("distribution matching currently requires ema.storage=sharded_fp32")


@dataclass(frozen=True)
class DistributionMatchingConfig:
    rollout: DmdRolloutConfig
    score: DmdScoreConfig
    schedule: DmdScheduleConfig
    ema: DmdEmaConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> DistributionMatchingConfig:
        mapping = dict(value)
        algorithm = str(mapping.pop("algorithm", "") or "")
        if algorithm != "distribution_matching":
            raise ValueError(
                "DistributionMatchingConfig requires algorithm='distribution_matching'"
            )
        expected = {"rollout", "score", "schedule", "ema"}
        unknown = set(mapping) - expected
        missing = expected - set(mapping)
        if unknown:
            raise ValueError(f"unknown DMD config sections: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing DMD config sections: {sorted(missing)}")
        rollout = config_section(mapping, "rollout")
        rollout["denoising_steps"] = integer_tuple(
            rollout.get("denoising_steps"),
            "dmd.rollout.denoising_steps",
        )
        score = config_section(mapping, "score")
        score["effective_timestep_clamp"] = integer_tuple(
            score.get("effective_timestep_clamp"),
            "dmd.score.effective_timestep_clamp",
            length=2,
        )
        return cls(
            rollout=DmdRolloutConfig(**rollout),
            score=DmdScoreConfig(**score),
            schedule=DmdScheduleConfig(**config_section(mapping, "schedule")),
            ema=DmdEmaConfig(**config_section(mapping, "ema")),
        )


def config_section(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    section = value.get(name)
    if not isinstance(section, Mapping):
        raise TypeError(f"dmd.{name} must be a mapping")
    return dict(section)


def integer_tuple(
    value: Any,
    name: str,
    *,
    length: int | None = None,
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(int(item) for item in value)
    if length is not None and len(result) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    return result


__all__ = [
    "BlockDistributionMatchingConfig",
    "DistributionMatchingConfig",
    "DmdEmaConfig",
    "DmdRolloutConfig",
    "DmdScheduleConfig",
    "DmdScoreConfig",
]


@dataclass(frozen=True)
class BlockRolloutConfig:
    denoising_steps: tuple[int, ...]
    same_step_across_blocks: bool = True
    backward_mode: str = "retained_graph"

    def __post_init__(self):
        DmdRolloutConfig(self.denoising_steps, True, True, 0.0)
        if self.backward_mode not in {"retained_graph", "block_replay"}:
            raise ValueError("block backward_mode must be retained_graph or block_replay")


@dataclass(frozen=True)
class BlockScoreConfig:
    num_train_timesteps: int
    sampled_timestep_min: int
    sampled_timestep_max: int
    component_guidance_scales: Mapping[str, float]
    component_weights: Mapping[str, float]

    def __post_init__(self):
        if (
            not 0
            <= self.sampled_timestep_min
            <= self.sampled_timestep_max
            < self.num_train_timesteps
        ):
            raise ValueError("block DMD score range is outside the schedule")
        if (
            not self.component_weights
            or any(w < 0 for w in self.component_weights.values())
            or not any(self.component_weights.values())
        ):
            raise ValueError(
                "block DMD requires non-negative component weights and a positive total"
            )
        if set(self.component_guidance_scales) != set(self.component_weights):
            raise ValueError("guidance and loss weight component keys must match")


@dataclass(frozen=True)
class BlockDistributionMatchingConfig:
    rollout: BlockRolloutConfig
    score: BlockScoreConfig
    schedule: DmdScheduleConfig
    ema: DmdEmaConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> BlockDistributionMatchingConfig:
        mapping = dict(value)
        if mapping.pop("algorithm", None) != "block_distribution_matching":
            raise ValueError(
                "BlockDistributionMatchingConfig requires algorithm='block_distribution_matching'"
            )
        if set(mapping) != {"rollout", "score", "schedule", "ema"}:
            raise ValueError("block DMD requires exactly rollout, score, schedule and ema sections")
        rollout = config_section(mapping, "rollout")
        rollout["denoising_steps"] = integer_tuple(
            rollout.get("denoising_steps"), "dmd.rollout.denoising_steps"
        )
        return cls(
            rollout=BlockRolloutConfig(**rollout),
            score=BlockScoreConfig(**config_section(mapping, "score")),
            schedule=DmdScheduleConfig(**config_section(mapping, "schedule")),
            ema=DmdEmaConfig(**config_section(mapping, "ema")),
        )


@dataclass(frozen=True)
class SegmentChainConfig:
    segment_slots: int = 9
    max_rollout_blocks: int = 57
    sink_blocks: int = 1
    local_blocks: int = 2

    def __post_init__(self):
        if any(
            type(value) is not int
            for value in (
                self.segment_slots,
                self.max_rollout_blocks,
                self.sink_blocks,
                self.local_blocks,
            )
        ):
            raise ValueError("segment chain sizes must be integers")
        if self.sink_blocks != 1 or self.local_blocks < 1:
            raise ValueError("segment chain requires one sink and at least one local block")
        if self.new_blocks < 1 or self.max_rollout_blocks <= self.segment_slots:
            raise ValueError("segment chain requires generated slots and more than one segment")

    @property
    def new_blocks(self):
        return self.segment_slots - self.sink_blocks - self.local_blocks

    @property
    def segments_per_sample(self):
        remaining = self.max_rollout_blocks - self.segment_slots
        return 1 + (remaining + self.new_blocks - 1) // self.new_blocks

    @property
    def actual_rollout_blocks(self):
        return self.segment_slots + (self.segments_per_sample - 1) * self.new_blocks


@dataclass(frozen=True)
class SegmentChainDistributionMatchingConfig(BlockDistributionMatchingConfig):
    chain: SegmentChainConfig

    @classmethod
    def from_mapping(cls, value):
        mapping = dict(value)
        if mapping.pop("algorithm", None) != "segment_chain_distribution_matching":
            raise ValueError("expected segment_chain_distribution_matching")
        chain = SegmentChainConfig(**config_section(mapping, "chain"))
        del mapping["chain"]
        base = BlockDistributionMatchingConfig.from_mapping(
            dict(mapping, algorithm="block_distribution_matching")
        )
        return cls(base.rollout, base.score, base.schedule, base.ema, chain)
