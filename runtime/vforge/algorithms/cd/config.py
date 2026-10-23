"""Validated configuration for causal consistency distillation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CdEmaConfig:
    decay: float
    start_step: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("cd.ema.decay must be in [0, 1)")
        if self.start_step < 0:
            raise ValueError("cd.ema.start_step must be non-negative")


@dataclass(frozen=True)
class CausalConsistencyConfig:
    discrete_levels: int
    sigma_grid: str
    pair_sampling: str
    consistency_shift: float
    guidance_scale: float
    ema: CdEmaConfig

    def __post_init__(self) -> None:
        if self.discrete_levels != 40:
            raise ValueError("Wan 5B causal consistency requires cd.discrete_levels=40")
        if self.sigma_grid != "wan":
            raise ValueError("causal consistency currently requires cd.sigma_grid='wan'")
        if self.pair_sampling != "adjacent_uniform":
            raise ValueError(
                "causal consistency currently requires cd.pair_sampling='adjacent_uniform'"
            )
        if self.consistency_shift != 5.0:
            raise ValueError("Wan 5B causal consistency requires cd.consistency_shift=5.0")
        if self.guidance_scale < 1.0:
            raise ValueError("cd.guidance_scale must be at least 1")
        if self.ema.decay != 0.995 or self.ema.start_step != 0:
            raise ValueError(
                "Wan 5B causal consistency requires cd.ema.decay=0.995 and cd.ema.start_step=0"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CausalConsistencyConfig:
        mapping = dict(value)
        algorithm = str(mapping.pop("algorithm", "") or "")
        if algorithm != "causal_consistency_tf":
            raise ValueError("CausalConsistencyConfig requires algorithm='causal_consistency_tf'")
        expected = {
            "discrete_levels",
            "sigma_grid",
            "pair_sampling",
            "consistency_shift",
            "guidance_scale",
            "ema",
        }
        unknown = set(mapping) - expected
        missing = expected - set(mapping)
        if unknown:
            raise ValueError(f"unknown CD config fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing CD config fields: {sorted(missing)}")
        ema = mapping.pop("ema")
        if not isinstance(ema, Mapping):
            raise TypeError("cd.ema must be a mapping")
        return cls(ema=CdEmaConfig(**dict(ema)), **mapping)


@dataclass(frozen=True)
class CausalConsistencyMoeConfig:
    discrete_levels: int
    sigma_grid: str
    pair_sampling: str
    consistency_shift: float
    guidance_scale: float
    boundary_ratio: float
    ema: CdEmaConfig

    def __post_init__(self) -> None:
        if self.discrete_levels != 40:
            raise ValueError("Wan A14B causal consistency requires cd.discrete_levels=40")
        if self.sigma_grid != "wan":
            raise ValueError("A14B causal consistency requires cd.sigma_grid='wan'")
        if self.pair_sampling != "adjacent_uniform_in_band":
            raise ValueError(
                "A14B causal consistency requires cd.pair_sampling='adjacent_uniform_in_band'"
            )
        if self.consistency_shift != 3.0:
            raise ValueError("Wan A14B causal consistency requires cd.consistency_shift=3.0")
        if self.guidance_scale < 1.0:
            raise ValueError("cd.guidance_scale must be at least 1")
        if self.boundary_ratio != 0.9:
            raise ValueError("Wan A14B causal consistency requires cd.boundary_ratio=0.9")
        if self.ema.decay != 0.995 or self.ema.start_step != 0:
            raise ValueError(
                "Wan A14B causal consistency requires cd.ema.decay=0.995 and cd.ema.start_step=0"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CausalConsistencyMoeConfig:
        mapping = dict(value)
        algorithm = str(mapping.pop("algorithm", "") or "")
        if algorithm != "causal_consistency_moe_tf":
            raise ValueError(
                "CausalConsistencyMoeConfig requires algorithm='causal_consistency_moe_tf'"
            )
        expected = {
            "discrete_levels",
            "sigma_grid",
            "pair_sampling",
            "consistency_shift",
            "guidance_scale",
            "boundary_ratio",
            "ema",
        }
        unknown = set(mapping) - expected
        missing = expected - set(mapping)
        if unknown:
            raise ValueError(f"unknown MoE CD config fields: {sorted(unknown)}")
        if missing:
            raise ValueError(f"missing MoE CD config fields: {sorted(missing)}")
        ema = mapping.pop("ema")
        if not isinstance(ema, Mapping):
            raise TypeError("cd.ema must be a mapping")
        return cls(ema=CdEmaConfig(**dict(ema)), **mapping)


__all__ = [
    "CausalConsistencyConfig",
    "CausalConsistencyMoeConfig",
    "CdEmaConfig",
]
