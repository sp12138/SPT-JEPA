"""Native vforge wrapper around Wan 2.2 A14B high/low-noise experts.

Unlike the single-tower Wan plugins this root does not register
``self.transformer``. It registers one or both published experts and routes a
step to whichever resident expert owns the sampled timestep.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vforge.models.factory import patch_diffusers_getattr as _patch_diffusers_getattr
from vforge.models.wan.model import (
    WanChannelLayout,
    WanModelRootBase,
    build_meta_module,
    published_z_dim,
    read_transformer_config,
)

#: Attribute names of the two experts, which are also their checkpoint key
#: prefixes. ``wan`` owns high noise, ``wan_2`` low noise, matching the
#: ``transformer`` / ``transformer_2`` split of the published checkpoint.
HIGH_NOISE_EXPERT = "wan"
LOW_NOISE_EXPERT = "wan_2"

#: Published subfolder backing each expert.
EXPERT_SUBFOLDERS = {
    HIGH_NOISE_EXPERT: "transformer",
    LOW_NOISE_EXPERT: "transformer_2",
}
ALL_EXPERTS = (HIGH_NOISE_EXPERT, LOW_NOISE_EXPERT)

DEFAULT_BOUNDARY_RATIO = 0.9
DEFAULT_NUM_TRAIN_TIMESTEPS = 1000


def read_published_configs(
    path: str | Path,
    *,
    expert_names: Sequence[str] = ALL_EXPERTS,
) -> dict[str, dict[str, Any]]:
    """Selected experts' published configs, keyed by their model paths."""
    from vforge.models.wan.v21.transformer import WanTransformer3DModel

    names = _validate_expert_names(expert_names)
    return {
        name: read_transformer_config(
            WanTransformer3DModel,
            path,
            subfolder=EXPERT_SUBFOLDERS[name],
        )
        for name in names
    }


def _validate_expert_names(expert_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(expert_names)
    if not names:
        raise ValueError("Wan22 A14B requires at least one expert")
    if len(set(names)) != len(names):
        raise ValueError(f"Wan22 A14B expert names must be unique; got {names}")
    unknown = set(names) - set(ALL_EXPERTS)
    if unknown:
        raise ValueError(f"unknown Wan22 A14B experts: {sorted(unknown)}")
    return names


def widen_expert_config(
    published: Mapping[str, Any],
    *,
    in_channels: int,
    **overrides: Any,
) -> tuple[dict[str, Any], WanChannelLayout]:
    """Widen one expert's patch embedding. Pure; reads nothing."""
    config = dict(published)
    z_dim = published_z_dim(config)
    declared_in = int(config["in_channels"])
    if int(in_channels) < declared_in:
        raise ValueError(
            f"the requested patch_embedding width {in_channels} is narrower than the "
            f"{declared_in} input channels the checkpoint declares; narrowing "
            "patch_embedding would discard published weights"
        )
    config["in_channels"] = int(in_channels)
    config["out_channels"] = z_dim
    # A14B has no CLIP/VAE image branch; the reference frame arrives as the rgb
    # condition stream instead, so the 2.1 transformer's image conv is dead
    # weight here and is left unbuilt.
    config.setdefault("use_vae_condition", False)
    config.update(overrides)
    layout = WanChannelLayout(
        z_dim=z_dim,
        pretrained_in_channels=declared_in,
        in_channels=int(config["in_channels"]),
    )
    return config, layout


class Wan22A14bModel(WanModelRootBase):
    """One or two Wan transformers behind one root, routed by timestep.

    Timesteps at or above ``boundary_ratio * num_train_timesteps`` belong to the
    high-noise expert, the rest to the low-noise one. Routing lives here rather
    than in the training step so the step stays a plain ``model(**inputs)`` call
    and evaluation gets the same dispatch for free.
    """

    def __init__(
        self,
        wan: nn.Module | None = None,
        wan_2: nn.Module | None = None,
        *,
        boundary_ratio: float = DEFAULT_BOUNDARY_RATIO,
        num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
    ) -> None:
        super().__init__()
        if not 0.0 < boundary_ratio < 1.0:
            raise ValueError(f"boundary_ratio must be in (0, 1); got {boundary_ratio}")
        if wan is None and wan_2 is None:
            raise ValueError("Wan22 A14B requires at least one expert")
        self.wan = wan
        self.wan_2 = wan_2
        for expert in self.cache_transformers():
            _patch_diffusers_getattr(expert)
        self.boundary_ratio = float(boundary_ratio)
        self.num_train_timesteps = int(num_train_timesteps)

    @classmethod
    def build_empty(
        cls,
        expert_configs: Mapping[str, Mapping[str, Any]],
        *,
        parameter_dtype: torch.dtype | None = None,
        boundary_ratio: float = DEFAULT_BOUNDARY_RATIO,
        num_train_timesteps: int = DEFAULT_NUM_TRAIN_TIMESTEPS,
        expert_names: Sequence[str] = ALL_EXPERTS,
    ) -> Wan22A14bModel:
        """Build only the selected model structures; weights load after sharding."""
        from vforge.models.wan.v21.transformer import WanTransformer3DModel

        names = _validate_expert_names(expert_names)
        missing = [name for name in names if name not in expert_configs]
        if missing:
            raise ValueError(f"Wan22 A14B is missing selected expert configs: {missing}")
        experts = {
            name: build_meta_module(
                WanTransformer3DModel,
                dict(expert_configs[name]),
                parameter_dtype=parameter_dtype,
            )
            for name in names
        }
        return cls(
            wan=experts.get(HIGH_NOISE_EXPERT),
            wan_2=experts.get(LOW_NOISE_EXPERT),
            boundary_ratio=boundary_ratio,
            num_train_timesteps=num_train_timesteps,
        )

    @property
    def boundary_timestep(self) -> float:
        return self.boundary_ratio * self.num_train_timesteps

    def primary_transformer(self) -> nn.Module:
        expert = self.wan if self.wan is not None else self.wan_2
        assert expert is not None
        return expert

    def cache_transformers(self) -> tuple[nn.Module, ...]:
        return tuple(expert for expert in (self.wan, self.wan_2) if expert is not None)

    def select_expert(self, timestep: torch.Tensor) -> nn.Module:
        """Return the expert owning ``timestep``.

        A micro-batch has to sit on one side of the boundary: the two experts
        are separate towers, so a batch that straddles it would need two
        forwards, and silently routing it by its first element would train the
        wrong expert on the rest.
        """
        if timestep.numel() == 0:
            raise ValueError("cannot route an empty timestep to a Wan22 A14B expert")
        first = timestep.flatten()[0]
        if timestep.numel() > 1 and not bool(torch.all(timestep == first)):
            raise ValueError(
                "Wan22 A14B routing needs one shared timestep per micro-batch; "
                "split the batch by high/low noise before the forward"
            )
        name = HIGH_NOISE_EXPERT if float(first) >= self.boundary_timestep else LOW_NOISE_EXPERT
        expert = getattr(self, name)
        if expert is None:
            label = "high" if name == HIGH_NOISE_EXPERT else "low"
            raise RuntimeError(
                f"Wan22 A14B was built for the other expert and cannot execute {label}-noise "
                "timesteps"
            )
        return expert

    def forward(self, *args: Any, timestep: torch.Tensor, **kwargs: Any):
        return self.select_expert(timestep)(*args, timestep=timestep, **kwargs)


__all__ = [
    "DEFAULT_BOUNDARY_RATIO",
    "DEFAULT_NUM_TRAIN_TIMESTEPS",
    "ALL_EXPERTS",
    "EXPERT_SUBFOLDERS",
    "HIGH_NOISE_EXPERT",
    "LOW_NOISE_EXPERT",
    "Wan22A14bModel",
    "read_published_configs",
    "widen_expert_config",
]
