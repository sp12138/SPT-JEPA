"""Native vforge wrapper around the Wan 2.2 TI2V transformer."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from vforge.models.wan.model import (
    WanChannelLayout,
    WanSingleTransformerModel,
    build_meta_module,
    published_z_dim,
    read_transformer_config,
)
from vforge.models.wan.v22.ti2v_5b.ct.transformer import WanTransformer3DModel


def read_published_config(path: str | Path) -> dict[str, Any]:
    """The published TI2V transformer config. Read this once per checkpoint."""
    return read_transformer_config(WanTransformer3DModel, path)


def widen_transformer_config(
    published: Mapping[str, Any],
    *,
    in_channels: int,
    **overrides: Any,
) -> tuple[dict[str, Any], WanChannelLayout]:
    """Widen a published config's patch embedding. Pure; reads nothing."""
    config = dict(published)
    z_dim = published_z_dim(config)
    declared_in = int(config["in_channels"])
    if declared_in not in {z_dim, in_channels}:
        raise ValueError(
            "Wan22 TI2V expects a transformer whose in_channels is either "
            f"z_dim ({z_dim}, official checkpoint) or the requested width "
            f"({in_channels}); got {declared_in}"
        )
    config["in_channels"] = int(in_channels)
    config["out_channels"] = z_dim
    config.update(overrides)
    layout = WanChannelLayout(
        z_dim=z_dim,
        pretrained_in_channels=declared_in,
        in_channels=int(config["in_channels"]),
    )
    return config, layout


class Wan22Ti2v5bModel(WanSingleTransformerModel):
    """Checkpoint-wrapper-friendly native Wan 2.2 TI2V transformer."""

    @classmethod
    def build_empty(
        cls,
        config: Mapping[str, Any],
        *,
        parameter_dtype: torch.dtype | None = None,
    ) -> Wan22Ti2v5bModel:
        """Materialize only model structure; weights are loaded after FSDP."""
        transformer = build_meta_module(
            WanTransformer3DModel,
            dict(config),
            parameter_dtype=parameter_dtype,
        )
        return cls(transformer)


__all__ = ["Wan22Ti2v5bModel", "read_published_config", "widen_transformer_config"]
