"""Shared helpers for Hydra task entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from vforge import conf
from vforge.hydra_config import write_resolved_config
from vforge.tasks.registry import require_registered
from vforge.tasks.translate import BuiltRuntime, build_runtime


def conf_dir() -> Path:
    return Path(conf.__file__).resolve().parent


def compose_standalone_config(
    config_name: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Compose a standalone Hydra config without changing CWD."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(conf_dir()),
    ):
        return compose(config_name=config_name, overrides=list(overrides or []))


def build_and_snapshot(
    cfg: DictConfig | dict[str, Any],
    *,
    world_size: int | None = None,
    write_snapshot: bool = True,
) -> BuiltRuntime:
    """Validate config, optionally persist resolved YAML, return runtime."""
    runtime = build_runtime(cfg, world_size=world_size)
    # Reject unknown model families before runtime init allocates anything.
    # A model recipe owns one ``model.name``; a DMD recipe owns explicitly
    # named model roles and intentionally leaves ``model.name`` empty.
    training = runtime.config.training
    if training.recipe in {"dmd", "cd"}:
        for role, model in training.models.items():
            try:
                require_registered(model.model)
            except ValueError as exc:
                raise ValueError(f"training.models.{role}: {exc}") from exc
    else:
        require_registered(runtime.config.model.name)
    if write_snapshot and _is_rank_zero():
        write_resolved_config(
            runtime.config,
            runtime.config.runtime.output_dir,
        )
        hash_path = Path(runtime.config.runtime.output_dir) / "resolved_config.hash"
        hash_path.write_text(
            runtime.resolved_config_hash + "\n",
            encoding="utf-8",
        )
    return runtime


def omega_to_container(cfg: DictConfig) -> dict[str, Any]:
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Hydra config must resolve to a mapping")
    return container


def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


__all__ = [
    "build_and_snapshot",
    "compose_standalone_config",
    "conf_dir",
    "omega_to_container",
]
