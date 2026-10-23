"""Checkpoint policy configuration and filename generation."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_DCP_STEP_RE = re.compile(r"^(\d+)\.dcp$")
_MCORE_STEP_RE = re.compile(r"^(\d+)\.mcore$")
_PT_STEP_RE = re.compile(r"^(\d+)\.pt$")


@dataclass(frozen=True)
class ResumeConfig:
    """Resume plan consumed by Trainer and storage loaders."""

    path: str = "auto"
    weights: bool = True
    optimizer: bool = True
    training_state: bool = True
    strict: bool | None = False
    stage_remote: bool = False

    def __post_init__(self) -> None:
        if not self.weights and (self.optimizer or self.training_state):
            raise ValueError(
                "resume.weights must be true when optimizer or training_state is enabled"
            )
        if self.training_state and not self.optimizer:
            raise ValueError("resume.training_state=true requires resume.optimizer=true")

    @property
    def effective_strict(self) -> bool:
        return self.training_state if self.strict is None else self.strict


def build_resume_config(
    resume: ResumeConfig | Mapping[str, Any],
    **overrides: Any,
) -> ResumeConfig:
    """Build one validated resume plan and reject removed partial-state fields."""
    values = (
        {
            field: getattr(resume, field)
            for field in (
                "path",
                "weights",
                "optimizer",
                "training_state",
                "strict",
                "stage_remote",
            )
        }
        if isinstance(resume, ResumeConfig)
        else dict(resume)
    )
    legacy_keys = {"progress", "scheduler", "dataloader", "rng"}.intersection(values)
    if legacy_keys:
        raise ValueError(
            "resume progress/scheduler/dataloader/rng were merged into "
            "training_state; remove legacy keys and set training_state explicitly"
        )
    values.update(overrides)
    return ResumeConfig(**values)


@dataclass(frozen=True)
class CheckpointConfig:
    """Checkpoint policy consumed by :class:`~vforge.training.trainer.Trainer`."""

    enabled: bool = False
    every_n_steps: int = 0
    dirpath: str = ""
    state: str = "weights"
    format: str = "pt"
    filename_format: str | None = None
    save_final: bool = True
    keep_last: int | None = None
    async_save: bool = False
    save_optimizer: bool = True
    save_trainer_state: bool = True
    empty_cache_before_save: bool = False
    remote_upload: bool = False

    def __post_init__(self) -> None:
        if self.state not in {"weights", "full"}:
            raise ValueError("checkpoint.state must be 'weights' or 'full'")
        if self.format not in {"pt", "dcp", "mcore"}:
            raise ValueError("checkpoint.format must be 'pt', 'dcp', or 'mcore'")
        if self.state == "full" and self.format == "pt":
            raise ValueError(
                "checkpoint combination full+pt is not supported; "
                "use full+dcp for training-state resume, or weights+pt "
                "for eval-compatible warm-start exports"
            )
        if self.enabled and self.every_n_steps <= 0 and not self.save_final:
            raise ValueError(
                "enabled checkpoint config requires every_n_steps > 0 or save_final=True"
            )
        if self.enabled and not self.dirpath:
            raise ValueError("enabled checkpoint config requires dirpath")
        if self.filename_format is not None:
            if "{step" not in self.filename_format:
                raise ValueError("checkpoint.filename_format must contain a {step} field")
            try:
                sample_name = self.filename_format.format(step=1)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    "checkpoint.filename_format must contain a valid {step} field"
                ) from exc
            pattern = {
                "pt": _PT_STEP_RE,
                "dcp": _DCP_STEP_RE,
                "mcore": _MCORE_STEP_RE,
            }[self.format]
            if pattern.match(sample_name) is None:
                raise ValueError(
                    "checkpoint.filename_format must produce a numeric basename "
                    f"ending in .{self.format}; got {sample_name!r}"
                )
        if self.keep_last is not None and self.keep_last < 1:
            raise ValueError("checkpoint.keep_last must be at least 1")
        if self.async_save and self.format not in {"dcp", "mcore"}:
            raise ValueError(
                "checkpoint.async_save requires format='dcp' or 'mcore'; "
                "PT exports remain synchronous"
            )
        if self.remote_upload and not self.async_save:
            raise ValueError("checkpoint.remote_upload requires async_save=true")
        if self.remote_upload and self.format != "dcp":
            raise ValueError("checkpoint.remote_upload requires format='dcp'")

    def should_save(self, completed_steps: int, *, is_final: bool = False) -> bool:
        if not self.enabled or completed_steps <= 0:
            return False
        if self.every_n_steps > 0 and completed_steps % self.every_n_steps == 0:
            return True
        return bool(is_final and self.save_final)


def checkpoint_path(config: CheckpointConfig, completed_steps: int) -> str:
    filename_format = config.filename_format
    if filename_format is None:
        filename_format = {
            "pt": "{step:05d}.pt",
            "dcp": "{step:05d}.dcp",
            "mcore": "{step:05d}.mcore",
        }[config.format]
    filename = filename_format.format(step=int(completed_steps))
    return os.path.join(config.dirpath, filename)


__all__ = [
    "CheckpointConfig",
    "ResumeConfig",
    "build_resume_config",
    "checkpoint_path",
]
