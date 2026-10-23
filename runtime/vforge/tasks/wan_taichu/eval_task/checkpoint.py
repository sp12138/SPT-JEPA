"""Wan Taichu evaluation-checkpoint compatibility callbacks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from vforge.models.loading import CompatibleLoadCallbacks
from vforge.tasks.wan_taichu.variants import WanTaichuVariant
from vforge.training.checkpoint import StateDictMapper


def checkpoint_load_callbacks(
    spec: WanTaichuVariant,
) -> CompatibleLoadCallbacks | None:
    """Return checkpoint-only compatibility without published after-load hooks."""
    if not spec.ref_token:
        return None
    return CompatibleLoadCallbacks(resolve_missing_key=warm_start_missing_ref_projection)


def a14b_checkpoint_mapper() -> StateDictMapper:
    """Accept both vforge ``wan.*`` and Taichu ``module.wan.*`` PT keys."""

    def passthrough(state: dict[str, Any]) -> dict[str, Any]:
        return dict(state)

    def import_(state: dict[str, Any]) -> dict[str, Any]:
        prefix = "module."
        if not any(key.startswith(prefix) for key in state):
            return dict(state)
        return {
            key[len(prefix) :] if key.startswith(prefix) else key: value
            for key, value in state.items()
        }

    return StateDictMapper(export=passthrough, import_=import_)


def warm_start_missing_ref_projection(
    key: str,
    target: torch.Tensor,
    resolved_state: Mapping[str, object],
) -> torch.Tensor | None:
    """Rebuild a missing checkpoint ref projection from its loaded main projection."""
    marker = "patch_embedding_ref."
    if marker not in key:
        return None
    main = resolved_state.get(key.replace(marker, "patch_embedding.", 1))
    if not isinstance(main, torch.Tensor):
        return None
    if key.endswith(".bias"):
        return main.clone() if tuple(main.shape) == tuple(target.shape) else None
    if not key.endswith(".weight") or main.ndim != target.ndim:
        return None
    if (
        tuple(main.shape[:1]) != tuple(target.shape[:1])
        or tuple(main.shape[2:]) != tuple(target.shape[2:])
        or int(main.shape[1]) < int(target.shape[1])
    ):
        return None
    return main[:, : target.shape[1]].clone()


__all__ = [
    "a14b_checkpoint_mapper",
    "checkpoint_load_callbacks",
    "warm_start_missing_ref_projection",
]
