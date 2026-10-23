"""Generic lazy model construction contracts."""

from __future__ import annotations

import types
from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn


def patch_diffusers_getattr(model: nn.Module) -> None:
    """Make diffusers config lookup coexist with registered ``nn.Module`` children.

    ``ModelMixin.__getattr__`` falls through the MRO to ``ConfigMixin``, which
    only consults ``_internal_dict`` and then raises, never reaching the module
    registry. Bind a per-instance override that tries the registry first.
    """
    original_getattr = type(model).__getattr__

    def _patched(self, name):
        try:
            return nn.Module.__getattr__(self, name)
        except AttributeError:
            return original_getattr(self, name)

    model.__getattr__ = types.MethodType(_patched, model)


@dataclass(frozen=True)
class ModelSpec:
    """Build a real model only when its runtime session starts."""

    build: Callable[[], nn.Module]


def resolve_model(model: nn.Module | ModelSpec) -> nn.Module:
    """Resolve a model specification while leaving concrete modules unchanged."""
    resolved = model.build() if isinstance(model, ModelSpec) else model
    if not isinstance(resolved, nn.Module):
        raise TypeError(f"ModelSpec.build() must return nn.Module, got {type(resolved).__name__}")
    return resolved


__all__ = ["ModelSpec", "patch_diffusers_getattr", "resolve_model"]
