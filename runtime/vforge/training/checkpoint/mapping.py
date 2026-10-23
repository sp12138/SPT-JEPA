"""Explicit state-dict key mappings for portable PT checkpoints."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

StateDict = dict[str, Any]


@dataclass(frozen=True)
class StateDictMapper:
    """Map between an internal module tree and an external PT key space."""

    export: Callable[[StateDict], StateDict]
    import_: Callable[[StateDict], StateDict]
    export_prefix: str | None = None

    @classmethod
    def strip_prefix(cls, prefix: str) -> StateDictMapper:
        def export(state: StateDict) -> StateDict:
            if state and not all(key.startswith(prefix) for key in state):
                raise ValueError(f"cannot export state dict: not all keys start with {prefix!r}")
            return {key[len(prefix) :]: value for key, value in state.items()}

        def import_(state: StateDict) -> StateDict:
            if any(key.startswith(prefix) for key in state):
                return state
            return {prefix + key: value for key, value in state.items()}

        return cls(
            export=export,
            import_=import_,
            export_prefix=prefix,
        )


__all__ = ["StateDict", "StateDictMapper"]
