"""Map ``data.sources[].kind`` onto a raw dataset class.

Entries are dotted paths, resolved on use, for the same reason
:mod:`vforge.tasks.registry` stores them that way: naming a kind in a config
should not drag decord and the whole decode stack into a process that only
wanted to validate the config.

Leaving ``kind`` empty is the encoded-cache path and never reaches this module.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

_REGISTRY: dict[str, str] = {
    "gameplay": "vforge.data.gameplay_dataset:GameplayDataset",
    "self_forcing": "vforge.data.self_forcing_dataset:SelfForcingDataset",
    "self_forcing_long_video": "vforge.data.self_forcing_dataset:SelfForcingLongVideoDataset",
}


def register(kind: str, path: str) -> None:
    """Add a kind, for tests and out-of-tree datasets."""
    if not kind:
        raise ValueError("dataset kind must be non-empty")
    _REGISTRY[kind] = path


def unregister(kind: str) -> None:
    _REGISTRY.pop(kind, None)


def dataset_kinds() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def resolve_dataset_class(kind: str) -> type:
    """Import and return the dataset class registered for ``kind``."""
    known = ", ".join(dataset_kinds()) or "<none>"
    if not kind:
        raise ValueError(f"dataset kind is required; known kinds: {known}")
    path = _REGISTRY.get(kind)
    if path is None:
        raise ValueError(f"unknown dataset kind={kind!r}; known kinds: {known}")
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise ValueError(f"dataset kind {kind!r} must map to 'package.module:attr', got {path!r}")
    target = getattr(importlib.import_module(module_name), attr, None)
    if target is None:
        raise ImportError(f"dataset kind {kind!r} target {path!r} does not exist")
    if not isinstance(target, type):
        raise TypeError(f"dataset kind {kind!r} target {path!r} is not a class")
    return target


def open_raw_dataset(source: Mapping[str, Any], *, mode: str = "train", index: int = 0):
    """Build one raw dataset from a translated data-source mapping.

    Every kind is constructed with the same keyword set, so registering one
    from outside this package means accepting that signature.
    """
    kind = str(source.get("kind") or "")
    dataset_class = resolve_dataset_class(kind)
    index_files = tuple(str(path) for path in (source.get("index_files") or ()))
    if not index_files:
        raise ValueError(f"data.sources[{index}] kind={kind!r} requires index_files")
    extras = source.get("extras") or {}
    return dataset_class(
        index_files=index_files,
        frames=int(source["frames"]),
        height=int(source["height"]),
        width=int(source["width"]),
        max_sequence_length=int(source.get("max_sequence_length", 512)),
        name=str(source.get("name") or f"sub_{index}"),
        mode=mode,
        extras=dict(extras),
    )


__all__ = [
    "dataset_kinds",
    "open_raw_dataset",
    "register",
    "resolve_dataset_class",
    "unregister",
]
