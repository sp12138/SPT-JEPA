"""Shared resolution contract for transformer-block keyword arguments."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypeAlias

TransformerBlockKwargs: TypeAlias = Mapping[str, Any]
TransformerBlockKwargsProvider: TypeAlias = (
    TransformerBlockKwargs | Callable[[int], TransformerBlockKwargs]
)


def resolve_transformer_block_kwargs(
    provider: TransformerBlockKwargsProvider | None,
    block_index: int,
) -> dict[str, Any]:
    """Resolve fixed or per-layer kwargs into a fresh mutable mapping."""
    if provider is None:
        return {}
    if isinstance(provider, Mapping):
        return dict(provider)
    return dict(provider(int(block_index)))


__all__ = [
    "TransformerBlockKwargs",
    "TransformerBlockKwargsProvider",
    "resolve_transformer_block_kwargs",
]
