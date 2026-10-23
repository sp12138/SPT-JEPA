"""Standalone evaluation runner package."""

from vforge.evaluation.multi_variant import (
    MultiVariantInference,
    MultiVariantOutputWriter,
    MultiVariantSharedOutput,
    NamedVariant,
)

__all__ = [
    "MultiVariantInference",
    "MultiVariantOutputWriter",
    "MultiVariantSharedOutput",
    "NamedVariant",
]
