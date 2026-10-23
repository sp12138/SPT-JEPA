"""Native model plugins for standalone vforge tasks."""

from .factory import ModelSpec, resolve_model
from .loading import PretrainedWeightSource

__all__ = ["ModelSpec", "PretrainedWeightSource", "resolve_model"]
