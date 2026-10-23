"""Framework contract shared by pluggable training algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from vforge.tasks.requests import TrainRequest
    from vforge.training.recipe import PhasedTrainComponents


class Algorithm[IntegrationT](ABC):
    """Build one training recipe from a model-integration contract."""

    name: ClassVar[str]

    @classmethod
    def _validate_definition(cls) -> None:
        if not getattr(cls, "name", ""):
            raise ValueError("algorithm name must be non-empty")

    @abstractmethod
    def validate_request(
        self,
        request: TrainRequest,
        *,
        engine: str,
        integration: IntegrationT,
    ) -> None:
        """Validate the complete recipe without allocating models or data."""

    @abstractmethod
    def build_components(
        self,
        request: TrainRequest,
        *,
        integration: IntegrationT,
    ) -> PhasedTrainComponents:
        """Build models, adapters, data, losses, and executable phases."""


__all__ = ["Algorithm"]
