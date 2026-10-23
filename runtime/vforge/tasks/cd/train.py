"""Public dispatcher for multi-model consistency distillation."""

from __future__ import annotations

from collections.abc import Callable

from vforge.algorithms.cd import create_algorithm
from vforge.tasks.registry import require_registered, resolve_cd_family
from vforge.tasks.requests import TrainRequest
from vforge.training.recipe import PhasedTrainComponents


def _algorithm_name(request: TrainRequest) -> str:
    name = str(request.cd.get("algorithm") or "").strip()
    if not name:
        raise ValueError("training.cd.algorithm is required")
    return name


def _model_family(request: TrainRequest) -> str:
    families = {model.model for model in request.training_models.values()}
    if not families:
        raise ValueError("training.recipe=cd requires training.models")
    if len(families) != 1:
        raise NotImplementedError(
            "this CD release requires every role to use the same model family; "
            f"got {sorted(families)}"
        )
    family = next(iter(families))
    require_registered(family)
    return family


def create_cd_components_builder(
    request: TrainRequest,
    *,
    engine: str,
) -> Callable[[], PhasedTrainComponents]:
    """Validate before runtime setup and defer all model/data allocation."""
    algorithm = create_algorithm(_algorithm_name(request))
    model_name = _model_family(request)
    model_family = resolve_cd_family(model_name)
    algorithm.validate_request(
        request,
        engine=engine,
        integration=model_family,
    )

    def build() -> PhasedTrainComponents:
        return algorithm.build_components(
            request,
            integration=model_family,
        )

    return build


__all__ = ["create_cd_components_builder"]
