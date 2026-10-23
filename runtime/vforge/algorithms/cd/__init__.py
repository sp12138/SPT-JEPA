"""Causal consistency distillation algorithm entry."""

from vforge.algorithms.base import Algorithm
from vforge.algorithms.cd.algorithm import (
    CausalConsistencyAlgorithm,
    CausalConsistencyMoeAlgorithm,
)
from vforge.algorithms.cd.model import (
    CdDataSupport,
    CdFamilySpec,
    CdModelFamily,
    CdNoisePair,
    CdSchedule,
    CdTeacherForcingAdapter,
    PreparedCdBatch,
)

_ALGORITHMS: dict[str, type[Algorithm]] = {
    CausalConsistencyAlgorithm.name: CausalConsistencyAlgorithm,
    CausalConsistencyMoeAlgorithm.name: CausalConsistencyMoeAlgorithm,
}


def register_algorithm(algorithm: type[Algorithm]) -> None:
    if not isinstance(algorithm, type) or not issubclass(algorithm, Algorithm):
        raise TypeError("algorithm registration requires an Algorithm subclass")
    algorithm._validate_definition()
    _ALGORITHMS[algorithm.name] = algorithm


def unregister_algorithm(name: str) -> None:
    _ALGORITHMS.pop(name, None)


def require_algorithm(name: str) -> type[Algorithm]:
    try:
        return _ALGORITHMS[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown CD algorithm {name!r}; expected one of {sorted(_ALGORITHMS)}"
        ) from exc


def create_algorithm(name: str) -> Algorithm:
    algorithm = require_algorithm(name)
    algorithm._validate_definition()
    return algorithm()


__all__ = [
    "CausalConsistencyAlgorithm",
    "CausalConsistencyMoeAlgorithm",
    "CdDataSupport",
    "CdFamilySpec",
    "CdModelFamily",
    "CdNoisePair",
    "CdSchedule",
    "CdTeacherForcingAdapter",
    "PreparedCdBatch",
    "create_algorithm",
    "register_algorithm",
]
