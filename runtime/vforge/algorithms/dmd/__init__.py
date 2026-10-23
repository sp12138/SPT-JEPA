"""Distribution Matching Distillation algorithm entry."""

from vforge.algorithms.base import Algorithm
from vforge.algorithms.dmd.algorithm import DistributionMatchingAlgorithm
from vforge.algorithms.dmd.blockwise import BlockDistributionMatchingAlgorithm
from vforge.algorithms.dmd.model import (
    DmdCriticAdapter,
    DmdDataSupport,
    DmdFamilySpec,
    DmdGeneratorAdapter,
    DmdModelFamily,
    DmdNoiseLevel,
    DmdRolloutBlock,
    LatentBundle,
    PreparedDmdBatch,
    RoleCapability,
)
from vforge.algorithms.dmd.segment_chain import SegmentChainDistributionMatchingAlgorithm

_ALGORITHMS: dict[str, type[Algorithm]] = {
    DistributionMatchingAlgorithm.name: DistributionMatchingAlgorithm,
    BlockDistributionMatchingAlgorithm.name: BlockDistributionMatchingAlgorithm,
    SegmentChainDistributionMatchingAlgorithm.name: SegmentChainDistributionMatchingAlgorithm,
}


def register_algorithm(algorithm: type[Algorithm]) -> None:
    """Register an in-tree or external training algorithm."""
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
            f"unknown DMD algorithm {name!r}; expected one of {sorted(_ALGORITHMS)}"
        ) from exc


def create_algorithm(name: str) -> Algorithm:
    algorithm = require_algorithm(name)
    algorithm._validate_definition()
    return algorithm()


__all__ = [
    "DmdCriticAdapter",
    "DmdDataSupport",
    "DmdFamilySpec",
    "DmdGeneratorAdapter",
    "DmdModelFamily",
    "DmdNoiseLevel",
    "DmdRolloutBlock",
    "DistributionMatchingAlgorithm",
    "LatentBundle",
    "PreparedDmdBatch",
    "RoleCapability",
    "create_algorithm",
    "register_algorithm",
]
