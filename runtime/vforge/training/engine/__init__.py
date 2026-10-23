"""Training engine selection.

Chosen once per process, not dispatched per call: parallel state, model
parameterization and checkpoint layout are all bound to the engine. An
unavailable one raises instead of falling back, which would invalidate any
comparison between engines. Imports are lazy so that an optional dependency
stays out of ``sys.modules`` until requested.
"""

from __future__ import annotations

from vforge.training.engine.base import (
    ComputeStep,
    EnginePolicy,
    EngineRunContext,
    EngineStepResult,
    PrecisionContext,
    TrainingEngine,
)

ENGINE_NAMES = ("fsdp2", "megatron")


def create_engine(name: str, policy: EnginePolicy) -> TrainingEngine:
    """Instantiate the engine named ``name``."""
    if name == "fsdp2":
        from vforge.training.engine.fsdp import FsdpEngine

        return FsdpEngine(policy)
    if name == "megatron":
        from vforge.training.engine.megatron import MegatronEngine

        return MegatronEngine(policy)
    raise ValueError(f"unsupported training engine {name!r}; expected one of {ENGINE_NAMES}")


__all__ = [
    "ENGINE_NAMES",
    "ComputeStep",
    "EnginePolicy",
    "EngineRunContext",
    "EngineStepResult",
    "PrecisionContext",
    "TrainingEngine",
    "create_engine",
]
