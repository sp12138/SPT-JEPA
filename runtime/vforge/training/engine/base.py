"""Training engine contract.

An engine owns model parallelization and one optimizer update. The Trainer owns
the global-step loop, phase scheduling, data, checkpointing and reporting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch
import torch.nn as nn

from vforge.distributed.parallelize import ParallelizeConfig
from vforge.training.diagnostics import DiagnosticsConfig, LogicalReduction, ParameterBucket
from vforge.training.parity import ParityStepConfig
from vforge.training.recipe import (
    GradientClipper,
    OptimizerSpec,
    SchedulerSpec,
    StepContext,
    StepOutput,
)

ComputeStep = Callable[[nn.Module, Any, int, StepContext], StepOutput]
PrecisionContext = Callable[[], AbstractContextManager[Any]]
_ForwardResult = TypeVar("_ForwardResult")


@dataclass(frozen=True)
class EnginePolicy:
    """Run configuration, mirrored so that an engine does not import the Trainer."""

    parallelize: ParallelizeConfig
    run_backward: bool = True
    gradient_clip_val: float = 0.0
    gradient_accumulation_steps: int = 1
    parity: ParityStepConfig | None = None
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    gradient_norm_dtype: str = "float32"

    def __post_init__(self) -> None:
        if (
            type(self.gradient_accumulation_steps) is not int
            or self.gradient_accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be a positive integer")


@dataclass(frozen=True)
class EngineRunContext:
    """Run-scoped collaborators an engine uses but does not own."""

    precision_context: PrecisionContext
    offload_manager: Any | None = None
    parameter_bucket: ParameterBucket | None = None
    profiler: Any | None = None
    logical_reduction: LogicalReduction | None = None
    gradient_clipper: GradientClipper | None = None


@dataclass(frozen=True)
class EngineStepResult:
    """Result of one engine-owned forward/backward/update transaction."""

    output: StepOutput
    diagnostics: Mapping[str, float]
    forward_s: float
    backward_s: float
    optimizer_s: float
    completed_at: float
    dataloader_s: float


class TrainingEngine(ABC):
    """One training engine, selected once per process.

    It rejects a policy it cannot execute when constructed, and speaks only about
    mechanisms it owns, not about what a given model supports.
    """

    name: str

    @abstractmethod
    def prepare_model(
        self,
        model: nn.Module,
        *,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> nn.Module:
        """Apply the engine's parallelization before optimizer construction."""
        raise NotImplementedError

    @abstractmethod
    def materialize_model(
        self,
        model: nn.Module,
        device: torch.device | str,
    ) -> nn.Module:
        """Allocate engine-ready parameters after sharding and rebuild metadata."""
        raise NotImplementedError

    @abstractmethod
    def prepare_data_parallel(self, model: nn.Module) -> nn.Module:
        """Apply replicated data parallelism after materialization and weight load."""
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(
        self,
        model: nn.Module,
        optimizer_spec: OptimizerSpec | None,
        scheduler_spec: SchedulerSpec | None,
    ) -> tuple[Any, Any] | None:
        """Build engine-owned optimizers, or return None for the recipe fallback."""
        raise NotImplementedError

    def initialize_optimizer_from_pretrained(
        self,
        model: nn.Module,
        optimizer: Any,
        sources: tuple[Any, ...],
    ) -> None:
        """Let an engine preserve pretrained precision in optimizer-owned state."""
        del model, optimizer, sources

    def run_forward(
        self,
        forward: Callable[[], _ForwardResult],
        *,
        run: EngineRunContext,
    ) -> _ForwardResult:
        """Execute one model-forward boundary under the engine runtime policy."""
        from vforge.training.profiling import record_region

        with record_region(run.profiler, "vforge/forward"):
            with run.precision_context():
                return forward()

    @abstractmethod
    def run_update(
        self,
        *,
        model: nn.Module,
        microbatches: Iterator[Any],
        step: int,
        context: StepContext,
        compute_step: ComputeStep,
        optimizer: Any,
        scheduler: Any,
        run: EngineRunContext,
        update_target: nn.Module | None = None,
    ) -> EngineStepResult:
        """Run one optimizer update.

        ``model`` is the forward root. ``update_target`` identifies the
        independently parallelized root that owns this optimizer's gradients;
        it defaults to ``model`` for ordinary single-model training.
        Consume exactly the policy's accumulation count from ``microbatches``,
        interleaving forward and backward. Leave later batches for the next
        update; exhaustion before the count is reached fails the update.
        All batches share ``context`` (including its continuous random streams)
        and the same optimizer ``step``.
        """
        raise NotImplementedError


__all__ = [
    "ComputeStep",
    "EnginePolicy",
    "EngineRunContext",
    "EngineStepResult",
    "PrecisionContext",
    "TrainingEngine",
]
