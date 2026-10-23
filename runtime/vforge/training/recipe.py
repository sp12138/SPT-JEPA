"""Training recipe contracts and input normalization."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import torch
import torch.nn as nn

from vforge.data.loader import (
    supports_bit_exact_resume as loader_supports_bit_exact,
)
from vforge.distributed.parallel_state import ParallelState
from vforge.distributed.parallelize import module_has_meta_tensors
from vforge.models.factory import ModelSpec, resolve_model
from vforge.models.loading import CompatibleLoadCallbacks, PretrainedWeightSource
from vforge.training.checkpoint.mapping import StateDictMapper
from vforge.utils.rng import RandomStreams

ScalarMetric = torch.Tensor | float | int


@dataclass(frozen=True)
class OptimizerSpec:
    """Engine-neutral optimizer configuration supplied by a recipe."""

    name: str = "adamw"
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.name != "adamw":
            raise ValueError(f"unsupported optimizer {self.name!r}")
        if self.learning_rate < 0:
            raise ValueError("optimizer learning_rate must be non-negative")
        if self.weight_decay < 0:
            raise ValueError("optimizer weight_decay must be non-negative")
        if len(self.betas) != 2 or any(beta < 0 or beta >= 1 for beta in self.betas):
            raise ValueError("optimizer betas must contain two values in [0, 1)")
        if self.eps <= 0:
            raise ValueError("optimizer eps must be positive")


@dataclass(frozen=True)
class SchedulerSpec:
    """Engine-neutral learning-rate schedule configuration."""

    name: str = "none"
    factor: float = 1.0
    warmup_steps: int = 0
    start_factor: float = 1.0
    end_factor: float = 0.1
    total_iters: int = 0

    def __post_init__(self) -> None:
        if self.name not in {"none", "constant", "constant_with_warmup", "linear"}:
            raise ValueError(f"unsupported scheduler {self.name!r}")
        if not 0 < self.factor <= 1:
            raise ValueError("scheduler factor must be in (0, 1]")
        if self.warmup_steps < 0:
            raise ValueError("scheduler warmup_steps must be non-negative")
        if self.name == "constant_with_warmup" and self.warmup_steps <= 0:
            raise ValueError("constant_with_warmup requires warmup_steps > 0")
        if self.name == "linear":
            if not 0 < self.start_factor <= 1:
                raise ValueError("linear start_factor must be in (0, 1]")
            if not 0 <= self.end_factor <= 1:
                raise ValueError("linear end_factor must be in [0, 1]")
            if self.total_iters <= 0:
                raise ValueError("linear total_iters must be positive")


@dataclass(frozen=True)
class StepContext:
    """Runtime state exposed to one model-specific training step."""

    device: torch.device
    parallel_state: ParallelState
    random_streams: RandomStreams = field(default_factory=RandomStreams)


# Backward-compatible alias for callers that prefer recipe terminology.
TrainContext = StepContext
StepFunction = Callable[
    [nn.Module, Any, int, StepContext],
    "StepOutput",
]
OptimizerFactory = Callable[[nn.Module], torch.optim.Optimizer | None]
SchedulerFactory = Callable[[torch.optim.Optimizer], Optional["Scheduler"]]
PhaseTarget = Callable[[nn.Module], nn.Module]
PhaseBatchSelector = Callable[[Any, str], Any]
PhaseUpdateHook = Callable[[nn.Module, int], None]
PhaseUpdateHookFactory = Callable[[nn.Module], PhaseUpdateHook]
GradientClipper = Callable[[nn.Module, Any, float, Any | None], float]
ModelInitializer = Callable[[nn.Module], None]
PostOptimizerStep = Callable[
    [nn.Module, int, StepContext, "StepOutput"],
    Mapping[str, ScalarMetric],
]


class Scheduler(Protocol):
    """Scheduler surface required by Trainer checkpoint and step handling."""

    def step(self) -> None: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


class StepReporter(Protocol):
    """Sink for canonical Trainer step payloads."""

    def report_step(self, payload: dict[str, Any]) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class StepOutput:
    """One business training step result consumed by Trainer."""

    loss: torch.Tensor
    metrics: Mapping[str, ScalarMetric] = field(default_factory=dict)
    workload: Mapping[str, float | int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Optional lazy forward losses. The engine backwards each before requesting
    # the next; loss remains the scalar reported for the whole optimizer update.
    backward_losses: Callable[[], Iterable[torch.Tensor]] | None = None


@dataclass(frozen=True)
class TrainComponents:
    """Explicit components needed to execute a training recipe."""

    model: nn.Module | ModelSpec
    dataloader: Iterable[Any]
    optimizer_factory: OptimizerFactory
    step_fn: StepFunction
    scheduler_factory: SchedulerFactory | None = None
    #: One source per published checkpoint the root is assembled from. A
    #: multi-tower model (Wan2.2 A14B) passes several, each narrowed to its own
    #: ``model_path``; single-tower models pass one and read the same.
    pretrained_weights: PretrainedWeightSource | Sequence[PretrainedWeightSource] | None = None
    state_dict_mapper: StateDictMapper | None = None
    compatible_load_callbacks: CompatibleLoadCallbacks | None = None
    dataloader_resume_contract: Mapping[str, Any] | None = None
    # Appended rather than inserted beside scheduler_factory so positional
    # callers of the existing public dataclass keep their argument mapping.
    optimizer_spec: OptimizerSpec | None = None
    scheduler_spec: SchedulerSpec | None = None
    # Optional task-owned clipping semantics. The default remains the engine's
    # logical global-norm implementation.
    gradient_clipper: GradientClipper | None = None
    # Task-owned baseline, applied after materialization when no weights are loaded.
    # TODO: Replace this transitional callback with a lifecycle where model
    # initializers see full, unsharded parameters and the framework owns
    # materialization and subsequent sharding.
    model_initializer: ModelInitializer | None = None
    # Runs after a successful optimizer update and before checkpointing. MoE
    # routing-bias updates need the post-update weights and the fsdp group.
    post_optimizer_step: PostOptimizerStep | None = None


@dataclass(frozen=True)
class PhaseSpec:
    """One independently optimized phase in a phased training recipe."""

    step_fn: StepFunction
    target: PhaseTarget
    optimizer_factory: OptimizerFactory
    scheduler_factory: SchedulerFactory | None = None
    every_n_steps: int = 1
    stream: str | None = None
    post_update_hook: PhaseUpdateHook | None = None
    post_update_hook_factory: PhaseUpdateHookFactory | None = None

    def __post_init__(self) -> None:
        if self.every_n_steps < 1:
            raise ValueError("phase every_n_steps must be at least 1")
        if self.stream is not None and not self.stream:
            raise ValueError("phase stream must be non-empty when declared")
        if self.post_update_hook is not None and self.post_update_hook_factory is not None:
            raise ValueError("declare post_update_hook or post_update_hook_factory, not both")


@dataclass(frozen=True)
class PhasedTrainComponents:
    """Components for an ordered collection of independently optimized phases."""

    model: nn.Module | ModelSpec
    dataloader: Iterable[Any]
    phases: Mapping[str, PhaseSpec]
    phase_order: Sequence[str]
    batch_selector: PhaseBatchSelector
    pretrained_weights: PretrainedWeightSource | Sequence[PretrainedWeightSource] | None = None
    state_dict_mapper: StateDictMapper | None = None
    compatible_load_callbacks: CompatibleLoadCallbacks | None = None
    dataloader_resume_contract: Mapping[str, Any] | None = None

    # Same clipping semantics for independently optimized phase targets.
    gradient_clipper_factory: Callable[[str], GradientClipper] | None = None

    def __post_init__(self) -> None:
        order = tuple(self.phase_order)
        if not order:
            raise ValueError("phase_order must contain at least one phase")
        if len(order) != len(set(order)):
            raise ValueError("phase_order must not contain duplicate phase names")
        if any(not isinstance(name, str) or not name for name in order):
            raise ValueError("phase names must be non-empty strings")
        declared = set(self.phases)
        ordered = set(order)
        if declared != ordered:
            raise ValueError(
                "phase_order must list every declared phase exactly once; "
                f"declared={sorted(declared)}, ordered={list(order)}"
            )
        if any(not isinstance(spec, PhaseSpec) for spec in self.phases.values()):
            raise TypeError("phases values must be PhaseSpec instances")
        object.__setattr__(self, "phase_order", order)


@dataclass(frozen=True)
class TrainingSession:
    """Normalized inputs consumed by the framework-owned loop."""

    setup: Callable[[torch.device, ParallelState], None]
    model: Callable[[], nn.Module]
    set_model: Callable[[nn.Module], None]
    dataloader: Callable[[], Any]
    configure_optimizers: Callable[[nn.Module], Any]
    before_step: Callable[[int], None]
    compute_step: Callable[[nn.Module, Any, int, StepContext], StepOutput]
    report_step: Callable[[dict[str, Any]], None]
    cleanup: Callable[[], None]
    start_step: Callable[[], int]
    resume_path: Callable[[], str | None]
    supports_bit_exact_resume: Callable[[Any], bool | None]
    dataloader_resume_contract: Callable[[Any], dict[str, Any] | None]
    pretrained_weights: Callable[[], tuple[PretrainedWeightSource, ...]]
    state_dict_mapper: Callable[[], StateDictMapper | None]
    compatible_load_callbacks: Callable[
        [],
        CompatibleLoadCallbacks | None,
    ] = field(default=lambda: None, repr=False)
    optimizer_spec: Callable[[], OptimizerSpec | None] = field(
        default=lambda: None,
        repr=False,
    )
    scheduler_spec: Callable[[], SchedulerSpec | None] = field(
        default=lambda: None,
        repr=False,
    )
    gradient_clipper: Callable[[], GradientClipper | None] = field(
        default=lambda: None,
        repr=False,
    )
    model_initializer: ModelInitializer | None = field(default=None, repr=False)
    post_optimizer_step: PostOptimizerStep | None = field(default=None, repr=False)


def _weight_sources(
    declared: PretrainedWeightSource | Sequence[PretrainedWeightSource] | None,
) -> tuple[PretrainedWeightSource, ...]:
    if declared is None:
        return ()
    if isinstance(declared, PretrainedWeightSource):
        return (declared,)
    return tuple(declared)


def session_from_components(
    components: TrainComponents,
    *,
    start_step: int,
    resume_path: str | None,
    reporters: Iterable[StepReporter],
) -> TrainingSession:
    """Normalize native recipe components."""
    model_holder = [components.model]
    reporter_tuple = tuple(reporters)

    def setup(device: torch.device, state: ParallelState) -> None:
        model = resolve_model(model_holder[0])
        if not module_has_meta_tensors(model):
            model = model.to(device)
        model_holder[0] = model
        del state

    def configure_optimizers(model: nn.Module):
        optimizer = components.optimizer_factory(model)
        scheduler = (
            components.scheduler_factory(optimizer)
            if optimizer is not None and components.scheduler_factory is not None
            else None
        )
        return optimizer, scheduler

    def compute_step(
        model: nn.Module,
        batch: Any,
        step: int,
        context: StepContext,
    ) -> StepOutput:
        output = components.step_fn(model, batch, step, context)
        if not isinstance(output, StepOutput):
            raise TypeError("step_fn must return StepOutput")
        return output

    def report_step(payload: dict[str, Any]) -> None:
        for reporter in reporter_tuple:
            reporter.report_step(payload)

    def cleanup() -> None:
        for reporter in reversed(reporter_tuple):
            reporter.close()

    return TrainingSession(
        setup=setup,
        model=lambda: model_holder[0],
        set_model=lambda model: model_holder.__setitem__(0, model),
        dataloader=lambda: components.dataloader,
        configure_optimizers=configure_optimizers,
        before_step=lambda _step: None,
        compute_step=compute_step,
        report_step=report_step,
        cleanup=cleanup,
        start_step=lambda: int(start_step),
        resume_path=lambda: resume_path,
        supports_bit_exact_resume=loader_supports_bit_exact,
        dataloader_resume_contract=lambda _loader: (
            dict(components.dataloader_resume_contract)
            if components.dataloader_resume_contract is not None
            else None
        ),
        pretrained_weights=lambda: _weight_sources(components.pretrained_weights),
        state_dict_mapper=lambda: components.state_dict_mapper,
        compatible_load_callbacks=lambda: components.compatible_load_callbacks,
        optimizer_spec=lambda: components.optimizer_spec,
        scheduler_spec=lambda: components.scheduler_spec,
        gradient_clipper=lambda: components.gradient_clipper,
        model_initializer=components.model_initializer,
        post_optimizer_step=components.post_optimizer_step,
    )


__all__ = [
    "GradientClipper",
    "ModelInitializer",
    "OptimizerFactory",
    "OptimizerSpec",
    "PhaseSpec",
    "PhasedTrainComponents",
    "PostOptimizerStep",
    "Scheduler",
    "SchedulerFactory",
    "SchedulerSpec",
    "StepContext",
    "StepFunction",
    "StepOutput",
    "StepReporter",
    "TrainComponents",
    "TrainContext",
]
