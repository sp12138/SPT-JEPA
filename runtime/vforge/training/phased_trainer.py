"""Ordered multi-phase training with independent optimizers."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from vforge.training.engine import EngineRunContext, EngineStepResult, TrainingEngine
from vforge.training.profiling import record_region
from vforge.training.recipe import (
    PhasedTrainComponents,
    StepContext,
    StepOutput,
    StepReporter,
    TrainComponents,
)
from vforge.training.trainer import Trainer, TrainerConfig


@dataclass
class NamedOptimizerState:
    """Live optimizer and scheduler objects keyed by phase name."""

    optimizers: dict[str, Any]
    schedulers: dict[str, Any] = field(default_factory=dict)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Expose the final phase LR to generic compatibility metrics."""

        for optimizer in reversed(tuple(self.optimizers.values())):
            if optimizer is not None:
                return optimizer.param_groups
        return []


class PhasedTrainer(Trainer):
    """Execute declared phases in order with independent optimizer state."""

    def __init__(self, config: TrainerConfig, **kwargs: Any) -> None:
        if config.engine != "fsdp2":
            raise NotImplementedError("PhasedTrainer first-release support is FSDP2 only")
        if config.gradient_accumulation_steps != 1:
            raise NotImplementedError("PhasedTrainer requires gradient_accumulation_steps=1")
        super().__init__(config, **kwargs)
        self._phased_components: PhasedTrainComponents | None = None
        self._post_update_hooks: dict[str, Any] = {}

    def fit(
        self,
        components: PhasedTrainComponents,
        *,
        start_step: int = 0,
        resume_path: str | None = None,
        reporters: Iterable[StepReporter] = (),
    ) -> None:
        """Normalize phased components through the shared Trainer lifecycle."""

        if self._phased_components is not None:
            raise RuntimeError("PhasedTrainer.fit is not re-entrant")
        self._phased_components = components
        try:
            adapter = TrainComponents(
                model=components.model,
                dataloader=components.dataloader,
                optimizer_factory=self._configure_named_optimizers,
                step_fn=self._unexpected_single_phase_step,
                pretrained_weights=components.pretrained_weights,
                state_dict_mapper=components.state_dict_mapper,
                compatible_load_callbacks=components.compatible_load_callbacks,
                dataloader_resume_contract=components.dataloader_resume_contract,
                gradient_clipper=(
                    components.gradient_clipper_factory(self.config.gradient_norm_dtype)
                    if components.gradient_clipper_factory is not None
                    else None
                ),
            )
            super().fit(
                adapter,
                start_step=start_step,
                resume_path=resume_path,
                reporters=reporters,
            )
        finally:
            self._post_update_hooks.clear()
            self._phased_components = None

    def _configure_named_optimizers(self, model: nn.Module) -> NamedOptimizerState:
        components = self._components()
        optimizers: dict[str, Any] = {}
        schedulers: dict[str, Any] = {}
        for phase_name in components.phase_order:
            phase = components.phases[phase_name]
            optimizer = phase.optimizer_factory(phase.target(model))
            if self.config.run_backward and optimizer is None:
                raise RuntimeError(f"training phase {phase_name!r} requires an optimizer")
            optimizers[phase_name] = optimizer
            if optimizer is not None and phase.scheduler_factory is not None:
                schedulers[phase_name] = phase.scheduler_factory(optimizer)
            if phase.post_update_hook_factory is not None:
                self._post_update_hooks[phase_name] = phase.post_update_hook_factory(model)
            elif phase.post_update_hook is not None:
                self._post_update_hooks[phase_name] = phase.post_update_hook
        return NamedOptimizerState(optimizers=optimizers, schedulers=schedulers)

    @staticmethod
    def _unexpected_single_phase_step(
        _model: nn.Module,
        _batch: Any,
        _step: int,
        _context: StepContext,
    ) -> StepOutput:
        raise RuntimeError("PhasedTrainer must execute declared phases through run_update")

    def _run_step(
        self,
        *,
        engine: TrainingEngine,
        model: nn.Module,
        microbatches: Iterator[Any],
        step: int,
        context: StepContext,
        compute_step: Any,
        optimizer: Any,
        scheduler: Any,
        run: EngineRunContext,
    ) -> EngineStepResult:
        del compute_step, scheduler
        load_started = time.perf_counter()
        batch = next(microbatches)
        dataloader_s = time.perf_counter() - load_started
        if not isinstance(optimizer, NamedOptimizerState):
            raise TypeError("PhasedTrainer expected NamedOptimizerState")
        components = self._components()
        phase_order = [
            name
            for name in components.phase_order
            if step % components.phases[name].every_n_steps == 0
        ]
        if not phase_order:
            raise RuntimeError(f"no training phase is active at global step {step}")

        phase_results: dict[str, EngineStepResult] = {}
        phase_durations: dict[str, float] = {}
        phase_learning_rates: dict[str, dict[str, float | None]] = {}

        for phase_name in phase_order:
            phase = components.phases[phase_name]
            phase_optimizer = optimizer.optimizers[phase_name]
            learning_rate_before = self._optimizer_learning_rate(phase_optimizer)
            phase_started_at = time.perf_counter()
            stream = phase.stream or phase_name
            phase_batch = components.batch_selector(batch, stream)

            def compute_phase(
                root: nn.Module,
                selected_batch: Any,
                global_step: int,
                phase_context: StepContext,
                *,
                _phase_name: str = phase_name,
            ) -> StepOutput:
                output = components.phases[_phase_name].step_fn(
                    root,
                    selected_batch,
                    global_step,
                    phase_context,
                )
                if not isinstance(output, StepOutput):
                    raise TypeError(f"training phase {_phase_name!r} must return StepOutput")
                return output

            with record_region(run.profiler, f"vforge/{phase_name}"):
                result = engine.run_update(
                    model=model,
                    microbatches=iter((phase_batch,)),
                    step=step,
                    context=context,
                    compute_step=compute_phase,
                    optimizer=phase_optimizer,
                    scheduler=optimizer.schedulers.get(phase_name),
                    run=run,
                    update_target=phase.target(model),
                )
            phase_results[phase_name] = result
            phase_durations[phase_name] = time.perf_counter() - phase_started_at
            phase_learning_rates[phase_name] = {
                "before": learning_rate_before,
                "after": self._optimizer_learning_rate(phase_optimizer),
            }
            hook = self._post_update_hooks.get(phase_name)
            if self.config.run_backward and hook is not None:
                hook(model, step)

        final_result = phase_results[phase_order[-1]]
        metrics: dict[str, Any] = {}
        diagnostics: dict[str, float] = {}
        metadata: dict[str, Any] = {
            "phase_order": tuple(phase_order),
            "phase_learning_rates": phase_learning_rates,
        }
        for phase_name in phase_order:
            result = phase_results[phase_name]
            metrics[f"{phase_name}/loss"] = result.output.loss.detach()
            metrics[f"{phase_name}/duration_s"] = phase_durations[phase_name]
            metrics.update(
                {f"{phase_name}/{name}": value for name, value in result.output.metrics.items()}
            )
            diagnostics.update(
                {f"{phase_name}/{name}": value for name, value in result.diagnostics.items()}
            )
            if result.output.metadata:
                metadata[phase_name] = dict(result.output.metadata)

        output = StepOutput(
            loss=final_result.output.loss,
            metrics=metrics,
            workload=final_result.output.workload,
            metadata=metadata,
        )
        return EngineStepResult(
            output=output,
            diagnostics=diagnostics,
            forward_s=sum(result.forward_s for result in phase_results.values()),
            backward_s=sum(result.backward_s for result in phase_results.values()),
            optimizer_s=sum(result.optimizer_s for result in phase_results.values()),
            completed_at=final_result.completed_at,
            dataloader_s=dataloader_s,
        )

    def _components(self) -> PhasedTrainComponents:
        if self._phased_components is None:
            raise RuntimeError("phased components are only available while fit is active")
        return self._phased_components

    @staticmethod
    def _optimizer_learning_rate(optimizer: Any) -> float | None:
        if optimizer is None or not optimizer.param_groups:
            return None
        return float(optimizer.param_groups[0]["lr"])


__all__ = ["NamedOptimizerState", "PhasedTrainer"]
