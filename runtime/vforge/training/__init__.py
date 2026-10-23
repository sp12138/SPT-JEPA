"""Framework-independent training loop and execution lifecycle."""

from vforge.checkpointing.offload import AttentionOffloadPolicy
from vforge.training.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    CheckpointManifest,
)
from vforge.training.diagnostics import DiagnosticsConfig
from vforge.training.ema import ShardedEmaState
from vforge.training.phased_trainer import PhasedTrainer
from vforge.training.recipe import (
    OptimizerFactory,
    OptimizerSpec,
    PhasedTrainComponents,
    PhaseSpec,
    PostOptimizerStep,
    Scheduler,
    SchedulerFactory,
    SchedulerSpec,
    StepContext,
    StepFunction,
    StepOutput,
    StepReporter,
    TrainComponents,
    TrainContext,
    TrainingSession,
)
from vforge.training.reporting import (
    WandbReporter,
    print_json_step,
    scalar_step_metrics,
    wandb_step_metrics,
)
from vforge.training.trainer import (
    Trainer,
    TrainerConfig,
)

__all__ = [
    "AttentionOffloadPolicy",
    "CheckpointConfig",
    "CheckpointManager",
    "CheckpointManifest",
    "DiagnosticsConfig",
    "OptimizerFactory",
    "OptimizerSpec",
    "PhaseSpec",
    "PhasedTrainComponents",
    "PhasedTrainer",
    "PostOptimizerStep",
    "Scheduler",
    "SchedulerFactory",
    "SchedulerSpec",
    "ShardedEmaState",
    "StepFunction",
    "StepContext",
    "StepOutput",
    "StepReporter",
    "Trainer",
    "TrainerConfig",
    "TrainComponents",
    "TrainContext",
    "TrainingSession",
    "WandbReporter",
    "print_json_step",
    "scalar_step_metrics",
    "wandb_step_metrics",
]
