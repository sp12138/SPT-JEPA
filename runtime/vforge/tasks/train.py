"""Hydra entrypoint for standalone training."""

from __future__ import annotations

import logging
import os
from dataclasses import replace

import hydra
from omegaconf import DictConfig

from vforge.distributed.runtime import destroy_runtime, initialize_runtime
from vforge.tasks.common import build_and_snapshot
from vforge.tasks.registry import resolve_builder, resolve_train_preflight
from vforge.tasks.translate import train_request
from vforge.training.checkpoint import resolve_checkpoint_path
from vforge.training.phased_trainer import PhasedTrainer
from vforge.training.recipe import PhasedTrainComponents, TrainComponents
from vforge.training.reporting import (
    ConsoleReporter,
    JsonlReporter,
    WandbReporter,
    resolve_wandb_run_name,
)
from vforge.training.trainer import Trainer

logger = logging.getLogger(__name__)


def run_train(cfg: DictConfig) -> None:
    """Compose-validated train path used by CLI and tests."""
    runtime = build_and_snapshot(cfg)
    logger.info(
        "resolved train config hash=%s model=%s latent_cache.mode=%s",
        runtime.resolved_config_hash,
        runtime.config.model.name,
        runtime.config.latent_cache.mode,
    )
    request = train_request(runtime.config)
    training_recipe = getattr(
        getattr(runtime.config, "training", None),
        "recipe",
        "model",
    )
    if training_recipe == "dmd":
        from vforge.tasks.dmd.train import (
            create_dmd_components_builder,
        )

        build = create_dmd_components_builder(
            request,
            engine=runtime.config.parallel.engine,
        )
        preflight = None
    elif training_recipe == "cd":
        from vforge.tasks.cd.train import create_cd_components_builder

        build = create_cd_components_builder(
            request,
            engine=runtime.config.parallel.engine,
        )
        preflight = None
    else:
        preflight = resolve_train_preflight(runtime.config.model.name)
        component_builder = resolve_builder(runtime.config.model.name, "train")

        def build():
            return component_builder(request)

    if preflight is not None:
        # A task preflight builds no model or dataloader. Model/engine
        # incompatibilities therefore fail before process groups, CUDA
        # parameter storage, or weight I/O exist.
        preflight(request, runtime.config.parallel.engine)
    resume_path = resolve_checkpoint_path(
        runtime.resume.path,
        runtime.trainer.checkpoint.dirpath,
    )
    ctx = initialize_runtime(runtime.parallel, runtime.ops)
    try:
        components = build()
        if not isinstance(components, (TrainComponents, PhasedTrainComponents)):
            raise TypeError(
                "train builder must return TrainComponents or PhasedTrainComponents, "
                f"got {type(components).__name__}"
            )
        trainer_class = PhasedTrainer if isinstance(components, PhasedTrainComponents) else Trainer
        trainer = trainer_class(
            config=replace(
                runtime.trainer,
                checkpoint_fingerprints={
                    **runtime.trainer.checkpoint_fingerprints,
                    "resolved_backend": ctx.resolved_backend,
                },
            ),
            runtime_context=ctx,
        )
        reporters = [
            ConsoleReporter(rank=ctx.parallel_state.global_rank),
        ]
        if runtime.config.logging.jsonl:
            reporters.append(
                JsonlReporter(
                    os.path.join(
                        runtime.config.runtime.output_dir,
                        runtime.config.logging.jsonl_filename,
                    ),
                    rank=ctx.parallel_state.global_rank,
                )
            )
        if runtime.config.logging.wandb_enabled:
            reporters.append(
                WandbReporter(
                    enabled=True,
                    project=runtime.config.logging.wandb_project,
                    entity=runtime.config.logging.wandb_entity,
                    group=runtime.config.logging.wandb_group,
                    name=resolve_wandb_run_name(
                        runtime.config.logging.wandb_name,
                        output_dir=runtime.config.runtime.output_dir,
                        model_name=runtime.config.model.name,
                    ),
                    tags=list(runtime.config.logging.wandb_tags),
                    mode=runtime.config.logging.wandb_mode,
                    save_dir=(
                        runtime.config.logging.wandb_save_dir
                        or os.path.join(
                            runtime.config.runtime.output_dir,
                            "wandb",
                        )
                    ),
                    rank=ctx.parallel_state.global_rank,
                    config={
                        "resolved_config_hash": runtime.resolved_config_hash,
                        "resolved_backend": ctx.resolved_backend,
                    },
                    include_optimizer_in_backward=(
                        runtime.config.logging.wandb_include_optimizer_in_backward
                    ),
                    exclude_dataloader_from_step=(
                        runtime.config.logging.wandb_exclude_dataloader_from_step
                    ),
                )
            )
        trainer.fit(
            components,
            resume_path=resume_path,
            reporters=tuple(reporters),
        )
        logger.info(
            "train finished device=%s backend=%s hash=%s",
            ctx.device,
            ctx.resolved_backend,
            runtime.resolved_config_hash,
        )
    except Exception:
        # Covers the setup that runs before Trainer.fit gets its own handler:
        # building components, staging weights, constructing reporters.
        logger.exception("train task failed on rank %s", ctx.parallel_state.global_rank)
        raise
    finally:
        destroy_runtime()


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    run_train(cfg)


if __name__ == "__main__":
    main()
