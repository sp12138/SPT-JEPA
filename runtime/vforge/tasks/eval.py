"""Hydra entrypoint for standalone evaluation."""

from __future__ import annotations

import logging
import os
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import hydra
from omegaconf import DictConfig

from vforge.distributed.runtime import destroy_runtime, initialize_runtime
from vforge.evaluation.runner import EvalComponents, run_evaluation
from vforge.tasks.common import build_and_snapshot
from vforge.tasks.registry import resolve_builder
from vforge.tasks.translate import eval_request
from vforge.training.checkpoint import (
    discover_checkpoints,
    read_manifest,
    resolve_checkpoint_path,
)
from vforge.training.engine import EnginePolicy, EngineRunContext, create_engine
from vforge.utils.precision import autocast_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _EvalCheckpoint:
    path: str | Mapping[str, str]
    step: int | None = None
    output_name: str | None = None


def run_eval(cfg: DictConfig) -> None:
    runtime = build_and_snapshot(cfg)
    if (
        runtime.config.evaluation.checkpoint_format == "mcore"
        and runtime.config.parallel.engine != "megatron"
    ):
        raise ValueError("evaluation.checkpoint_format=mcore requires parallel.engine=megatron")
    logger.info(
        "resolved eval config hash=%s model=%s checkpoint=%s",
        runtime.resolved_config_hash,
        runtime.config.model.name,
        runtime.config.evaluation.checkpoint_path,
    )
    checkpoints = _resolve_eval_checkpoints(runtime.config)
    request = eval_request(runtime.config)
    ctx = initialize_runtime(runtime.parallel, runtime.ops)
    try:
        engine = create_engine(
            runtime.config.parallel.engine,
            EnginePolicy(
                parallelize=runtime.trainer.parallelize,
                run_backward=False,
            ),
        )
        engine_run = EngineRunContext(
            precision_context=lambda: autocast_context(
                ctx.device,
                enabled=runtime.trainer.autocast,
                precision=runtime.trainer.precision,
            ),
        )
        build = resolve_builder(runtime.config.model.name, "eval")
        output_root = (
            runtime.config.evaluation.output_dir or f"{runtime.config.runtime.output_dir}/eval"
        )
        for checkpoint in checkpoints:
            current_request = replace(request, checkpoint_path=checkpoint.path)
            components = build(current_request)
            if not isinstance(components, EvalComponents):
                raise TypeError(
                    f"eval builder must return EvalComponents, got {type(components).__name__}"
                )
            manifest_context = {
                "resolved_config_hash": runtime.resolved_config_hash,
                "ops_policy": asdict(runtime.ops),
                "resolved_backend": ctx.resolved_backend,
                "model": runtime.config.model.name,
                "data_source": runtime.config.evaluation.data_source,
                "manifest": runtime.config.evaluation.manifest,
                "sources": [asdict(source) for source in runtime.config.evaluation.sources],
                "topology": asdict(runtime.parallel),
                "limitations": asdict(runtime.config.limitations),
                "checkpoint_step": checkpoint.step,
                "evaluate_all_checkpoints": bool(
                    runtime.config.evaluation.evaluate_all_checkpoints
                ),
            }
            output_dir = (
                os.path.join(output_root, checkpoint.output_name)
                if checkpoint.output_name is not None
                else output_root
            )
            manifest = run_evaluation(
                components,
                engine=engine,
                engine_run=engine_run,
                device=ctx.device,
                output_dir=output_dir,
                seed=runtime.config.evaluation.seed,
                skip_existing=runtime.config.evaluation.skip_existing,
                weights_strict=runtime.config.runtime.weights_strict,
                manifest_context=manifest_context,
            )
            logger.info(
                "eval finished checkpoint=%s output_dir=%s samples=%s",
                checkpoint.path,
                manifest.get("output_dir"),
                manifest.get("num_samples"),
            )
    finally:
        destroy_runtime()


def _resolve_eval_checkpoints(config: Any) -> tuple[_EvalCheckpoint, ...]:
    evaluation = config.evaluation
    configured = evaluation.checkpoint_path
    checkpoint_dir = str(config.checkpoint.dirpath).strip()
    if not evaluation.evaluate_all_checkpoints:
        if isinstance(configured, str) and configured.strip().lower() in {"auto", "latest"}:
            if not checkpoint_dir:
                raise ValueError(
                    "evaluation.checkpoint_path=latest|auto requires checkpoint.dirpath"
                )
            resolved = resolve_checkpoint_path(configured, checkpoint_dir)
            if resolved is None:
                raise FileNotFoundError(f"no complete checkpoint found in {checkpoint_dir}")
            if not _checkpoint_matches_format(resolved, evaluation.checkpoint_format):
                matching = [
                    path
                    for _step, path in discover_checkpoints(checkpoint_dir)
                    if _checkpoint_matches_format(path, evaluation.checkpoint_format)
                ]
                if not matching:
                    raise FileNotFoundError(
                        f"no complete {evaluation.checkpoint_format} checkpoint found "
                        f"in {checkpoint_dir}"
                    )
                resolved = matching[-1]
            return (_EvalCheckpoint(path=resolved),)
        return (_EvalCheckpoint(path=configured),)

    if isinstance(configured, Mapping):
        raise TypeError("evaluation.evaluate_all_checkpoints does not support checkpoint mappings")
    if str(configured).strip().lower() not in {"", "auto", "latest"}:
        raise ValueError(
            "evaluation.evaluate_all_checkpoints discovers checkpoint.dirpath; "
            "evaluation.checkpoint_path must be empty, auto, or latest"
        )
    if not checkpoint_dir:
        raise ValueError("evaluation.evaluate_all_checkpoints requires checkpoint.dirpath")

    discovered = [
        (step, path)
        for step, path in discover_checkpoints(checkpoint_dir)
        if _checkpoint_matches_format(path, evaluation.checkpoint_format)
    ]
    if not discovered:
        raise FileNotFoundError(
            f"no complete {evaluation.checkpoint_format} checkpoints found in {checkpoint_dir}"
        )
    step_counts = Counter(step for step, _path in discovered)
    return tuple(
        _EvalCheckpoint(
            path=path,
            step=step,
            output_name=_checkpoint_output_name(
                step,
                path,
                include_format=step_counts[step] > 1,
            ),
        )
        for step, path in discovered
    )


def _checkpoint_matches_format(path: str, requested: str) -> bool:
    if requested == "auto":
        return True
    if os.path.isfile(path):
        return (requested == "pt" and path.endswith(".pt")) or (
            requested == "safetensors" and path.endswith(".safetensors")
        )
    return read_manifest(path).format == requested


def _checkpoint_output_name(step: int, path: str, *, include_format: bool) -> str:
    name = f"checkpoint-{step:05d}"
    if not include_format:
        return name
    checkpoint_format = "pt" if os.path.isfile(path) else read_manifest(path).format
    return f"{name}-{checkpoint_format}"


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="eval",
)
def main(cfg: DictConfig) -> None:
    run_eval(cfg)


if __name__ == "__main__":
    main()
