"""Torch-native trainer owning vforge execution and step lifecycle."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from vforge.checkpointing.offload import (
    ActivationOffloadManager,
    AttentionOffloadPolicy,
)
from vforge.data.loader import TrainingDataLoader, is_resumable_dataloader
from vforge.distributed.parallelize import (
    ParallelizeConfig,
    module_has_meta_tensors,
)
from vforge.distributed.runtime import (
    RuntimeContext,
    destroy_runtime,
    ensure_runtime,
    get_runtime,
)
from vforge.models.loading import load_pretrained_weights, stage_pretrained_weights
from vforge.ops.config import OpsConfig
from vforge.training.checkpoint import (
    CheckpointConfig,
    CheckpointManager,
    ResumeConfig,
    build_dataloader_resume_contract,
    execution_fingerprint,
    fast_forward_scheduler,
    load_model_weights,
    load_training_state,
    parallel_fingerprint,
    prepare_staged_dcp_path,
    read_manifest,
    restore_scheduler,
    validate_bit_exact_dataloader,
    validate_dataloader_resume_contract,
    validate_resume_fingerprints,
)
from vforge.training.diagnostics import (
    DiagnosticsConfig,
    LogicalReduction,
    ParameterBucket,
    global_param_value_stats,
    gradient_clipping_fingerprint,
    log_local_parameter_stats,
    validate_gradient_norm_dtype,
)
from vforge.training.engine import (
    ENGINE_NAMES,
    EnginePolicy,
    EngineRunContext,
    TrainingEngine,
    create_engine,
)
from vforge.training.parity import ParityStepConfig
from vforge.training.profiling import (
    advance_profiler,
    build_torch_profiler,
    record_region,
    stop_profiler,
)
from vforge.training.recipe import (
    StepContext,
    StepReporter,
    TrainComponents,
    TrainingSession,
    session_from_components,
)
from vforge.training.reporting import (
    compute_throughput_metrics,
    global_mean_step_payload,
    scalar_step_metrics,
)
from vforge.utils.precision import autocast_context
from vforge.utils.rng import seed_process
from vforge.utils.s3_staging import (
    StorageConfig,
    activate_storage_config,
    resolve_s3_source,
    restore_storage_config,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerConfig:
    max_steps: int
    parallelize: ParallelizeConfig
    seed: int = 42
    precision: str = "bf16"
    autocast: bool = False
    run_backward: bool = True
    attention_offload: AttentionOffloadPolicy = field(default_factory=AttentionOffloadPolicy)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    weights_strict: bool = False
    storage: StorageConfig = field(default_factory=StorageConfig)
    gradient_accumulation_steps: int = 1
    gradient_clip_val: float = 0.0
    log_every_n_steps: int = 1
    log_timing: bool = True
    parity: ParityStepConfig | None = None
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    diagnostic_parameter_bucket: ParameterBucket | None = None
    checkpoint_fingerprints: dict[str, Any] = field(default_factory=dict)
    runtime_ops: OpsConfig | None = None
    engine: str = "fsdp2"
    gradient_norm_dtype: str = "float32"

    def __post_init__(self) -> None:
        validate_gradient_norm_dtype(
            self.gradient_norm_dtype,
            engine=self.engine,
            tp_size=self.parallelize.parallel.tp_size,
        )
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.engine not in ENGINE_NAMES:
            raise ValueError(f"unsupported engine {self.engine!r}; expected one of {ENGINE_NAMES}")
        if (
            type(self.gradient_accumulation_steps) is not int
            or self.gradient_accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        if self.gradient_clip_val < 0:
            raise ValueError("gradient_clip_val must be non-negative")
        if self.log_every_n_steps < 1:
            raise ValueError("log_every_n_steps must be at least 1")
        if (
            self.checkpoint.enabled
            and self.checkpoint.state == "full"
            and self.checkpoint.save_optimizer
            and self.engine == "megatron"
            and self.checkpoint.format != "mcore"
        ):
            # MCore DistributedOptimizer owns FP32 parameter and moment shards
            # whose layout is independent of the model state-dict hooks. Passing
            # its wrapper to torch.distributed.checkpoint's torch-optimizer path
            # would either fail or silently describe the wrong logical tensors.
            # Model-only Torch DCP remains valid. Full optimizer state uses the
            # MCore-native format so MCore owns its master-weight and moment
            # layout instead of exposing it as a torch optimizer state dict.
            raise NotImplementedError(
                "the megatron engine saves its distributed optimizer with "
                "checkpoint.format='mcore', not Torch DCP. got "
                f"tp_size={self.parallelize.parallel.tp_size} with "
                "checkpoint.state=full and checkpoint.save_optimizer=true. Model "
                "weights are supported: set save_optimizer=false, or "
                "checkpoint.state=weights, select format=mcore, or run tp on "
                "the fsdp2 engine"
            )
        if (
            self.checkpoint.enabled
            and self.checkpoint.format == "mcore"
            and self.engine != "megatron"
        ):
            raise ValueError("checkpoint.format='mcore' requires engine='megatron'")
        if self.autocast:
            logger.warning(
                "Global autocast is enabled (runtime.autocast=true). It can lower "
                "the compute precision of modules intended to remain FP32. "
                "Set runtime.autocast=false and use module-level mixed precision policies."
            )


class Trainer:
    """Own distributed execution for native recipes and normalized sessions."""

    def __init__(
        self,
        config: TrainerConfig,
        *,
        runtime_context: RuntimeContext | None = None,
    ) -> None:
        self.config = config
        self.runtime_context = runtime_context

    def fit(
        self,
        components: TrainComponents,
        *,
        start_step: int = 0,
        resume_path: str | None = None,
        reporters: Iterable[StepReporter] = (),
    ) -> None:
        session = session_from_components(
            components,
            start_step=start_step,
            resume_path=resume_path,
            reporters=reporters,
        )
        self._fit_session(session)

    def fit_session(self, session: TrainingSession) -> None:
        """Execute an already normalized session."""
        self._fit_session(session)

    def _fit_session(self, session: TrainingSession) -> None:
        # Constructed first so an unusable engine fails before collectives and
        # the model exist.
        gradient_clipping_fingerprint(session.gradient_clipper(), self.config.gradient_norm_dtype)
        engine = create_engine(self.config.engine, self._engine_policy())
        had_runtime = get_runtime() is not None
        if self.runtime_context is not None:
            runtime = ensure_runtime(
                self.config.parallelize.parallel,
                self.runtime_context.ops,
                device=self.runtime_context.device,
            )
            if runtime is not self.runtime_context:
                raise RuntimeError(
                    "Trainer received a RuntimeContext that is not the active runtime"
                )
        else:
            ops = self.config.runtime_ops
            if ops is None:
                ops = OpsConfig()
            runtime = ensure_runtime(
                self.config.parallelize.parallel,
                ops,
            )
        owns_runtime = not had_runtime
        device = runtime.device
        state = runtime.parallel_state
        offload_manager: ActivationOffloadManager | None = None
        checkpoint_manager: CheckpointManager | None = None
        profiler: Any | None = None
        storage_token = activate_storage_config(self.config.storage)
        train_context = StepContext(device=device, parallel_state=state)
        try:
            seed_process(self.config.seed)
            checkpoint_manager = CheckpointManager(
                self.config.checkpoint,
                state_dict_mapper=session.state_dict_mapper(),
                random_streams=train_context.random_streams,
            )
            session.setup(device, state)

            offload_manager = self._build_offload_manager()
            gradient_checkpointing_kwargs = (
                None
                if offload_manager is None
                else {
                    "use_reentrant": False,
                    "context_fn": offload_manager.checkpoint_contexts,
                }
            )
            model = engine.prepare_model(
                session.model(),
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
            )
            materialized_from_meta = module_has_meta_tensors(model)
            model = engine.materialize_model(model, device)
            resume_path = session.resume_path()
            compatible_overlay = (
                resume_path is not None
                and not self.config.weights_strict
                and (not os.path.isdir(resume_path) or not self.config.resume.effective_strict)
            )
            sources = session.pretrained_weights()
            staged_sources = (
                tuple(stage_pretrained_weights(source) for source in sources)
                if resume_path is None or compatible_overlay
                else ()
            )
            if resume_path is None or compatible_overlay:
                if staged_sources:
                    for source in staged_sources:
                        load_pretrained_weights(
                            model,
                            source,
                            strict=self.config.weights_strict,
                        )
                elif session.model_initializer is not None:
                    # TODO: Move initialization before sharding, coordinating
                    # meta materialization and peak memory in the lifecycle refactor.
                    session.model_initializer(model)
                elif materialized_from_meta:
                    raise RuntimeError(
                        "the meta-initialized model has no pretrained weights or "
                        "scratch initializer baseline"
                    )
            parameter_bucket = self.config.diagnostic_parameter_bucket
            model = engine.prepare_data_parallel(model)
            session.set_model(model)
            logical_reduction = LogicalReduction.for_model(model, state)
            optimizer, scheduler = self._configure_optimizers(session, model, engine)
            if resume_path is None and staged_sources:
                engine.initialize_optimizer_from_pretrained(model, optimizer, staged_sources)
            source_loader = session.dataloader()
            dataloader_resume_contract = build_dataloader_resume_contract(
                source_loader,
                session.dataloader_resume_contract(source_loader),
            )
            loader = TrainingDataLoader(source_loader, device=device)
            execution_contract = execution_fingerprint(
                seed=self.config.seed,
                precision=self.config.precision,
                autocast=self.config.autocast,
                gradient_accumulation_steps=(self.config.gradient_accumulation_steps),
                gradient_clip_val=self.config.gradient_clip_val,
                gradient_norm_dtype=self.config.gradient_norm_dtype,
                gradient_clipper=session.gradient_clipper(),
                parity=self.config.parity,
                optimizer=optimizer,
                scheduler=scheduler,
                parallelize=self.config.parallelize,
            )
            parallel_contract = parallel_fingerprint(self.config.parallelize)
            rank_state: dict[str, Any] | None = None
            dcp_manifest = None
            resume_plan = self.config.resume
            resume_bit_exact = resume_plan.effective_strict
            if resume_path is not None and os.path.isdir(resume_path):
                resume_path = prepare_staged_dcp_path(
                    resume_path,
                    enabled=(
                        resume_plan.stage_remote or resolve_s3_source(resume_path) is not None
                    ),
                )
                dcp_manifest = read_manifest(resume_path)
                if dcp_manifest.format == "mcore" and self.config.engine != "megatron":
                    raise ValueError(
                        "MCore native checkpoints can only be loaded by engine='megatron'"
                    )
                if (
                    resume_plan.optimizer
                    and self.config.engine == "megatron"
                    and dcp_manifest.format != "mcore"
                ):
                    raise ValueError(
                        "the megatron engine cannot restore optimizer state from a "
                        "Torch DCP checkpoint; use checkpoint.format='mcore' or set "
                        "resume.optimizer=false"
                    )
                if resume_plan.effective_strict:
                    resume_bit_exact = validate_resume_fingerprints(
                        dcp_manifest,
                        execution_policy=execution_contract,
                        topology=parallel_contract,
                        fingerprints={
                            key: value
                            for key, value in self.config.checkpoint_fingerprints.items()
                            if key != "resolved_config_hash"
                        },
                    )
                dcp_manifest, rank_state = load_training_state(
                    model,
                    optimizer if resume_plan.optimizer else None,
                    resume_path,
                    resume=resume_plan,
                    warn_on_non_bit_exact=not resume_plan.effective_strict,
                    weights_strict=self.config.weights_strict,
                )
                start_step = int(dcp_manifest.completed_steps) if resume_plan.training_state else 0
            elif resume_path is not None:
                if resume_plan.optimizer or resume_plan.training_state:
                    if state.global_rank == 0:
                        print(
                            "VFORGE_PT_WARMSTART ignoring non-weight resume "
                            "settings because PT checkpoints contain weights only",
                            flush=True,
                        )
                load_model_weights(
                    model,
                    resume_path,
                    format="pt",
                    state_dict_mapper=session.state_dict_mapper(),
                    stage_remote=resume_plan.stage_remote,
                    strict=self.config.weights_strict,
                    compatible_load_callbacks=(session.compatible_load_callbacks()),
                )
                # PT is an explicit weights-only warm-start. It has no atomic
                # training state, so filename/application counters must not
                # create a partial resume with a fresh optimizer/data/RNG.
                start_step = 0
            else:
                start_step = self._resolve_start_step(session)
            if self.config.diagnostics.log_local_parameter_stats:
                log_local_parameter_stats(
                    model,
                    phase="post_load",
                    rank=state.global_rank,
                )
            if os.environ.get("VFORGE_WEIGHT_TRACE") == "1":
                weight_stats = global_param_value_stats(model, logical_reduction)
                if state.global_rank == 0:
                    print(
                        "VFORGE_WEIGHT_TRACE " + json.dumps(weight_stats, sort_keys=True),
                        flush=True,
                    )
            if start_step > int(self.config.max_steps):
                raise ValueError(
                    f"start_step={start_step} exceeds max_steps={self.config.max_steps}"
                )
            restore_training_state = (
                dcp_manifest is not None and resume_plan.training_state and rank_state is not None
            )
            if restore_training_state:
                assert rank_state is not None
                resume_bit_exact = bool(rank_state.get("_vforge_bit_exact", resume_bit_exact))
                if resume_bit_exact:
                    validate_bit_exact_dataloader(
                        source_loader,
                        session.supports_bit_exact_resume,
                    )
                restore_scheduler(scheduler, rank_state.get("scheduler"))
                saved_loader_state = rank_state.get("dataloader")
                resume_bit_exact = validate_dataloader_resume_contract(
                    (
                        saved_loader_state.get("contract")
                        if isinstance(saved_loader_state, dict)
                        else None
                    ),
                    dataloader_resume_contract,
                    bit_exact=resume_bit_exact,
                )
                loader.load_state_dict(saved_loader_state)
            else:
                # Fresh scheduler when not restoring scheduler state. Do not
                # assume every scheduler can be reconstructed by fast-forward.
                if resume_plan.training_state and start_step > 0:
                    fast_forward_scheduler(scheduler, start_step)
            if self.config.checkpoint.enabled and self.config.checkpoint.state == "full":
                if not is_resumable_dataloader(loader):
                    raise RuntimeError(
                        "full DCP requires StatefulDataLoader or "
                        "LeaderBroadcastDataLoader; replay-based resume is not supported"
                    )
                validate_bit_exact_dataloader(
                    source_loader,
                    session.supports_bit_exact_resume,
                )
            batches = iter(loader)
            if (
                restore_training_state
                and rank_state is not None
                and rank_state.get("_vforge_restore_rng", True)
            ):
                train_context.random_streams.load_state_dict(rank_state["rng"])

            profiler = build_torch_profiler(rank=state.global_rank)
            if profiler is not None:
                profiler.start()
            engine_run = EngineRunContext(
                precision_context=lambda: self._precision_context(device),
                offload_manager=offload_manager,
                parameter_bucket=parameter_bucket,
                profiler=profiler,
                logical_reduction=logical_reduction,
                gradient_clipper=session.gradient_clipper(),
            )
            for step in range(start_step, int(self.config.max_steps)):
                checkpoint_manager.poll()
                started_at = time.perf_counter()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                session.before_step(step)
                learning_rate_before_step = self._learning_rate(optimizer)

                step_result = self._run_step(
                    engine=engine,
                    model=model,
                    microbatches=batches,
                    step=step,
                    context=train_context,
                    compute_step=session.compute_step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    run=engine_run,
                )
                step_output = step_result.output
                loss = step_output.loss
                metrics = dict(step_output.metrics)
                if self.config.run_backward and session.post_optimizer_step is not None:
                    post_step_metrics = dict(
                        session.post_optimizer_step(
                            model,
                            step,
                            train_context,
                            step_output,
                        )
                    )
                    duplicate_metrics = set(metrics) & set(post_step_metrics)
                    if duplicate_metrics:
                        raise ValueError(
                            "post_optimizer_step returned duplicate metric names: "
                            f"{sorted(duplicate_metrics)}"
                        )
                    metrics.update(post_step_metrics)
                diagnostics = step_result.diagnostics
                optimizer_at = step_result.completed_at
                learning_rate_after_step = self._learning_rate(optimizer)

                completed_steps = step + 1
                is_final = completed_steps >= int(self.config.max_steps)
                checkpoint_at = optimizer_at
                checkpoint_s = 0.0
                saved_path: str | None = None
                pending_checkpoint = None
                if checkpoint_manager.should_save(
                    completed_steps,
                    is_final=is_final,
                ):
                    checkpoint_result = checkpoint_manager.save(
                        model,
                        optimizer,
                        scheduler,
                        completed_steps,
                        dataloader_state={
                            **loader.state_dict(),
                            "contract": dataloader_resume_contract,
                        },
                        execution_policy=execution_contract,
                        topology=parallel_contract,
                        fingerprints=self.config.checkpoint_fingerprints,
                    )
                    saved_path = checkpoint_result.path
                    pending_checkpoint = checkpoint_result.pending
                    checkpoint_at = time.perf_counter()
                    checkpoint_s = checkpoint_at - optimizer_at

                scalar_metrics = scalar_step_metrics(metrics)
                loss_backward = float(loss.detach().float().cpu())
                loss_scalar = float(scalar_metrics.get("loss_scalar", loss_backward))
                payload: dict[str, Any] = {
                    "event": "step",
                    "step": step,
                    "rank": state.global_rank,
                    "world_size": state.world_size,
                    "dp_size": state.config.dp_size,
                    "usp_size": state.config.usp_size,
                    "loss": loss_scalar,
                    "loss_backward": loss_backward,
                    # The optimizer update consumed the pre-scheduler value.
                    "learning_rate": learning_rate_before_step,
                    "learning_rate_before_step": learning_rate_before_step,
                    "learning_rate_after_step": learning_rate_after_step,
                    "metrics": scalar_metrics,
                    "step_metadata": dict(step_output.metadata),
                }
                payload["metrics"].update(diagnostics)
                if saved_path is not None:
                    payload["checkpoint_path"] = saved_path
                    payload["checkpoint_step"] = completed_steps
                if pending_checkpoint is not None:
                    payload["checkpoint_pending_path"] = pending_checkpoint.path
                    payload["checkpoint_step"] = completed_steps
                stats = offload_manager.pop_stats() if offload_manager is not None else None
                if stats is not None:
                    payload["attention_offload"] = stats
                if device.type == "cuda":
                    gib = float(1024**3)
                    payload["memory"] = {
                        "allocated_gb": (torch.cuda.memory_allocated(device) / gib),
                        "reserved_gb": (torch.cuda.memory_reserved(device) / gib),
                        "peak_allocated_gb": (torch.cuda.max_memory_allocated(device) / gib),
                        "peak_reserved_gb": (torch.cuda.max_memory_reserved(device) / gib),
                    }
                if self.config.log_timing:
                    # End-to-end step time includes checkpoint gather/write on
                    # save steps; step_compute excludes it.
                    payload["timing"] = {
                        "dataloader_s": step_result.dataloader_s,
                        "forward_s": step_result.forward_s,
                        "backward_s": step_result.backward_s,
                        "optimizer_s": step_result.optimizer_s,
                        "checkpoint_s": checkpoint_s,
                        "step_compute_s": optimizer_at - started_at,
                        "step_s": checkpoint_at - started_at,
                    }
                    if pending_checkpoint is not None:
                        payload["timing"].update(pending_checkpoint.timing_dict())
                workload = step_output.workload
                if workload:
                    payload["perf"] = compute_throughput_metrics(
                        local_batch=workload["local_batch"],
                        local_tokens=workload["local_tokens"],
                        dp_size=state.config.dp_size,
                        world_size=state.world_size,
                        timing=payload.get("timing"),
                    )
                if (step + 1) % self.config.log_every_n_steps == 0:
                    with record_region(profiler, "vforge/report_step"):
                        session.report_step(
                            global_mean_step_payload(
                                payload,
                                device=device,
                            )
                        )
                advance_profiler(profiler, device)
        except Exception:
            logger.exception("training failed on rank %s", state.global_rank)
            # Skip the orderly teardown: joining the checkpoint upload and
            # destroying the process groups both block until ranks that are
            # still inside a collective show up, which trades one rank's
            # traceback for a watchdog timeout on all of them. Flush the
            # reporter so the failure reaches wandb, drop the runtime
            # bookkeeping without touching NCCL, and let the process die.
            session.cleanup()
            if owns_runtime:
                destroy_runtime(destroy_process_group=False)
            raise
        else:
            from vforge.training.checkpoint.storage import cleanup_checkpoint_runtime

            stop_profiler(profiler)
            if offload_manager is not None:
                offload_manager.abort_step()
            if checkpoint_manager is not None:
                checkpoint_manager.close()
            session.cleanup()
            cleanup_checkpoint_runtime()
            if owns_runtime:
                destroy_runtime()
        finally:
            restore_storage_config(storage_token)

    def _engine_policy(self) -> EnginePolicy:
        return EnginePolicy(
            parallelize=self.config.parallelize,
            run_backward=self.config.run_backward,
            gradient_clip_val=self.config.gradient_clip_val,
            gradient_norm_dtype=self.config.gradient_norm_dtype,
            gradient_accumulation_steps=(self.config.gradient_accumulation_steps),
            parity=self.config.parity,
            diagnostics=self.config.diagnostics,
        )

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
    ):
        """Execute one business step.

        Specialized trainers override this hook to compose several engine
        phases while retaining the shared setup, resume, reporting, and
        checkpoint lifecycle.
        """

        return engine.run_update(
            model=model,
            microbatches=microbatches,
            step=step,
            context=context,
            compute_step=compute_step,
            optimizer=optimizer,
            scheduler=scheduler,
            run=run,
        )

    def _build_offload_manager(self) -> ActivationOffloadManager | None:
        """Create the manager injected into Native GC before model preparation."""
        policy = self.config.attention_offload
        if not policy.enable:
            return None
        return ActivationOffloadManager(
            policy.config,
            preserve_call_indices=policy.preserve_fa_call_indices,
        )

    def _configure_optimizers(
        self,
        session: TrainingSession,
        model: nn.Module,
        engine: TrainingEngine,
    ):
        configured = engine.configure_optimizers(
            model,
            session.optimizer_spec(),
            session.scheduler_spec(),
        )
        if configured is None:
            configured = session.configure_optimizers(model)
        if configured is None:
            return None, None
        if isinstance(configured, dict):
            optimizer = configured["optimizer"]
            scheduler_config = configured.get("lr_scheduler")
            scheduler = (
                scheduler_config.get("scheduler")
                if isinstance(scheduler_config, dict)
                else scheduler_config
            )
            return optimizer, scheduler
        if isinstance(configured, (tuple, list)):
            optimizer = configured[0]
            scheduler = configured[1] if len(configured) > 1 else None
            return optimizer, scheduler
        return configured, None

    def _resolve_start_step(self, session: TrainingSession) -> int:
        start_step = session.start_step()
        if start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {start_step}")
        return start_step

    @staticmethod
    def _learning_rate(optimizer: Any) -> float | None:
        if optimizer is None or not optimizer.param_groups:
            return None
        return float(optimizer.param_groups[0]["lr"])

    def _precision_context(self, device: torch.device):
        return autocast_context(
            device,
            enabled=self.config.autocast,
            precision=self.config.precision,
        )


__all__ = [
    "Trainer",
    "TrainerConfig",
]
