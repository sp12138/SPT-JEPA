"""Turn a resolved Hydra config into the runtime objects each consumer needs.

Every schema -> runtime mapping lives here: a model builder receives a request,
the trainer receives a ``TrainerConfig``, and neither reads the schema itself.
Keeping the mapping in the entry layer rather than in :mod:`vforge.hydra_config`
is what lets the schema module stay free of runtime imports, and what makes
"which Hydra field feeds which runtime knob" answerable in one file.

The request mapping is deliberately dumb -- field to field, no defaulting, no
derivation.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from typing import Any

from omegaconf import DictConfig

from vforge.checkpointing.offload import (
    ActivationOffloadConfig,
    AttentionOffloadPolicy,
)
from vforge.distributed.mcore_data_parallel import MCoreDataParallelConfig
from vforge.distributed.parallel_state import ParallelConfig
from vforge.distributed.parallelize import (
    FSDP2Config,
    MixedPrecisionConfig,
    ParallelizeConfig,
)
from vforge.hydra_config import (
    AttentionHydraConfig,
    DiagnosticsHydraConfig,
    ParallelHydraConfig,
    ResumeHydraConfig,
    StandaloneConfig,
    StorageHydraConfig,
    config_from_dict,
    resolved_config_hash,
    to_plain,
)
from vforge.ops.config import OpsConfig
from vforge.tasks.requests import (
    DataSourceSpec,
    EvalRequest,
    TrainingModelRequest,
    TrainRequest,
    validate_eval_source,
)
from vforge.training.checkpoint.config import (
    CheckpointConfig,
    ResumeConfig,
    build_resume_config,
)
from vforge.training.diagnostics import DiagnosticsConfig
from vforge.training.recipe import OptimizerSpec
from vforge.training.trainer import TrainerConfig
from vforge.utils.s3_staging import S3MountConfig, StorageConfig


@dataclass(frozen=True)
class BuiltRuntime:
    """Frozen runtime objects produced from a resolved Hydra config."""

    config: StandaloneConfig
    trainer: TrainerConfig
    ops: OpsConfig
    resume: ResumeConfig
    parallel: ParallelConfig
    resolved_config_hash: str


def resolve_dp_size(
    parallel: ParallelHydraConfig,
    *,
    world_size: int,
) -> int:
    """Derive concrete DP size from WORLD_SIZE and model-parallel dimensions.

    Ulysses, context, and tensor parallelism share a batch rather than split it,
    so they consume ranks that DP does not get to use. Note that Megatron's own
    "sequence parallelism" is a different mechanism that shares the tp group and
    costs no ranks at all; were we to adopt it, it would be a flag rather than a
    term in this product.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if parallel.usp_size < 1 or parallel.cp_size < 1 or parallel.tp_size < 1:
        raise ValueError(
            "parallel.usp_size, parallel.cp_size, and parallel.tp_size must be >= 1, got "
            f"{parallel.usp_size}, {parallel.cp_size}, and {parallel.tp_size}"
        )
    ranks_per_batch = parallel.usp_size * parallel.cp_size * parallel.tp_size
    if world_size % ranks_per_batch != 0:
        raise ValueError(
            "WORLD_SIZE must be divisible by parallel.usp_size * parallel.cp_size * "
            "parallel.tp_size "
            f"({world_size} % {ranks_per_batch} != 0)"
        )
    inferred = world_size // ranks_per_batch
    if parallel.dp_size == "auto":
        return inferred
    dp_size = int(parallel.dp_size)
    if dp_size != inferred:
        raise ValueError(
            "parallel.dp_size must equal WORLD_SIZE / (usp_size * cp_size * tp_size) "
            f"({dp_size} != {inferred})"
        )
    return dp_size


def build_parallel_config(
    parallel: ParallelHydraConfig,
    *,
    world_size: int | None = None,
) -> ParallelConfig:
    size = int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else world_size
    dp_size = resolve_dp_size(parallel, world_size=size)
    return ParallelConfig(
        dp_size=dp_size,
        dp_replicate_size=parallel.dp_replicate_size,
        dp_shard_size=parallel.dp_shard_size,
        tp_size=parallel.tp_size,
        cp_size=parallel.cp_size,
        usp_size=parallel.usp_size,
        dp_mode=parallel.dp_mode,
        include_usp_in_fsdp=parallel.include_usp_in_fsdp,
        extra_parallel_names=("ep",),
        extra_parallel_sizes=(parallel.ep_size,),
    ).normalized(size)


def build_ops_config(attention: AttentionHydraConfig, *, deterministic: bool) -> OpsConfig:
    """Combine attention selection and runtime determinism in the ops configuration."""
    return OpsConfig(**asdict(attention), deterministic=deterministic)


def build_resume_plan(resume: ResumeHydraConfig) -> ResumeConfig:
    return build_resume_config(asdict(resume))


def build_storage(storage: StorageHydraConfig) -> StorageConfig:
    """Expand mount paths and order them; both belong to the runtime object."""
    return StorageConfig(
        s3_mounts=tuple(S3MountConfig(**asdict(mount)) for mount in storage.s3_mounts),
    )


def build_diagnostics_config(diagnostics: DiagnosticsHydraConfig) -> DiagnosticsConfig:
    return DiagnosticsConfig(**asdict(diagnostics))


def build_parallelize_config(
    cfg: StandaloneConfig,
    *,
    world_size: int | None = None,
) -> ParallelizeConfig:
    """Resolve the parallelism plan.

    Split out of :func:`build_trainer_config` because evaluation needs only this
    much, and deriving it requires the world size, which a model cannot know.
    """
    parallel = build_parallel_config(cfg.parallel, world_size=world_size)
    return ParallelizeConfig(
        parallel=parallel,
        fsdp2=FSDP2Config(
            enable=parallel.dp_mode == "fsdp2",
            reshard_after_forward=cfg.parallel.reshard_after_forward,
            mixed_precision=MixedPrecisionConfig(
                enable=cfg.parallel.mixed_precision.enabled,
                param_dtype=cfg.parallel.mixed_precision.param_dtype,
                reduce_dtype=cfg.parallel.mixed_precision.reduce_dtype,
                output_dtype=cfg.parallel.mixed_precision.output_dtype,
                buffer_dtype=cfg.parallel.mixed_precision.buffer_dtype,
                cast_forward_inputs=(cfg.parallel.mixed_precision.cast_forward_inputs),
                preserve_module_paths=(cfg.parallel.mixed_precision.preserve_module_paths),
                preserve_forward_input_paths=(
                    cfg.parallel.mixed_precision.preserve_forward_input_paths
                ),
                contract_mode=cfg.parallel.mixed_precision.contract_mode,
            ),
            leaf_module_paths=cfg.parallel.fsdp_leaf_module_paths,
        ),
        mcore_ddp=MCoreDataParallelConfig(
            overlap_grad_reduce=cfg.parallel.mcore_ddp.overlap_grad_reduce,
            overlap_param_gather=cfg.parallel.mcore_ddp.overlap_param_gather,
        ),
        gradient_checkpointing=cfg.parallel.gradient_checkpointing,
    )


def build_trainer_config(
    cfg: StandaloneConfig,
    *,
    world_size: int | None = None,
) -> TrainerConfig:
    """Map Hydra blocks onto the existing TrainerConfig dataclass."""
    checkpoint_dir = cfg.checkpoint.dirpath or os.path.join(
        cfg.runtime.output_dir,
        "checkpoints",
    )
    attention_offload = AttentionOffloadPolicy(
        enable=cfg.attention_offload.enable,
        preserve_fa_call_indices=cfg.attention_offload.preserve_fa_call_indices,
        config=ActivationOffloadConfig(
            min_tensor_bytes=cfg.attention_offload.min_tensor_bytes,
            pin_memory=cfg.attention_offload.pin_memory,
            max_cpu_bytes=cfg.attention_offload.max_cpu_bytes,
            async_transfer=cfg.attention_offload.async_transfer,
        ),
    )
    return TrainerConfig(
        max_steps=cfg.runtime.max_steps,
        run_backward=cfg.runtime.run_backward,
        engine=cfg.parallel.engine,
        parallelize=build_parallelize_config(cfg, world_size=world_size),
        seed=cfg.runtime.seed,
        precision=cfg.runtime.precision,
        autocast=cfg.runtime.autocast,
        attention_offload=attention_offload,
        checkpoint=CheckpointConfig(
            enabled=cfg.checkpoint.enabled,
            every_n_steps=cfg.checkpoint.every_n_steps,
            dirpath=checkpoint_dir,
            state=cfg.checkpoint.state,
            format=cfg.checkpoint.format,
            save_final=cfg.checkpoint.save_final,
            keep_last=cfg.checkpoint.keep_last,
            async_save=cfg.checkpoint.async_save,
            save_optimizer=cfg.checkpoint.save_optimizer,
            save_trainer_state=cfg.checkpoint.save_trainer_state,
            empty_cache_before_save=cfg.checkpoint.empty_cache_before_save,
        ),
        resume=build_resume_plan(cfg.resume),
        weights_strict=cfg.runtime.weights_strict,
        storage=build_storage(cfg.storage),
        diagnostics=build_diagnostics_config(cfg.diagnostics),
        gradient_clip_val=cfg.runtime.gradient_clip_val,
        gradient_norm_dtype=cfg.runtime.gradient_norm_dtype,
        gradient_accumulation_steps=cfg.runtime.gradient_accumulation_steps,
        log_every_n_steps=cfg.runtime.log_every_n_steps,
        log_timing=cfg.runtime.log_timing,
        runtime_ops=build_ops_config(cfg.attention, deterministic=cfg.runtime.deterministic),
    )


def build_runtime(
    cfg: StandaloneConfig | DictConfig | dict[str, Any],
    *,
    world_size: int | None = None,
) -> BuiltRuntime:
    """Validate Hydra config and produce frozen runtime dataclasses."""
    if isinstance(cfg, StandaloneConfig):
        standalone = cfg
    else:
        standalone = config_from_dict(cfg)
    trainer = build_trainer_config(standalone, world_size=world_size)
    ops = build_ops_config(standalone.attention, deterministic=standalone.runtime.deterministic)
    resume = build_resume_plan(standalone.resume)
    parallel = trainer.parallelize.parallel
    config_hash = resolved_config_hash(standalone)
    semantic_model = to_plain(standalone.model)
    semantic_extras = dict(semantic_model.get("extras") or {})
    for operational_key in (
        "dual_encode_probe",
        "skip_existing",
    ):
        semantic_extras.pop(operational_key, None)
    semantic_model["extras"] = semantic_extras
    trainer = replace(
        trainer,
        checkpoint_fingerprints={
            "resolved_config_hash": config_hash,
            "ops": to_plain(ops),
            "data_config": to_plain(standalone.data),
            "semantic_config": {
                "model": semantic_model,
                "latent_cache": to_plain(standalone.latent_cache),
                "optimizer": to_plain(standalone.optimizer),
                "optimizers": to_plain(standalone.optimizers),
                "scheduler": to_plain(standalone.scheduler),
                "limitations": to_plain(standalone.limitations),
            },
        },
    )
    return BuiltRuntime(
        config=standalone,
        trainer=trainer,
        ops=ops,
        resume=resume,
        parallel=parallel,
        resolved_config_hash=config_hash,
    )


def _source_spec(
    source: Any,
    *,
    path_replacements: Any = (),
) -> DataSourceSpec:
    extras = dict(source.extras or {})
    if path_replacements:
        extras["path_replacements"] = [
            [str(source_root), str(target_root)] for source_root, target_root in path_replacements
        ]
    return DataSourceSpec(
        name=source.name,
        cache_dir=source.cache_dir,
        manifest=source.manifest,
        weight=source.weight,
        min_latents=source.min_latents,
        max_latents=source.max_latents,
        kind=source.kind,
        index_files=source.index_files,
        frames=source.frames,
        height=source.height,
        width=source.width,
        max_sequence_length=source.max_sequence_length,
        extras=extras,
    )


def train_request(cfg: StandaloneConfig) -> TrainRequest:
    return TrainRequest(
        extras=dict(cfg.model.extras or {}),
        pretrained_path=cfg.model.pretrained_path,
        init_device=cfg.model.init_device,
        weight_loading=cfg.model.weight_loading,
        training_recipe=cfg.training.recipe,
        training_models={
            role: TrainingModelRequest(
                model=model.model,
                config=dict(model.config or {}),
                pretrained_path=model.pretrained_path,
                init_weights=model.init_weights,
                init_device=model.init_device,
                weight_loading=model.weight_loading,
            )
            for role, model in cfg.training.models.items()
        },
        dmd=dict(cfg.training.dmd or {}),
        cd=dict(cfg.training.cd or {}),
        output_dir=cfg.runtime.output_dir,
        train_manifest=cfg.data.train_manifest,
        sources=tuple(_source_spec(source) for source in cfg.data.sources),
        mix_seed=cfg.data.mix_seed,
        steps_per_epoch=cfg.data.steps_per_epoch,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=cfg.data.shuffle,
        drop_last=cfg.data.drop_last,
        pin_memory=cfg.data.pin_memory,
        persistent_workers=cfg.data.persistent_workers,
        prefetch_factor=cfg.data.prefetch_factor,
        base_seed=cfg.data.base_seed,
        sampler_seed=cfg.data.sampler_seed,
        learning_rate=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        eps=cfg.optimizer.eps,
        scheduler_name=cfg.scheduler.name,
        scheduler_factor=cfg.scheduler.factor,
        warmup_steps=cfg.scheduler.warmup_steps,
        scheduler_start_factor=cfg.scheduler.start_factor,
        scheduler_end_factor=cfg.scheduler.end_factor,
        scheduler_total_iters=cfg.scheduler.total_iters,
        optimizers={
            name: OptimizerSpec(
                name=optimizer.name,
                learning_rate=optimizer.learning_rate,
                weight_decay=optimizer.weight_decay,
                betas=optimizer.betas,
                eps=optimizer.eps,
            )
            for name, optimizer in cfg.optimizers.items()
        },
        max_steps=cfg.runtime.max_steps,
        seed=cfg.runtime.seed,
        precision=cfg.runtime.precision,
        latent_cache_mode=cfg.latent_cache.mode,
        usp_size=cfg.parallel.usp_size,
        tp_size=cfg.parallel.tp_size,
        cp_size=cfg.parallel.cp_size,
    )


def eval_request(cfg: StandaloneConfig) -> EvalRequest:
    data_source = str(cfg.evaluation.data_source).strip()
    manifest = str(cfg.evaluation.manifest).strip()
    legacy_sampling = sorted(
        key
        for key in ("num_inference_steps", "guidance_scale", "negative_prompt")
        if key in cfg.model.extras
    )
    if legacy_sampling:
        replacements = ", ".join(
            f"model.extras.{key} -> evaluation.{key}" for key in legacy_sampling
        )
        raise ValueError(f"eval sampling fields moved to evaluation: {replacements}")
    legacy_execution = sorted(
        key
        for key in ("eval_workflow", "gridnormal_variants", "rollout_mode")
        if key in cfg.model.extras
    )
    if legacy_execution:
        replacements = {
            "eval_workflow": (
                "model.extras.variant (causal model shape) or evaluation.gridnormal.enabled"
            ),
            "gridnormal_variants": "evaluation.gridnormal.variants",
            "rollout_mode": "evaluation.rollout_mode",
        }
        detail = ", ".join(f"model.extras.{key} -> {replacements[key]}" for key in legacy_execution)
        raise ValueError(f"eval execution fields moved to evaluation: {detail}")
    if not data_source and not manifest:
        legacy = [
            name
            for name, value in (
                ("data.val_manifest", cfg.data.val_manifest),
                ("data.sources", cfg.data.sources),
            )
            if value
        ]
        if legacy:
            migration = {
                "data.val_manifest": (
                    "evaluation.data_source=validation and evaluation.manifest=<data.val_manifest>"
                ),
                "data.sources": (
                    "evaluation.data_source=cache and evaluation.manifest=<cache_dir>/manifest.json"
                ),
            }
            replacements = ", ".join(f"{name}: {migration[name]}" for name in legacy)
            raise ValueError(
                "data.val_manifest and the training data sources are not standalone eval "
                f"inputs; set the corresponding evaluation fields instead ({replacements})"
            )
    request = EvalRequest(
        extras=dict(cfg.model.extras or {}),
        pretrained_path=cfg.model.pretrained_path,
        init_device=cfg.model.init_device,
        data_source=data_source,
        manifest=manifest,
        sources=tuple(
            _source_spec(
                source,
                path_replacements=cfg.evaluation.path_replacements,
            )
            for source in cfg.evaluation.sources
        ),
        checkpoint_path=cfg.evaluation.checkpoint_path,
        checkpoint_format=cfg.evaluation.checkpoint_format,
        num_samples=cfg.evaluation.num_samples,
        sample_index=cfg.evaluation.sample_index,
        sample_selection=cfg.evaluation.sample_selection,
        num_inference_steps=cfg.evaluation.num_inference_steps,
        guidance_scale=cfg.evaluation.guidance_scale,
        negative_prompt=cfg.evaluation.negative_prompt,
        rollout_mode=str(cfg.evaluation.rollout_mode),
        gridnormal_enabled=bool(cfg.evaluation.gridnormal.enabled),
        gridnormal_variants=(
            None
            if cfg.evaluation.gridnormal.variants is None
            else tuple(str(name) for name in cfg.evaluation.gridnormal.variants)
        ),
        online_ref_num_frames=cfg.evaluation.online_ref.num_frames,
        online_ref_frame_indices=(
            None
            if cfg.evaluation.online_ref.frame_indices is None
            else tuple(int(index) for index in cfg.evaluation.online_ref.frame_indices)
        ),
        seed=cfg.evaluation.seed,
        precision=cfg.runtime.precision,
        latent_cache_mode=cfg.latent_cache.mode,
        usp_size=cfg.parallel.usp_size,
    )
    validate_eval_source(request)
    return request


# What the Hydra entrypoints consume. The per-block builders above are steps of
# build_runtime, not a menu: a caller assembling them by hand would be rebuilding
# build_runtime and would miss the checkpoint fingerprints it stamps on.
__all__ = [
    "BuiltRuntime",
    "build_runtime",
    "eval_request",
    "train_request",
]
