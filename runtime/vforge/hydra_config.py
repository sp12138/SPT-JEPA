"""Hydra structured schema for standalone vforge tasks.

Schema, parsing and serialization only: every block here describes what a YAML
file may contain, never what the runtime does with it. Turning a resolved config
into runtime objects belongs to the entry layer -- see
:mod:`vforge.tasks.translate`.

Nothing in vforge is imported from here on purpose. Borrowing a runtime type as
a field annotation used to drag ``torch`` into a module of field declarations,
and it made the schema move whenever the runtime object it borrowed moved. So
blocks whose runtime counterpart lives elsewhere get a plain schema twin -- the
``attention``/``resume``/``storage``/``diagnostics`` classes below mirror
``OpsConfig``, ``ResumeConfig``, ``StorageConfig`` and ``DiagnosticsConfig`` --
exactly as ``CheckpointHydraConfig`` already mirrors ``CheckpointConfig``.

The twins carry no validation. An invariant belongs to the type that owns it and
would only drift if restated here, so the runtime constructor in ``translate``
is what rejects a bad combination, one call later in the same command.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

# Mirrors vforge.training.engine.ENGINE_NAMES, which cannot be imported here
# without pulling the training stack into the schema. A test pins them equal.
ENGINE_NAMES = ("fsdp2", "megatron")

# Plain strings keep OmegaConf.structured compatible across omegaconf versions.
AttentionBackend = str
WeightLoading = str
InitDevice = str
ConditioningMode = str


@dataclass(frozen=True)
class ModelConfig:
    name: str = ""
    init_device: str = "meta"
    weight_loading: str = "auto"
    pretrained_path: str = ""
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.init_device not in {"meta", "cpu"}:
            raise ValueError(f"unsupported model.init_device={self.init_device!r}")
        if self.weight_loading not in {
            "auto",
            "distributed",
            "rank0_broadcast",
            "replicated",
        }:
            raise ValueError(f"unsupported model.weight_loading={self.weight_loading!r}")


@dataclass(frozen=True)
class TrainingModelHydraConfig:
    """One explicitly named model role in a multi-model training recipe."""

    model: str = ""
    config: dict = field(default_factory=dict)
    pretrained_path: str = ""
    init_weights: str = ""
    init_device: str = "meta"
    weight_loading: str = "auto"

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("training.models.<role>.model must be non-empty")
        if self.init_device not in {"meta", "cpu"}:
            raise ValueError(f"unsupported training model init_device={self.init_device!r}")
        if self.weight_loading not in {
            "auto",
            "distributed",
            "rank0_broadcast",
            "replicated",
        }:
            raise ValueError(f"unsupported training model weight_loading={self.weight_loading!r}")


@dataclass(frozen=True)
class TrainingHydraConfig:
    """Select a training paradigm and its explicitly named model roles."""

    recipe: str = "model"
    models: dict[str, TrainingModelHydraConfig] = field(default_factory=dict)
    dmd: dict = field(default_factory=dict)
    cd: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.recipe not in {"model", "dmd", "cd"}:
            raise ValueError(f"unsupported training.recipe={self.recipe!r}")
        if self.recipe in {"dmd", "cd"} and not self.models:
            raise ValueError(f"training.recipe={self.recipe} requires training.models")
        if self.recipe == "dmd" and not str(self.dmd.get("algorithm") or "").strip():
            raise ValueError("training.recipe=dmd requires training.dmd.algorithm")
        if self.recipe == "cd" and not str(self.cd.get("algorithm") or "").strip():
            raise ValueError("training.recipe=cd requires training.cd.algorithm")
        if any(not name for name in self.models):
            raise ValueError("training model role names must be non-empty")


@dataclass(frozen=True)
class LatentCacheConfig:
    """When the encoders run: ahead of training, or inside the step.

    ``offline`` reads a cache that prepare already wrote, so the training
    process never loads a VAE or a text encoder. ``online`` carries them and
    encodes every step, which is what prepare itself uses and what any
    pixel-space augmentation needs -- a cache is read back verbatim, so
    ``normal_rot_aug_*`` and friends are silently inert against one.

    The cache holds the diffusion target (``frame_latents``) alongside the
    conditioning tensors, which is why this is named for the cache rather than
    for conditioning.
    """

    mode: str = "offline"

    def __post_init__(self) -> None:
        if self.mode not in {"online", "offline"}:
            raise ValueError(f"unsupported latent_cache.mode={self.mode!r}")


@dataclass(frozen=True)
class DataSourceConfig:
    """One train/eval data source: an encoded cache, or raw video to decode.

    ``data.sources`` is mixed by ``weight``; ``evaluation.sources`` is concatenated.
    ``min_latents`` / ``max_latents`` are optional per-entry bounds the
    bidirectional CT recipe reads at the training step.

    ``kind`` picks between the two modes. Empty means an encoded cache, read
    from ``cache_dir``. Anything else names a raw dataset in
    :mod:`vforge.data.registry`, which decodes ``index_files`` at
    ``frames`` x ``height`` x ``width``.

    A raw source may also name a ``cache_dir``, and that combination means one
    thing only: prepare encodes this raw source into that directory. Training
    from it would be ambiguous -- cache or pixels? -- so the train builder
    rejects it rather than picking one.

    ``extras`` carries whatever that one dataset class understands and no other
    does -- ``use_partial_normal_aug`` for gameplay, say. Promoting those to
    named fields would put every dataset's knobs in every source's schema, and
    the combinations that type-check would still mostly be nonsense. The cost
    is that a typo inside ``extras`` is not caught here; the dataset class
    validates its own.
    """

    name: str = ""
    cache_dir: str = ""
    manifest: str = ""
    weight: float = 1.0
    min_latents: int | None = None
    max_latents: int | None = None
    kind: str = ""
    index_files: tuple[str, ...] = ()
    frames: int = 0
    height: int = 0
    width: int = 0
    max_sequence_length: int = 512
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError(f"source weight must be >= 0, got {self.weight}")
        raw_only = {
            "index_files": self.index_files,
            "frames": self.frames,
            "height": self.height,
            "width": self.width,
            "extras": self.extras,
        }
        if not self.kind:
            declared = sorted(field_name for field_name, value in raw_only.items() if value)
            if declared:
                raise ValueError(
                    f"source entry sets {', '.join(declared)} without a kind; "
                    "these belong to raw sources, while an encoded cache is read from cache_dir"
                )
            return
        if self.manifest:
            raise ValueError(
                f"source entry sets kind={self.kind!r} together with manifest; "
                "manifest lists the sample ids of an encoded cache, which a raw source has none of"
            )
        if not self.index_files:
            raise ValueError(f"source entry kind={self.kind!r} requires index_files")
        for field_name in ("frames", "height", "width"):
            if getattr(self, field_name) < 1:
                raise ValueError(
                    f"source entry kind={self.kind!r} requires {field_name} >= 1, "
                    f"got {getattr(self, field_name)}"
                )


@dataclass(frozen=True)
class DataConfig:
    """Where training samples come from, and how the loader walks them.

    ``sources`` is the list; ``data.cache_dir`` in YAML is a shorthand that
    fills it with one encoded cache. Both spellings land here, so nothing
    downstream has to know which one the config used.
    """

    train_manifest: str = ""
    val_manifest: str = ""
    sources: tuple[DataSourceConfig, ...] = ()
    mix_seed: int = 20260726
    steps_per_epoch: int = 1_000_000_000
    base_seed: int = 1234567
    sampler_seed: int = 0
    batch_size: int = 1
    num_workers: int = 0
    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None

    def __post_init__(self) -> None:
        if self.steps_per_epoch < 1:
            raise ValueError("data.steps_per_epoch must be >= 1")


def prepare_cache_dir(data: DataConfig) -> str:
    """Cache directory prepare writes; requires exactly one source with one."""
    if len(data.sources) != 1 or not data.sources[0].cache_dir:
        raise ValueError(
            "prepare writes one cache: set data.cache_dir to its path, "
            "or give data.sources a single entry that names a cache_dir"
        )
    return data.sources[0].cache_dir


@dataclass(frozen=True)
class OptimizerHydraConfig:
    """One optimizer definition.

    ``optimizer`` remains the single-optimizer compatibility surface. Complex
    trainers may instead consume entries from ``optimizers`` by name.
    """

    name: str = "adamw"
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.name != "adamw":
            raise ValueError(f"unsupported optimizer.name={self.name!r}")


@dataclass(frozen=True)
class SchedulerHydraConfig:
    name: str = "none"
    factor: float = 1.0
    warmup_steps: int = 0
    start_factor: float = 1.0
    end_factor: float = 0.1
    total_iters: int = 0

    def __post_init__(self) -> None:
        if self.name not in {"none", "constant", "constant_with_warmup", "linear"}:
            raise ValueError(f"unsupported scheduler.name={self.name!r}")
        if self.warmup_steps < 0:
            raise ValueError("scheduler.warmup_steps must be non-negative")
        if self.name == "constant_with_warmup" and self.warmup_steps <= 0:
            raise ValueError("scheduler.constant_with_warmup requires warmup_steps > 0")
        if self.name == "linear":
            if not 0 < self.start_factor <= 1:
                raise ValueError("scheduler.linear start_factor must be in (0, 1]")
            if not 0 <= self.end_factor <= 1:
                raise ValueError("scheduler.linear end_factor must be in [0, 1]")
            if self.total_iters <= 0:
                raise ValueError("scheduler.linear total_iters must be positive")


@dataclass(frozen=True)
class LoggingHydraConfig:
    jsonl: bool = True
    jsonl_filename: str = "train_steps.jsonl"
    wandb_enabled: bool = True
    wandb_project: str = "vforge"
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_name: str = ""
    wandb_tags: tuple[str, ...] = ()
    wandb_mode: str | None = None
    wandb_save_dir: str = ""
    wandb_include_optimizer_in_backward: bool = False
    wandb_exclude_dataloader_from_step: bool = False

    def __post_init__(self) -> None:
        if self.wandb_enabled and not self.wandb_project:
            raise ValueError("logging.wandb_project is required when WandB is enabled")


@dataclass(frozen=True)
class MCoreDDPHydraConfig:
    overlap_grad_reduce: bool = True
    overlap_param_gather: bool = True


@dataclass(frozen=True)
class MixedPrecisionHydraConfig:
    """Explicit FSDP2 mixed-precision policy and model overrides."""

    enabled: bool = True
    param_dtype: str | None = "bfloat16"
    reduce_dtype: str | None = "float32"
    output_dtype: str | None = None
    buffer_dtype: str | None = None
    cast_forward_inputs: bool = True
    preserve_module_paths: tuple[str, ...] = ()
    preserve_forward_input_paths: tuple[str, ...] = ()
    contract_mode: str = "append"

    def __post_init__(self) -> None:
        if self.contract_mode not in {"append", "replace"}:
            raise ValueError("parallel.mixed_precision.contract_mode must be 'append' or 'replace'")


@dataclass(frozen=True)
class ParallelHydraConfig:
    ep_size: int = 1
    usp_size: int = 1
    cp_size: int = 1
    tp_size: int = 1
    dp_size: Any = "auto"
    dp_replicate_size: int = 1
    dp_shard_size: int = 1
    dp_mode: str = "fsdp2"
    include_usp_in_fsdp: bool = True
    gradient_checkpointing: bool = False
    reshard_after_forward: bool = True
    fsdp_leaf_module_paths: tuple[str, ...] = ()
    mixed_precision: MixedPrecisionHydraConfig = field(default_factory=MixedPrecisionHydraConfig)
    engine: str = "fsdp2"
    mcore_ddp: MCoreDDPHydraConfig = field(default_factory=MCoreDDPHydraConfig)

    def __post_init__(self) -> None:
        if self.ep_size < 1:
            raise ValueError("parallel.ep_size must be >= 1")
        if self.ep_size > 1 and self.engine != "fsdp2":
            raise ValueError("EP requires parallel.engine=fsdp2")
        if self.engine not in ENGINE_NAMES:
            raise ValueError(
                f"unsupported parallel.engine={self.engine!r}; expected one of {ENGINE_NAMES}"
            )
        if self.usp_size < 1:
            raise ValueError(f"parallel.usp_size must be >= 1, got {self.usp_size}")
        if self.cp_size < 1:
            raise ValueError(f"parallel.cp_size must be >= 1, got {self.cp_size}")
        if self.tp_size < 1:
            raise ValueError(f"parallel.tp_size must be >= 1, got {self.tp_size}")
        if self.dp_replicate_size < 1:
            raise ValueError(
                f"parallel.dp_replicate_size must be >= 1, got {self.dp_replicate_size}"
            )
        if self.dp_shard_size < 1:
            raise ValueError(f"parallel.dp_shard_size must be >= 1, got {self.dp_shard_size}")
        if self.dp_mode not in {"ddp", "fsdp2"}:
            raise ValueError(f"unsupported parallel.dp_mode={self.dp_mode!r}")
        if self.dp_replicate_size > 1:
            if self.dp_shard_size == 1:
                raise ValueError(
                    "parallel.dp_replicate_size > 1 requires parallel.dp_shard_size > 1 for HSDP"
                )
            if self.engine != "fsdp2" or self.dp_mode != "fsdp2":
                raise ValueError("HSDP requires parallel.engine=fsdp2 and parallel.dp_mode=fsdp2")


@dataclass(frozen=True)
class RuntimeHydraConfig:
    max_steps: int = 0
    run_backward: bool = True
    seed: int = 42
    deterministic: bool = False
    precision: str = "bf16"
    autocast: bool = False
    gradient_clip_val: float = 0.0
    gradient_accumulation_steps: int = 1
    gradient_norm_dtype: str = "float32"
    log_every_n_steps: int = 1
    log_timing: bool = True
    output_dir: str = "outputs"
    weights_strict: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.gradient_accumulation_steps) is not int
            or self.gradient_accumulation_steps < 1
        ):
            raise ValueError("runtime.gradient_accumulation_steps must be a positive integer")
        if self.gradient_norm_dtype not in {"float32", "float64"}:
            raise ValueError("runtime.gradient_norm_dtype must be float32 or float64")


@dataclass(frozen=True)
class LimitationsConfig:
    """Declared first-release scope persisted in every resolved config."""

    expert_parallel: bool = False
    ring_attention: bool = False
    train_condition_encoders: bool = False
    cross_attention_ulysses: bool = False
    padded_cross_attention_requires_varlen: bool = True

    def __post_init__(self) -> None:
        expected = {
            "ring_attention": False,
            "train_condition_encoders": False,
            "cross_attention_ulysses": False,
            "padded_cross_attention_requires_varlen": True,
        }
        actual = {name: getattr(self, name) for name in expected}
        if actual != expected:
            raise ValueError(
                "standalone limitations are immutable in the first release; "
                f"expected {expected}, got {actual}"
            )


@dataclass(frozen=True)
class CheckpointHydraConfig:
    enabled: bool = True
    every_n_steps: int = 0
    dirpath: str = ""
    state: str = "full"
    format: str = "dcp"
    save_final: bool = True
    keep_last: int | None = 2
    async_save: bool = True
    save_optimizer: bool = True
    save_trainer_state: bool = True
    empty_cache_before_save: bool = True

    def __post_init__(self) -> None:
        if self.state not in {"weights", "full"}:
            raise ValueError(f"unsupported checkpoint.state={self.state!r}")
        if self.format not in {"pt", "dcp", "mcore"}:
            raise ValueError(f"unsupported checkpoint.format={self.format!r}")


@dataclass(frozen=True)
class AttentionHydraConfig:
    """Attention backend fields of :class:`vforge.ops.config.OpsConfig`."""

    backend: str = "auto"
    allow_fallback: bool = True
    debug: bool = False
    blas_backend: str = "default"


@dataclass(frozen=True)
class AttentionOffloadHydraConfig:
    enable: bool = False
    preserve_fa_call_indices: tuple[int, ...] | None = (0,)
    min_tensor_bytes: int = 1 << 20
    pin_memory: bool = True
    max_cpu_bytes: int | None = None
    async_transfer: bool = False


@dataclass(frozen=True)
class ResumeHydraConfig:
    """Hydra surface of :class:`vforge.training.checkpoint.config.ResumeConfig`."""

    path: str = "none"
    weights: bool = True
    optimizer: bool = True
    training_state: bool = True
    strict: bool | None = False
    stage_remote: bool = False


@dataclass(frozen=True)
class S3MountHydraConfig:
    mount_path: str = ""
    bucket: str = ""
    region: str = "us-west-2"
    key_prefix: str = ""
    endpoint_url: str | None = None
    addressing_style: str = "auto"


@dataclass(frozen=True)
class StorageHydraConfig:
    """Hydra surface of :class:`vforge.utils.s3_staging.StorageConfig`.

    Mount paths stay exactly as written; the runtime object is what expands and
    normalizes them, so the resolved snapshot keeps reproducing the input.
    """

    s3_mounts: tuple[S3MountHydraConfig, ...] = ()


@dataclass(frozen=True)
class DiagnosticsHydraConfig:
    """Hydra surface of :class:`vforge.training.diagnostics.DiagnosticsConfig`."""

    level: str = "basic"
    log_local_parameter_stats: bool = False


@dataclass(frozen=True)
class GridNormalEvaluationHydraConfig:
    """Optional multi-variant grid/normal evaluation matrix."""

    enabled: bool = False
    variants: list[str] | None = None

    def __post_init__(self) -> None:
        if self.variants is not None and not self.enabled:
            raise ValueError(
                "evaluation.gridnormal.variants requires evaluation.gridnormal.enabled=true"
            )
        if self.variants is not None and not self.variants:
            raise ValueError("evaluation.gridnormal.variants must not be empty")


@dataclass(frozen=True)
class OnlineReferenceEvaluationHydraConfig:
    """Optional deterministic overrides for target-video reference sampling."""

    num_frames: int | None = None
    frame_indices: list[int] | None = None

    def __post_init__(self) -> None:
        if self.num_frames is not None and self.num_frames < 0:
            raise ValueError("evaluation.online_ref.num_frames must be non-negative")
        if self.frame_indices is not None:
            if not self.frame_indices:
                raise ValueError("evaluation.online_ref.frame_indices must not be empty")
            if any(index < 0 for index in self.frame_indices):
                raise ValueError(
                    "evaluation.online_ref.frame_indices must contain non-negative integers"
                )
            if len(set(self.frame_indices)) != len(self.frame_indices):
                raise ValueError("evaluation.online_ref.frame_indices must be unique")
        if self.num_frames is not None and self.frame_indices is not None:
            raise ValueError(
                "evaluation.online_ref.num_frames and frame_indices are mutually exclusive"
            )


@dataclass(frozen=True)
class EvaluationHydraConfig:
    checkpoint_format: str = "auto"
    # OmegaConf rejects unions containing a container, so this remains ``Any``
    # at the structured-schema boundary and is validated immediately below.
    # Most models accept a string; composite models may interpret a mapping.
    checkpoint_path: Any = ""
    evaluate_all_checkpoints: bool = False
    data_source: str = ""
    manifest: str = ""
    sources: tuple[DataSourceConfig, ...] = ()
    path_replacements: tuple[tuple[str, str], ...] = ()
    output_dir: str = ""
    num_samples: int | None = None
    sample_index: int | None = None
    sample_selection: str = "first"
    seed: int = 42
    skip_existing: bool = True
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    negative_prompt: str | None = None
    rollout_mode: str = ""
    gridnormal: GridNormalEvaluationHydraConfig = field(
        default_factory=GridNormalEvaluationHydraConfig
    )
    online_ref: OnlineReferenceEvaluationHydraConfig = field(
        default_factory=OnlineReferenceEvaluationHydraConfig
    )

    def __post_init__(self) -> None:
        if self.checkpoint_format not in {"auto", "pt", "dcp", "mcore", "safetensors"}:
            raise ValueError(f"unsupported evaluation.checkpoint_format={self.checkpoint_format!r}")
        path = self.checkpoint_path
        if not isinstance(path, (str, Mapping)):
            raise TypeError("evaluation.checkpoint_path must be a string or a mapping")
        if isinstance(path, Mapping) and any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in path.items()
        ):
            raise TypeError(
                "evaluation.checkpoint_path mapping must contain string keys and values"
            )
        if self.sample_index is not None and self.sample_index < 0:
            raise ValueError("evaluation.sample_index must be non-negative")
        if self.sample_index is not None and self.num_samples not in {None, 1}:
            raise ValueError(
                "evaluation.sample_index selects one sample, so evaluation.num_samples "
                "must be unset or 1"
            )
        if self.sample_selection not in {"first", "evenly_spaced"}:
            raise ValueError(
                "evaluation.sample_selection must be first or evenly_spaced; "
                f"got {self.sample_selection!r}"
            )
        if self.num_inference_steps is not None and self.num_inference_steps < 1:
            raise ValueError("evaluation.num_inference_steps must be at least 1")
        if self.guidance_scale is not None and self.guidance_scale < 0:
            raise ValueError("evaluation.guidance_scale must be non-negative")


@dataclass(frozen=True)
class ConvertCheckpointHydraConfig:
    input_path: str = ""
    output_path: str = ""
    dry_run: bool = False
    component_path: str = ""
    weights: str = "raw"

    def __post_init__(self) -> None:
        if self.weights not in {"raw", "ema"}:
            raise ValueError("convert_checkpoint.weights must be raw or ema")
        if self.weights == "ema" and not self.component_path:
            raise ValueError("convert_checkpoint.weights=ema requires component_path")


@dataclass(frozen=True)
class StandaloneConfig:
    """Top-level structured config shared by train/prepare/eval/convert."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingHydraConfig = field(default_factory=TrainingHydraConfig)
    latent_cache: LatentCacheConfig = field(default_factory=LatentCacheConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerHydraConfig = field(default_factory=OptimizerHydraConfig)
    optimizers: dict[str, OptimizerHydraConfig] = field(default_factory=dict)
    scheduler: SchedulerHydraConfig = field(default_factory=SchedulerHydraConfig)
    logging: LoggingHydraConfig = field(default_factory=LoggingHydraConfig)
    runtime: RuntimeHydraConfig = field(default_factory=RuntimeHydraConfig)
    limitations: LimitationsConfig = field(default_factory=LimitationsConfig)
    parallel: ParallelHydraConfig = field(default_factory=ParallelHydraConfig)
    attention: AttentionHydraConfig = field(default_factory=AttentionHydraConfig)
    attention_offload: AttentionOffloadHydraConfig = field(
        default_factory=AttentionOffloadHydraConfig
    )
    checkpoint: CheckpointHydraConfig = field(default_factory=CheckpointHydraConfig)
    resume: ResumeHydraConfig = field(default_factory=ResumeHydraConfig)
    storage: StorageHydraConfig = field(default_factory=StorageHydraConfig)
    diagnostics: DiagnosticsHydraConfig = field(default_factory=DiagnosticsHydraConfig)
    evaluation: EvaluationHydraConfig = field(default_factory=EvaluationHydraConfig)
    convert_checkpoint: ConvertCheckpointHydraConfig = field(
        default_factory=ConvertCheckpointHydraConfig
    )


def to_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def config_from_dict(data: dict[str, Any] | DictConfig) -> StandaloneConfig:
    """Build a validated :class:`StandaloneConfig` from a mapping."""
    plain = OmegaConf.to_container(OmegaConf.create(data), resolve=True)
    if not isinstance(plain, dict):
        raise TypeError("Hydra config must resolve to a mapping")
    # Prefer plain defaults over OmegaConf.structured(frozen dataclass): older
    # omegaconf versions mark structured frozen nodes read-only and reject merges.
    defaults = to_plain(StandaloneConfig())
    merged = OmegaConf.merge(OmegaConf.create(defaults), OmegaConf.create(plain))
    container = OmegaConf.to_container(merged, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Hydra config must resolve to a mapping")
    return _standalone_from_plain(container)


_LEGACY_RESUME_KEYS = frozenset({"progress", "scheduler", "dataloader", "rng"})


def _resume_from_plain(data: dict[str, Any]) -> ResumeHydraConfig:
    if _LEGACY_RESUME_KEYS.intersection(data):
        raise ValueError(
            "resume progress/scheduler/dataloader/rng were merged into "
            "training_state; remove legacy keys and set training_state explicitly"
        )
    return ResumeHydraConfig(**data)


def _data_source_from_mapping(raw: Mapping[str, Any]) -> DataSourceConfig:
    payload = dict(raw)
    index_files = payload.get("index_files")
    if index_files is not None:
        if isinstance(index_files, (str, bytes)):
            index_files = [index_files]
        payload["index_files"] = tuple(str(path) for path in index_files)
    return DataSourceConfig(**payload)


def _coerce_sources(
    raw: Any,
    *,
    field_name: str = "data.sources",
) -> tuple[DataSourceConfig, ...]:
    """Accept one path, one mapping, or a sequence of either."""
    if raw is None or raw == "" or raw == () or raw == []:
        return ()
    if isinstance(raw, (str, bytes)):
        return (DataSourceConfig(cache_dir=str(raw)),)
    if isinstance(raw, Mapping):
        return (_data_source_from_mapping(raw),)
    if not isinstance(raw, Iterable):
        raise TypeError(f"{field_name} must be a path, a mapping, or a sequence of those")
    entries: list[DataSourceConfig] = []
    for item in raw:
        if isinstance(item, (str, bytes)):
            entries.append(DataSourceConfig(cache_dir=str(item)))
            continue
        if not isinstance(item, Mapping):
            raise TypeError(f"each {field_name} item must be a path or a mapping")
        entries.append(_data_source_from_mapping(item))
    return tuple(entries)


def _data_from_plain(data: dict[str, Any]) -> DataConfig:
    """Fold the ``cache_dir`` shorthand and ``sources`` into one source list.

    ``data.cache_dir`` stays a plain path so ``data.cache_dir=/path`` keeps
    working from the CLI, where a list is painful to type. Anything richer --
    several caches, or a raw dataset with its index and geometry -- is
    ``data.sources``, which is what the list has always meant.
    """
    payload = dict(data)
    cache_dir = payload.pop("cache_dir", "") or ""
    sources = payload.pop("sources", ()) or ()
    if not isinstance(cache_dir, (str, bytes)):
        raise ValueError(
            "data.cache_dir is the single-cache shorthand and takes a path string; "
            "write a list of entries under data.sources instead"
        )
    if cache_dir and sources:
        raise ValueError(
            "data.cache_dir and data.sources both name training data; "
            "if data.sources is the override, clear the inherited shorthand "
            'with data.cache_dir: ""'
        )
    return DataConfig(sources=_coerce_sources(sources or cache_dir), **payload)


def _storage_from_plain(data: dict[str, Any]) -> StorageHydraConfig:
    mounts_value: Any = data.get("s3_mounts", ())
    if isinstance(mounts_value, (str, bytes)) or not isinstance(mounts_value, Iterable):
        raise TypeError("storage.s3_mounts must be a sequence of mappings")
    mounts = []
    for item in mounts_value:
        if not isinstance(item, Mapping):
            raise TypeError("each storage.s3_mounts item must be a mapping")
        mounts.append(S3MountHydraConfig(**dict(item)))
    return StorageHydraConfig(s3_mounts=tuple(mounts))


def _limitations_from_plain(data: dict[str, Any]) -> LimitationsConfig:
    legacy_multi_node = data.pop("multi_node", None)
    if legacy_multi_node not in (None, False):
        raise ValueError(
            "limitations.multi_node is no longer a feature toggle; remove it and "
            "configure the distributed topology under parallel"
        )
    return LimitationsConfig(**data)


def _parallel_from_plain(data: dict[str, Any]) -> ParallelHydraConfig:
    mcore_ddp_data = dict(data.pop("mcore_ddp", None) or {})
    mixed_precision_data = dict(data.pop("mixed_precision", None) or {})
    for name in ("preserve_module_paths", "preserve_forward_input_paths"):
        if name in mixed_precision_data:
            mixed_precision_data[name] = tuple(mixed_precision_data[name])
    if "fsdp_leaf_module_paths" in data:
        data["fsdp_leaf_module_paths"] = tuple(data["fsdp_leaf_module_paths"])
    return ParallelHydraConfig(
        **data,
        mixed_precision=MixedPrecisionHydraConfig(**mixed_precision_data),
        mcore_ddp=MCoreDDPHydraConfig(**mcore_ddp_data),
    )


def _evaluation_from_plain(data: dict[str, Any]) -> EvaluationHydraConfig:
    gridnormal = GridNormalEvaluationHydraConfig(**dict(data.pop("gridnormal", None) or {}))
    online_ref = OnlineReferenceEvaluationHydraConfig(**dict(data.pop("online_ref", None) or {}))
    sources = _coerce_sources(
        data.pop("sources", ()) or (),
        field_name="evaluation.sources",
    )
    return EvaluationHydraConfig(
        **data,
        sources=sources,
        gridnormal=gridnormal,
        online_ref=online_ref,
    )


def _standalone_from_plain(data: dict[str, Any]) -> StandaloneConfig:
    """Construct nested dataclasses without OmegaConf.to_object(structured)."""
    optimizer_data = dict(data.get("optimizer") or {})
    if "betas" in optimizer_data:
        optimizer_data["betas"] = tuple(optimizer_data["betas"])
    named_optimizers: dict[str, OptimizerHydraConfig] = {}
    for name, raw in dict(data.get("optimizers") or {}).items():
        values = dict(raw or {})
        if "betas" in values:
            values["betas"] = tuple(values["betas"])
        named_optimizers[str(name)] = OptimizerHydraConfig(**values)
    parallel_data = dict(data.get("parallel") or {})
    attention_offload_data = dict(data.get("attention_offload") or {})
    if attention_offload_data.get("preserve_fa_call_indices") is not None:
        attention_offload_data["preserve_fa_call_indices"] = tuple(
            attention_offload_data["preserve_fa_call_indices"]
        )
    training_data = dict(data.get("training") or {})
    training_models = {
        str(role): TrainingModelHydraConfig(**dict(raw or {}))
        for role, raw in dict(training_data.pop("models", None) or {}).items()
    }
    return StandaloneConfig(
        model=ModelConfig(**dict(data.get("model") or {})),
        training=TrainingHydraConfig(
            **training_data,
            models=training_models,
        ),
        latent_cache=LatentCacheConfig(**dict(data.get("latent_cache") or {})),
        data=_data_from_plain(dict(data.get("data") or {})),
        optimizer=OptimizerHydraConfig(**optimizer_data),
        optimizers=named_optimizers,
        scheduler=SchedulerHydraConfig(**dict(data.get("scheduler") or {})),
        logging=LoggingHydraConfig(**dict(data.get("logging") or {})),
        runtime=RuntimeHydraConfig(**dict(data.get("runtime") or {})),
        limitations=_limitations_from_plain(dict(data.get("limitations") or {})),
        parallel=_parallel_from_plain(parallel_data),
        attention=AttentionHydraConfig(**dict(data.get("attention") or {})),
        attention_offload=AttentionOffloadHydraConfig(**attention_offload_data),
        checkpoint=CheckpointHydraConfig(**dict(data.get("checkpoint") or {})),
        resume=_resume_from_plain(dict(data.get("resume") or {})),
        storage=_storage_from_plain(dict(data.get("storage") or {})),
        diagnostics=DiagnosticsHydraConfig(**dict(data.get("diagnostics") or {})),
        evaluation=_evaluation_from_plain(dict(data.get("evaluation") or {})),
        convert_checkpoint=ConvertCheckpointHydraConfig(
            **dict(data.get("convert_checkpoint") or {})
        ),
    )


def resolved_config_hash(cfg: StandaloneConfig | dict[str, Any]) -> str:
    payload = to_plain(cfg)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_resolved_config(
    cfg: StandaloneConfig | DictConfig | dict[str, Any],
    output_dir: str | Path,
    *,
    filename: str = "resolved_config.yaml",
) -> Path:
    """Atomically write the fully resolved config for resume/debug parity."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / filename
    if isinstance(cfg, StandaloneConfig):
        container = to_plain(cfg)
    else:
        container = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    yaml_text = OmegaConf.to_yaml(OmegaConf.create(container))
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(yaml_text, encoding="utf-8")
    os.replace(tmp, target)
    return target


__all__ = [
    "ENGINE_NAMES",
    "AttentionHydraConfig",
    "AttentionOffloadHydraConfig",
    "CheckpointHydraConfig",
    "LatentCacheConfig",
    "ConvertCheckpointHydraConfig",
    "DataConfig",
    "DataSourceConfig",
    "DiagnosticsHydraConfig",
    "EvaluationHydraConfig",
    "GridNormalEvaluationHydraConfig",
    "LimitationsConfig",
    "LoggingHydraConfig",
    "ModelConfig",
    "MCoreDDPHydraConfig",
    "OnlineReferenceEvaluationHydraConfig",
    "OptimizerHydraConfig",
    "ParallelHydraConfig",
    "ResumeHydraConfig",
    "RuntimeHydraConfig",
    "S3MountHydraConfig",
    "SchedulerHydraConfig",
    "StandaloneConfig",
    "StorageHydraConfig",
    "TrainingHydraConfig",
    "TrainingModelHydraConfig",
    "config_from_dict",
    "prepare_cache_dir",
    "resolved_config_hash",
    "to_plain",
    "write_resolved_config",
]
