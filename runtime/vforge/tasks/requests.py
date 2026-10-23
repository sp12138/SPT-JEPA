"""What a model builder receives, per verb.

A request carries only the values a model actually reads, so a model builder
never sees the entry schema: the entry translates its own configuration into
one of these (see :mod:`vforge.tasks.translate`), the model turns it into a
components object, and the trainer or the evaluation runner consumes that. The
request itself is dead past the builder, which is why it lives here in the
entry layer instead of in the training or evaluation packages.

This module deliberately imports no configuration schema, so that importing a
model builder does not drag ``StandaloneConfig`` along with it.

Every field is defaulted so a test or a script can state just the two or three
values it cares about. Translators are expected to set all of them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from vforge.training.recipe import OptimizerSpec

EVAL_DATA_SOURCES = ("validation", "inference", "cache")


@dataclass(frozen=True)
class DataSourceSpec:
    """Twin of the Hydra ``DataSourceConfig`` that a train recipe reads.

    Kept here so a model builder never imports the entry schema.
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
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingModelRequest:
    """Resolved runtime inputs for one named model role."""

    model: str
    config: Mapping[str, Any] = field(default_factory=dict)
    pretrained_path: str = ""
    init_weights: str = ""
    init_device: str = "meta"
    weight_loading: str = "auto"


@dataclass(frozen=True)
class TrainRequest:
    """What a model needs in order to build a training recipe.

    Notably absent are the trainer's own knobs, which reach the loop through
    ``TrainerConfig`` rather than through a model.
    """

    # Where the weights and the model-specific knobs come from.
    extras: Mapping[str, Any] = field(default_factory=dict)
    pretrained_path: str = ""
    init_device: str = "meta"
    weight_loading: str = "auto"
    training_recipe: str = "model"
    training_models: Mapping[str, TrainingModelRequest] = field(default_factory=dict)
    dmd: Mapping[str, Any] = field(default_factory=dict)
    cd: Mapping[str, Any] = field(default_factory=dict)
    # Which samples to train on and how to batch them.
    train_manifest: str = ""
    sources: tuple[DataSourceSpec, ...] = ()
    mix_seed: int = 20260726
    steps_per_epoch: int = 1_000_000_000
    batch_size: int = 1
    num_workers: int = 0
    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int | None = None
    base_seed: int = 1234567
    sampler_seed: int = 0
    # Optimization.
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    scheduler_name: str = "none"
    scheduler_factor: float = 1.0
    warmup_steps: int = 0
    scheduler_start_factor: float = 1.0
    scheduler_end_factor: float = 0.1
    scheduler_total_iters: int = 0
    optimizers: Mapping[str, OptimizerSpec] = field(default_factory=dict)
    # Execution shape the model has to agree with.
    max_steps: int = 0
    output_dir: str = "outputs/train"
    seed: int = 42
    precision: str = "bf16"
    latent_cache_mode: str = "offline"
    usp_size: int = 1
    tp_size: int = 1
    cp_size: int = 1


@dataclass(frozen=True)
class EvalRequest:
    """What a model needs in order to build an evaluation."""

    extras: Mapping[str, Any] = field(default_factory=dict)
    pretrained_path: str = ""
    init_device: str = "meta"
    data_source: str = ""
    manifest: str = ""
    sources: tuple[DataSourceSpec, ...] = ()
    checkpoint_path: str | Mapping[str, str] = ""
    checkpoint_format: str = "dcp"
    num_samples: int | None = None
    sample_index: int | None = None
    sample_selection: str = "first"
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    negative_prompt: str | None = None
    rollout_mode: str = ""
    gridnormal_enabled: bool = False
    gridnormal_variants: tuple[str, ...] | None = None
    online_ref_num_frames: int | None = None
    online_ref_frame_indices: tuple[int, ...] | None = None
    seed: int = 42
    precision: str = "bf16"
    latent_cache_mode: str = "offline"
    usp_size: int = 1


def single_offline_source(request: TrainRequest, *, recipe: str) -> DataSourceSpec | None:
    """Zero or one encoded cache; recipes that cannot mix call this."""
    if request.sources and request.latent_cache_mode != "offline":
        raise ValueError(
            f"{recipe}: data.sources holds offline encoded caches; set latent_cache.mode=offline"
        )
    if len(request.sources) > 1:
        raise ValueError(f"{recipe} does not mix encoded caches; use a single data.cache_dir path")
    if request.latent_cache_mode != "offline":
        return None
    return request.sources[0] if request.sources else None


def validate_eval_source(request: EvalRequest) -> None:
    source = str(request.data_source).strip()
    manifest = str(request.manifest).strip()
    raw_sources = tuple(request.sources)
    if not source or (not manifest and not raw_sources):
        raise ValueError(
            "evaluation.data_source and exactly one input are required; set either "
            "evaluation.manifest or evaluation.sources; "
            f"data_source={source!r} manifest={manifest!r} sources={len(raw_sources)}"
        )
    if manifest and raw_sources:
        raise ValueError("evaluation.manifest and evaluation.sources are mutually exclusive")
    if source not in EVAL_DATA_SOURCES:
        raise ValueError(
            f"unsupported evaluation.data_source={source!r}; "
            f"expected one of {', '.join(EVAL_DATA_SOURCES)}"
        )
    if raw_sources and source == "validation" and any(not item.kind for item in raw_sources):
        raise ValueError(
            "evaluation.sources with data_source=validation accepts raw datasets only; "
            "set kind and index_files for every source"
        )
    if raw_sources and source == "cache":
        if len(raw_sources) != 1:
            raise ValueError(
                "evaluation.data_source=cache currently accepts exactly one encoded source"
            )
        item = raw_sources[0]
        if not item.cache_dir:
            raise ValueError("evaluation cache source requires cache_dir")
        if item.kind or item.index_files:
            raise ValueError("evaluation cache source is encoded; remove kind and index_files")
    if raw_sources and source == "inference":
        raise ValueError(
            "evaluation.sources is not valid with evaluation.data_source=inference; "
            "use evaluation.manifest"
        )
    if request.sample_index is not None and request.sample_index < 0:
        raise ValueError("evaluation.sample_index must be non-negative")
    if request.sample_index is not None and request.num_samples not in {None, 1}:
        raise ValueError(
            "evaluation.sample_index selects one sample, so evaluation.num_samples "
            "must be unset or 1"
        )
    if request.sample_selection not in {"first", "evenly_spaced"}:
        raise ValueError(
            "evaluation.sample_selection must be first or evenly_spaced; "
            f"got {request.sample_selection!r}"
        )
    if request.num_inference_steps is not None and request.num_inference_steps < 1:
        raise ValueError("evaluation.num_inference_steps must be at least 1")
    if request.guidance_scale is not None and request.guidance_scale < 0:
        raise ValueError("evaluation.guidance_scale must be non-negative")
    if request.gridnormal_variants is not None and not request.gridnormal_enabled:
        raise ValueError(
            "evaluation.gridnormal.variants requires evaluation.gridnormal.enabled=true"
        )
    if request.gridnormal_variants is not None and not request.gridnormal_variants:
        raise ValueError("evaluation.gridnormal.variants must not be empty")
    if request.online_ref_num_frames is not None and request.online_ref_num_frames < 0:
        raise ValueError("evaluation.online_ref.num_frames must be non-negative")
    if request.online_ref_frame_indices is not None:
        if not request.online_ref_frame_indices:
            raise ValueError("evaluation.online_ref.frame_indices must not be empty")
        if any(index < 0 for index in request.online_ref_frame_indices):
            raise ValueError(
                "evaluation.online_ref.frame_indices must contain non-negative integers"
            )
        if len(set(request.online_ref_frame_indices)) != len(request.online_ref_frame_indices):
            raise ValueError("evaluation.online_ref.frame_indices must be unique")
    if request.online_ref_num_frames is not None and request.online_ref_frame_indices is not None:
        raise ValueError(
            "evaluation.online_ref.num_frames and frame_indices are mutually exclusive"
        )


__all__ = [
    "EVAL_DATA_SOURCES",
    "DataSourceSpec",
    "EvalRequest",
    "TrainingModelRequest",
    "TrainRequest",
    "single_offline_source",
    "validate_eval_source",
]
