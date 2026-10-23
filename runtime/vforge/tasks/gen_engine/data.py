"""Assemble resolved gen-engine data inputs into the shared DataSpec contract.

The task selects MMGenDataset and its model-owned sample validation. The core
MultiDataset only routes addresses, and its sampler owns both shuffle layers.
This module consumes TrainRequest; YAML parsing stays in the entry layer.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import fields
from fractions import Fraction
from functools import partial
from typing import Any

from vforge.data.multi_source import (
    ALGORITHM,
    MAX_BLOCK_SIZE,
    DatasetSource,
    MultiDataset,
    build_multi_source_sampler,
)
from vforge.data.spec import DataSpec
from vforge.tasks.gen_engine.dataset import MMGenDataset, dataset_input_options
from vforge.tasks.gen_engine.shared import target_modalities
from vforge.tasks.requests import DataSourceSpec, TrainRequest


def _integer_weights(sources: Sequence[DataSourceSpec]) -> tuple[int, ...]:
    """Preserve decimal configuration ratios exactly, without rounding quotas."""
    fractions: list[Fraction] = []
    for source in sources:
        try:
            value = Fraction(str(source.weight))
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(
                f"source {source.name!r}: weight must be finite and nonnegative"
            ) from error
        if value < 0:
            raise ValueError(f"source {source.name!r}: weight must be finite and nonnegative")
        fractions.append(value)
    denominator = math.lcm(*(value.denominator for value in fractions))
    weights = tuple(value.numerator * (denominator // value.denominator) for value in fractions)
    divisor = math.gcd(*weights)
    if divisor == 0:
        raise ValueError("at least one source weight must be positive")
    normalized = tuple(weight // divisor for weight in weights)
    if sum(normalized) > MAX_BLOCK_SIZE:
        raise ValueError(
            f"normalized quota block size {sum(normalized)} exceeds "
            f"MAX_BLOCK_SIZE={MAX_BLOCK_SIZE}; configure a smaller exact integer ratio"
        )
    return normalized


def _validate_source(source: DataSourceSpec) -> None:
    if not isinstance(source.name, str) or not source.name.strip():
        raise ValueError("each gen_engine source requires a nonempty name")
    if not isinstance(source.manifest, str) or not source.manifest.strip():
        raise ValueError(f"source {source.name!r}: gen_engine requires a JSONL manifest")
    defaults = DataSourceSpec()
    unsupported = [
        field.name
        for field in fields(DataSourceSpec)
        if field.name not in {"name", "manifest", "weight"}
        and getattr(source, field.name) != getattr(defaults, field.name)
    ]
    if unsupported:
        raise ValueError(
            f"source {source.name!r}: gen_engine JSONL sources do not support "
            + ", ".join(unsupported)
        )


def build_data_spec(
    request: TrainRequest,
    *,
    collate_fn: Callable[[list[Any]], Any] | None,
    max_reference_index: int | None = None,
    total_samples: int | None = None,
) -> tuple[DataSpec, dict[str, Any]]:
    """Build live data components and the data portion of the resume fingerprint.

    A single train_manifest retains its previous DistributedSampler semantics
    and fingerprint fields. Explicit sources use a finite global sample budget
    supplied by the task (max_steps * local batch size * DP size). The caller
    adds model packing and loader settings to this fingerprint.
    """
    if request.train_manifest and request.sources:
        raise ValueError("data.train_manifest and data.sources are mutually exclusive")
    if not request.train_manifest and not request.sources:
        raise ValueError("gen_engine training requires data.train_manifest or data.sources")

    modalities = target_modalities(request.extras)
    require_digests = request.extras.get("require_digests", True)
    if not isinstance(require_digests, bool):
        raise ValueError("model.extras.require_digests must be true or false")
    common = {
        "target_modalities": list(modalities or ()),
        "require_digests": require_digests,
    }
    input_options = dataset_input_options(request.extras)

    def manifest_dataset(path: str) -> MMGenDataset:
        return MMGenDataset(
            path,
            target_modalities=modalities,
            require_digests=require_digests,
            sample_kind="target",
            max_reference_index=max_reference_index,
            **input_options,
        )

    if request.train_manifest:
        dataset = manifest_dataset(request.train_manifest)
        input_contract = dataset.input_contract()
        if input_contract is not None:
            common["input"] = input_contract
        return DataSpec(dataset, collate_fn=collate_fn), {
            "source_kind": "mmgen_precomputed",
            "manifest": str(dataset.manifest_path),
            "manifest_digest": dataset.content_digest(exact=True),
            "sample_count": len(dataset),
            **common,
        }

    if type(total_samples) is not int or total_samples < 1:
        raise ValueError("gen_engine multi-source training requires a positive total_samples")
    if not request.shuffle:
        raise ValueError("gen_engine multi-source training requires data.shuffle=true")
    for source in request.sources:
        _validate_source(source)
    weights = _integer_weights(request.sources)
    children = [manifest_dataset(source.manifest) for source in request.sources]
    multi_dataset = MultiDataset(
        [
            DatasetSource(source.name, child, child.data_version(), weight)
            for source, child, weight in zip(request.sources, children, weights, strict=True)
        ]
    )
    sampler_factory = partial(
        build_multi_source_sampler,
        total_samples=total_samples,
        mix_seed=request.mix_seed,
    )
    return DataSpec(multi_dataset, sampler_factory, collate_fn), {
        "source_kind": "mmgen_precomputed_multi_source",
        "dataset": multi_dataset.resume_contract(),
        "manifests": [str(child.manifest_path) for child in children],
        "sample_count": len(multi_dataset),
        "sampling": {
            "algorithm": ALGORITHM,
            "total_samples": total_samples,
            "mix_seed": request.mix_seed,
            "shuffle_seed": request.sampler_seed,
            "block_multiplier": 1,
            "stage_id": "stage-0",
        },
        **common,
    }


__all__ = ["build_data_spec"]
