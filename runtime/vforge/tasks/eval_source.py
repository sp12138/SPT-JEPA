"""Resolve the two-field standalone evaluation input contract."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import asdict

from torch.utils.data import ConcatDataset, Dataset, Subset

from vforge.data.cache import EncodedBatchCodec, read_cache_manifest
from vforge.data.dataset import EncodedCacheDataset, InferenceRequestDataset, RawSampleDataset
from vforge.data.registry import open_raw_dataset
from vforge.tasks.requests import DataSourceSpec, EvalRequest, validate_eval_source

EncodedEvalSourceOpener = Callable[[DataSourceSpec, EncodedBatchCodec], Dataset]


class _IndexedEvalDataset(Dataset):
    """Attach the original global manifest index before any DP sharding."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        source_groups: tuple[tuple[int, ...], ...] = (),
    ) -> None:
        self.dataset = dataset
        self.eval_source_groups = source_groups

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]
        if not isinstance(sample, dict):
            raise TypeError(
                f"evaluation datasets must yield dictionaries, got {type(sample).__name__}"
            )
        seed_offset = sample.get("eval_seed_offset", index)
        if isinstance(seed_offset, bool):
            raise ValueError("eval_seed_offset must be a non-negative integer")
        try:
            seed_offset = int(seed_offset)
        except (TypeError, ValueError) as exc:
            raise ValueError("eval_seed_offset must be a non-negative integer") from exc
        if seed_offset < 0:
            raise ValueError("eval_seed_offset must be a non-negative integer")
        return {
            **sample,
            # Original position before sample selection and DP sharding. Artifacts
            # use it as stable provenance for the source manifest.
            "__vforge_eval_index__": int(index),
            # Seed offset normally matches the manifest index, but an exported
            # mixed dataset may preserve its original compound index explicitly.
            "__vforge_eval_seed_offset__": seed_offset,
        }


def _source_label(record: Mapping[str, object]) -> str:
    source = record.get("source")
    meta = record.get("meta")
    if (source is None or source == "") and isinstance(meta, Mapping):
        source = meta.get("source")
    return "" if source is None else str(source)


def _manifest_source_groups(dataset: Dataset) -> tuple[tuple[int, ...], ...]:
    """Group flat manifest indices by first-seen source for eval selection."""
    if isinstance(dataset, RawSampleDataset):
        records: list[Mapping[str, object]] = [
            dict(sample.extras or {}) for sample in dataset.samples
        ]
    elif isinstance(dataset, InferenceRequestDataset):
        records = dataset.records
    else:
        return ()

    labels = [_source_label(record) for record in records]
    if not any(labels):
        return ()
    grouped: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        grouped.setdefault(label, []).append(index)
    return tuple(tuple(indices) for indices in grouped.values())


def _open_raw_eval_sources(request: EvalRequest) -> tuple[Dataset, tuple[tuple[int, ...], ...]]:
    datasets = [
        open_raw_dataset(asdict(source), mode="val", index=index)
        for index, source in enumerate(request.sources)
    ]
    groups: list[tuple[int, ...]] = []
    offset = 0
    for dataset in datasets:
        groups.append(tuple(range(offset, offset + len(dataset))))
        offset += len(dataset)
    if len(datasets) == 1:
        return datasets[0], tuple(groups)
    return ConcatDataset(datasets), tuple(groups)


def open_eval_dataset(
    request: EvalRequest,
    *,
    codec: EncodedBatchCodec,
    open_encoded_source: EncodedEvalSourceOpener | None = None,
    manifest_dataset_factory: Callable[[str], Dataset] | None = None,
) -> Dataset:
    """Open the dataset selected by ``data_source`` and ``manifest``."""
    validate_eval_source(request)
    source_groups: tuple[tuple[int, ...], ...]
    if request.sources and request.data_source == "validation":
        dataset, source_groups = _open_raw_eval_sources(request)
    elif request.sources:
        if open_encoded_source is None:
            raise NotImplementedError(
                "this model does not support evaluation cache sources without manifest.json"
            )
        dataset = open_encoded_source(request.sources[0], codec)
        source_groups = ()
    elif manifest_dataset_factory is not None and request.data_source in {
        "validation",
        "inference",
    }:
        dataset = manifest_dataset_factory(request.manifest)
        if not isinstance(dataset, Dataset):
            raise TypeError("evaluation manifest dataset factory must return a torch Dataset")
        source_groups = _manifest_source_groups(dataset)
    elif request.data_source == "validation":
        dataset: Dataset = RawSampleDataset.from_manifest(request.manifest)
        source_groups = _manifest_source_groups(dataset)
    elif request.data_source == "inference":
        dataset = InferenceRequestDataset.from_manifest(request.manifest)
        source_groups = _manifest_source_groups(dataset)
    else:
        manifest_path = os.path.abspath(os.path.expanduser(request.manifest))
        if os.path.basename(manifest_path) != "manifest.json":
            raise ValueError(
                "evaluation.manifest must point directly to the cache root manifest.json "
                "when evaluation.data_source=cache"
            )
        cache_dir = os.path.dirname(manifest_path)
        manifest = read_cache_manifest(cache_dir, codec=codec)
        if manifest is None:
            raise ValueError("evaluation.manifest must point to a compatible cache manifest.json")
        if manifest.completed != len(manifest.sample_ids):
            raise ValueError(
                "evaluation cache manifest is incomplete: "
                f"completed={manifest.completed} samples={len(manifest.sample_ids)}"
            )
        dataset = EncodedCacheDataset(cache_dir, manifest.sample_ids, codec)
        source_groups = ()

    indexed = _IndexedEvalDataset(
        dataset,
        source_groups=source_groups,
    )
    if request.sample_index is None:
        return indexed
    if request.sample_index >= len(dataset):
        raise IndexError(
            f"evaluation.sample_index={request.sample_index} is outside dataset length "
            f"{len(dataset)}"
        )
    return Subset(indexed, [request.sample_index])


__all__ = ["EncodedEvalSourceOpener", "open_eval_dataset"]
