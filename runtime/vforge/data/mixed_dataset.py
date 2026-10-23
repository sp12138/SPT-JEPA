"""Weighted mixture of datasets: assembly, sampling, and the loader that runs it.

Everything a mixed run needs lives here. :func:`open_mixed_cache_dataset` and
:func:`open_mixed_raw_dataset` build the same :class:`MixedDataset` shape and
the same fingerprint contract, one from encoded caches and one from raw
sources, so the loader and the resume check do not care which one opened the
data. :func:`build_mixed_resumable_loader` then installs :class:`MixedSampler`,
which the generic loader builder cannot do because it always reaches for a
``DistributedSampler``.

At every data batch every rank samples from the same sub-dataset, chosen by
a pure function of ``(mix_seed, batch_index)`` so no cross-rank broadcast is needed.
This position advances with data consumption, independently of optimizer updates.
Different DP ranks then take disjoint slices of that sub's permutation, which
keeps:

* the SP-group same-batch invariant (SP peers share ``dp_rank``);
* the DP-group differ-batch invariant;
* batch-level homogeneity (one sub per batch), which default collate and
  per-sub latent bounds both require.

Within each sub, Random Reshuffling (without-replacement shuffled epochs)
consumes a permutation of ``[0, N_i)`` and reshuffles when the cursor wraps.

The openers take already-translated source mappings; nothing here loads YAML or
reads a :class:`TrainRequest`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from vforge.data.cache import EncodedBatchCodec, digest_sample_ids, read_cache_manifest
from vforge.data.dataset import EncodedCacheDataset, load_manifest_sample_ids
from vforge.data.loader import LeaderBroadcastDataLoader
from vforge.data.registry import open_raw_dataset
from vforge.data.taichu_index_cache import (
    indexed_path_resolver,
    scan_indexed_caches_on_leader,
)

# Compound index encoding: ``sub_id * _SUB_OFFSET + local_index``.
# 1e10 leaves head-room for O(billions) of samples per sub-dataset while
# fitting in a native int64.
_SUB_OFFSET = 10_000_000_000
_DEFAULT_MIX_SEED = 20260726
_DEFAULT_STEPS_PER_EPOCH = 1_000_000_000
_U64_MASK = (1 << 64) - 1


def _u64(*parts: int) -> np.uint64:
    """XOR ``parts`` as wrapping uint64 values, matching numpy's overflow."""
    acc = 0
    for part in parts:
        acc ^= int(part) & _U64_MASK
    return np.uint64(acc)


class MixedDataset(Dataset):
    """Route ``__getitem__`` to one of N sub-datasets by compound index."""

    def __init__(
        self,
        subs: Sequence[Dataset],
        weights: Sequence[float],
        *,
        mix_seed: int = _DEFAULT_MIX_SEED,
        steps_per_epoch: int = _DEFAULT_STEPS_PER_EPOCH,
        names: Sequence[str] | None = None,
        cache_dirs: Sequence[str | None] | None = None,
    ) -> None:
        subs = list(subs)
        weights = [float(weight) for weight in weights]
        if not subs:
            raise ValueError("MixedDataset requires at least one sub-dataset")
        if len(weights) != len(subs):
            raise ValueError(
                f"MixedDataset got {len(subs)} sub-datasets and {len(weights)} weights"
            )
        if any(weight < 0 for weight in weights):
            raise ValueError("MixedDataset weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("MixedDataset weights must sum to a positive value")
        for index, sub in enumerate(subs):
            length = len(sub)
            if length <= 0:
                raise ValueError(f"MixedDataset sub-dataset {index} is empty")
            if length >= _SUB_OFFSET:
                raise RuntimeError(
                    f"MixedDataset sub-dataset {index} has {length} samples, "
                    f"exceeding the compound-index encoding capacity ({_SUB_OFFSET})"
                )
        self._sub_datasets = subs
        self._sub_weights = weights
        self._mix_seed = int(mix_seed)
        self._steps_per_epoch = int(steps_per_epoch)
        if self._steps_per_epoch < 1:
            raise ValueError("MixedDataset steps_per_epoch must be >= 1")
        self._sub_names = (
            [str(name) for name in names]
            if names is not None
            else [f"sub_{index}" for index in range(len(subs))]
        )
        if len(self._sub_names) != len(subs):
            raise ValueError("MixedDataset names must match the number of sub-datasets")
        self._sub_cache_dirs: list[str | None] = (
            list(cache_dirs) if cache_dirs is not None else [None] * len(subs)
        )
        if len(self._sub_cache_dirs) != len(subs):
            raise ValueError("MixedDataset cache_dirs must match the number of sub-datasets")
        weight_array = np.asarray(self._sub_weights, dtype=np.float64)
        self._sub_probs = weight_array / weight_array.sum()
        self._num_subs = len(subs)

    def pick_sub_id(self, step_id: int) -> int:
        """Return the sub-dataset index for global step ``step_id``.

        Deterministic across ranks: only depends on ``mix_seed`` and
        ``step_id``. Used by :class:`MixedSampler` to align every rank onto the
        same sub-dataset per step.
        """
        rng = np.random.default_rng(_u64(self._mix_seed, int(step_id) * 1315423911))
        return int(rng.choice(self._num_subs, p=self._sub_probs))

    @staticmethod
    def encode_index(sub_id: int, local_idx: int) -> int:
        return int(sub_id) * _SUB_OFFSET + int(local_idx)

    @staticmethod
    def decode_index(compound: int) -> tuple[int, int]:
        return int(compound // _SUB_OFFSET), int(compound % _SUB_OFFSET)

    def __len__(self) -> int:
        return int(sum(len(sub) for sub in self._sub_datasets))

    def __getitem__(self, index: int) -> Any:
        sub_id, local_idx = self.decode_index(index)
        if sub_id < 0 or sub_id >= self._num_subs:
            raise IndexError(
                f"MixedDataset: decoded sub_id={sub_id} out of range "
                f"[0, {self._num_subs}). compound_index={index}."
            )
        sub = self._sub_datasets[sub_id]
        return sub[local_idx % len(sub)]

    @property
    def num_sub_datasets(self) -> int:
        return self._num_subs

    @property
    def sub_lengths(self) -> list[int]:
        return [len(sub) for sub in self._sub_datasets]

    @property
    def sub_names(self) -> list[str]:
        return list(self._sub_names)

    @property
    def sub_probs(self) -> list[float]:
        return [float(prob) for prob in self._sub_probs]

    @property
    def steps_per_epoch(self) -> int:
        return self._steps_per_epoch

    @property
    def mix_seed(self) -> int:
        return self._mix_seed

    def resume_contract(self) -> dict[str, Any]:
        """Stable metadata describing this mixed dataset composition.

        Sampler and loader settings are excluded; the sampler combines this
        with the knobs that define sample order.
        """
        return {
            "schema_version": 1,
            "mix_seed": self.mix_seed,
            "steps_per_epoch": self.steps_per_epoch,
            "sub_names": self.sub_names,
            "sub_lengths": self.sub_lengths,
            "sub_weights": [float(weight) for weight in self._sub_weights],
            "sub_probs": [float(prob) for prob in self.sub_probs],
            "sub_types": [
                f"{type(sub).__module__}.{type(sub).__qualname__}" for sub in self._sub_datasets
            ],
            "sub_cache_dirs": [
                os.path.realpath(str(path)) if path else None for path in self._sub_cache_dirs
            ],
        }


class MixedSampler(Sampler[int]):
    """Deterministic per-batch sampler for :class:`MixedDataset` (per-sub RR).

    For step ``t`` (``0 <= t < steps_per_epoch``), let
    ``global_step = t + self._epoch * steps_per_epoch``. Here ``step`` counts
    data batches, not optimizer updates. Then:

    * Pick ``sub_id = mixed_dataset.pick_sub_id(global_step)`` (same on every
      rank; a pure function of ``mix_seed`` and ``global_step``).
    * Consume the next ``batch_size * dp_world_size`` indices from the current
      permutation of that sub. DP rank ``r`` takes
      ``[r * B : (r + 1) * B]``.
    * When a sub's permutation cursor wraps, bump that sub's local
      ``sub_epoch`` and regenerate its permutation from
      ``(mix_seed, seed, sub_id, sub_epoch)``.
    * Yield the compound indices.

    The DataLoader is configured with ``batch_size=B``; this sampler yields
    ``B * steps_per_epoch`` indices, which the DataLoader chunks into
    homogeneous batches of size ``B``.
    """

    def __init__(
        self,
        mixed_dataset: MixedDataset,
        *,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        if not isinstance(mixed_dataset, MixedDataset):
            raise TypeError(
                f"MixedSampler requires a MixedDataset, got {type(mixed_dataset).__name__}"
            )
        if dp_world_size < 1:
            raise ValueError(f"MixedSampler dp_world_size must be >= 1, got {dp_world_size}")
        if not 0 <= dp_rank < dp_world_size:
            raise ValueError(f"MixedSampler dp_rank must be in [0, {dp_world_size}), got {dp_rank}")
        if batch_size < 1:
            raise ValueError(f"MixedSampler batch_size must be >= 1, got {batch_size}")
        self.mixed_dataset = mixed_dataset
        self.dp_rank = int(dp_rank)
        self.dp_world_size = int(dp_world_size)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def resume_contract(self) -> dict[str, Any]:
        """Describe the deterministic sample stream emitted by this sampler."""
        return {
            "dataset": self.mixed_dataset.resume_contract(),
            "seed": self.seed,
            "shuffle": self.shuffle,
            "batch_size": self.batch_size,
            "dp_world_size": self.dp_world_size,
        }

    def __len__(self) -> int:
        return self.mixed_dataset.steps_per_epoch * self.batch_size

    def _permutation_for(self, sub_id: int, sub_epoch: int) -> np.ndarray:
        """Return the shuffle permutation for ``(sub_id, sub_epoch)``.

        Fully determined by ``(mix_seed, seed, sub_id, sub_epoch)``; every rank
        computes the same array without any communication.
        """
        sub_len = len(self.mixed_dataset._sub_datasets[sub_id])
        if not self.shuffle:
            return np.arange(sub_len, dtype=np.int64)
        rng = np.random.default_rng(
            _u64(
                self.mixed_dataset.mix_seed,
                int(self.seed) * 6364136223846793005,
                int(sub_id) * 1_000_003,
                int(sub_epoch) * 2654435761,
            )
        )
        return rng.permutation(sub_len).astype(np.int64, copy=False)

    def __iter__(self) -> Iterator[int]:
        md = self.mixed_dataset
        steps = md.steps_per_epoch
        epoch_offset = self._epoch * steps
        num_subs = md.num_sub_datasets
        per_step = self.batch_size * self.dp_world_size

        sub_epochs = [0] * num_subs
        cursors = [0] * num_subs
        perms: list[np.ndarray] = [self._permutation_for(index, 0) for index in range(num_subs)]

        my_slice_start = self.dp_rank * self.batch_size
        my_slice_end = my_slice_start + self.batch_size

        for step in range(steps):
            global_step = step + epoch_offset
            sub_id = md.pick_sub_id(global_step)
            sub_len = len(md._sub_datasets[sub_id])

            gathered = np.empty(per_step, dtype=np.int64)
            filled = 0
            while filled < per_step:
                perm = perms[sub_id]
                cursor = cursors[sub_id]
                take = min(per_step - filled, sub_len - cursor)
                gathered[filled : filled + take] = perm[cursor : cursor + take]
                filled += take
                cursor += take
                if cursor >= sub_len:
                    sub_epochs[sub_id] += 1
                    perms[sub_id] = self._permutation_for(sub_id, sub_epochs[sub_id])
                    cursor = 0
                cursors[sub_id] = cursor

            local_ids = gathered[my_slice_start:my_slice_end]
            for local_idx in local_ids.tolist():
                yield MixedDataset.encode_index(sub_id, int(local_idx))


class _BoundsDataset(Dataset):
    """Copy a sample and stamp homogeneous latent-length bounds onto it."""

    def __init__(
        self,
        inner: Dataset,
        *,
        min_latents: int | None,
        max_latents: int | None,
    ) -> None:
        self._inner = inner
        self._min_latents = min_latents
        self._max_latents = max_latents

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self._inner[index])
        if self._min_latents is not None:
            item["min_latents"] = int(self._min_latents)
        if self._max_latents is not None:
            item["max_latents"] = int(self._max_latents)
        return item

    @property
    def sample_ids(self) -> list[str]:
        return list(self._inner.sample_ids)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def wrap_latent_bounds(
    dataset: Dataset,
    *,
    min_latents: int | None,
    max_latents: int | None,
) -> Dataset:
    """Stamp per-source bounds on samples so :class:`MixedDataset` never sees them.

    The bounds belong to one source, not to the mixture, and a batch is
    homogeneous by construction -- so carrying them on the sample is what lets
    the training step read them without asking which sub it drew from.
    """
    if min_latents is None and max_latents is None:
        return dataset
    return _BoundsDataset(dataset, min_latents=min_latents, max_latents=max_latents)


def _cache_source_mapping(source: Mapping[str, Any], *, index: int) -> dict[str, Any]:
    cache_dir = str(source.get("cache_dir") or "")
    if not cache_dir:
        raise ValueError(f"data.sources[{index}] requires a cache_dir path")
    name = str(source.get("name") or f"sub_{index}")
    return {
        "name": name,
        "cache_dir": cache_dir,
        "manifest": str(source.get("manifest") or ""),
        "weight": float(source.get("weight", 1.0)),
        "min_latents": _optional_int(source.get("min_latents")),
        "max_latents": _optional_int(source.get("max_latents")),
    }


def open_mixed_cache_dataset(
    sources: Sequence[Mapping[str, Any]],
    *,
    codec: EncodedBatchCodec,
    mix_seed: int = _DEFAULT_MIX_SEED,
    steps_per_epoch: int = _DEFAULT_STEPS_PER_EPOCH,
) -> tuple[MixedDataset, dict[str, Any]]:
    """Open N encoded caches as one :class:`MixedDataset` plus a fingerprint.

    Sources that already have a Native ``manifest.json`` or an explicit
    ``manifest`` path skip the indexed scan. The remaining indexed caches are
    scanned on the leader and shared with one ``broadcast_object_list``.
    """
    specs = [_cache_source_mapping(source, index=index) for index, source in enumerate(sources)]
    if not specs:
        raise ValueError("open_mixed_cache_dataset requires at least one source")

    resolved: list[dict[str, Any] | None] = [None] * len(specs)
    scan_indices: list[int] = []
    scan_dirs: list[str] = []

    for index, spec in enumerate(specs):
        cache_dir = spec["cache_dir"]
        train_manifest = spec["manifest"]
        native_manifest = read_cache_manifest(cache_dir, codec=codec)
        if native_manifest is not None:
            resolved[index] = {
                "sample_ids": list(native_manifest.sample_ids),
                "path_for": None,
            }
            continue
        indexed_dir = os.path.join(cache_dir, "train")
        if not os.path.isdir(indexed_dir) and not train_manifest:
            raise ValueError(
                f"cache has neither manifest.json nor indexed train/ directory: {cache_dir}"
            )
        if train_manifest:
            resolved[index] = {
                "sample_ids": load_manifest_sample_ids(train_manifest),
                "path_for": indexed_path_resolver(),
            }
            continue
        scan_indices.append(index)
        scan_dirs.append(cache_dir)

    if scan_dirs:
        scanned = scan_indexed_caches_on_leader(scan_dirs)
        for index, sample_ids in zip(scan_indices, scanned, strict=True):
            resolved[index] = {
                "sample_ids": sample_ids,
                "path_for": indexed_path_resolver(),
            }

    datasets: list[Dataset] = []
    weights: list[float] = []
    names: list[str] = []
    cache_dirs: list[str | None] = []
    fingerprint_sources: list[dict[str, Any]] = []
    for spec, opened in zip(specs, resolved, strict=True):
        assert opened is not None
        sample_ids = opened["sample_ids"]
        inner = EncodedCacheDataset(
            spec["cache_dir"],
            sample_ids,
            codec,
            path_for=opened["path_for"],
        )
        datasets.append(
            wrap_latent_bounds(
                inner,
                min_latents=spec["min_latents"],
                max_latents=spec["max_latents"],
            )
        )
        weights.append(spec["weight"])
        names.append(spec["name"])
        cache_dirs.append(spec["cache_dir"])
        fingerprint_sources.append(
            {
                "name": spec["name"],
                "cache_dir": spec["cache_dir"],
                "manifest": spec["manifest"],
                "weight": spec["weight"],
                "min_latents": spec["min_latents"],
                "max_latents": spec["max_latents"],
                "sample_count": len(sample_ids),
                "sample_ids_digest": digest_sample_ids(sample_ids),
            }
        )

    mixed = MixedDataset(
        datasets,
        weights,
        mix_seed=mix_seed,
        steps_per_epoch=steps_per_epoch,
        names=names,
        cache_dirs=cache_dirs,
    )
    fingerprint = {
        "source_kind": "mixed_encoded_cache",
        "mix_seed": int(mix_seed),
        "steps_per_epoch": int(steps_per_epoch),
        "sources": fingerprint_sources,
    }
    return mixed, fingerprint


def _index_file_stamps(index_files: Sequence[str]) -> list[dict[str, Any]]:
    """Record identity of each index file, so an edited index fails resume."""
    stamps: list[dict[str, Any]] = []
    for path in index_files:
        real = os.path.realpath(str(path))
        try:
            stat = os.stat(real)
        except OSError:
            stamps.append({"path": real, "size": None, "mtime_ns": None})
            continue
        stamps.append({"path": real, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return stamps


def open_mixed_raw_dataset(
    sources: Sequence[Mapping[str, Any]],
    *,
    mode: str = "train",
    mix_seed: int = _DEFAULT_MIX_SEED,
    steps_per_epoch: int = _DEFAULT_STEPS_PER_EPOCH,
) -> tuple[MixedDataset, dict[str, Any]]:
    """Open N raw sources as one :class:`MixedDataset` plus a fingerprint.

    A single source still goes through :class:`MixedDataset`, which keeps the
    sampler and the resume contract identical whether a run mixes or not.

    Fingerprinting differs from the cache twin in what it can hash. A cache has
    content-addressed sample ids to digest; a raw source has an index file whose
    contents could change under a stable path, so the fingerprint records the
    file's size and mtime alongside the decode parameters that determine the
    pixels.
    """
    if not sources:
        raise ValueError("open_mixed_raw_dataset requires at least one source")

    datasets: list[Dataset] = []
    weights: list[float] = []
    names: list[str] = []
    fingerprint_sources: list[dict[str, Any]] = []

    for index, source in enumerate(sources):
        name = str(source.get("name") or f"sub_{index}")
        inner = open_raw_dataset(source, mode=mode, index=index)
        min_latents = _optional_int(source.get("min_latents"))
        max_latents = _optional_int(source.get("max_latents"))
        datasets.append(wrap_latent_bounds(inner, min_latents=min_latents, max_latents=max_latents))
        weights.append(float(source.get("weight", 1.0)))
        names.append(name)

        index_files = [str(path) for path in (source.get("index_files") or ())]
        fingerprint_sources.append(
            {
                "name": name,
                "kind": str(source.get("kind") or ""),
                "index_files": _index_file_stamps(index_files),
                "weight": float(source.get("weight", 1.0)),
                "frames": int(source["frames"]),
                "height": int(source["height"]),
                "width": int(source["width"]),
                "max_sequence_length": int(source.get("max_sequence_length", 512)),
                "min_latents": min_latents,
                "max_latents": max_latents,
                "extras": dict(source.get("extras") or {}),
                "sample_count": len(inner),
                "sample_ids_digest": digest_sample_ids(inner.sample_ids),
            }
        )

    mixed = MixedDataset(
        datasets,
        weights,
        mix_seed=mix_seed,
        steps_per_epoch=steps_per_epoch,
        names=names,
        cache_dirs=[None] * len(datasets),
    )
    fingerprint = {
        "source_kind": "mixed_raw_video",
        "mode": mode,
        "mix_seed": int(mix_seed),
        "steps_per_epoch": int(steps_per_epoch),
        "sources": fingerprint_sources,
    }
    return mixed, fingerprint


def build_mixed_resumable_loader(
    dataset: MixedDataset,
    *,
    parallel_state: Any,
    batch_size: int,
    num_workers: int,
    drop_last: bool,
    seed: int,
    shuffle: bool = True,
    collate_fn: Callable | None = None,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    broadcast_batches: bool = True,
) -> StatefulDataLoader | LeaderBroadcastDataLoader:
    """Build a leader-owned loader that samples one mixed sub-dataset per step.

    Sampler, StatefulDataLoader and leader-broadcast wrapping are assembled here
    rather than in the generic builder so that path does not grow mix-shaped
    hooks.
    """
    if not isinstance(dataset, MixedDataset):
        raise TypeError(
            f"build_mixed_resumable_loader requires a MixedDataset, got {type(dataset).__name__}"
        )
    group = parallel_state.batch_group
    is_leader = bool(parallel_state.is_batch_leader)
    leader_loader = None
    if is_leader:
        sampler = None
        if len(dataset) > 0:
            sampler = MixedSampler(
                dataset,
                dp_rank=int(parallel_state.dp_rank),
                dp_world_size=int(parallel_state.config.dp_size),
                batch_size=batch_size,
                shuffle=shuffle,
                seed=seed,
            )
        leader_loader = StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            generator=torch.Generator().manual_seed(seed),
        )
    if group is None:
        assert leader_loader is not None
        return leader_loader
    return LeaderBroadcastDataLoader(
        leader_loader,
        group=group,
        is_leader=is_leader,
        group_size=dist.get_world_size(group),
        broadcast_batches=broadcast_batches,
    )


__all__ = [
    "MixedDataset",
    "MixedSampler",
    "build_mixed_resumable_loader",
    "open_mixed_cache_dataset",
    "open_mixed_raw_dataset",
    "wrap_latent_bounds",
]
