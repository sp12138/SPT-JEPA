"""Taichu production indexed-cache semantics for the Native Wan task.

The generic cache reader treats the files that actually exist as the dataset.
Taichu production training instead keeps the raw metadata index space dense:
missing or corrupt entries remain sampler positions and are replaced at read
time by a random position from the same source.  That distinction changes the
Random-Reshuffling permutation even when only one cache entry is absent, so it
belongs to the numerical migration contract rather than to generic data code.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from vforge.data.cache import (
    EncodedBatchCodec,
    digest_sample_ids,
    load_latent_bundle,
    read_cache_manifest,
)
from vforge.data.dataset import CachePathResolver, EncodedCacheDataset, load_manifest_sample_ids
from vforge.data.mixed_dataset import MixedDataset, wrap_latent_bounds
from vforge.data.taichu_index_cache import indexed_sample_ids

logger = logging.getLogger(__name__)

_DEFAULT_MIX_SEED = 20260726
_DEFAULT_STEPS_PER_EPOCH = 1_000_000_000
_DEFAULT_MAX_CACHE_MISS_RETRIES = 32
_INDEXED_WIDTH = 8
_INDEXED_SUFFIX = ".pt"
_SHARD_SIZE = 1000


def _indexed_path_resolver(
    *,
    split: str = "train",
    sample_id_is_index: bool = False,
) -> CachePathResolver:
    """Resolve Taichu's indexed layout without extending generic data semantics."""

    def path_for(cache_dir: str, sample_id: str, index: int) -> str:
        written_index = int(sample_id) if sample_id_is_index else index
        shard = f"{written_index // _SHARD_SIZE:04d}"
        return str(
            Path(cache_dir) / split / shard / f"{written_index:0{_INDEXED_WIDTH}d}{_INDEXED_SUFFIX}"
        )

    return path_for


def _scan_written_ids(cache_dir: str, *, split: str = "train") -> list[str]:
    """Return sparse written IDs; dense reconstruction is a task-level choice."""
    root = os.path.join(cache_dir, split)
    if not os.path.isdir(root):
        raise ValueError(f"indexed cache has no {split}/ directory under {cache_dir}")

    indices: set[int] = set()
    with os.scandir(root) as shards:
        for shard in shards:
            if not shard.is_dir():
                continue
            with os.scandir(shard.path) as entries:
                for entry in entries:
                    if not entry.name.endswith(_INDEXED_SUFFIX):
                        continue
                    stem = entry.name[: -len(_INDEXED_SUFFIX)]
                    if len(stem) == _INDEXED_WIDTH and stem.isdigit():
                        indices.add(int(stem))
    if not indices:
        raise ValueError(
            f"indexed cache {root} holds no {_INDEXED_WIDTH}-digit {_INDEXED_SUFFIX} bundles"
        )
    return [f"{index:0{_INDEXED_WIDTH}d}" for index in sorted(indices)]


def _scan_written_ids_on_leader(cache_dirs: Sequence[str]) -> list[list[str]]:
    """Scan all Taichu sources on rank zero and broadcast their sparse IDs once."""
    dirs = [str(cache_dir) for cache_dir in cache_dirs]
    if not dirs:
        return []
    if not (dist.is_available() and dist.is_initialized()):
        return [_scan_written_ids(cache_dir) for cache_dir in dirs]

    payload: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = [_scan_written_ids(cache_dir) for cache_dir in dirs]
        except Exception as error:
            payload[0] = error
    dist.broadcast_object_list(payload, src=0)
    scanned = payload[0]
    if isinstance(scanned, BaseException):
        raise scanned
    return [[str(sample_id) for sample_id in sample_ids] for sample_ids in scanned]


class TaichuIndexedCacheDataset(Dataset):
    """Read a dense indexed cache with Taichu's bounded miss fallback."""

    def __init__(
        self,
        inner: EncodedCacheDataset,
        *,
        cache_dir: str,
        load_ref_sidecar: bool = False,
        max_cache_miss_retries: int = _DEFAULT_MAX_CACHE_MISS_RETRIES,
    ) -> None:
        if len(inner) <= 0:
            raise ValueError("Taichu indexed cache must contain at least one logical sample")
        retries = int(max_cache_miss_retries)
        if retries < 0:
            raise ValueError("max_cache_miss_retries must be non-negative")
        self._inner = inner
        self._cache_dir = str(cache_dir)
        self._load_ref_sidecar = bool(load_ref_sidecar)
        self._max_cache_miss_retries = retries
        self._cache_miss_warned = False

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def sample_ids(self) -> list[str]:
        return self._inner.sample_ids

    def __getitem__(self, index: int) -> dict[str, Any]:
        logical_index = int(index)
        logical_sample_id = self._inner.sample_ids[logical_index]
        current = logical_index
        last_error: Exception | None = None
        for _attempt in range(self._max_cache_miss_retries + 1):
            try:
                item = dict(self._inner[current])
            except Exception as error:
                last_error = error
                if not self._cache_miss_warned:
                    logger.warning(
                        "Wan Taichu indexed cache miss at logical index %d (%s); "
                        "falling back to a random index in the same source. Further "
                        "misses from this worker are logged at debug level.",
                        current,
                        error,
                    )
                    self._cache_miss_warned = True
                else:
                    logger.debug(
                        "Wan Taichu indexed cache miss at logical index %d: %s",
                        current,
                        error,
                    )
                current = random.randrange(len(self))
                continue
            # Taichu's dataset identity is the requested dense metadata index,
            # even if the payload was supplied by a fallback cache entry.
            self._maybe_attach_ref_sidecar(item, current)
            item["sample_id"] = logical_sample_id
            return item
        raise FileNotFoundError(
            "Wan Taichu indexed cache could not load a bundle after "
            f"{self._max_cache_miss_retries} random retries; "
            f"last logical index={current}, last error={last_error}"
        ) from last_error

    def _maybe_attach_ref_sidecar(self, item: dict[str, Any], resolved_index: int) -> None:
        if not self._load_ref_sidecar or item.get("ref_frame_latents") is not None:
            return
        shard = f"{resolved_index // 1000:04d}"
        path = os.path.join(
            self._cache_dir,
            "ref_images",
            "train",
            shard,
            f"{resolved_index:08d}.pt",
        )
        if not os.path.isfile(path):
            return
        try:
            sidecar = load_latent_bundle(path)
        except Exception as error:
            logger.debug("Wan Taichu ref sidecar load failed at %s: %s", path, error)
            return
        latent = sidecar.get("ref_latent")
        if isinstance(latent, torch.Tensor):
            item["ref_frame_latents"] = latent


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _dense_ids_from_scan(
    surviving_ids: Sequence[str],
    *,
    explicit_count: int | None = None,
) -> list[str]:
    """Restore the dense metadata index space from a sparse cache scan."""
    if not surviving_ids:
        raise ValueError("Taichu indexed cache scan returned no bundles")
    largest_written = max(int(sample_id) for sample_id in surviving_ids)
    count = largest_written + 1 if explicit_count is None else int(explicit_count)
    if count <= largest_written:
        raise ValueError(
            "indexed_sample_count must be greater than every written cache index; "
            f"got count={count}, largest_written={largest_written}"
        )
    return indexed_sample_ids(count)


def _indexed_sample_count(source: Mapping[str, Any]) -> int | None:
    extras = source.get("extras") or {}
    if not isinstance(extras, Mapping):
        raise TypeError("data source extras must be a mapping")
    return _optional_int(extras.get("indexed_sample_count"))


def _wrap_indexed(
    cache_dir: str,
    sample_ids: list[str],
    codec: EncodedBatchCodec,
    *,
    sample_id_is_index: bool,
    load_ref_sidecar: bool = False,
) -> TaichuIndexedCacheDataset:
    return TaichuIndexedCacheDataset(
        EncodedCacheDataset(
            cache_dir,
            sample_ids,
            codec,
            path_for=_indexed_path_resolver(sample_id_is_index=sample_id_is_index),
        ),
        cache_dir=cache_dir,
        load_ref_sidecar=load_ref_sidecar,
    )


def open_taichu_encoded_cache_dataset(
    cache_dir: str,
    train_manifest: str,
    *,
    codec: EncodedBatchCodec,
    load_ref_sidecar: bool = False,
    indexed_sample_count: int | None = None,
) -> Dataset:
    """Open one cache, applying dense fallback only to Taichu indexed layout."""
    manifest = read_cache_manifest(cache_dir, codec=codec) if cache_dir else None
    if manifest is not None:
        return EncodedCacheDataset(cache_dir, manifest.sample_ids, codec)
    if not cache_dir:
        raise ValueError("opening an encoded cache requires data.cache_dir")
    if not os.path.isdir(os.path.join(cache_dir, "train")):
        raise ValueError(
            f"cache has neither manifest.json nor indexed train/ directory: {cache_dir}"
        )
    if train_manifest:
        return _wrap_indexed(
            cache_dir,
            load_manifest_sample_ids(train_manifest),
            codec,
            sample_id_is_index=False,
            load_ref_sidecar=load_ref_sidecar,
        )
    surviving_ids = _scan_written_ids_on_leader([cache_dir])[0]
    return _wrap_indexed(
        cache_dir,
        _dense_ids_from_scan(surviving_ids, explicit_count=indexed_sample_count),
        codec,
        sample_id_is_index=True,
        load_ref_sidecar=load_ref_sidecar,
    )


def open_taichu_mixed_cache_dataset(
    sources: Sequence[Mapping[str, Any]],
    *,
    codec: EncodedBatchCodec,
    mix_seed: int = _DEFAULT_MIX_SEED,
    steps_per_epoch: int = _DEFAULT_STEPS_PER_EPOCH,
    load_ref_sidecar: bool = False,
) -> tuple[MixedDataset, dict[str, Any]]:
    """Open mixed caches while preserving Taichu's dense indexed lengths."""
    if not sources:
        raise ValueError("Wan Taichu mixed cache requires at least one source")

    specs: list[dict[str, Any]] = []
    resolved: list[dict[str, Any] | None] = [None] * len(sources)
    scan_indices: list[int] = []
    scan_dirs: list[str] = []
    for index, source in enumerate(sources):
        cache_dir = str(source.get("cache_dir") or "")
        if not cache_dir:
            raise ValueError(f"data.sources[{index}] requires a cache_dir path")
        spec = {
            "name": str(source.get("name") or f"sub_{index}"),
            "cache_dir": cache_dir,
            "manifest": str(source.get("manifest") or ""),
            "weight": float(source.get("weight", 1.0)),
            "min_latents": _optional_int(source.get("min_latents")),
            "max_latents": _optional_int(source.get("max_latents")),
            "indexed_sample_count": _indexed_sample_count(source),
        }
        specs.append(spec)

        native_manifest = read_cache_manifest(cache_dir, codec=codec)
        if native_manifest is not None:
            resolved[index] = {
                "sample_ids": list(native_manifest.sample_ids),
                "path_for": None,
                "indexed": False,
                "written_ids": list(native_manifest.sample_ids),
            }
            continue
        if not os.path.isdir(os.path.join(cache_dir, "train")) and not spec["manifest"]:
            raise ValueError(
                f"cache has neither manifest.json nor indexed train/ directory: {cache_dir}"
            )
        if spec["manifest"]:
            sample_ids = load_manifest_sample_ids(spec["manifest"])
            resolved[index] = {
                "sample_ids": sample_ids,
                "path_for": _indexed_path_resolver(),
                "indexed": True,
                "written_ids": sample_ids,
            }
            continue
        scan_indices.append(index)
        scan_dirs.append(cache_dir)

    if scan_dirs:
        scanned_sources = _scan_written_ids_on_leader(scan_dirs)
        for index, surviving_ids in zip(scan_indices, scanned_sources, strict=True):
            resolved[index] = {
                "sample_ids": _dense_ids_from_scan(
                    surviving_ids,
                    explicit_count=specs[index]["indexed_sample_count"],
                ),
                "path_for": _indexed_path_resolver(sample_id_is_index=True),
                "indexed": True,
                "written_ids": list(surviving_ids),
            }

    datasets: list[Dataset] = []
    weights: list[float] = []
    names: list[str] = []
    cache_dirs: list[str] = []
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
        dataset: Dataset = (
            TaichuIndexedCacheDataset(
                inner,
                cache_dir=spec["cache_dir"],
                load_ref_sidecar=load_ref_sidecar,
            )
            if opened["indexed"]
            else inner
        )
        datasets.append(
            wrap_latent_bounds(
                dataset,
                min_latents=spec["min_latents"],
                max_latents=spec["max_latents"],
            )
        )
        weights.append(spec["weight"])
        names.append(spec["name"])
        cache_dirs.append(spec["cache_dir"])
        written_ids = opened["written_ids"]
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
                "written_sample_count": len(written_ids),
                "written_sample_ids_digest": digest_sample_ids(written_ids),
                "indexed_miss_fallback": bool(opened["indexed"]),
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
    return mixed, {
        "source_kind": "mixed_encoded_cache",
        "mix_seed": int(mix_seed),
        "steps_per_epoch": int(steps_per_epoch),
        "sources": fingerprint_sources,
    }


__all__ = [
    "TaichuIndexedCacheDataset",
    "open_taichu_encoded_cache_dataset",
    "open_taichu_mixed_cache_dataset",
]
