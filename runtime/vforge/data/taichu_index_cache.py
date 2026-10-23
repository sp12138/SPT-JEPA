"""The indexed cache layout Wan's offline caches are written in.

A taichu-era cache shards its bundles as ``<split>/<i//1000:04d>/<i:08d>.pt``
and carries no root manifest, so both its ordering and its length are
properties of the directory tree rather than of a side-car file. Every piece of
code that knows that rule lives here; :class:`vforge.data.dataset.EncodedCacheDataset`
takes the path rule as an argument and stays layout-agnostic.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch.distributed as dist

from vforge.data.cache import EncodedBatchCodec, read_cache_manifest
from vforge.data.dataset import (
    CachePathResolver,
    EncodedCacheDataset,
    load_manifest_sample_ids,
)

_INDEXED_WIDTH = 8
_INDEXED_SUFFIX = ".pt"
_SHARD_SIZE = 1000


def indexed_sample_ids(count: int) -> list[str]:
    """The sample IDs a dense indexed cache of ``count`` bundles carries.

    The IDs are the zero-padded indices, so the whole list follows from the
    count. Callers that already know the count -- notably ranks that received
    it from the leader instead of scanning -- rebuild the list from here rather
    than repeating the padding.
    """
    if count < 0:
        raise ValueError(f"indexed cache sample count must be non-negative; got {count}")
    return [f"{index:0{_INDEXED_WIDTH}d}" for index in range(count)]


def indexed_path_resolver(*, split: str = "train") -> CachePathResolver:
    """Resolve a bundle path from its position rather than from its ID.

    An indexed cache names bundles after the index they were written at, so the
    sample ID carries no information the position does not already have.
    """

    def path_for(cache_dir: str, sample_id: str, index: int) -> str:
        del sample_id
        shard = f"{index // _SHARD_SIZE:04d}"
        return str(Path(cache_dir) / split / shard / f"{index:0{_INDEXED_WIDTH}d}{_INDEXED_SUFFIX}")

    return path_for


def scan_indexed_cache(cache_dir: str | Path, *, split: str = "train") -> list[str]:
    """Derive ordered sample IDs from a dense indexed cache, rejecting gaps.

    The sample count is already a property of the directory tree, and a producer
    that finished writing leaves ``0..N-1`` with no holes. Reading the count back
    from the tree keeps the cache self-describing: no side-car manifest has to be
    kept in sync with it.

    A gap means the cache was only partially written. That fails here rather than
    resampling around it, because silently training on a shorter epoch would break
    the resume contract, which pins the sample count and ID digest.
    """
    root = os.path.join(str(cache_dir), split)
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

    expected = max(indices) + 1
    missing = [index for index in range(expected) if index not in indices]
    if missing:
        preview = ", ".join(f"{index:0{_INDEXED_WIDTH}d}" for index in missing[:10])
        if len(missing) > 10:
            preview += f", ... ({len(missing)} total)"
        raise ValueError(
            f"indexed cache {root} is missing {len(missing)} of {expected} bundles "
            f"[{preview}]; re-run prepare for those indices or point data.cache_dir "
            "at a complete cache"
        )
    return indexed_sample_ids(expected)


def scan_indexed_caches_on_leader(cache_dirs: Sequence[str]) -> list[list[str]]:
    """Scan N indexed caches with one collective rather than N of them.

    Mixed training opens several caches. Walking each tree on every rank, or
    even broadcasting each count on its own, stacks NFS traffic and
    collectives. The leader scans every directory that needs it and ships one
    list of counts; followers rebuild the ID lists from those integers.

    A failed scan travels as the exception object, so every rank raises the
    leader's message instead of hanging on the next collective.
    """
    dirs = [str(cache_dir) for cache_dir in cache_dirs]
    if not dirs:
        return []
    if not (dist.is_available() and dist.is_initialized()):
        return [scan_indexed_cache(cache_dir) for cache_dir in dirs]
    payload: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = [len(scan_indexed_cache(cache_dir)) for cache_dir in dirs]
        except Exception as error:
            payload[0] = error
    dist.broadcast_object_list(payload, src=0)
    scanned = payload[0]
    if isinstance(scanned, BaseException):
        raise scanned
    return [indexed_sample_ids(int(count)) for count in scanned]


def scan_indexed_cache_on_leader(cache_dir: str) -> list[str]:
    """Scan the shard tree once per job rather than once per rank.

    A dense scan reads every shard directory -- 583 of them for the 20s-era 5s
    cache, about 90 seconds from one process -- and every rank walking the same
    network mount at once is worse than that. The result follows entirely from
    the bundle count, so the leader scans and the rest rebuild the same list
    from one integer.

    A failed scan travels as the exception object, so every rank raises the
    leader's message -- the missing-index list included -- instead of hanging on
    the next collective.
    """
    return scan_indexed_caches_on_leader([cache_dir])[0]


def open_encoded_cache_dataset(
    cache_dir: str,
    train_manifest: str,
    *,
    codec: EncodedBatchCodec,
) -> EncodedCacheDataset:
    """Open a prepared cache by manifest, falling back to the indexed layout.

    An indexed cache carries no root manifest, so its ordering comes from either
    ``train_manifest`` when one is supplied or from the shard tree itself. The
    scan is the default because it cannot drift out of sync with the bundles;
    it runs on the leader and is shared, so it costs one tree walk per job.
    """
    manifest = read_cache_manifest(cache_dir, codec=codec) if cache_dir else None
    if manifest is not None:
        return EncodedCacheDataset(cache_dir, manifest.sample_ids, codec)
    if not cache_dir:
        raise ValueError("opening an encoded cache requires data.cache_dir")
    indexed_dir = os.path.join(cache_dir, "train")
    if not os.path.isdir(indexed_dir):
        raise ValueError(
            f"cache has neither manifest.json nor indexed train/ directory: {cache_dir}"
        )
    sample_ids = (
        load_manifest_sample_ids(train_manifest)
        if train_manifest
        else scan_indexed_cache_on_leader(cache_dir)
    )
    return EncodedCacheDataset(
        cache_dir,
        sample_ids,
        codec,
        path_for=indexed_path_resolver(),
    )


__all__ = [
    "indexed_path_resolver",
    "indexed_sample_ids",
    "open_encoded_cache_dataset",
    "scan_indexed_cache",
    "scan_indexed_cache_on_leader",
    "scan_indexed_caches_on_leader",
]
