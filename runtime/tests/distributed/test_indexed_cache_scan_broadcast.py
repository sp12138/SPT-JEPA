"""One rank walks the indexed shard tree; the rest are handed the answer.

A dense scan of the production 5s cache reads 583 shard directories holding
582k bundles and takes about 90 seconds from a single process. Every rank doing
that against the same network mount at job start is the cost this broadcast
removes, and it is invisible from one process: a per-rank scan returns the right
list too, just N times over.

So both properties here need real processes. That a follower never touches the
tree is asserted by making the walk raise on followers -- if the broadcast
regressed to a local scan, the follower fails instead of quietly getting slow.
That a bad cache still fails everywhere is the other half: an exception raised
only on the leader would leave the followers blocked on the next collective,
turning a clear error into a hang.

gloo on CPU: the question is which rank read the filesystem, not how fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 3
DENSE = 1005


def _write_indexed_shards(cache_dir: Path, indices: list[int]) -> None:
    for index in indices:
        path = cache_dir / "train" / f"{index // 1000:04d}" / f"{index:08d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")


def _forbid_scan_off_leader(rank: int) -> None:
    from vforge.data import taichu_index_cache as indexed_cache

    if rank == 0:
        return

    def _forbidden(*args: object, **kwargs: object) -> list[str]:
        raise AssertionError(f"rank {rank} walked the shard tree instead of taking the broadcast")

    indexed_cache.scan_indexed_cache = _forbidden


def _worker_dense(rank: int, world_size: int, init_file: str, cache_dir: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
    )
    try:
        from vforge.data.taichu_index_cache import scan_indexed_cache_on_leader

        _forbid_scan_off_leader(rank)
        sample_ids = scan_indexed_cache_on_leader(cache_dir)

        assert len(sample_ids) == DENSE, (rank, len(sample_ids))
        assert sample_ids[0] == "00000000", (rank, sample_ids[0])
        assert sample_ids[-1] == f"{DENSE - 1:08d}", (rank, sample_ids[-1])
    finally:
        dist.destroy_process_group()


def _worker_holes(rank: int, world_size: int, init_file: str, cache_dir: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
    )
    try:
        from vforge.data.taichu_index_cache import scan_indexed_cache_on_leader

        _forbid_scan_off_leader(rank)
        with pytest.raises(ValueError, match="missing 2 of 10 bundles") as excinfo:
            scan_indexed_cache_on_leader(cache_dir)

        # The leader's own message travels, so a follower can name the gap.
        assert "00000003" in str(excinfo.value), (rank, str(excinfo.value))
        assert "00000007" in str(excinfo.value), (rank, str(excinfo.value))
    finally:
        dist.destroy_process_group()


def test_followers_take_the_leader_scan(tmp_path):
    cache_dir = tmp_path / "cache"
    _write_indexed_shards(cache_dir, list(range(DENSE)))

    mp.spawn(
        _worker_dense,
        args=(WORLD, str(tmp_path / "dense-pg"), str(cache_dir)),
        nprocs=WORLD,
        join=True,
    )


def test_scan_failure_reaches_every_rank(tmp_path):
    cache_dir = tmp_path / "cache"
    _write_indexed_shards(cache_dir, [index for index in range(10) if index not in (3, 7)])

    mp.spawn(
        _worker_holes,
        args=(WORLD, str(tmp_path / "holes-pg"), str(cache_dir)),
        nprocs=WORLD,
        join=True,
    )
