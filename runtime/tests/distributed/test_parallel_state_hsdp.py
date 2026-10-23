"""Real-process tests for the FSDP2 hybrid-shard mesh."""

from __future__ import annotations

import os

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp


def _hsdp_mesh_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
    )
    os.environ["LOCAL_RANK"] = str(rank)
    try:
        from vforge.distributed.parallel_state import ParallelConfig, init_parallel_state

        state = init_parallel_state(
            ParallelConfig(
                dp_size=4,
                dp_replicate_size=2,
                dp_shard_size=2,
                device_type="cpu",
            )
        )

        assert state.device_mesh is not None
        assert state.device_mesh.mesh_dim_names == ("dp_replicate", "dp_shard")
        assert state.fsdp_mesh is not None
        assert state.fsdp_mesh.mesh_dim_names == ("dp_replicate", "dp_shard_usp")
        assert tuple(state.fsdp_mesh.mesh.shape) == (2, 2)

        replicate_peers = dist.get_process_group_ranks(state.device_mesh.get_group("dp_replicate"))
        shard_peers = dist.get_process_group_ranks(state.device_mesh.get_group("dp_shard"))
        dp_peers = dist.get_process_group_ranks(state.dp_group)

        assert replicate_peers == [rank % 2, rank % 2 + 2]
        assert shard_peers == [rank - rank % 2, rank - rank % 2 + 1]
        assert dp_peers == list(range(world_size))
        assert state.dp_replicate_rank == rank // 2
    finally:
        dist.destroy_process_group()


def test_four_rank_hsdp_mesh_groups(tmp_path) -> None:
    init_file = str(tmp_path / "hsdp-pg")
    mp.spawn(_hsdp_mesh_worker, args=(4, init_file), nprocs=4, join=True)


def test_programmatic_hsdp_rejects_invalid_topologies() -> None:
    from vforge.distributed.parallel_state import ParallelConfig

    with pytest.raises(ValueError, match="dp_shard_size > 1"):
        ParallelConfig(
            dp_size=2,
            dp_replicate_size=2,
        ).normalized(world_size=2)
    with pytest.raises(ValueError, match="requires dp_mode='fsdp2'"):
        ParallelConfig(
            dp_size=4,
            dp_replicate_size=2,
            dp_shard_size=2,
            dp_mode="ddp",
        ).normalized(world_size=4)


def _hsdp_ep_mesh_worker(rank: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", world_size=4, rank=rank)
    try:
        from vforge.distributed.parallel_state import ParallelConfig, init_parallel_state

        state = init_parallel_state(
            ParallelConfig(
                dp_size=4,
                dp_replicate_size=2,
                dp_shard_size=2,
                extra_parallel_sizes=(2,),
                device_type="cpu",
            )
        )
        assert state.ep_fsdp_mesh.mesh_dim_names == ("ep_replicate", "ep_fsdp")
        assert tuple(state.ep_fsdp_mesh.shape) == (2, 1)
        assert dist.get_process_group_ranks(state.ep_group) == [rank // 2 * 2, rank // 2 * 2 + 1]
        assert dist.get_process_group_ranks(state.ep_fsdp_mesh.get_group("ep_replicate")) == [
            rank % 2,
            rank % 2 + 2,
        ]
        assert state.ep_fsdp_mesh.size(-1) == 1
    finally:
        dist.destroy_process_group()


def test_hsdp_expert_replication_mesh(tmp_path):
    mp.spawn(_hsdp_ep_mesh_worker, args=(str(tmp_path / "hsdp-ep-pg"),), nprocs=4, join=True)


def test_ep_must_fit_inside_one_hsdp_shard_domain():
    from vforge.distributed.parallel_state import ParallelConfig

    with pytest.raises(ValueError, match="ep_size must divide dp_shard_size"):
        ParallelConfig(
            dp_size=4,
            dp_replicate_size=2,
            dp_shard_size=2,
            extra_parallel_sizes=(4,),
        ).normalized(4)
