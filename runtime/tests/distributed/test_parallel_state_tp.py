"""Mesh topology under tensor parallelism.

These run on gloo because the question is rank bookkeeping, not kernels: which
process lands in which group. Getting that wrong does not crash, it trains on
mismatched data, so it is worth spawning real processes to check.
"""

from __future__ import annotations

import os

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp


def _init(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
    )
    os.environ["LOCAL_RANK"] = str(rank)


def _tp_mesh_worker(rank: int, world_size: int, init_file: str) -> None:
    _init(rank, world_size, init_file)
    try:
        from vforge.distributed.parallel_state import ParallelConfig, init_parallel_state

        state = init_parallel_state(
            ParallelConfig(dp_size=2, tp_size=2, device_type="cpu"),
        )

        # The mesh lists tp last, so tp is the fastest varying dimension and
        # global_rank == tp_rank + dp_rank * tp_size. That is byte for byte the
        # layout mcore's RankGenerator produces for its default "tp-cp-ep-dp-pp"
        # order, which is what lets the megatron engine reuse this topology.
        assert state.dp_rank == rank // 2, rank
        assert state.tp_rank == rank % 2, rank
        assert state.tp_enabled

        tp_peers = dist.get_process_group_ranks(state.tp_group)
        dp_peers = dist.get_process_group_ranks(state.dp_group)
        assert tp_peers == [rank - rank % 2, rank - rank % 2 + 1], (rank, tp_peers)
        assert dp_peers == [rank % 2, rank % 2 + 2], (rank, dp_peers)

        # With no sequence parallelism the batch is shared exactly across the tp
        # group, so batch_group must be that group itself and not a flattened
        # copy of it, which would be a second communicator over the same ranks.
        assert state.batch_group is state.tp_group
        assert dist.get_process_group_ranks(state.batch_group) == tp_peers
        assert state.is_batch_leader is (state.tp_rank == 0)
    finally:
        dist.destroy_process_group()


def _sp_only_worker(rank: int, world_size: int, init_file: str) -> None:
    _init(rank, world_size, init_file)
    try:
        from vforge.distributed.parallel_state import ParallelConfig, init_parallel_state

        state = init_parallel_state(
            ParallelConfig(dp_size=2, usp_size=2, device_type="cpu"),
        )

        # Backward compatibility: without tp the batch group has to be the SP
        # group itself, or switching the dataloader over to it would change
        # behaviour for every run that exists today.
        assert not state.tp_enabled
        assert state.tp_rank == 0
        assert state.tp_group is None
        assert state.batch_group is state.usp_group
        assert state.is_batch_leader is (state.usp_rank == 0)
    finally:
        dist.destroy_process_group()


def _cp_tp_mesh_worker(rank: int, world_size: int, init_file: str) -> None:
    _init(rank, world_size, init_file)
    try:
        from vforge.distributed.parallel_state import ParallelConfig, init_parallel_state

        state = init_parallel_state(
            ParallelConfig(dp_size=1, cp_size=2, tp_size=2, dp_mode="ddp", device_type="cpu"),
        )

        assert state.cp_rank == rank // 2
        assert state.tp_rank == rank % 2
        assert dist.get_process_group_ranks(state.cp_group) == [rank % 2, rank % 2 + 2]
        assert dist.get_process_group_ranks(state.tp_group) == [
            rank - rank % 2,
            rank - rank % 2 + 1,
        ]
        assert dist.get_process_group_ranks(state.dp_cp_group) == [rank % 2, rank % 2 + 2]
        assert dist.get_process_group_ranks(state.batch_group) == list(range(world_size))
        assert state.is_batch_leader is (rank == 0)
    finally:
        dist.destroy_process_group()


def _single_process_worker() -> None:
    from vforge.distributed.parallel_state import (
        ParallelConfig,
        destroy_parallel_state,
        init_parallel_state,
    )

    destroy_parallel_state()
    state = init_parallel_state(ParallelConfig(device_type="cpu"))
    assert state.device_mesh is None
    assert state.tp_rank == 0
    assert state.tp_group is None
    assert state.batch_group is None
    assert state.is_batch_leader
    destroy_parallel_state()


@pytest.mark.parametrize(
    "worker",
    [_tp_mesh_worker, _sp_only_worker, _cp_tp_mesh_worker],
    ids=["tp2_dp2", "ulysses2_dp2", "cp2_tp2"],
)
def test_four_rank_mesh_groups(tmp_path, worker):
    init_file = str(tmp_path / f"{worker.__name__}-pg")
    mp.spawn(worker, args=(4, init_file), nprocs=4, join=True)


def test_single_process_state_has_no_tp_groups():
    _single_process_worker()


def test_batch_group_refuses_tp_and_ulysses_together():
    """Better to stop than to hand back whichever group happens to exist.

    Nothing runs this combination today, so there is no right answer to return;
    picking one silently would leave the other group's peers without a batch.
    """
    from vforge.distributed.parallel_state import ParallelConfig, ParallelState

    state = ParallelState(config=ParallelConfig(dp_size=1, usp_size=2, tp_size=2))
    with pytest.raises(NotImplementedError, match="Ulysses with tensor or context"):
        _ = state.batch_group
    with pytest.raises(NotImplementedError, match="Ulysses with tensor or context"):
        _ = state.is_batch_leader


def test_replicated_ddp_refuses_ulysses_until_both_axes_are_reduced():
    from vforge.distributed.parallel_state import ParallelConfig

    with pytest.raises(NotImplementedError, match="DDP and Ulysses"):
        ParallelConfig(dp_mode="ddp", usp_size=2).normalized(world_size=2)


def test_tensor_parallel_refuses_ulysses_until_both_axes_have_a_layout():
    from vforge.distributed.parallel_state import ParallelConfig

    with pytest.raises(NotImplementedError, match="tensor parallelism and Ulysses"):
        ParallelConfig(tp_size=2, usp_size=2).normalized(world_size=4)


def test_context_parallel_refuses_ulysses_because_both_shard_tokens():
    from vforge.distributed.parallel_state import ParallelConfig

    with pytest.raises(NotImplementedError, match="context parallelism and Ulysses"):
        ParallelConfig(cp_size=2, usp_size=2).normalized(world_size=4)
