"""Every rank sharing a dp coordinate must be handed the same batch.

This is the failure ``batch_group`` exists to prevent, and it is invisible from
a single process. Sampling indexes ``dp``; a tensor-parallel peer
sits at the same dp coordinate but a different tp one, so if it draws from its
own loader it gets a different sample, computes a gradient for a *different*
example against its shard of the weights, and the all-reduce inside
``RowParallelLinear`` averages two unrelated things. Nothing raises. The loss
even goes down. So the check has to be that two processes observed equal bytes,
which means real processes.

gloo on CPU: the question is which rank received what, not how fast.
"""

from __future__ import annotations

import os

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
SAMPLES = 8


def _values(batches: list) -> list[int]:
    return [int(batch["value"].item()) for batch in batches]


def _worker(rank: int, world_size: int, init_file: str, topology: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
    )
    os.environ["LOCAL_RANK"] = str(rank)
    try:
        import torch

        from vforge.data.loader import (
            LeaderBroadcastDataLoader,
            build_resumable_loader,
        )
        from vforge.distributed.parallel_state import (
            ParallelConfig,
            destroy_parallel_state,
            init_parallel_state,
        )

        # Both topologies put two ranks on every dp coordinate. Under ulysses
        # that has always been true and the loader already handled it; under tp
        # it is new. The point of running both is that the tp case must reach
        # the same answer the ulysses case has been reaching all along.
        extra = {
            "tp": {"tp_size": 2},
            "ulysses": {"usp_size": 2},
            "cp": {"cp_size": 2, "dp_mode": "ddp"},
        }[topology]
        state = init_parallel_state(ParallelConfig(dp_size=2, device_type="cpu", **extra))
        assert dist.get_world_size(state.batch_group) == 2

        dataset = [{"value": torch.tensor([index])} for index in range(SAMPLES)]
        loader = build_resumable_loader(
            dataset,
            parallel_state=state,
            batch_size=1,
            num_workers=0,
            drop_last=True,
            seed=0,
        )
        assert isinstance(loader, LeaderBroadcastDataLoader)

        # Only one rank per group may hold a real loader. Two live cursors on
        # one dp coordinate would advance independently and desynchronise even
        # if the first batch happened to match.
        assert loader.state_dict()["owner"] is state.is_batch_leader

        seen = _values(list(loader))
        assert len(seen) == SAMPLES // 2, (rank, seen)

        gathered: list[list[int] | None] = [None] * world_size
        dist.all_gather_object(gathered, seen)

        for leader in range(0, world_size, 2):
            assert gathered[leader] == gathered[leader + 1], (topology, leader, gathered)

        # Guards the assertion above against passing for the wrong reason: if
        # every rank saw one identical stream, dp would have collapsed and the
        # equality would be trivially true.
        assert set(gathered[0] or []).isdisjoint(gathered[2] or []), gathered

        destroy_parallel_state()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "topology",
    ["tp", "ulysses", "cp"],
    ids=["tp2_dp2", "ulysses2_dp2", "cp2_dp2"],
)
def test_replica_peers_receive_identical_batches(tmp_path, topology):
    init_file = str(tmp_path / f"replica-loader-{topology}-pg")
    mp.spawn(_worker, args=(WORLD, init_file, topology), nprocs=WORLD, join=True)
