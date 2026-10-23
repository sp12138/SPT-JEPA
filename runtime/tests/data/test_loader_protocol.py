"""CPU-oriented loader protocol tests.

Distributed replica broadcast parity belongs on the cloud regression job.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_dp1_loader_uses_seeded_distributed_sampler() -> None:
    from torch.utils.data.distributed import DistributedSampler

    from vforge.data.loader import build_resumable_loader

    state = SimpleNamespace(
        config=SimpleNamespace(dp_size=1),
        dp_rank=0,
        batch_group=None,
        is_batch_leader=True,
    )
    loader = build_resumable_loader(
        list(range(8)),
        parallel_state=state,
        batch_size=1,
        num_workers=0,
        drop_last=True,
        seed=0,
    )

    assert isinstance(loader.sampler, DistributedSampler)
    assert loader.sampler.seed == 0


@pytest.mark.parametrize("dp_size", [1, 2, 4])
def test_shuffle_off_walks_the_dataset_in_index_order(dp_size: int) -> None:
    """Rank r sees r, r+dp, r+2*dp...

    Ordered consumption is what lets a run be compared step-by-step against a
    producer whose shuffle is a different permutation scheme, so it has to be a
    plain stride rather than merely "some deterministic order".
    """
    from vforge.data.loader import build_resumable_sp_loader

    for rank in range(dp_size):
        loader = build_resumable_sp_loader(
            list(range(24)),
            parallel_state=SimpleNamespace(
                config=SimpleNamespace(dp_size=dp_size, usp_size=1),
                dp_rank=rank,
            ),
            batch_size=1,
            num_workers=0,
            drop_last=True,
            seed=0,
            shuffle=False,
        )

        seen = [int(batch[0]) for batch in loader]
        assert seen == list(range(rank, 24, dp_size))


def test_shuffle_on_stays_the_default_and_permutes() -> None:
    from vforge.data.loader import build_resumable_sp_loader

    def indices(**overrides):
        loader = build_resumable_sp_loader(
            list(range(24)),
            parallel_state=SimpleNamespace(
                config=SimpleNamespace(dp_size=1, usp_size=1),
                dp_rank=0,
            ),
            batch_size=1,
            num_workers=0,
            drop_last=True,
            seed=0,
            **overrides,
        )
        return [int(batch[0]) for batch in loader]

    assert indices() == indices(shuffle=True)
    assert indices() != list(range(24))
    assert sorted(indices()) == list(range(24))


def test_split_and_rebuild_batch_roundtrip() -> None:
    from vforge.data.loader import _rebuild_batch, _split_batch

    batch = {
        "latents": torch.randn(2, 4, 8, 8),
        "ids": [1, 2],
        "nested": {"mask": torch.ones(2, 8), "flag": True},
        "tuple_field": (torch.zeros(3), "x"),
    }
    meta, tensors = _split_batch(batch)
    assert len(tensors) == 3
    rebuilt = _rebuild_batch(meta, tensors)
    assert torch.equal(rebuilt["latents"], batch["latents"])
    assert rebuilt["ids"] == [1, 2]
    assert torch.equal(rebuilt["nested"]["mask"], batch["nested"]["mask"])
    assert rebuilt["nested"]["flag"] is True
    assert torch.equal(rebuilt["tuple_field"][0], batch["tuple_field"][0])
    assert rebuilt["tuple_field"][1] == "x"


def test_broadcast_batch_single_process_preserves_training_tree() -> None:
    from vforge.data.loader import broadcast_batch

    batch = {
        "model_inputs": {
            "hidden_states": torch.randn(1, 4, 2, 2, 2),
            "encoder_hidden_states_image_vae": None,
        },
        "targets": torch.randn(1, 4, 2, 2, 2),
        "loss_weights": torch.ones(1),
    }
    result = broadcast_batch(
        batch,
        group=None,
        is_source=True,
        target_device=torch.device("cpu"),
    )
    assert torch.equal(result["targets"], batch["targets"])
    assert result["model_inputs"]["encoder_hidden_states_image_vae"] is None


def test_broadcast_batch_accepts_the_previous_source_arguments() -> None:
    from vforge.data.loader import broadcast_batch

    batch = {"value": torch.tensor([1])}
    with pytest.warns(FutureWarning, match="deprecated"):
        result = broadcast_batch(
            batch,
            group=None,
            src_rank=0,
            group_rank=0,
        )

    assert torch.equal(result["value"], batch["value"])


def test_stateful_loader_helpers_reject_unrelated_state_methods() -> None:
    from torchdata.stateful_dataloader import StatefulDataLoader

    from vforge.data.loader import (
        is_resumable_dataloader,
        supports_bit_exact_resume,
    )

    class _Loader:
        def state_dict(self):
            return {"cursor": 1}

        def load_state_dict(self, state):
            self.state = state

    loader = StatefulDataLoader(
        list(range(4)),
        batch_size=1,
        shuffle=False,
    )

    assert is_resumable_dataloader(loader)
    assert supports_bit_exact_resume(loader) is True
    assert not is_resumable_dataloader(_Loader())
    assert supports_bit_exact_resume(_Loader()) is None


def test_eval_sampler_never_pads_or_duplicates_tail_samples() -> None:
    from vforge.data.loader import DistributedEvalSampler

    dataset = list(range(7))
    shards = [list(DistributedEvalSampler(dataset, num_replicas=3, rank=rank)) for rank in range(3)]
    real = [[index for index, padding in shard if not padding] for shard in shards]

    assert real == [[0, 3, 6], [1, 4], [2, 5]]
    assert sorted(index for shard in real for index in shard) == list(range(7))
    assert [len(shard) for shard in shards] == [3, 3, 3]
    assert [sum(padding for _index, padding in shard) for shard in shards] == [0, 1, 1]


def test_eval_sampler_applies_global_limit_before_sharding() -> None:
    from vforge.data.loader import DistributedEvalSampler

    dataset = list(range(9))
    shards = [
        list(DistributedEvalSampler(dataset, num_replicas=2, rank=rank, limit=5))
        for rank in range(2)
    ]
    real = [[index for index, padding in shard if not padding] for shard in shards]

    assert real == [[0, 2, 4], [1, 3]]
    assert [len(shard) for shard in shards] == [3, 3]


def test_eval_sampler_selects_evenly_spaced_indices_before_sharding() -> None:
    from vforge.data.loader import DistributedEvalSampler

    dataset = list(range(10))
    shards = [
        list(
            DistributedEvalSampler(
                dataset,
                num_replicas=2,
                rank=rank,
                limit=4,
                selection="evenly_spaced",
            )
        )
        for rank in range(2)
    ]
    real = [[index for index, padding in shard if not padding] for shard in shards]

    assert real == [[0, 6], [3, 9]]
    assert sorted(index for shard in real for index in shard) == [0, 3, 6, 9]


def test_eval_sampler_balances_evenly_spaced_selection_across_sources() -> None:
    from vforge.data.loader import DistributedEvalSampler

    dataset = list(range(10))
    source_groups = ((0, 1, 2, 3, 4, 5), (6, 7), (8, 9))
    shards = [
        list(
            DistributedEvalSampler(
                dataset,
                num_replicas=2,
                rank=rank,
                limit=5,
                selection="evenly_spaced",
                source_groups=source_groups,
            )
        )
        for rank in range(2)
    ]
    real = [[index for index, padding in shard if not padding] for shard in shards]

    # Front-heavy quota [2, 2, 1], then evenly spaced selection within each source.
    assert real == [[0, 6, 8], [5, 7]]
    assert sorted(index for shard in real for index in shard) == [0, 5, 6, 7, 8]


def test_eval_sampler_rejects_invalid_source_partition() -> None:
    from vforge.data.loader import DistributedEvalSampler

    with pytest.raises(ValueError, match="partition every dataset index"):
        DistributedEvalSampler(
            list(range(4)),
            num_replicas=1,
            rank=0,
            limit=2,
            selection="evenly_spaced",
            source_groups=((0, 1), (1, 2)),
        )


def test_leader_broadcast_loader_requires_a_group() -> None:
    from vforge.data.loader import LeaderBroadcastDataLoader

    with pytest.raises(ValueError, match="requires a process group"):
        LeaderBroadcastDataLoader(
            object(),
            group=None,
            is_leader=True,
            group_size=2,
        )


def test_follower_rejects_owner_checkpoint() -> None:
    from vforge.data.loader import LeaderBroadcastDataLoader

    loader = LeaderBroadcastDataLoader(
        None,
        group=object(),
        is_leader=False,
        group_size=2,
    )

    with pytest.raises(RuntimeError, match="requires a follower stub"):
        loader.load_state_dict(
            {
                "owner": True,
                "loader": {"cursor": 1},
                "group_size": 2,
            }
        )


def test_resume_refuses_a_differently_sized_group() -> None:
    """A cursor is only meaningful against the topology that produced it."""
    from vforge.data.loader import LeaderBroadcastDataLoader

    loader = LeaderBroadcastDataLoader(
        None,
        group=object(),
        is_leader=False,
        group_size=2,
    )

    with pytest.raises(RuntimeError, match="changed size"):
        loader.load_state_dict({"owner": False, "group_size": 4})


def test_leader_loader_accepts_the_previous_usp_checkpoint_key() -> None:
    from vforge.data.loader import LeaderBroadcastDataLoader

    loader = LeaderBroadcastDataLoader(
        None,
        group=object(),
        is_leader=False,
        group_size=2,
    )

    loader.load_state_dict({"owner": False, "usp_size": 2})


def test_previous_sp_loader_api_remains_available_for_one_rank() -> None:
    from torchdata.stateful_dataloader import StatefulDataLoader

    from vforge.data.loader import SPBroadcastDataLoader

    concrete = StatefulDataLoader(list(range(2)), batch_size=1)
    with pytest.warns(FutureWarning, match="deprecated"):
        loader = SPBroadcastDataLoader(
            concrete,
            group=None,
            usp_rank=0,
            usp_size=1,
        )

    assert [batch.item() for batch in loader] == [0, 1]
    assert loader.state_dict()["usp_size"] == 1


def test_previous_sp_loader_builder_forwards_to_the_generic_builder(monkeypatch) -> None:
    import vforge.data.loader as loader_module

    sentinel = object()
    monkeypatch.setattr(loader_module, "build_resumable_loader", lambda *args, **kwargs: sentinel)

    with pytest.warns(FutureWarning, match="deprecated"):
        result = loader_module.build_resumable_sp_loader(object(), parallel_state=object())

    assert result is sentinel


def test_follower_does_not_construct_stateful_loader(monkeypatch) -> None:
    """A follower holding its own cursor is the bug the leader election prevents.

    Which ranks are followers is a topology question answered by
    ``ParallelState.batch_group``; here the state is faked, so all this pins
    down is that the builder honours the answer instead of second-guessing it.
    """
    import vforge.data.loader as loader_module

    class _UnexpectedLoader:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("a follower must not construct a StatefulDataLoader")

    monkeypatch.setattr(loader_module, "StatefulDataLoader", _UnexpectedLoader)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 2)
    state = SimpleNamespace(
        config=SimpleNamespace(dp_size=1),
        dp_rank=0,
        batch_group=object(),
        is_batch_leader=False,
    )

    loader = loader_module.build_resumable_loader(
        list(range(8)),
        parallel_state=state,
        batch_size=1,
        num_workers=0,
        drop_last=True,
        seed=0,
    )

    assert isinstance(loader, loader_module.LeaderBroadcastDataLoader)
