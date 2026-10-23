"""Shared-loader integration contracts for task-owned DataSpec components."""

from __future__ import annotations

import copy
from functools import partial
from types import SimpleNamespace

import pytest
from torch.utils.data import Dataset, DistributedSampler, Sampler

import vforge.data.loader as loader_module
from vforge.data.loader import build_resumable_loader
from vforge.data.multi_source import (
    DatasetSource,
    MultiDataset,
    MultiSourceSampler,
    build_multi_source_sampler,
)
from vforge.data.spec import DataSpec


class _TokenItems(Dataset):
    def __init__(self, name: str, size: int) -> None:
        self.name, self.size = name, size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[str, int]:
        if not 0 <= index < self.size:
            raise IndexError(index)
        return self.name, index


class _FixedSampler(Sampler[int]):
    def __init__(self, indices: tuple[int, ...]) -> None:
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def _collate_items(samples):
    return list(samples)


def _parallel(rank=0, size=1):
    return SimpleNamespace(
        config=SimpleNamespace(dp_size=size),
        dp_rank=rank,
        batch_group=None,
        is_batch_leader=True,
    )


def _spec(*, total=97, version="ordered-v1") -> DataSpec:
    dataset = MultiDataset(
        (
            DatasetSource("A", _TokenItems("A", 7), version, 11),
            DatasetSource("B", _TokenItems("B", 11), "ordered-v1", 17),
        )
    )
    return DataSpec(
        dataset,
        sampler_factory=partial(build_multi_source_sampler, total_samples=total, mix_seed=73),
        collate_fn=_collate_items,
    )


def _loader(dataset, **overrides):
    kwargs = {
        "parallel_state": _parallel(),
        "batch_size": 8,
        "num_workers": 0,
        "drop_last": False,
        "seed": 11,
    }
    kwargs.update(overrides)
    return build_resumable_loader(dataset, **kwargs)


def _shutdown(loader) -> None:
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()


@pytest.mark.parametrize("wrap_in_spec", [False, True])
def test_existing_dataset_and_default_spec_keep_distributed_sampler_path(wrap_in_spec):
    dataset = list(range(14))
    loader = _loader(
        DataSpec(dataset) if wrap_in_spec else dataset,
        parallel_state=_parallel(rank=1, size=2),
        batch_size=2,
        shuffle=False,
    )
    assert isinstance(loader.sampler, DistributedSampler)
    assert len(loader) == 4
    assert [value for batch in loader for value in batch.tolist()] == list(range(1, 14, 2))


def test_data_spec_factory_receives_context_and_owns_all_sharding() -> None:
    calls = []
    dataset = list(range(10))
    sampler = _FixedSampler((7, 2, 5, 0))

    def factory(actual_dataset, **kwargs):
        calls.append((actual_dataset, kwargs))
        return sampler

    loader = _loader(
        DataSpec(dataset, sampler_factory=factory, collate_fn=_collate_items),
        parallel_state=_parallel(rank=3, size=4),
        batch_size=2,
        drop_last=True,
        shuffle=True,
        seed=19,
    )
    assert loader.sampler is sampler
    assert list(loader) == [[7, 2], [5, 0]]
    assert len(calls) == 1
    assert calls[0][0] is dataset
    assert calls[0][1] == {
        "dp_rank": 3,
        "dp_world_size": 4,
        "batch_size": 2,
        "seed": 19,
        "shuffle": True,
        "drop_last": True,
    }


def test_spec_rejects_conflicting_collate_and_non_sampler_factory_results() -> None:
    with pytest.raises(ValueError, match="collate_fn"):
        _loader(DataSpec([1], collate_fn=_collate_items), collate_fn=_collate_items)
    with pytest.raises(TypeError, match="Sampler"):
        _loader(DataSpec([1], sampler_factory=lambda *_args, **_kwargs: [0]))


def test_follower_never_constructs_the_task_sampler(monkeypatch) -> None:
    def unexpected_factory(*_args, **_kwargs):
        raise AssertionError("follower constructed a task sampler")

    monkeypatch.setattr(loader_module.dist, "get_world_size", lambda _group: 2)
    state = SimpleNamespace(
        config=SimpleNamespace(dp_size=2),
        dp_rank=0,
        batch_group=object(),
        is_batch_leader=False,
    )
    loader = _loader(DataSpec([1], sampler_factory=unexpected_factory), parallel_state=state)
    assert isinstance(loader, loader_module.LeaderBroadcastDataLoader)
    assert loader.sampler is None


@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("world_size,rank,total", [(1, 0, 97), (2, 1, 96)])
def test_multi_source_loader_restores_committed_batches_with_prefetch(
    num_workers, world_size, rank, total
):
    options = {
        "parallel_state": _parallel(rank, world_size),
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers else None,
    }
    spec = _spec(total=total)
    first = _loader(spec, **options)
    resumed = None
    try:
        assert isinstance(first.sampler, MultiSourceSampler)
        assert first.sampler.shuffle_seed == 11
        assert len(first) == math_ceil_div(total // world_size, 8)
        reference = list(
            MultiSourceSampler(
                spec.dataset,
                total_samples=total,
                mix_seed=73,
                shuffle_seed=11,
                dp_rank=rank,
                dp_world_size=world_size,
                batch_size=8,
            )
        )
        iterator = iter(first)
        consumed = [next(iterator) for _ in range(3)]
        assert [item for batch in consumed for item in batch] == reference[:24]
        state = copy.deepcopy(first.state_dict())
        expected = list(iterator)
        assert [item for batch in expected for item in batch] == reference[24:]

        resumed = _loader(_spec(total=total), **options)
        resumed.load_state_dict(state)
        assert list(resumed) == expected
    finally:
        _shutdown(first)
        if resumed is not None:
            _shutdown(resumed)


def math_ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


@pytest.mark.parametrize("change", ["version", "total", "batch_size", "seed", "mix_seed"])
def test_loader_resume_refuses_changed_sampling_contract(change) -> None:
    first = _loader(_spec())
    iterator = iter(first)
    next(iterator)
    state = copy.deepcopy(first.state_dict())
    spec_kwargs, loader_kwargs = {}, {}
    if change == "version":
        spec_kwargs["version"] = "ordered-v2"
    elif change == "total":
        spec_kwargs["total"] = 101
    elif change == "batch_size":
        loader_kwargs["batch_size"] = 4
    elif change == "seed":
        loader_kwargs["seed"] = 12
    spec = _spec(**spec_kwargs)
    if change == "mix_seed":
        spec = DataSpec(
            spec.dataset,
            sampler_factory=partial(build_multi_source_sampler, total_samples=97, mix_seed=74),
            collate_fn=_collate_items,
        )
    changed = _loader(spec, **loader_kwargs)
    with pytest.raises((ValueError, RuntimeError)):
        changed.load_state_dict(state)
        next(iter(changed))


def test_data_spec_multi_source_dp_batches_reconstruct_global_dataset_addresses() -> None:
    spec = _spec(total=72)
    per_rank = []
    for rank in range(3):
        loader = _loader(spec, parallel_state=_parallel(rank, 3), batch_size=2)
        assert isinstance(loader.sampler, MultiSourceSampler)
        assert len(loader) == 12
        per_rank.append(list(loader))
    reconstructed = [
        address for batch in range(12) for rank in range(3) for address in per_rank[rank][batch]
    ]
    assert reconstructed == list(
        MultiSourceSampler(spec.dataset, total_samples=72, mix_seed=73, shuffle_seed=11)
    )


def test_multi_source_factory_rejects_shuffle_off_and_discarded_plan_tail() -> None:
    with pytest.raises(ValueError, match="shuffle"):
        _loader(_spec(total=64), shuffle=False)
    with pytest.raises(ValueError, match="drop_last"):
        _loader(_spec(total=65), drop_last=True)
    exact = _loader(_spec(total=64), drop_last=True)
    assert sum(len(batch) for batch in exact) == 64
