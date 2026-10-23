"""Mathematical, replay and DP contracts for the finite multi-source sampler."""

from __future__ import annotations

import copy
import json
import math
from collections import Counter
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction
from itertools import islice

import pytest
import torch
from torch.utils.data import Dataset

from vforge.data.multi_source import DatasetSource, MultiDataset, MultiSourceSampler


class _Items(Dataset):
    def __init__(self, name: str, size: int) -> None:
        self.items = [{"source": name, "index": index} for index in range(size)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.items[index]


def _dataset(weights=(11, 17, 23), sizes=None) -> MultiDataset:
    sizes = sizes or tuple(3 + 2 * i for i in range(len(weights)))
    return MultiDataset(
        tuple(
            DatasetSource(f"S{i}", _Items(f"S{i}", size), "ordered-v1", weight)
            for i, (weight, size) in enumerate(zip(weights, sizes, strict=True))
        )
    )


def _sampler(dataset=None, **overrides) -> MultiSourceSampler:
    kwargs = {"total_samples": 257, "mix_seed": 7, "shuffle_seed": 42}
    kwargs.update(overrides)
    return MultiSourceSampler(dataset if dataset is not None else _dataset(), **kwargs)


def _allocation(weights, total) -> tuple[int, ...]:
    """Exact rational apportionment oracle, independent of sampler internals."""
    ideals = [Fraction(total * weight, sum(weights)) for weight in weights]
    counts = [math.floor(ideal) for ideal in ideals]
    order = sorted(range(len(weights)), key=lambda i: (ideals[i] - counts[i], -i), reverse=True)
    for i in order[: total - sum(counts)]:
        counts[i] += 1
    return tuple(counts)


def _counts(addresses, source_count) -> tuple[int, ...]:
    counts = Counter(source_id for source_id, _index in addresses)
    return tuple(counts[f"S{i}"] for i in range(source_count))


def test_multi_dataset_routes_explicit_addresses_without_modulo() -> None:
    dataset = _dataset((1, 2), (2, 3))
    assert len(dataset) == 5
    assert dataset[("S0", 1)] is dataset.sources[0].dataset[1]
    assert dataset[("S1", 2)] == {"source": "S1", "index": 2}
    with pytest.raises(FrozenInstanceError):
        dataset.sources[0].weight = 99

    for address in (1, ["S0", 0], ("S0",), ("S0", 0, 1), (0, 0), ("S0", True)):
        with pytest.raises(TypeError):
            dataset[address]
    with pytest.raises(KeyError):
        dataset[("unknown", 0)]
    for index in (-1, 2, 100):
        with pytest.raises(IndexError):
            dataset[("S0", index)]


@pytest.mark.parametrize("weight", [-1, 1.5, True])
def test_source_rejects_invalid_integer_weight(weight) -> None:
    with pytest.raises(ValueError):
        DatasetSource("S0", _Items("S0", 2), "v1", weight)


def test_source_definitions_require_versions_unique_names_and_positive_total() -> None:
    for field in ("source_id", "version"):
        values = {"source_id": "S0", "dataset": _Items("S0", 2), "version": "v1", "weight": 1}
        values[field] = ""
        with pytest.raises(ValueError):
            DatasetSource(**values)
    with pytest.raises(ValueError):
        DatasetSource("S0", _Items("S0", 0), "v1", 1)
    source = DatasetSource("S0", _Items("S0", 2), "v1", 1)
    with pytest.raises(ValueError):
        MultiDataset((source, source))
    with pytest.raises(ValueError):
        _dataset((0, 0))


@pytest.mark.parametrize(
    "weights,total,multiplier",
    [
        ((1, 2), 8, 1),
        ((11, 12, 17, 75, 63), 201, 1),
        ((47, 53), 200, 1),
        ((0, 1, 0, 2), 2, 1),
        ((1, 1, 1), 2, 1),
        ((0, 1, 1), 1, 1),
        ((2, 4), 17, 3),
        ((1, 0, 0), 7, 1),
        ((0, 11, 17, 0), 31, 2),
        ((3, 7, 11), 73, 3),
    ],
)
def test_blocks_and_corrected_tail_match_independent_allocation(weights, total, multiplier):
    sampler = _sampler(_dataset(weights), total_samples=total, block_multiplier=multiplier)
    addresses = list(sampler)
    quotas = tuple(weight // math.gcd(*weights) * multiplier for weight in weights)
    length = sum(quotas)
    complete, tail_size = divmod(total, length)

    assert len(addresses) == len(sampler) == total
    assert sampler.block_size == length
    assert sampler.tail_quotas == _allocation(weights, tail_size)
    assert sampler.plan_quotas == _allocation(weights, total)
    assert _counts(addresses, len(weights)) == sampler.plan_quotas
    for start in range(0, complete * length, length):
        assert _counts(addresses[start : start + length], len(weights)) == quotas
    assert _counts(addresses[complete * length :], len(weights)) == sampler.tail_quotas
    for count, weight in zip(sampler.plan_quotas, weights, strict=True):
        assert abs(count - Fraction(total * weight, sum(weights))) < 1


def test_source_epochs_are_permutations_and_can_repeat_across_epochs() -> None:
    dataset = _dataset((2, 3), (2, 5))
    addresses = list(_sampler(dataset, total_samples=103))
    for source in dataset.sources:
        indices = [index for source_id, index in addresses if source_id == source.source_id]
        size = len(source.dataset)
        complete, tail = divmod(len(indices), size)
        assert complete >= 2
        for epoch in range(complete):
            assert sorted(indices[epoch * size : (epoch + 1) * size]) == list(range(size))
        assert len(set(indices[complete * size :])) == tail
        assert len(set(indices)) < len(indices)


def test_iterator_replays_full_addresses_and_keeps_independent_cursors() -> None:
    sampler = _sampler()
    reference = list(sampler)
    iterator = iter(sampler)
    assert list(islice(iterator, 73)) == reference[:73]
    assert iterator.position == 73
    assert iterator.consumed == _counts(reference[:73], 3)
    saved = json.loads(json.dumps(iterator.state_dict()))
    assert list(islice(iterator, 31)) == reference[73:104]

    restored = iter(_sampler())
    restored.load_state_dict(saved)
    assert list(restored) == reference[73:]
    assert restored.position == sampler.total_samples
    assert restored.consumed == sampler.plan_quotas
    with pytest.raises(StopIteration):
        next(restored)
    restored.seek(0)
    assert list(islice(restored, 9)) == reference[:9]
    assert list(sampler) == reference
    assert iterator.position == 104


def test_two_shuffle_layers_are_independent_of_global_torch_rng() -> None:
    dataset = _dataset((2, 3), (7, 11))
    torch.manual_seed(100)
    rng_before = torch.get_rng_state().clone()
    first = list(_sampler(dataset, total_samples=153))
    assert torch.equal(torch.get_rng_state(), rng_before)
    torch.manual_seed(999)
    torch.rand(37)
    assert list(_sampler(dataset, total_samples=153)) == first

    changed_shuffle = list(_sampler(dataset, total_samples=153, shuffle_seed=43))
    assert [source for source, _ in changed_shuffle] == [source for source, _ in first]
    assert changed_shuffle != first
    changed_sizes = list(_sampler(_dataset((2, 3), (13, 17)), total_samples=153))
    assert [source for source, _ in changed_sizes] == [source for source, _ in first]


@pytest.mark.parametrize(
    "override",
    [
        {"total_samples": 100},
        {"mix_seed": 8},
        {"shuffle_seed": 43},
        {"stage_id": "another-stage"},
        {"block_multiplier": 2},
        {"dp_rank": 1},
        {"dp_world_size": 3},
        {"batch_size": 4},
    ],
)
def test_resume_refuses_configuration_or_topology_changes(override) -> None:
    config = {"total_samples": 96, "batch_size": 2, "dp_world_size": 2}
    first = iter(_sampler(**config))
    list(islice(first, 3))
    resumed = iter(_sampler(**(config | override)))
    before = copy.deepcopy(resumed.state_dict())
    with pytest.raises(ValueError):
        resumed.load_state_dict(first.state_dict())
    assert resumed.state_dict() == before


@pytest.mark.parametrize("change", ["version", "size", "order", "weight", "runtime"])
def test_resume_validates_dataset_identity_and_runtime(change) -> None:
    dataset = _dataset((1, 2), (3, 5))
    saved = iter(_sampler(dataset)).state_dict()
    sources = list(dataset.sources)
    if change == "version":
        sources[0] = replace(sources[0], version="ordered-v2")
    elif change == "size":
        sources[0] = replace(sources[0], dataset=_Items("S0", 4))
    elif change == "order":
        sources.reverse()
    elif change == "weight":
        sources[0] = replace(sources[0], weight=2)
    else:
        saved["contract"]["runtime"]["torch"] = "different-build"
    resumed = iter(_sampler(MultiDataset(sources)))
    with pytest.raises(ValueError):
        resumed.load_state_dict(saved)


@pytest.mark.parametrize("position", [-1, True, 258])
def test_invalid_restore_position_leaves_live_cursor_unchanged(position) -> None:
    iterator = iter(_sampler())
    list(islice(iterator, 5))
    before = iterator.state_dict()
    invalid = copy.deepcopy(before)
    invalid["position"] = position
    with pytest.raises(ValueError):
        iterator.load_state_dict(invalid)
    assert iterator.state_dict() == before


def test_huge_plan_seek_and_restore_allocate_only_one_small_group(monkeypatch) -> None:
    sizes = []
    original = torch.randperm

    def bounded_randperm(size, *args, **kwargs):
        sizes.append(size)
        assert size <= 8, "seek attempted to allocate the complete historical plan"
        assert len(sizes) <= 12, "seek scanned historical quota groups"
        return original(size, *args, **kwargs)

    monkeypatch.setattr(torch, "randperm", bounded_randperm)
    position = 10**12
    sampler = _sampler(_dataset((3, 5), (3, 5)), total_samples=position + 7)
    iterator = iter(sampler)
    iterator.seek(position)
    assert iterator.position == position
    assert iterator.consumed == (position // 8 * 3, position // 8 * 5)
    assert sizes == []
    saved = iterator.state_dict()
    expected = list(islice(iterator, 7))
    restored = iter(sampler)
    restored.load_state_dict(saved)
    assert list(islice(restored, 7)) == expected
    assert restored.position == sampler.total_samples


@pytest.mark.parametrize("world_size,batch_size", [(2, 1), (2, 3), (3, 2), (4, 3)])
def test_dp_rank_batches_reassemble_the_unchanged_global_plan(world_size, batch_size) -> None:
    dataset = _dataset((2, 3), (2, 3))
    total = 120
    global_addresses = list(_sampler(dataset, total_samples=total))
    shards = []
    for rank in range(world_size):
        sampler = _sampler(
            dataset,
            total_samples=total,
            dp_world_size=world_size,
            dp_rank=rank,
            batch_size=batch_size,
        )
        shard = list(sampler)
        expected = [
            global_addresses[position]
            for start in range(0, total, batch_size * world_size)
            for position in range(start + rank * batch_size, start + (rank + 1) * batch_size)
        ]
        assert shard == expected
        assert len(shard) == len(sampler) == total // world_size
        iterator = iter(sampler)
        list(islice(iterator, 5))
        resumed = iter(sampler)
        resumed.load_state_dict(iterator.state_dict())
        assert list(resumed) == shard[5:]
        shards.append(shard)

    reconstructed = [
        address
        for batch in range(total // (batch_size * world_size))
        for rank in range(world_size)
        for address in shards[rank][batch * batch_size : (batch + 1) * batch_size]
    ]
    assert reconstructed == global_addresses


def test_dp_does_not_deduplicate_valid_cross_epoch_physical_samples() -> None:
    dataset = _dataset((1,), (1,))
    for rank in range(3):
        sampler = _sampler(dataset, total_samples=12, dp_world_size=3, dp_rank=rank, batch_size=2)
        assert list(sampler) == [("S0", 0)] * 4


def test_dp_refuses_partial_global_batch_and_nonowned_seek() -> None:
    with pytest.raises(ValueError):
        _sampler(total_samples=11, dp_world_size=2, batch_size=3)
    iterator = iter(_sampler(total_samples=12, dp_world_size=2, dp_rank=1, batch_size=3))
    before = iterator.state_dict()
    with pytest.raises(ValueError):
        iterator.seek(0)
    assert iterator.state_dict() == before
    iterator.seek(12)
    assert list(iterator) == []
