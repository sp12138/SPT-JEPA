"""Committed cursor roundtrips for torchdata StatefulDataLoader."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch.utils.data import Dataset, DistributedSampler, TensorDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from vforge.data.loader import is_resumable_dataloader
from vforge.data.mixed_dataset import MixedDataset, MixedSampler
from vforge.data.multi_stream_loader import MultiStreamStatefulDataLoader


def _loader(*, num_workers: int = 0) -> StatefulDataLoader:
    return StatefulDataLoader(
        TensorDataset(torch.arange(16)),
        batch_size=2,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def test_stateful_loader_restores_committed_cursor() -> None:
    first = _loader()
    iterator = iter(first)
    assert torch.equal(next(iterator)[0], torch.tensor([0, 1]))
    assert torch.equal(next(iterator)[0], torch.tensor([2, 3]))

    resumed = _loader()
    resumed.load_state_dict(first.state_dict())

    assert torch.equal(next(iter(resumed))[0], torch.tensor([4, 5]))


def test_stateful_loader_worker_prefetch_roundtrip() -> None:
    first = _loader(num_workers=2)
    iterator = iter(first)
    assert torch.equal(next(iterator)[0], torch.tensor([0, 1]))
    assert torch.equal(next(iterator)[0], torch.tensor([2, 3]))

    resumed = _loader(num_workers=2)
    resumed.load_state_dict(first.state_dict())

    assert torch.equal(next(iter(resumed))[0], torch.tensor([4, 5]))


def test_native_stateful_loader_is_detected_for_direct_resume() -> None:
    loader = StatefulDataLoader(
        TensorDataset(torch.arange(4)),
        batch_size=2,
        shuffle=False,
    )

    assert is_resumable_dataloader(loader)


class _ScalarItems(Dataset):
    def __init__(self, length: int) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(int(index))


def _mixed_loader() -> StatefulDataLoader:
    mixed = MixedDataset(
        [_ScalarItems(16)],
        weights=[1.0],
        mix_seed=1,
        steps_per_epoch=16,
    )
    sampler = MixedSampler(
        mixed,
        dp_rank=0,
        dp_world_size=1,
        batch_size=2,
        shuffle=False,
        seed=0,
    )
    return StatefulDataLoader(mixed, batch_size=2, sampler=sampler, drop_last=True)


def test_mixed_sampler_state_dict_roundtrip() -> None:
    first = _mixed_loader()
    iterator = iter(first)
    assert torch.equal(next(iterator), torch.tensor([0, 1]))
    assert torch.equal(next(iterator), torch.tensor([2, 3]))

    resumed = _mixed_loader()
    resumed.load_state_dict(first.state_dict())
    assert torch.equal(next(iter(resumed)), torch.tensor([4, 5]))


def _multi_stream_loader(sampler_type):
    loaders = {}
    for name, length in [("generator", 3), ("critic", 5)]:
        dataset = list(range(length))
        sampler = sampler_type(dataset, num_replicas=1, rank=0, seed=31)
        loaders[name] = StatefulDataLoader(dataset, sampler=sampler, batch_size=None)
    return MultiStreamStatefulDataLoader(loaders)


@pytest.mark.parametrize("sampler_type", [DistributedSampler, StatefulDistributedSampler])
@pytest.mark.parametrize("steps", [3, 8])
def test_multi_stream_restores_independent_epochs_and_next_batches(steps, sampler_type):
    live = _multi_stream_loader(sampler_type)
    iterator = iter(live)
    for _ in range(steps):
        next(iterator)
    saved = deepcopy(live.state_dict())
    expected = [next(iterator) for _ in range(12)]
    restored = _multi_stream_loader(sampler_type)
    restored.load_state_dict(saved)
    resumed = iter(restored)
    assert [next(resumed) for _ in range(12)] == expected
    assert restored.state_dict()["epochs"] == live.state_dict()["epochs"]
    assert restored.state_dict()["batches"] == live.state_dict()["batches"]
