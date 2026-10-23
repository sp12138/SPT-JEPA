"""Worker RNG must resume at the fetch+collate snapshot despite prefetch."""

from __future__ import annotations

import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

from vforge.data.loader import build_resumable_loader
from vforge.models.gen_engine.codec import DiagonalGaussianSampling


class RandomItems(Dataset):
    def __init__(self):
        self.reads = 0

    def __len__(self):
        return 80

    def __getitem__(self, index):
        self.reads += 1
        return (index, self.reads, torch.rand(()).item(), random.random(), np.random.rand())

    def __getitems__(self, indices):
        return [(*self[index], "batched") for index in indices]

    def state_dict(self):
        return {"reads": self.reads}

    def load_state_dict(self, state):
        self.reads = state["reads"]


class RandomCollate:
    def __init__(self):
        self.calls = 0

    def __call__(self, items):
        self.calls += 1
        # Exercise the unchanged production posterior sampler inside collate.
        noise = DiagonalGaussianSampling().sample(
            torch.zeros(2, 1, 2, 2), channel_dim=0, latent_channels=1
        )
        return (items, self.calls, noise.flatten().tolist(), random.random(), np.random.rand())

    def state_dict(self):
        return {"calls": self.calls}

    def load_state_dict(self, state):
        self.calls = state["calls"]


def make_loader(workers=2, persistent=True, prefetch=2, rank=0):
    return build_resumable_loader(
        RandomItems(),
        parallel_state=SimpleNamespace(
            config=SimpleNamespace(dp_size=2),
            dp_rank=rank,
            batch_group=None,
            is_batch_leader=True,
        ),
        batch_size=4,
        num_workers=workers,
        drop_last=True,
        seed=42,
        collate_fn=RandomCollate(),
        persistent_workers=persistent,
        prefetch_factor=prefetch,
    )


def shutdown(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()


@pytest.mark.parametrize(
    "workers,persistent,prefetch,cut,interval,rank",
    [
        (1, False, 1, 0, 1, 0),
        (2, False, 2, 1, 1, 1),
        (2, True, 2, 3, 1, 0),
        (2, True, 1, 5, 4, 1),
        (2, True, 2, 10, 1, 0),
    ],
)
def test_worker_rng_replay_and_resume(workers, persistent, prefetch, cut, interval, rank):
    loaders = []
    try:
        first = make_loader(workers, persistent, prefetch, rank)
        loaders.append(first)
        first.snapshot_every_n_steps = interval
        iterator = iter(first)
        prefix = [next(iterator) for _ in range(cut)]
        saved = copy.deepcopy(first.state_dict())
        suffix = list(iterator)
        shutdown(first)

        # A fresh loader must not depend on parent RNG state.
        torch.manual_seed(991)
        random.seed(733)
        np.random.seed(127)
        replay = make_loader(workers, persistent, prefetch, rank)
        loaders.append(replay)
        assert list(replay) == prefix + suffix
        shutdown(replay)

        restored = make_loader(workers, persistent, prefetch, rank)
        loaders.append(restored)
        restored.snapshot_every_n_steps = interval
        restored.load_state_dict(saved)
        assert list(restored) == suffix
    finally:
        for loader in loaders:
            shutdown(loader)


def test_worker_resume_rejects_missing_state_or_changed_workers():
    first = make_loader()
    try:
        saved = copy.deepcopy(first.state_dict())
    finally:
        shutdown(first)
    legacy = {k: v for k, v in saved.items() if k != "_vforge_worker_rng"}
    with pytest.raises(ValueError, match="worker RNG checkpoint"):
        make_loader().load_state_dict(legacy)
    with pytest.raises(ValueError, match="worker RNG checkpoint"):
        make_loader(workers=1).load_state_dict(saved)


def test_worker_rng_continues_across_persistent_epochs():
    first, restored = make_loader(), make_loader()
    try:
        list(first)
        first.sampler.set_epoch(1)
        iterator = iter(first)
        next(iterator)
        saved = copy.deepcopy(first.state_dict())
        expected = list(iterator)
        restored.sampler.set_epoch(1)
        restored.load_state_dict(saved)
        assert list(restored) == expected
    finally:
        shutdown(first)
        shutdown(restored)


def test_worker_wrapper_preserves_native_full_run_and_supports_spawn():
    # Match the rank-zero, DP=2 address sequence explicitly in the native loader.
    wrapped = make_loader(persistent=False)
    native = StatefulDataLoader(
        RandomItems(),
        sampler=wrapped.sampler,
        batch_size=4,
        num_workers=2,
        collate_fn=RandomCollate(),
        generator=torch.Generator().manual_seed(42),
    )
    wrapped.multiprocessing_context = "spawn"
    resumed = make_loader(persistent=False)
    resumed.multiprocessing_context = "spawn"
    try:
        expected = list(native)
        iterator = iter(wrapped)
        prefix = [next(iterator) for _ in range(3)]
        saved = copy.deepcopy(wrapped.state_dict())
        assert prefix + list(iterator) == expected
        resumed.load_state_dict(saved)
        assert list(resumed) == expected[3:]
    finally:
        for loader in (native, wrapped, resumed):
            shutdown(loader)
