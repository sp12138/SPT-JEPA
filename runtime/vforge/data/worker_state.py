"""Worker CPU RNG snapshots integrated with torchdata's dataset state hooks.

Torchdata snapshots the dataset after fetch (including collate), and restores
it before replaying pending work. Keep RNG state at that same boundary rather
than reading a worker's live, potentially prefetched state from the parent.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.stateful import Stateful


class WorkerStateDataset(Dataset):
    """Transparent map-style reads with dataset/collate state and CPU RNG hooks."""

    def __init__(self, dataset: Any, collate_fn: Any) -> None:
        self.dataset = dataset
        self.collate_fn = collate_fn

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: Any) -> Any:
        return self.dataset[index]

    def __getitems__(self, indices: list[Any]) -> Any:
        batched = getattr(self.dataset, "__getitems__", None)
        if callable(batched):
            return batched(indices)
        return [self.dataset[index] for index in indices]

    def state_dict(self) -> dict[str, Any]:
        if get_worker_info() is None:
            raise RuntimeError("worker RNG snapshots must be taken inside the worker")
        # Copy component state before capturing RNG: state hooks must not leave
        # references that subsequent prefetches can mutate.
        components = {
            name: copy.deepcopy(obj.state_dict()) if isinstance(obj, Stateful) else None
            for name, obj in (("dataset", self.dataset), ("collate", self.collate_fn))
        }
        numpy_state = np.random.get_state()
        return {
            "schema": 1,
            **components,
            "torch_cpu": torch.get_rng_state().clone(),
            "python": random.getstate(),
            # Lists allow torchdata's incremental-state equality comparisons.
            "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if get_worker_info() is None or state.get("schema") != 1:
            raise ValueError("invalid worker RNG snapshot")
        for name, obj in (("dataset", self.dataset), ("collate", self.collate_fn)):
            if isinstance(obj, Stateful):
                obj.load_state_dict(state[name])
            elif state[name] is not None:
                raise ValueError(f"worker {name} no longer supports state restoration")
        # Restore last, so component restoration cannot advance the saved RNG.
        random.setstate(state["python"])
        numpy_state = state["numpy"]
        np.random.set_state(
            (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
        )
        torch.set_rng_state(state["torch_cpu"])


class WorkerStatefulDataLoader(StatefulDataLoader):
    """Version the added worker state without replacing torchdata iteration."""

    _STATE_KEY = "_vforge_worker_rng"

    def _worker_contract(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "seed": self.generator.initial_seed(),
        }

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), self._STATE_KEY: self._worker_contract()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict and state_dict.get(self._STATE_KEY) != self._worker_contract():
            raise ValueError(
                "worker RNG checkpoint contract mismatch or missing; use the same worker "
                "configuration and a checkpoint saved with worker RNG support"
            )
        native_state = {key: value for key, value in state_dict.items() if key != self._STATE_KEY}
        super().load_state_dict(native_state)
