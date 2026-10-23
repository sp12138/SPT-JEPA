"""Stateful lockstep loading for named training streams."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class MultiStreamStatefulDataLoader:
    """Cycle resumable named loaders and advance each once per returned batch.

    Streams own independent epochs; the infinite outer iterator intentionally
    exposes no sampler for TrainingDataLoader to reset.
    """

    _vforge_resumable_dataloader = True
    _vforge_bit_exact_resume = True

    def __init__(self, loaders: Mapping[str, Any]) -> None:
        if not loaders:
            raise ValueError("multi-stream loader requires at least one stream")
        for name, loader in loaders.items():
            if not isinstance(name, str) or not name:
                raise ValueError("stream names must be non-empty strings")
            if not callable(getattr(loader, "state_dict", None)):
                raise TypeError(f"{name} loader must provide state_dict()")
            if not callable(getattr(loader, "load_state_dict", None)):
                raise TypeError(f"{name} loader must provide load_state_dict()")
        self.loaders = dict(loaders)
        self._iterators: dict[str, Any] = {}
        self._epochs = dict.fromkeys(self.loaders, 0)
        self._batches = dict.fromkeys(self.loaders, 0)

    def set_epoch(self, epoch: int) -> None:
        for name, loader in self.loaders.items():
            self._epochs[name] = int(epoch)
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and callable(getattr(sampler, "set_epoch", None)):
                sampler.set_epoch(int(epoch))

    def __iter__(self) -> MultiStreamStatefulDataLoader:
        for name, loader in self.loaders.items():
            if name not in self._iterators:
                self._iterators[name] = iter(loader)
        return self

    def __next__(self) -> dict[str, Any]:
        return {name: self._next_stream(name) for name in self.loaders}

    def state_dict(self) -> dict[str, Any]:
        states = {name: loader.state_dict() for name, loader in self.loaders.items()}
        return {
            "version": 1,
            "epochs": dict(self._epochs),
            "batches": dict(self._batches),
            "streams": states,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if int(state_dict.get("version", 0)) != 1:
            raise ValueError("unsupported multi-stream dataloader state version")
        states = state_dict.get("streams")
        if states is None and all(name in state_dict for name in self.loaders):
            states = {name: state_dict[name] for name in self.loaders}
        if not isinstance(states, Mapping) or set(states) != set(self.loaders):
            raise ValueError("multi-stream dataloader state streams must match configured streams")
        self._epochs = self._validated_counters(state_dict["epochs"], "epochs")
        self._batches = self._validated_counters(state_dict["batches"], "batches")
        for name, loader in self.loaders.items():
            # Stateless samplers need the saved epoch to reconstruct their order.
            # Set it before restoring a stateful sampler's committed cursor.
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and callable(getattr(sampler, "set_epoch", None)):
                sampler.set_epoch(self._epochs[name])
            loader.load_state_dict(states[name])
        self._iterators.clear()

    def _next_stream(self, name: str) -> Any:
        loader = self.loaders[name]
        iterator = self._iterators.get(name)
        if iterator is None:
            iterator = iter(loader)
            self._iterators[name] = iterator
        try:
            batch = next(iterator)
        except StopIteration:
            self._epochs[name] += 1
            sampler = getattr(loader, "sampler", None)
            if sampler is not None and callable(getattr(sampler, "set_epoch", None)):
                sampler.set_epoch(self._epochs[name])
            iterator = iter(loader)
            self._iterators[name] = iterator
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise RuntimeError(f"{name} loader is empty and cannot be cycled") from error
        self._batches[name] += 1
        return batch

    def _validated_counters(self, value: Any, name: str) -> dict[str, int]:
        if not isinstance(value, Mapping) or set(value) != set(self.loaders):
            raise ValueError(f"multi-stream dataloader {name} must match configured streams")
        counters = {key: int(item) for key, item in value.items()}
        if any(item < 0 for item in counters.values()):
            raise ValueError(f"multi-stream dataloader {name} must be non-negative")
        return counters


__all__ = ["MultiStreamStatefulDataLoader"]
