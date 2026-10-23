"""Task-assembled data components consumed by the shared loader builder.

A DataSpec contains live objects, not YAML or a second configuration schema.
The task owns the sample contract and collate function. The loader supplies the
DP coordinates when it calls the sampler factory, so sampling is sharded once.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from torch.utils.data import Dataset, Sampler


class SamplerFactory(Protocol):
    """Create the sampler for one DP rank; the result already owns sharding."""

    def __call__(
        self,
        dataset: Dataset,
        *,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        seed: int,
        shuffle: bool,
        drop_last: bool,
    ) -> Sampler: ...


@dataclass(frozen=True)
class DataSpec:
    """A dataset, its optional sampler factory, and its batch collate function.

    ``dataset`` is the single indexable object presented to DataLoader; for
    multiple sources that object is a MultiDataset. A missing sampler factory
    preserves the shared builder's DistributedSampler behavior. Worker count,
    prefetch and batch size remain arguments to the loader builder.

    Custom sampler factories and collate functions should be module-level
    callables (or picklable partials) when used with worker processes.
    """

    dataset: Dataset
    sampler_factory: SamplerFactory | None = None
    collate_fn: Callable[[list[Any]], Any] | None = None

    def __post_init__(self) -> None:
        if not hasattr(self.dataset, "__len__") or not hasattr(self.dataset, "__getitem__"):
            raise TypeError("DataSpec.dataset must support len() and indexed reads")
        if self.sampler_factory is not None and not callable(self.sampler_factory):
            raise TypeError("DataSpec.sampler_factory must be callable")
        if self.collate_fn is not None and not callable(self.collate_fn):
            raise TypeError("DataSpec.collate_fn must be callable")


__all__ = ["DataSpec", "SamplerFactory"]
