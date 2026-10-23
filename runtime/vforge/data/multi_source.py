"""Finite multi-source sampling with quota blocks and source-local shuffles.

The global plan has T samples. GCD-normalized integer weights times a block
multiplier give quotas q and L=sum(q). Each complete block shuffles L tickets;
only the final T % L tickets use largest-remainder quotas. Seed encodings match
the quota/tail sampling experiments. Source-local epoch permutations depend on
shuffle_seed, source identity/version/size and epoch, never DP or batch size.

Selection caches retain one block per iterator, bounded by MAX_BLOCK_SIZE.
Each visited source additionally retains one CPU int64 permutation of its full
size; that memory is NOT bounded by MAX_BLOCK_SIZE. Prefix reconstruction uses
O(L + K) work and temporary memory for K sources, without scanning old blocks.
Dataset contents and ordered index mappings must remain immutable for a version.
New source epochs are reshuffled, but distinct epochs can produce the same order.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from bisect import bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset, Sampler

ALGORITHM = "multi-source-quota-blocks-largest-remainder-tail-cpu-randperm-v1"
MAX_BLOCK_SIZE = 1_000_000
SampleAddress = tuple[str, int]


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _u64(domain: str, *parts: object) -> int:
    # Frozen seed protocol: compact ASCII-escaped JSON array, UTF-8, then the
    # first eight SHA-256 digest bytes interpreted as an unsigned big-endian int.
    encoded = json.dumps([domain, *parts], ensure_ascii=True, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(encoded.encode("utf-8")).digest()[:8], "big")


def _randperm(size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(size, generator=generator, device="cpu", dtype=torch.int64)


@dataclass(frozen=True)
class DatasetSource:
    """A stable source identity and ordered map-style dataset, with integer weight."""

    source_id: str
    dataset: Dataset
    version: str
    weight: int

    def __post_init__(self) -> None:
        _nonempty_string(self.source_id, "source_id")
        _nonempty_string(self.version, "version")
        _integer(self.weight, "weight")
        if not isinstance(self.dataset, Dataset) or isinstance(self.dataset, IterableDataset):
            raise TypeError("source dataset must be a map-style torch Dataset")
        try:
            size = len(self.dataset)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("source dataset must have a finite positive length") from error
        _integer(size, "source size", minimum=1)


class MultiDataset(Dataset):
    """Route explicit (source_id, local_index) addresses without changing samples.

    len(dataset) is the sum of physical source lengths, not the training budget.
    Repetition and source-local epochs belong to MultiSourceSampler. Invalid
    addresses are rejected rather than wrapped, retried, or replaced.
    """

    def __init__(self, sources: Sequence[DatasetSource]) -> None:
        self.sources = tuple(sources)
        if not self.sources:
            raise ValueError("at least one source is required")
        if any(not isinstance(source, DatasetSource) for source in self.sources):
            raise TypeError("sources must contain DatasetSource objects")
        identifiers = [source.source_id for source in self.sources]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("source_id values must be unique")
        if not any(source.weight > 0 for source in self.sources):
            raise ValueError("at least one source weight must be positive")
        self.sizes = tuple(len(source.dataset) for source in self.sources)
        for size in self.sizes:
            _integer(size, "source size", minimum=1)
        self._source_indices = {name: index for index, name in enumerate(identifiers)}

    def __len__(self) -> int:
        return sum(self.sizes)

    def __getitem__(self, address: SampleAddress) -> Any:
        if not isinstance(address, tuple) or len(address) != 2:
            raise TypeError("sample address must be (source_id, local_index)")
        source_id, local_index = address
        if not isinstance(source_id, str) or type(local_index) is not int:
            raise TypeError("sample address requires a string source_id and integer local_index")
        source_index = self._source_indices[source_id]
        if not 0 <= local_index < self.sizes[source_index]:
            raise IndexError(f"local_index {local_index} is out of range for source {source_id!r}")
        return self.sources[source_index].dataset[local_index]

    def resume_contract(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "sources": [
                {
                    "source_id": source.source_id,
                    "version": source.version,
                    "size": size,
                    "weight": source.weight,
                }
                for source, size in zip(self.sources, self.sizes, strict=True)
            ],
        }


class _QuotaPlan:
    """Regenerable source-selection plan; never allocate total_samples tickets."""

    def __init__(self, weights: tuple[int, ...], mix_seed: int, total: int, multiplier: int):
        divisor = math.gcd(*weights)
        self.weights = tuple(weight // divisor for weight in weights)
        self.mix_seed, self.total, self.multiplier = mix_seed, total, multiplier
        self.quotas = tuple(weight * multiplier for weight in self.weights)
        self.block_size = sum(self.quotas)
        if self.block_size > MAX_BLOCK_SIZE:
            raise ValueError(
                f"normalized quota block size {self.block_size} exceeds "
                f"MAX_BLOCK_SIZE={MAX_BLOCK_SIZE}; weights are not approximated"
            )
        self.full_blocks, self.tail_size = divmod(total, self.block_size)
        self.tail_start = self.full_blocks * self.block_size
        total_weight = sum(self.weights)
        divisions = [divmod(self.tail_size * weight, total_weight) for weight in self.weights]
        tail = [floor for floor, _ in divisions]
        order = sorted(
            (i for i, weight in enumerate(self.weights) if weight > 0),
            key=lambda i: (-divisions[i][1], i),
        )
        for index in order[: self.tail_size - sum(tail)]:
            tail[index] += 1
        self.tail_quotas = tuple(tail)
        self.plan_quotas = tuple(
            self.full_blocks * quota + last
            for quota, last in zip(self.quotas, self.tail_quotas, strict=True)
        )
        self.ends = tuple(accumulate(self.quotas))
        self.tail_ends = tuple(accumulate(self.tail_quotas))
        self._cache: tuple[int, torch.Tensor] | None = None

    def _permutation(self, block: int) -> torch.Tensor:
        if self._cache is None or self._cache[0] != block:
            if block < self.full_blocks:
                seed = _u64("mix-quota-v1", self.mix_seed, self.weights, self.multiplier, block)
                size = self.block_size
            else:
                seed = _u64(
                    "mix-quota-tail-v1",
                    self.mix_seed,
                    self.weights,
                    self.multiplier,
                    self.full_blocks,
                    self.tail_size,
                )
                size = self.tail_size
            self._cache = (block, _randperm(size, seed))
        return self._cache[1]

    def source_at(self, position: int) -> int:
        block, offset = divmod(position, self.block_size)
        ends = self.ends if position < self.tail_start else self.tail_ends
        return bisect_right(ends, int(self._permutation(block)[offset]))

    def prefix_counts(self, position: int) -> tuple[int, ...]:
        _integer(position, "position")
        if position > self.total:
            raise ValueError("position exceeds total_samples")
        if position == self.total:
            return self.plan_quotas
        block, offset = divmod(position, self.block_size)
        complete = tuple(block * quota for quota in self.quotas)
        if offset == 0:
            return complete
        permutation = self._permutation(block)
        ends = self.ends if position < self.tail_start else self.tail_ends
        # Count selected ticket numbers, then sum each source's ticket interval.
        # Unlike sorting/mapping all historical samples, this is O(L + K).
        histogram = torch.bincount(permutation[:offset], minlength=permutation.numel())
        cumulative = torch.cat((histogram.new_zeros(1), histogram.cumsum(0)))
        boundaries = torch.tensor((0, *ends), dtype=torch.int64, device="cpu")
        partial = torch.diff(cumulative[boundaries]).tolist()
        return tuple(a + b for a, b in zip(complete, partial, strict=True))


class MultiSourceSampler(Sampler[SampleAddress]):
    """A finite, replayable global sample plan, sliced into contiguous DP batches.

    For local ordinal n, B=local batch_size, and D=dp_world_size, a rank emits
    global position (n//B)*(B*D) + dp_rank*B + n%B. Thus changing B/D changes
    ownership only, not the global sequence. Checkpoints nevertheless require
    unchanged B/D/rank: topology migration needs an explicit caller policy.

    Every iter(sampler) owns independent cursor and permutation caches. Iterator
    snapshots track sampler production; StatefulDataLoader must snapshot them
    alongside its worker/prefetch state to recover trainer-visible progress.
    DP skips reconstruct prefix counts once per local batch in O(L + K) work;
    they do not load samples or build source permutations for other ranks.
    """

    def __init__(
        self,
        dataset: MultiDataset,
        *,
        total_samples: int,
        mix_seed: int,
        shuffle_seed: int,
        block_multiplier: int = 1,
        stage_id: str = "stage-0",
        dp_rank: int = 0,
        dp_world_size: int = 1,
        batch_size: int = 1,
    ) -> None:
        if not isinstance(dataset, MultiDataset):
            raise TypeError("MultiSourceSampler requires a MultiDataset")
        self.dataset = dataset
        self.total_samples = _integer(total_samples, "total_samples", minimum=1)
        self.mix_seed = _integer(mix_seed, "mix_seed")
        self.shuffle_seed = _integer(shuffle_seed, "shuffle_seed")
        self.block_multiplier = _integer(block_multiplier, "block_multiplier", minimum=1)
        self.stage_id = _nonempty_string(stage_id, "stage_id")
        self.dp_world_size = _integer(dp_world_size, "dp_world_size", minimum=1)
        self.dp_rank = _integer(dp_rank, "dp_rank")
        if self.dp_rank >= self.dp_world_size:
            raise ValueError("dp_rank must be smaller than dp_world_size")
        self.batch_size = _integer(batch_size, "batch_size", minimum=1)
        self.global_batch_size = self.batch_size * self.dp_world_size
        if self.dp_world_size > 1 and self.total_samples % self.global_batch_size:
            raise ValueError("DP requires total_samples divisible by the global batch size")
        self._plan = self._new_plan()
        self.normalized_weights = self._plan.weights
        self.quotas = self._plan.quotas
        self.block_size = self._plan.block_size
        self.tail_quotas = self._plan.tail_quotas
        self.plan_quotas = self._plan.plan_quotas
        self.tail_start = self._plan.tail_start

    def _new_plan(self) -> _QuotaPlan:
        return _QuotaPlan(
            tuple(source.weight for source in self.dataset.sources),
            self.mix_seed,
            self.total_samples,
            self.block_multiplier,
        )

    def __len__(self) -> int:
        return self.total_samples // self.dp_world_size

    def __iter__(self) -> _MultiSourceIterator:
        return _MultiSourceIterator(self)

    def set_epoch(self, epoch: int) -> None:
        if _integer(epoch, "epoch") != 0:
            raise ValueError(
                "finite multi-source plans cannot advance epoch; configure a new Stage"
            )

    def prefix_counts(self, position: int) -> tuple[int, ...]:
        return self._plan.prefix_counts(position)

    def resume_contract(self) -> dict[str, Any]:
        return {
            "algorithm": ALGORITHM,
            "dataset": self.dataset.resume_contract(),
            "total_samples": self.total_samples,
            "stage_id": self.stage_id,
            "mix_seed": self.mix_seed,
            "shuffle_seed": self.shuffle_seed,
            "block_multiplier": self.block_multiplier,
            "normalized_weights": list(self.normalized_weights),
            "block_size": self.block_size,
            "tail_quotas": list(self.tail_quotas),
            "plan_quotas": list(self.plan_quotas),
            "dp_rank": self.dp_rank,
            "dp_world_size": self.dp_world_size,
            "batch_size": self.batch_size,
            "runtime": {
                "torch": str(torch.__version__),
                "torch_git": torch.version.git_version,
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
                "permutation_device": "cpu",
                "permutation_dtype": "int64",
            },
        }


class _MultiSourceIterator(Iterator[SampleAddress]):
    def __init__(self, sampler: MultiSourceSampler) -> None:
        self._sampler = sampler
        self._plan = sampler._new_plan()
        self._cache: dict[int, tuple[int, torch.Tensor]] = {}
        self.seek(sampler.dp_rank * sampler.batch_size)

    @property
    def consumed(self) -> tuple[int, ...]:
        """Global prefix counts before position, including other ranks' samples."""
        return tuple(self._consumed)

    def __next__(self) -> SampleAddress:
        sampler = self._sampler
        if self.position == sampler.total_samples:
            raise StopIteration
        index = self._plan.source_at(self.position)
        source = sampler.dataset.sources[index]
        size = sampler.dataset.sizes[index]
        epoch, offset = divmod(self._consumed[index], size)
        cached = self._cache.get(index)
        if cached is None or cached[0] != epoch:
            seed = _u64(
                "shuffle-v1", sampler.shuffle_seed, source.source_id, source.version, size, epoch
            )
            self._cache[index] = (epoch, _randperm(size, seed))
        address = (source.source_id, int(self._cache[index][1][offset]))
        next_position = self.position + 1
        if (
            sampler.dp_world_size > 1
            and next_position % sampler.global_batch_size
            == ((sampler.dp_rank + 1) * sampler.batch_size) % sampler.global_batch_size
        ):
            next_position = min(
                next_position + sampler.global_batch_size - sampler.batch_size,
                sampler.total_samples,
            )
        if next_position != self.position + 1:
            counts = list(self._plan.prefix_counts(next_position))
        else:
            counts = list(self._consumed)
            counts[index] += 1
        self.position, self._consumed = next_position, counts
        return address

    def seek(self, global_position: int) -> None:
        """Seek to this rank's next sample or T, rebuilding counts but no source I/O."""
        position = _integer(global_position, "global_position")
        sampler = self._sampler
        if position > sampler.total_samples:
            raise ValueError("global_position exceeds total_samples")
        if position != sampler.total_samples:
            owner = (position % sampler.global_batch_size) // sampler.batch_size
            if owner != sampler.dp_rank:
                raise ValueError("global_position does not belong to this DP rank")
        counts = self._plan.prefix_counts(position)
        self.position, self._consumed = position, list(counts)
        self._cache.clear()

    def state_dict(self) -> dict[str, Any]:
        return {"schema": 1, "contract": self._sampler.resume_contract(), "position": self.position}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            not isinstance(state, dict)
            or type(state.get("schema")) is not int
            or state["schema"] != 1
        ):
            raise ValueError("invalid multi-source iterator state schema")
        try:
            # JSON equality distinguishes booleans from integer configuration values.
            saved = json.dumps(state.get("contract"), sort_keys=True, allow_nan=False)
            expected = json.dumps(self._sampler.resume_contract(), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid multi-source iterator contract") from error
        if saved != expected:
            raise ValueError("multi-source iterator configuration, algorithm or runtime mismatch")
        self.seek(state.get("position"))


def build_multi_source_sampler(
    dataset: MultiDataset,
    *,
    dp_rank: int,
    dp_world_size: int,
    batch_size: int,
    seed: int,
    shuffle: bool,
    drop_last: bool,
    total_samples: int,
    mix_seed: int,
    block_multiplier: int = 1,
    stage_id: str = "stage-0",
) -> MultiSourceSampler:
    """DataSpec sampler factory; seed controls source-local shuffle only."""
    if type(shuffle) is not bool or not shuffle:
        raise ValueError("multi-source quota sampling requires shuffle=True")
    if type(drop_last) is not bool:
        raise ValueError("drop_last must be a boolean")
    sampler = MultiSourceSampler(
        dataset,
        total_samples=total_samples,
        mix_seed=mix_seed,
        shuffle_seed=seed,
        block_multiplier=block_multiplier,
        stage_id=stage_id,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
    )
    if drop_last and sampler.total_samples % sampler.global_batch_size:
        raise ValueError(
            "drop_last would discard the finite quota tail; align total_samples or disable it"
        )
    return sampler


__all__ = [
    "ALGORITHM",
    "MAX_BLOCK_SIZE",
    "DatasetSource",
    "MultiDataset",
    "MultiSourceSampler",
    "build_multi_source_sampler",
]
