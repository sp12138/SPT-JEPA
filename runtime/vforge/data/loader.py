"""Resumable distributed loader and leader-to-peers batch broadcast.

Sampling indexes the ``dp`` coordinate, so every rank sharing that coordinate
has to be handed the same batch. Which ranks those are depends on the engine --
sequence-parallel peers under fsdp, tensor-parallel peers under megatron -- so
callers take the group from ``ParallelState.batch_group`` rather than deciding
for themselves.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from vforge.data.spec import DataSpec
from vforge.data.worker_state import WorkerStateDataset, WorkerStatefulDataLoader


class TrainingDataLoader(Iterator[Any]):
    """Yield one device-ready batch at a time, cycling a finite source by epoch.

    Iteration and device transfer run in the training process, after the source
    loader's workers. ``device=None`` preserves the source batch's devices.
    An already-cycling source (such as a multi-stream loader) keeps ownership of
    its own epochs; the outer cursor only counts batches returned by that source.
    """

    def __init__(self, loader: Iterable[Any], *, device: torch.device | None = None) -> None:
        self.loader = loader
        self.device = device
        self._epoch = 0
        self._batches_in_epoch = 0
        self._iterator: Iterator[Any] | None = None
        self._set_sampler_epoch()

    def __iter__(self) -> TrainingDataLoader:
        if self._iterator is None:
            self._iterator = iter(self.loader)
        return self

    def __next__(self) -> Any:
        if self._iterator is None:
            iter(self)
        assert self._iterator is not None
        try:
            batch = next(self._iterator)
        except StopIteration:
            self._epoch += 1
            self._batches_in_epoch = 0
            self._set_sampler_epoch()
            self._iterator = iter(self.loader)
            try:
                batch = next(self._iterator)
            except StopIteration as exc:
                raise ValueError("training dataloader must not be empty") from exc
        if self.device is not None:
            batch = _move_to_device(batch, self.device)
        self._batches_in_epoch += 1
        return batch

    def state_dict(self) -> dict[str, Any]:
        if not self._vforge_resumable_dataloader:
            raise TypeError("training dataloader source is not resumable")
        return {
            "epoch": self._epoch,
            "batches_in_epoch": self._batches_in_epoch,
            "state_dict": self.loader.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("missing training dataloader state")
        epoch = state.get("epoch")
        batches = state.get("batches_in_epoch")
        if type(epoch) is not int or epoch < 0 or type(batches) is not int or batches < 0:
            raise ValueError("invalid dataloader cursor state")
        if state.get("state_dict") is None:
            raise ValueError("missing underlying dataloader state_dict; replay is not supported")
        if not self._vforge_resumable_dataloader:
            raise TypeError("training dataloader source is not resumable")
        self._epoch = epoch
        self._batches_in_epoch = batches
        # Set the sampler's epoch before restoring its committed position.
        self._set_sampler_epoch()
        self.loader.load_state_dict(state["state_dict"])
        self._iterator = None

    def _set_sampler_epoch(self) -> None:
        sampler = getattr(self.loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self._epoch)

    @property
    def _vforge_resumable_dataloader(self) -> bool:
        return is_resumable_dataloader(self.loader)

    @property
    def _vforge_bit_exact_resume(self) -> bool | None:
        return supports_bit_exact_resume(self.loader)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _split_batch(value: Any) -> tuple[Any, list[torch.Tensor]]:
    """Replace tensors with placeholders; return metadata tree and tensor list."""
    tensors: list[torch.Tensor] = []

    def _walk(node: Any) -> Any:
        if isinstance(node, torch.Tensor):
            index = len(tensors)
            tensors.append(node.detach())
            return {"__vforge_tensor__": index, "shape": list(node.shape), "dtype": str(node.dtype)}
        if isinstance(node, Mapping):
            return {key: _walk(item) for key, item in node.items()}
        if isinstance(node, tuple):
            return {"__vforge_tuple__": [_walk(item) for item in node]}
        if isinstance(node, list):
            return [_walk(item) for item in node]
        if isinstance(node, (str, bytes, int, float, bool, type(None))):
            return node
        raise TypeError(
            "LeaderBroadcastDataLoader only supports nested dict/list/tuple batches "
            f"of tensors and python scalars; got {type(node)!r}"
        )

    return _walk(value), tensors


def _rebuild_batch(meta: Any, tensors: list[torch.Tensor]) -> Any:
    if isinstance(meta, dict):
        if "__vforge_tensor__" in meta:
            return tensors[int(meta["__vforge_tensor__"])]
        if "__vforge_tuple__" in meta:
            return tuple(_rebuild_batch(item, tensors) for item in meta["__vforge_tuple__"])
        return {key: _rebuild_batch(item, tensors) for key, item in meta.items()}
    if isinstance(meta, list):
        return [_rebuild_batch(item, tensors) for item in meta]
    return meta


def broadcast_batch(
    batch: Any | None,
    *,
    group: dist.ProcessGroup | None,
    is_source: bool | None = None,
    target_device: torch.device | None = None,
    src_rank: int | None = None,
    group_rank: int | None = None,
) -> Any:
    """Broadcast a nested batch: metadata via object list, tensors via dist.broadcast.

    The source is always local rank 0 of ``group``, so it is derived here rather
    than asked of the caller, who would only ever compute the same thing.
    """
    using_legacy_source = src_rank is not None or group_rank is not None
    if using_legacy_source:
        if is_source is not None:
            raise TypeError("pass is_source or the deprecated src_rank/group_rank pair, not both")
        if src_rank is None or group_rank is None:
            raise TypeError("deprecated broadcast arguments require both src_rank and group_rank")
        warnings.warn(
            "broadcast_batch(src_rank=..., group_rank=...) is deprecated; use is_source=...",
            FutureWarning,
            stacklevel=2,
        )
        is_source = group_rank == 0
    elif is_source is None:
        raise TypeError("broadcast_batch requires is_source")

    if not dist.is_available() or not dist.is_initialized():
        if batch is None:
            raise RuntimeError("broadcast_batch requires a batch when dist is inactive")
        return batch
    if target_device is None and dist.get_backend(group) == "nccl":
        target_device = torch.device("cuda", torch.cuda.current_device())
    group_source_rank = 0 if group is None else int(dist.get_global_rank(group, 0))
    if src_rank is not None and src_rank != group_source_rank:
        raise ValueError(
            f"src_rank={src_rank} is not local rank 0 of the supplied group ({group_source_rank})"
        )

    if is_source:
        if batch is None:
            raise RuntimeError("broadcast source rank must provide a batch")
        meta, tensors = _split_batch(batch)
        if target_device is not None:
            tensors = [tensor.to(target_device) for tensor in tensors]
        spec = [
            {
                "shape": list(tensor.shape),
                "dtype": tensor.dtype,
                "device": tensor.device.type,
            }
            for tensor in tensors
        ]
        payload = [("batch", meta, spec)]
    else:
        payload = [None]
        tensors = []

    dist.broadcast_object_list(payload, src=group_source_rank, group=group)
    kind, meta, spec = payload[0]
    if kind != "batch":
        raise RuntimeError(f"unexpected broadcast payload kind: {kind!r}")

    if not is_source:
        tensors = []
        for item in spec:
            dtype = item["dtype"]
            device = target_device or torch.device(item["device"])
            tensors.append(torch.empty(item["shape"], dtype=dtype, device=device))

    for tensor in tensors:
        dist.broadcast(tensor.contiguous(), src=group_source_rank, group=group)
    return _rebuild_batch(meta, tensors)


class LeaderBroadcastDataLoader:
    """One rank advances a real loader; its peers receive broadcasts."""

    def __init__(
        self,
        loader: Any | None,
        *,
        group: dist.ProcessGroup | None,
        is_leader: bool,
        group_size: int,
        broadcast_batches: bool = True,
        global_length: int | None = None,
        resumable: bool = True,
    ) -> None:
        if group_size < 1:
            raise ValueError(f"leader broadcast needs at least one rank, got {group_size}")
        if group_size > 1 and group is None:
            raise ValueError("leader broadcast requires a process group")
        if group_size == 1 and not is_leader:
            raise ValueError("a single-rank loader must be its own leader")
        if is_leader and loader is None:
            raise ValueError("the leader requires a concrete loader")
        if not is_leader and loader is not None:
            raise ValueError("followers must not own a concrete loader")
        self._loader = loader
        self._group = group
        self._is_leader = bool(is_leader)
        self._group_size = int(group_size)
        self._broadcast_batches = bool(broadcast_batches)
        self._global_length = global_length
        self._resumable = bool(resumable)

    def __iter__(self) -> Iterator[Any]:
        if self._group_size == 1:
            assert self._loader is not None
            yield from self._loader
            return

        iterator: Iterator[Any] | None = None

        while True:
            leader_error: BaseException | None = None
            if self._is_leader:
                assert self._loader is not None
                try:
                    if iterator is None:
                        iterator = iter(self._loader)
                    batch = next(iterator)
                    control = [("continue", None, None)]
                except StopIteration:
                    batch = None
                    control = [("stop", None, None)]
                except BaseException as exc:
                    batch = None
                    leader_error = exc
                    control = [
                        ("error", type(exc).__name__, str(exc)),
                    ]
            else:
                batch = None
                control = [None]

            dist.broadcast_object_list(
                control,
                src=int(dist.get_global_rank(self._group, 0)),
                group=self._group,
            )
            kind, error_type, error_message = control[0]
            if kind == "stop":
                break
            if kind == "error":
                if leader_error is not None:
                    raise leader_error
                raise RuntimeError(f"leader dataloader failed: {error_type}: {error_message}")
            if not self._broadcast_batches:
                yield batch
                continue
            yield broadcast_batch(
                batch,
                group=self._group,
                is_source=self._is_leader,
            )

    def __len__(self) -> int:
        if self._loader is None:
            raise TypeError("follower loader length is owned by its leader")
        return len(self._loader)

    @property
    def global_length(self) -> int:
        if self._global_length is None:
            raise TypeError("this loader does not declare a global sample count")
        return int(self._global_length)

    def state_dict(self) -> dict[str, Any]:
        if not self._resumable:
            raise TypeError("this leader broadcast loader is not resumable")
        if self._is_leader:
            assert self._loader is not None
            return {
                "owner": True,
                "loader": self._loader.state_dict(),
                "group_size": self._group_size,
            }
        return {
            "owner": False,
            "group_size": self._group_size,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not self._resumable:
            raise TypeError("this leader broadcast loader is not resumable")
        saved_group_size = state.get("group_size", state.get("usp_size"))
        if saved_group_size is None:
            raise RuntimeError("loader checkpoint is missing group_size")
        if int(saved_group_size) != self._group_size:
            raise RuntimeError(
                "the group sharing a batch changed size; refuse to restore "
                "dataloader cursor under strict topology"
            )
        if self._is_leader:
            assert self._loader is not None
            if not state.get("owner", False):
                raise RuntimeError("leader resume requires owner loader state, not a follower stub")
            self._loader.load_state_dict(state["loader"])
        elif state.get("owner", False):
            raise RuntimeError("follower resume requires a follower stub, not owner loader state")

    @property
    def sampler(self) -> Any:
        if self._loader is None:
            return None
        return getattr(self._loader, "sampler", None)

    def set_epoch(self, epoch: int) -> None:
        if self._loader is not None and hasattr(self._loader, "set_epoch"):
            self._loader.set_epoch(epoch)

    @property
    def num_workers(self) -> int:
        return int(getattr(self._loader, "num_workers", 0))

    @property
    def generator(self) -> Any:
        return getattr(self._loader, "generator", None)


class SPBroadcastDataLoader(LeaderBroadcastDataLoader):
    """Deprecated compatibility wrapper for the former Ulysses-only API."""

    def __init__(
        self,
        loader: Any | None,
        *,
        group: dist.ProcessGroup | None,
        usp_rank: int,
        usp_size: int,
        broadcast_batches: bool = True,
    ) -> None:
        warnings.warn(
            "SPBroadcastDataLoader is deprecated; use LeaderBroadcastDataLoader",
            FutureWarning,
            stacklevel=2,
        )
        if usp_size < 1:
            raise ValueError(f"usp_size must be >= 1, got {usp_size}")
        if not 0 <= usp_rank < usp_size:
            raise ValueError(f"usp_rank must be in [0, {usp_size}), got {usp_rank}")
        super().__init__(
            loader,
            group=group,
            is_leader=usp_rank == 0,
            group_size=usp_size,
            broadcast_batches=broadcast_batches,
        )

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["usp_size"] = state.pop("group_size")
        return state


class DistributedEvalSampler(Sampler[tuple[int, bool]]):
    """Assign real indices once and mark collective-only padding explicitly."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        num_replicas: int,
        rank: int,
        limit: int | None = None,
        selection: str = "first",
        source_groups: Sequence[Sequence[int]] = (),
    ) -> None:
        if num_replicas < 1:
            raise ValueError("eval sampler num_replicas must be at least one")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"eval sampler rank must be in [0, {num_replicas}), got {rank}")
        total = len(dataset)
        selected = total if limit is None else min(total, max(0, int(limit)))
        if selection == "first":
            indices = list(range(selected))
        elif selection == "evenly_spaced":
            if selected == 0:
                indices = []
            elif selected == 1:
                indices = [0]
            elif selected == total:
                indices = list(range(total))
            elif source_groups:
                normalized_groups = tuple(
                    tuple(int(index) for index in group) for group in source_groups
                )
                flattened = [index for group in normalized_groups for index in group]
                if sorted(flattened) != list(range(total)):
                    raise ValueError(
                        "eval source_groups must partition every dataset index exactly once"
                    )
                base, extra = divmod(selected, len(normalized_groups))
                indices = []
                for group_index, group in enumerate(normalized_groups):
                    count = base + (1 if group_index < extra else 0)
                    if count <= 0 or not group:
                        continue
                    if count >= len(group):
                        indices.extend(group)
                        continue
                    if count == 1:
                        indices.append(group[0])
                        continue
                    step = (len(group) - 1) / (count - 1)
                    indices.extend(group[round(offset * step)] for offset in range(count))
            else:
                step = (total - 1) / (selected - 1)
                indices = sorted({round(offset * step) for offset in range(selected)})
        else:
            raise ValueError(
                f"eval sampler selection must be first or evenly_spaced; got {selection!r}"
            )
        self.indices = tuple(indices)
        self.total = len(self.indices)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)

    def __iter__(self) -> Iterator[tuple[int, bool]]:
        real = list(self.indices[self.rank :: self.num_replicas])
        target = len(self)
        for index in real:
            yield index, False
        if self.total:
            for offset in range(target - len(real)):
                yield self.indices[(self.rank + offset) % self.total], True

    def __len__(self) -> int:
        if self.total == 0:
            return 0
        return (self.total + self.num_replicas - 1) // self.num_replicas


class _EvalDatasetView(Dataset):
    """Decode sampler tokens and attach the physical-padding marker."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, token: tuple[int, bool]) -> Any:
        index, padding = token
        sample = self.dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError(
                f"distributed evaluation datasets must yield mappings; got {type(sample).__name__}"
            )
        return {**sample, "__vforge_eval_padding__": bool(padding)}


def build_eval_loader(
    dataset: Dataset,
    *,
    parallel_state: Any,
    collate_fn: Callable[[list[Any]], Any],
    num_samples: int | None = None,
    sample_selection: str = "first",
    num_workers: int = 0,
    pin_memory: bool = False,
) -> LeaderBroadcastDataLoader:
    """Build a leader-owned, no-padding distributed evaluation loader."""
    group = parallel_state.batch_group
    is_leader = bool(parallel_state.is_batch_leader)
    source_groups = getattr(dataset, "eval_source_groups", ())
    sampler = DistributedEvalSampler(
        dataset,
        num_replicas=int(parallel_state.config.dp_size),
        rank=int(parallel_state.dp_rank),
        limit=num_samples,
        selection=sample_selection,
        source_groups=source_groups,
    )
    total = sampler.total
    leader_loader = None
    if is_leader:
        view = _EvalDatasetView(dataset)
        leader_loader = DataLoader(
            view,
            batch_size=1,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
    group_size = 1 if group is None else dist.get_world_size(group)
    return LeaderBroadcastDataLoader(
        leader_loader,
        group=group,
        is_leader=is_leader,
        group_size=group_size,
        global_length=total,
        resumable=False,
    )


def build_distributed_sampler(
    dataset: Dataset,
    *,
    num_replicas: int,
    rank: int,
    shuffle: bool = True,
    seed: int = 0,
    drop_last: bool = True,
) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )


def build_resumable_loader(
    dataset: Dataset | DataSpec,
    *,
    parallel_state: Any,
    batch_size: int,
    num_workers: int,
    drop_last: bool,
    seed: int,
    shuffle: bool = True,
    collate_fn: Callable | None = None,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    broadcast_batches: bool = True,
) -> StatefulDataLoader | LeaderBroadcastDataLoader:
    """Build the generic DP-sampled, leader-owned resumable loader.

    Tasks may pass a DataSpec to install their own sampler and collate. The
    sampler factory receives this rank's DP coordinates and owns all sharding;
    its result is never wrapped in another DistributedSampler. Existing calls
    passing only a Dataset retain their original behavior.

    Map-style datasets with workers snapshot CPU RNGs after fetch and collate
    through torchdata's dataset hooks. Resume requires the same worker setup;
    old multi-worker checkpoints without these RNG snapshots are rejected.
    Iterable datasets keep their own state protocol. With zero workers the
    Trainer continues to own process RNG restoration.
    """
    sampler_factory = None
    if isinstance(dataset, DataSpec):
        spec = dataset
        if collate_fn is not None and spec.collate_fn is not None:
            raise ValueError("provide collate_fn in DataSpec or the loader call, not both")
        dataset = spec.dataset
        sampler_factory = spec.sampler_factory
        collate_fn = spec.collate_fn if spec.collate_fn is not None else collate_fn
    group = parallel_state.batch_group
    is_leader = bool(parallel_state.is_batch_leader)
    leader_loader = None
    if is_leader:
        dp_size = int(parallel_state.config.dp_size)
        sampler = None
        if sampler_factory is not None:
            sampler = sampler_factory(
                dataset,
                dp_rank=int(parallel_state.dp_rank),
                dp_world_size=dp_size,
                batch_size=batch_size,
                seed=seed,
                shuffle=shuffle,
                drop_last=drop_last,
            )
            if not isinstance(sampler, Sampler):
                raise TypeError("DataSpec.sampler_factory must return a torch Sampler")
        elif len(dataset) > 0:
            sampler = build_distributed_sampler(
                dataset,
                num_replicas=dp_size,
                rank=int(parallel_state.dp_rank),
                shuffle=shuffle,
                seed=seed,
                drop_last=drop_last,
            )
        loader_type = StatefulDataLoader
        loader_dataset = dataset
        if num_workers > 0 and not isinstance(dataset, IterableDataset):
            # Build the sampler against the original dataset; only worker reads
            # need the wrapper. Torchdata snapshots it after collate completes.
            loader_dataset = WorkerStateDataset(dataset, collate_fn)
            loader_type = WorkerStatefulDataLoader
        leader_loader = loader_type(
            loader_dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None and len(dataset) > 0,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            generator=torch.Generator().manual_seed(seed),
        )
    if group is None:
        assert leader_loader is not None
        return leader_loader
    return LeaderBroadcastDataLoader(
        leader_loader,
        group=group,
        is_leader=is_leader,
        group_size=dist.get_world_size(group),
        broadcast_batches=broadcast_batches,
    )


def build_resumable_sp_loader(*args: Any, **kwargs: Any) -> Any:
    """Deprecated Ulysses-specific wrapper for :func:`build_resumable_loader`.

    Older callers supply only ``usp_*`` topology attributes.  Keep accepting
    that shape while routing fully initialized parallel states through the
    generic batch-group implementation.
    """
    warnings.warn(
        "build_resumable_sp_loader is deprecated; use build_resumable_loader",
        FutureWarning,
        stacklevel=2,
    )
    parallel_state = kwargs.get("parallel_state")
    if (
        parallel_state is not None
        and not hasattr(parallel_state, "batch_group")
        and hasattr(parallel_state, "config")
        and hasattr(parallel_state.config, "usp_size")
    ):
        usp_size = int(parallel_state.config.usp_size)
        usp_rank = int(getattr(parallel_state, "usp_rank", 0))
        usp_group = getattr(parallel_state, "usp_group", None) if usp_size > 1 else None

        class _LegacyBatchState:
            config = parallel_state.config
            dp_rank = parallel_state.dp_rank
            batch_group = usp_group
            is_batch_leader = usp_rank == 0

        kwargs["parallel_state"] = _LegacyBatchState()
    return build_resumable_loader(*args, **kwargs)


def is_resumable_dataloader(loader: Any) -> bool:
    return (
        isinstance(loader, StatefulDataLoader)
        or (isinstance(loader, LeaderBroadcastDataLoader) and loader._resumable)
        or bool(getattr(loader, "_vforge_resumable_dataloader", False))
    )


def supports_bit_exact_resume(loader: Any) -> bool | None:
    if not is_resumable_dataloader(loader):
        return None
    return bool(getattr(loader, "_vforge_bit_exact_resume", True))


__all__ = [
    "DistributedEvalSampler",
    "LeaderBroadcastDataLoader",
    "SPBroadcastDataLoader",
    "TrainingDataLoader",
    "broadcast_batch",
    "build_distributed_sampler",
    "build_eval_loader",
    "build_resumable_loader",
    "build_resumable_sp_loader",
    "is_resumable_dataloader",
    "supports_bit_exact_resume",
]
