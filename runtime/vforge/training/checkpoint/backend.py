"""Checkpoint format backends that are not represented by Torch state dicts."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

import torch
import torch.distributed as dist
import torch.nn as nn

from vforge.distributed.mcore_data_parallel import (
    is_mcore_data_parallel,
    mcore_process_group_collection,
)
from vforge.distributed.parallel_state import get_parallel_state
from vforge.distributed.parallelize import unwrap_replicated_data_parallel
from vforge.distributed.tensor_parallel import (
    suspend_tensor_parallel_state_dict_post_hooks,
    tensor_parallel_shard_axes,
)

from .config import ResumeConfig
from .manifest import CheckpointManifest

_MCORE_COMPONENT = "mcore"
_MCORE_CONTENT_METADATA = {
    "vforge_checkpoint_format": "mcore",
    "vforge_checkpoint_schema": 1,
    "optimizer_sharding": "dp_reshardable",
}


class CheckpointBackend(Protocol):
    """Format-specific I/O while :class:`CheckpointManager` owns policy."""

    format: str

    def prepare_save(
        self,
        model: nn.Module,
        optimizer: Any | None,
        path: str,
        *,
        manifest: CheckpointManifest,
        rank_state: dict[str, Any] | None,
        save_optimizer: bool,
        save_trainer_state: bool,
        empty_cache_before_save: bool,
    ) -> Any: ...

    def save_sync(self, prepared: Any) -> str | None: ...

    def schedule_async(
        self,
        prepared: Any,
        *,
        finalize: Callable[[], None],
    ) -> int: ...

    def poll(self, call_index: int) -> bool: ...

    def wait(self, call_index: int) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PreparedMCoreSave:
    """Native state and durable paths prepared on the training thread."""

    path: str
    incomplete: str
    payload_path: str
    manifest: CheckpointManifest
    state: dict[str, Any]
    state_dict_s: float
    sidecar_s: float


def build_mcore_model_sharded_state_dict(model: nn.Module) -> dict[str, Any]:
    """Describe a mixed Wan/MCore module tree using MCore checkpoint objects.

    Calling MCore's default helper on the Diffusers root would flatten the
    entire ordinary module subtree and hide the TP axes declared by nested
    MCore linears. Recursing here lets native MCore modules describe their own
    subtrees while vForge describes ordinary containers one module at a time.
    """
    if is_mcore_data_parallel(model):
        groups = mcore_process_group_collection(model)
        tp_group = groups.tp
        dp_cp_group = groups.dp_cp
    else:
        # Inference applies the engine's TP/CP parameterization but deliberately
        # skips MCore DDP: there are no gradients to reduce, and allocating its
        # contiguous buffers for a two-tower A14B model would waste substantial
        # memory. The checkpoint layout only needs the process groups, which
        # vForge's immutable ParallelState already owns.
        parallel = get_parallel_state()
        tp_group = parallel.attention_tp_group
        dp_cp_group = parallel.dp_cp_group
        if tp_group is None or dp_cp_group is None:
            raise RuntimeError(
                "MCore checkpoint model state requires initialized TP and DP x CP process groups"
            )
    logical_model = unwrap_replicated_data_parallel(model)
    metadata = {"dp_cp_group": dp_cp_group}

    from vforge.mcore.transformer.utils import make_sharded_tensors_for_checkpoint

    def recurse(module: nn.Module, prefix: str) -> dict[str, Any]:
        native_builder = getattr(module, "sharded_state_dict", None)
        if callable(native_builder):
            return native_builder(prefix=prefix, sharded_offsets=(), metadata=metadata)

        own_state: dict[str, Any] = {}
        module._save_to_state_dict(own_state, "", keep_vars=True)
        result = make_sharded_tensors_for_checkpoint(
            own_state,
            prefix,
            tensor_parallel_shard_axes(module),
            tp_group=tp_group,
            dp_cp_group=dp_cp_group,
        )
        for name, child in module.named_children():
            result.update(recurse(child, f"{prefix}{name}."))
        return result

    with suspend_tensor_parallel_state_dict_post_hooks(logical_model):
        return recurse(logical_model, "")


def _require_mcore_optimizer(optimizer: Any):
    from vforge.training.mcore_optimizer import MCoreDistributedOptimizer

    if not isinstance(optimizer, MCoreDistributedOptimizer):
        raise TypeError(
            "MCore optimizer checkpointing requires MCoreDistributedOptimizer; "
            f"got {type(optimizer).__name__}"
        )
    return optimizer


class MCoreCheckpointBackend:
    """Thin adapter over MCore distributed checkpoint serialization."""

    format = "mcore"

    def __init__(self, *, async_save: bool) -> None:
        self._async_save = bool(async_save)
        self._save_strategy: Any = None
        self._async_queue: Any = None
        if self._async_save:
            self._require_nvrx()

    @staticmethod
    def _require_nvrx() -> None:
        from vforge.mcore.dist_checkpointing.strategies.nvrx import has_nvrx_async_support

        if not has_nvrx_async_support():
            raise ModuleNotFoundError(
                "checkpoint.format='mcore' with async_save=true requires "
                "nvidia-resiliency-ext>=0.6,<0.7"
            )

    def _strategy(self):
        if self._save_strategy is None:
            from vforge.mcore.dist_checkpointing.strategies.fully_parallel import (
                FullyParallelSaveStrategyWrapper,
            )
            from vforge.mcore.dist_checkpointing.strategies.torch import (
                TorchDistSaveShardedStrategy,
            )

            self._save_strategy = FullyParallelSaveStrategyWrapper(
                TorchDistSaveShardedStrategy(cached_metadata=True),
                parallelization_group=dist.group.WORLD,
                do_cache_distribution=True,
            )
        return self._save_strategy

    def _queue(self):
        if self._async_queue is None:
            from vforge.mcore.dist_checkpointing.strategies.torch import get_async_strategy

            _, modules = get_async_strategy("nvrx")
            self._async_queue = modules["AsyncCallsQueue"]()
        return self._async_queue

    def prepare_save(
        self,
        model: nn.Module,
        optimizer: Any | None,
        path: str,
        *,
        manifest: CheckpointManifest,
        rank_state: dict[str, Any] | None,
        save_optimizer: bool,
        save_trainer_state: bool,
        empty_cache_before_save: bool,
    ) -> PreparedMCoreSave:
        from . import storage

        if manifest.format != self.format:
            raise ValueError(f"MCore backend received manifest format={manifest.format!r}")
        if save_optimizer and optimizer is None:
            raise ValueError("full MCore checkpoint requires an optimizer")
        if empty_cache_before_save and torch.cuda.is_available():
            torch.cuda.empty_cache()

        incomplete = storage._prepare_dcp_directory(path, manifest)
        payload_path = os.path.join(incomplete, _MCORE_COMPONENT)
        os.makedirs(payload_path, exist_ok=True)

        trainer_saved = rank_state is not None and save_trainer_state
        sidecar_started = time.perf_counter()
        if trainer_saved:
            storage._write_rank_sidecar(
                incomplete,
                manifest,
                rank_state,
                trainer_directory="trainer",
            )
        sidecar_s = time.perf_counter() - sidecar_started

        state_dict_started = time.perf_counter()
        # With overlap_param_gather, an optimizer step may return while later
        # parameter buckets are still being gathered. The next forward would
        # normally wait through module pre-hooks, but checkpointing happens
        # before that forward. Establish one coherent model/optimizer snapshot
        # before MCore captures references for synchronous or NVRx staging.
        from vforge.distributed.mcore_data_parallel import (
            finish_mcore_data_parallel_parameter_sync,
        )

        finish_mcore_data_parallel_parameter_sync(model)
        model_state = build_mcore_model_sharded_state_dict(model)
        state: dict[str, Any] = {"model": model_state}
        if save_optimizer:
            mcore_optimizer = _require_mcore_optimizer(optimizer)
            state["optimizer"] = mcore_optimizer.checkpoint_sharded_state_dict(
                model_state,
                is_loading=False,
            )
        state_dict_s = time.perf_counter() - state_dict_started

        components = {"weights": _MCORE_COMPONENT}
        components_present = {
            "weights": True,
            "optimizer": save_optimizer,
            "trainer": trainer_saved,
        }
        if save_optimizer:
            components["optimizer"] = _MCORE_COMPONENT
        if trainer_saved:
            components["trainer"] = "trainer"
        prepared_manifest = replace(
            manifest,
            components=components,
            components_present=components_present,
        )
        return PreparedMCoreSave(
            path=path,
            incomplete=incomplete,
            payload_path=payload_path,
            manifest=prepared_manifest,
            state=state,
            state_dict_s=state_dict_s,
            sidecar_s=sidecar_s,
        )

    def save_sync(self, prepared: PreparedMCoreSave) -> str | None:
        from vforge.mcore import dist_checkpointing

        from . import storage

        dist_checkpointing.save(
            prepared.state,
            prepared.payload_path,
            sharded_strategy=self._strategy(),
            content_metadata=_MCORE_CONTENT_METADATA,
        )
        if dist.is_initialized():
            dist.barrier()
        if storage.is_global_zero():
            storage._commit_dcp_directory(
                prepared.path,
                prepared.incomplete,
                prepared.manifest,
            )
        if dist.is_initialized():
            dist.barrier()
        return prepared.path if storage.is_global_zero() else None

    def schedule_async(
        self,
        prepared: PreparedMCoreSave,
        *,
        finalize: Callable[[], None],
    ) -> int:
        from vforge.mcore import dist_checkpointing

        request = dist_checkpointing.save(
            prepared.state,
            prepared.payload_path,
            sharded_strategy=self._strategy(),
            async_sharded_save=True,
            async_strategy="nvrx",
            content_metadata=_MCORE_CONTENT_METADATA,
        )
        if request is None:
            raise RuntimeError("MCore async checkpoint save returned no request")
        request.finalize_fns.append(finalize)
        return int(self._queue().schedule_async_request(request))

    def poll(self, call_index: int) -> bool:
        return call_index in self._queue().maybe_finalize_async_calls(blocking=False)

    def wait(self, call_index: int) -> None:
        # The queue reports calls finalized *by this invocation*. A callback
        # may complete after CheckpointManager observes Future.done() == false
        # but before this blocking flush acquires the queue, in which case the
        # returned list legitimately omits ``call_index``. The manager-owned
        # Future is the durable completion/error source; here we only need to
        # flush outstanding NVRx work before it reads that Future.
        self._queue().maybe_finalize_async_calls(blocking=True)

    def close(self) -> None:
        if self._async_queue is not None:
            self._async_queue.close()
            self._async_queue = None


def _validate_mcore_topology(manifest: CheckpointManifest) -> bool:
    """Return whether DP is unchanged; CP/TP/PP must always match."""
    from vforge.distributed.parallel_state import get_parallel_state

    current = get_parallel_state().config
    saved = manifest.topology
    for name in ("cp_size", "tp_size", "pp_size"):
        saved_value = saved.get(name)
        current_value = getattr(current, name)
        if saved_value != current_value:
            raise ValueError(
                "MCore optimizer checkpoint only reshards the DP dimension; "
                f"{name} changed from {saved_value} to {current_value}"
            )
    return saved.get("dp_size") == current.dp_size


def load_mcore_training_state(
    model: nn.Module,
    optimizer: Any | None,
    path: str,
    *,
    manifest: CheckpointManifest,
    resume: ResumeConfig,
    weights_strict: bool,
    integrity: str = "metadata",
) -> tuple[CheckpointManifest, dict[str, Any] | None]:
    """Load native MCore model/optimizer shards and vForge's trainer sidecar."""
    from vforge.mcore import dist_checkpointing

    from . import storage

    if manifest.format != "mcore":
        raise ValueError(f"not an MCore checkpoint: {path}")
    same_dp = _validate_mcore_topology(manifest)
    if not resume.weights:
        return manifest, None
    if resume.optimizer and optimizer is None:
        raise ValueError("resume.optimizer=true requires an optimizer")
    if (resume.optimizer or resume.training_state) and manifest.state != "full":
        raise ValueError("checkpoint has weights only; requested training-state resume")
    if resume.training_state and not same_dp:
        raise ValueError(
            "MCore DP reshard restores model and optimizer only; set "
            "resume.training_state=false when changing DP"
        )
    if not same_dp and (not dist.is_initialized() or dist.get_rank() == 0):
        print(
            "VFORGE_MCORE_OPTIMIZER_RESHARD_RESUME "
            f"saved_dp={manifest.topology.get('dp_size')} status=loading",
            flush=True,
        )

    # MCore owns payload interpretation, while vForge still owns the outer
    # checkpoint contract. Validate the physical native payload before asking
    # MCore to deserialize it. Model and optimizer deliberately share this
    # component, so checking the weights alias once covers both logical states.
    storage._validate_component_integrity(
        path,
        manifest,
        "weights",
        distributed=True,
        mode=integrity,
    )

    model_state = build_mcore_model_sharded_state_dict(model)
    target: dict[str, Any] = {"model": model_state}
    mcore_optimizer = None
    if resume.optimizer:
        mcore_optimizer = _require_mcore_optimizer(optimizer)
        target["optimizer"] = mcore_optimizer.checkpoint_sharded_state_dict(
            model_state,
            is_loading=True,
        )
    payload_path = os.path.join(path, str(manifest.components.get("weights", _MCORE_COMPONENT)))
    if resume.optimizer:
        strict_handling = "raise_all" if resume.effective_strict else "log_all"
    else:
        # A full native payload may intentionally be read as weights-only. In
        # that case optimizer entries in the checkpoint are not an error, but
        # every model key requested by the live topology still must exist when
        # the caller requested strict weights.
        strict_handling = (
            "raise_unexpected" if weights_strict or resume.effective_strict else "log_unexpected"
        )
    loaded = dist_checkpointing.load(
        target,
        payload_path,
        strict=strict_handling,
    )
    logical_model = unwrap_replicated_data_parallel(model)
    logical_model.load_state_dict(
        loaded["model"],
        strict=bool(weights_strict or resume.effective_strict),
    )
    if mcore_optimizer is not None:
        mcore_optimizer.load_checkpoint_sharded_state_dict(loaded["optimizer"])

    if not resume.training_state:
        return manifest, None
    if not storage._component_present(manifest, "trainer"):
        raise ValueError("checkpoint does not contain trainer state")
    storage._validate_component_integrity(
        path,
        manifest,
        "trainer",
        distributed=True,
        mode=integrity,
    )
    rank = dist.get_rank() if dist.is_initialized() else 0
    sidecar_path = storage._trainer_sidecar_path(path, manifest, rank)
    if not os.path.isfile(sidecar_path):
        raise ValueError(f"missing rank-local state: {sidecar_path}")
    rank_state = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    selected = dict(rank_state)
    selected["_vforge_bit_exact"] = bool(resume.effective_strict)
    selected["_vforge_restore_rng"] = True
    return manifest, selected


__all__ = [
    "build_mcore_model_sharded_state_dict",
    "CheckpointBackend",
    "load_mcore_training_state",
    "MCoreCheckpointBackend",
    "PreparedMCoreSave",
]
