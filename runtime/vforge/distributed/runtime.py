"""Shared device / distributed / ParallelState / ops lifecycle."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from vforge.distributed.parallel_state import (
    ParallelConfig,
    ParallelState,
    destroy_parallel_state,
    get_parallel_state,
    init_parallel_state,
)
from vforge.ops.config import OpsConfig


@dataclass(frozen=True)
class RuntimeContext:
    """Process runtime owned by train/prepare/eval entrypoints."""

    device: torch.device
    parallel_state: ParallelState
    ops: OpsConfig
    owns_process_group: bool
    resolved_backend: str | None = None


_RUNTIME: RuntimeContext | None = None


def get_runtime() -> RuntimeContext | None:
    return _RUNTIME


def _resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank)


def _ensure_process_group(device: torch.device) -> bool:
    if dist.is_available() and dist.is_initialized():
        return False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False
    backend = "nccl" if device.type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)
    return True


def initialize_runtime(
    parallel: ParallelConfig,
    ops: OpsConfig,
    *,
    device: torch.device | None = None,
) -> RuntimeContext:
    """Idempotently initialize device, process group, ParallelState, and ops.

    Same configuration may be re-applied; a conflicting configuration raises.
    """
    global _RUNTIME
    selected_device = device if device is not None else _resolve_device()
    owns_process_group = _ensure_process_group(selected_device)
    parallel_state = init_parallel_state(parallel)

    from vforge.ops import initialize_ops

    resolved_backend = initialize_ops(
        ops,
        device=selected_device,
        parallel_state=parallel_state,
    )
    runtime = RuntimeContext(
        device=selected_device,
        parallel_state=parallel_state,
        ops=ops,
        owns_process_group=owns_process_group,
        resolved_backend=resolved_backend,
    )
    if _RUNTIME is not None and _RUNTIME != runtime:
        # Compare policy-bearing fields only; ParallelState identity may differ
        # across idempotent re-init with identical config.
        if (
            _RUNTIME.ops != runtime.ops
            or _RUNTIME.parallel_state.config != runtime.parallel_state.config
            or _RUNTIME.device != runtime.device
        ):
            raise RuntimeError(
                "initialize_runtime() called with a conflicting configuration; "
                "one process may only host one immutable runtime policy"
            )
        return _RUNTIME
    _RUNTIME = runtime
    return runtime


def destroy_runtime(*, destroy_process_group: bool | None = None) -> None:
    """Clear process runtime; destroy process group only when owned."""
    global _RUNTIME
    runtime = _RUNTIME
    _RUNTIME = None
    from vforge.ops import reset_ops

    reset_ops()
    from vforge.distributed.context_parallel import reset_context_parallel

    reset_context_parallel()
    destroy_parallel_state()
    should_destroy = (
        runtime.owns_process_group
        if destroy_process_group is None and runtime is not None
        else bool(destroy_process_group)
    )
    if should_destroy and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def ensure_runtime(
    parallel: ParallelConfig,
    ops: OpsConfig,
    *,
    device: torch.device | None = None,
) -> RuntimeContext:
    """Return the existing runtime or initialize a new one."""
    current = get_runtime()
    if current is not None:
        # ParallelState stores the normalized config (implicit DP shard sizes
        # filled in), so the caller's raw config has to be normalized too or
        # any DP>1 mesh would look like a conflict.
        requested = parallel.normalized(current.parallel_state.world_size)
        if current.ops != ops or current.parallel_state.config != requested:
            raise RuntimeError("ensure_runtime() found an incompatible existing RuntimeContext")
        return current
    return initialize_runtime(parallel, ops, device=device)


__all__ = [
    "RuntimeContext",
    "destroy_runtime",
    "ensure_runtime",
    "get_parallel_state",
    "get_runtime",
    "initialize_runtime",
]
