"""Opt-in PyTorch profiler support for focused cloud diagnostics."""

from __future__ import annotations

import logging
import os
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_CUDA_PROFILER_WARMED = False


def _warmup_cuda_profiler() -> None:
    global _CUDA_PROFILER_WARMED
    if _CUDA_PROFILER_WARMED:
        return

    # Consume the broken first Kineto session so the real profiler captures
    # CUDA kernels. PyTorch 2.9.x+cu130 can drop all CUDA activity from its
    # first profiling session on B300 (pytorch/pytorch#169464) because the fix
    # in pytorch/kineto#1115 was not included in that release. A scheduled
    # warmup cannot help because it still runs inside that same broken session.
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ):
        probe = torch.empty(1, device=torch.cuda.current_device())
        probe.add_(1)
        torch.cuda.synchronize()
    _CUDA_PROFILER_WARMED = True
    print("[vforge-profiler] completed CUDA profiler prewarm", flush=True)


def build_torch_profiler(*, rank: int) -> Any | None:
    """Build a scheduled CPU/CUDA profiler from environment variables."""
    if os.environ.get("VFORGE_TORCH_PROFILE") != "1":
        return None
    target_rank = int(os.environ.get("VFORGE_TORCH_PROFILE_RANK", "0"))
    if rank != target_rank:
        return None

    output_template = os.environ.get("VFORGE_TORCH_PROFILE_OUTPUT")
    if not output_template:
        raise ValueError("VFORGE_TORCH_PROFILE_OUTPUT is required when profiling is enabled")
    output_dir = Path(output_template.format(rank=rank))
    output_dir.mkdir(parents=True, exist_ok=True)

    wait = int(os.environ.get("VFORGE_TORCH_PROFILE_WAIT", "1"))
    warmup = int(os.environ.get("VFORGE_TORCH_PROFILE_WARMUP", "1"))
    active = int(os.environ.get("VFORGE_TORCH_PROFILE_ACTIVE", "2"))
    if min(wait, warmup, active) < 0 or active < 1:
        raise ValueError("profiler wait/warmup must be non-negative and active must be >= 1")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        _warmup_cuda_profiler()
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_emitted = False

    def trace_handler(profiler: Any) -> None:
        nonlocal trace_emitted
        if trace_emitted:
            print(
                "[vforge-profiler] ignoring duplicate final trace callback",
                flush=True,
            )
            return
        trace_emitted = True
        step = int(profiler.step_num)
        profiler.export_chrome_trace(str(output_dir / f"trace-step{step}.json"))
        table = profiler.key_averages().table(
            sort_by=(
                "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
            ),
            row_limit=200,
        )
        (output_dir / f"key-averages-step{step}.txt").write_text(
            table + "\n",
            encoding="utf-8",
        )

    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=1,
        ),
        on_trace_ready=trace_handler,
        record_shapes=(os.environ.get("VFORGE_TORCH_PROFILE_RECORD_SHAPES", "0") == "1"),
        profile_memory=(os.environ.get("VFORGE_TORCH_PROFILE_MEMORY", "0") == "1"),
        with_stack=(os.environ.get("VFORGE_TORCH_PROFILE_STACK", "0") == "1"),
    )


def advance_profiler(profiler: Any | None, device: torch.device) -> None:
    """Flush active CUDA work before advancing a scheduled profiler cycle."""
    if profiler is None:
        return
    recording_actions = {
        torch.profiler.ProfilerAction.RECORD,
        torch.profiler.ProfilerAction.RECORD_AND_SAVE,
    }
    if torch.cuda.is_available() and getattr(profiler, "current_action", None) in recording_actions:
        torch.cuda.synchronize(device)
    profiler.step()


def stop_profiler(profiler: Any | None) -> None:
    """Stop an optional profiler without disrupting trainer cleanup."""
    if profiler is None:
        return
    try:
        profiler.stop()
    except Exception:
        logger.warning(
            "Failed to stop profiler cleanly",
            exc_info=True,
        )


def record_region(
    profiler: Any | None,
    name: str,
) -> AbstractContextManager[Any]:
    """Create a named profiler range only when profiling is active."""
    if profiler is None:
        return nullcontext()
    return torch.profiler.record_function(name)


__all__ = [
    "advance_profiler",
    "build_torch_profiler",
    "record_region",
    "stop_profiler",
]
