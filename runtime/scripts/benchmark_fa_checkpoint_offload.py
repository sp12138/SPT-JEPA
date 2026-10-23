#!/usr/bin/env python3
"""Benchmark standard checkpoint replay against FA4 recompute offload."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from functools import partial

import torch
from torch.utils.checkpoint import checkpoint

from vforge.checkpointing.offload import (
    ActivationOffloadConfig,
    ActivationOffloadManager,
)
from vforge.ops._attention import dispatch as attention


def _one_step(
    attention_fn,
    templates: tuple[torch.Tensor, ...],
    controller: ActivationOffloadManager | None,
) -> tuple[float, float, dict[str, object] | None]:
    inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in templates)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = torch.cuda.Event(enable_timing=True)
    finished = torch.cuda.Event(enable_timing=True)
    started.record()

    if controller is None:
        output = checkpoint(
            attention_fn,
            *inputs,
            use_reentrant=False,
        )
    else:
        controller.begin_step()
        output = checkpoint(
            attention_fn,
            *inputs,
            use_reentrant=False,
            context_fn=controller.checkpoint_contexts,
        )
        controller.start_backward_prefetch()
    try:
        output.float().square().mean().backward()
    finally:
        if controller is not None:
            controller.finish_backward_prefetch()

    finished.record()
    finished.synchronize()
    elapsed_ms = started.elapsed_time(finished)
    peak_gb = torch.cuda.max_memory_allocated() / float(1024**3)
    stats = controller.pop_stats() if controller is not None else None
    del output, inputs
    return elapsed_ms, peak_gb, stats


def _benchmark(
    name: str,
    attention_fn,
    templates: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    controller = ActivationOffloadManager(
        ActivationOffloadConfig(
            min_tensor_bytes=1,
            pin_memory=True,
            async_transfer=True,
        )
    )
    results: dict[str, object] = {"case": name}
    for mode, active_controller in (
        ("checkpoint", None),
        ("recompute_offload", controller),
    ):
        timings = []
        peaks = []
        last_stats = None
        for iteration in range(warmup + iterations):
            elapsed_ms, peak_gb, stats = _one_step(
                attention_fn,
                templates,
                active_controller,
            )
            if iteration >= warmup:
                timings.append(elapsed_ms)
                peaks.append(peak_gb)
            last_stats = stats
        results[mode] = {
            "median_ms": statistics.median(timings),
            "min_ms": min(timings),
            "max_ms": max(timings),
            "peak_allocated_gb": max(peaks),
            "offload": last_stats,
        }
        gc.collect()
        torch.cuda.empty_cache()

    checkpoint_ms = results["checkpoint"]["median_ms"]
    offload_ms = results["recompute_offload"]["median_ms"]
    results["speedup_percent"] = (checkpoint_ms / offload_ms - 1.0) * 100.0
    results["peak_allocated_delta_gb"] = (
        results["recompute_offload"]["peak_allocated_gb"]
        - results["checkpoint"]["peak_allocated_gb"]
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(42)
    templates = tuple(
        torch.randn(
            args.batch,
            args.sequence_length,
            args.heads,
            args.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(3)
    )
    mask = torch.ones(
        args.batch,
        1,
        1,
        args.sequence_length,
        device="cuda",
        dtype=torch.bool,
    )
    for batch_index in range(args.batch):
        padding = 17 * (batch_index + 1)
        mask[batch_index, :, :, -padding:] = False

    cases = [
        (
            "dense",
            attention.unified_attention,
        ),
        (
            "uniform_tail_mask",
            partial(attention.unified_attention, attn_mask=mask),
        ),
    ]
    for name, attention_fn in cases:
        result = _benchmark(
            name,
            attention_fn,
            templates,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print("FA4_BENCHMARK " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
