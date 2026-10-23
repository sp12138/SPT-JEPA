#!/usr/bin/env python3
"""Benchmark production-shape FA4 teacher-forcing attention on one GPU.

The tensors model the layout *after* Ulysses sequence-to-head exchange:
``(batch, full_sequence, local_heads, head_dim)``. This isolates FA4 from
Ulysses and FSDP communication so local-head performance cliffs are visible.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
from collections.abc import Sequence

import torch

from vforge.ops._attention import dispatch as attention
from vforge.ops._attention.teacher_forcing import (
    TeacherForcingSparsity,
    _dense_to_ordered,
    _tile_matrices,
    build_metadata,
    fa4_tile_size,
    teacher_forcing_available,
    teacher_forcing_sparsity_stats,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heads", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-latent-frames", type=int, default=99)
    parser.add_argument("--tokens-per-frame", type=int, default=3600)
    parser.add_argument("--frame-per-block", type=int, default=3)
    parser.add_argument("--local-attn-size", type=int, default=30)
    parser.add_argument("--sink-size", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Audit backward metadata against an independent matrix transpose and exit.",
    )
    return parser.parse_args()


def _tensor_summary(tensor: torch.Tensor) -> dict[str, object]:
    cpu = tensor.detach().cpu().contiguous()
    return {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sum": int(cpu.to(torch.int64).sum()),
        "min": int(cpu.min()) if cpu.numel() else None,
        "max": int(cpu.max()) if cpu.numel() else None,
        "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
    }


def _audit_backward_metadata(spec: TeacherForcingSparsity) -> dict[str, object]:
    tile = fa4_tile_size(spec.seqlen)
    partial, full = _tile_matrices(spec, tile)
    forward, backward = build_metadata(
        spec,
        device="cuda",
        need_bwd=True,
        deterministic=False,
        tile=tile,
    )
    if backward is None:
        raise AssertionError("need_bwd=True unexpectedly returned no backward metadata")

    # Independent reference: backward rows are KV tiles and their columns are
    # Q tiles, so they must be the exact transpose of the forward tile graph.
    expected_mask_cnt, expected_mask_idx = _dense_to_ordered(partial.T.contiguous(), "cuda", 1, 1)
    expected_full_cnt, expected_full_idx = _dense_to_ordered(full.T.contiguous(), "cuda", 1, 1)
    comparisons = {
        "mask_block_cnt": torch.equal(backward.mask_block_cnt, expected_mask_cnt),
        "mask_block_idx": torch.equal(backward.mask_block_idx, expected_mask_idx),
        "full_block_cnt": torch.equal(backward.full_block_cnt, expected_full_cnt),
        "full_block_idx": torch.equal(backward.full_block_idx, expected_full_idx),
    }
    result: dict[str, object] = {
        "event": "fa4_teacher_forcing_metadata_audit",
        "seqlen": spec.seqlen,
        "tokens_per_frame": spec.tokens_per_frame,
        "tile": list(tile),
        "all_equal_manual_transpose": all(comparisons.values()),
        "comparisons": comparisons,
        "forward": {
            "mask_block_cnt": _tensor_summary(forward.mask_block_cnt),
            "full_block_cnt": _tensor_summary(forward.full_block_cnt),
        },
        "backward": {
            "mask_block_cnt": _tensor_summary(backward.mask_block_cnt),
            "full_block_cnt": _tensor_summary(backward.full_block_cnt),
            "mask_block_idx": _tensor_summary(backward.mask_block_idx),
            "full_block_idx": _tensor_summary(backward.full_block_idx),
        },
        "expected_backward": {
            "mask_block_cnt": _tensor_summary(expected_mask_cnt),
            "full_block_cnt": _tensor_summary(expected_full_cnt),
            "mask_block_idx": _tensor_summary(expected_mask_idx),
            "full_block_idx": _tensor_summary(expected_full_idx),
        },
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    if not result["all_equal_manual_transpose"]:
        raise AssertionError("FA4 backward metadata differs from the manual transpose")
    return result


def _event_pair() -> tuple[torch.cuda.Event, torch.cuda.Event]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _run_once(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad: torch.Tensor,
    spec: TeacherForcingSparsity,
) -> tuple[float, float]:
    fwd_start, fwd_end = _event_pair()
    bwd_start, bwd_end = _event_pair()

    fwd_start.record()
    output = attention.unified_attention(q, k, v, sparsity=spec)
    fwd_end.record()
    bwd_start.record()
    output.backward(grad)
    bwd_end.record()
    torch.cuda.synchronize()

    forward_s = fwd_start.elapsed_time(fwd_end) / 1_000.0
    backward_s = bwd_start.elapsed_time(bwd_end) / 1_000.0
    q.grad = None
    k.grad = None
    v.grad = None
    del output
    return forward_s, backward_s


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def _benchmark_heads(
    *,
    heads: int,
    args: argparse.Namespace,
    spec: TeacherForcingSparsity,
) -> dict[str, object]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed)

    shape = (1, spec.seqlen, heads, args.head_dim)
    q, k, v = (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    )
    grad = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    warmup_times = []
    for _ in range(args.warmup):
        warmup_times.append(_run_once(q, k, v, grad, spec))

    measured = []
    for _ in range(args.repeats):
        measured.append(_run_once(q, k, v, grad, spec))

    forward = [value[0] for value in measured]
    backward = [value[1] for value in measured]
    result: dict[str, object] = {
        "event": "fa4_teacher_forcing_benchmark",
        "heads": heads,
        "head_dim": args.head_dim,
        "shape": list(shape),
        "warmup_s": [
            {"forward": round(value[0], 6), "backward": round(value[1], 6)}
            for value in warmup_times
        ],
        "samples_s": [
            {"forward": round(value[0], 6), "backward": round(value[1], 6)} for value in measured
        ],
        "forward_median_s": round(_median(forward), 6),
        "backward_median_s": round(_median(backward), 6),
        "total_median_s": round(_median([sum(value) for value in measured]), 6),
        "backward_over_forward": round(_median(backward) / _median(forward), 6),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 6),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 6),
    }
    print(json.dumps(result, sort_keys=True), flush=True)

    del q, k, v, grad
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    available, reason = teacher_forcing_available()
    if not available:
        raise RuntimeError(f"FA4 teacher-forcing sparsity is unavailable: {reason}")
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError("--warmup and --repeats must both be >= 1")
    if any(heads < 1 for heads in args.heads):
        raise ValueError("--heads values must all be >= 1")

    attention.set_flash_attention_backend("flash_attention_4")
    attention.set_flash_attention_deterministic(False)
    spec = TeacherForcingSparsity(
        num_latent_frames=args.num_latent_frames,
        tokens_per_frame=args.tokens_per_frame,
        frame_per_block=args.frame_per_block,
        first_frame_own_block=False,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
    )
    started = time.perf_counter()
    stats = teacher_forcing_sparsity_stats(spec)
    print(
        json.dumps(
            {
                "event": "fa4_teacher_forcing_setup",
                "device": torch.cuda.get_device_name(),
                "dtype": "bfloat16",
                "seqlen": spec.seqlen,
                "heads": args.heads,
                "sparsity": stats,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _audit_backward_metadata(spec)
    if args.metadata_only:
        return
    results = [_benchmark_heads(heads=heads, args=args, spec=spec) for heads in args.heads]
    print(
        json.dumps(
            {
                "event": "fa4_teacher_forcing_summary",
                "elapsed_s": round(time.perf_counter() - started, 6),
                "results": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
