"""Validate overlapping MCore DDP over arbitrary DP x CP x TP topology.

The probe uses one scalar logical shard per TP rank. DP ranks contribute
different samples, CP ranks contribute different sequence pieces, and MCore
DDP must reduce only replicas with the same TP coordinate. It then exercises
vForge's logical norm, clipping, distributed optimizer, and unused-parameter
handling across consecutive steps. The first step learns MCore's hook pattern;
the second exercises steady-state gradient and parameter overlap.
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.tensor.parallel import ColwiseParallel


class _ShardedScalarModel(torch.nn.Module):
    def __init__(self, initial_weight: float) -> None:
        super().__init__()
        from vforge.distributed.tensor_parallel import mark_tp_sharded

        self.weight = torch.nn.Parameter(torch.tensor([initial_weight]))
        self.unused = torch.nn.Parameter(torch.tensor([7.0]))
        mark_tp_sharded(self, ColwiseParallel())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.weight * value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--overlap-grad-reduce",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--overlap-param-gather",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    expected_world = args.dp_size * args.cp_size * args.tp_size
    if world_size != expected_world:
        raise ValueError(f"WORLD_SIZE must be {expected_world}, got {world_size}")
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from vforge.distributed.mcore_data_parallel import (
        MCoreDataParallelConfig,
        apply_mcore_data_parallel,
        finish_mcore_data_parallel_gradient_sync,
        prepare_mcore_data_parallel_backward,
    )
    from vforge.distributed.parallel_state import (
        ParallelConfig,
        destroy_parallel_state,
        init_parallel_state,
    )
    from vforge.distributed.tensor_parallel import (
        restore_mcore_tensor_parallel_parameter_attributes,
    )
    from vforge.training.diagnostics import DiagnosticsConfig, LogicalReduction
    from vforge.training.mcore_optimizer import (
        backward_mcore_update,
        build_mcore_distributed_optimizer,
    )
    from vforge.training.recipe import OptimizerSpec

    state = init_parallel_state(
        ParallelConfig(
            dp_size=args.dp_size,
            cp_size=args.cp_size,
            tp_size=args.tp_size,
            dp_mode="ddp",
            device_type="cuda",
        )
    )
    initial_weight = float(state.tp_rank + 1)
    local_model = _ShardedScalarModel(initial_weight).to(device)
    # The production engine performs this immediately after materialization.
    # This direct DDP probe must establish the same precondition itself.
    restore_mcore_tensor_parallel_parameter_attributes(local_model)
    model = apply_mcore_data_parallel(
        local_model,
        config=MCoreDataParallelConfig(
            overlap_grad_reduce=args.overlap_grad_reduce,
            overlap_param_gather=args.overlap_param_gather,
        ),
    )
    if set(get_model_state_dict(model)) != {"unused", "weight"}:
        raise AssertionError("MCore DDP changed the logical model state-dict keys")
    optimizer, scheduler = build_mcore_distributed_optimizer(
        model,
        OptimizerSpec(learning_rate=0.1, weight_decay=0.5),
        None,
        clip_grad=args.max_norm,
    )
    local_contribution = float((state.dp_rank + 1) * (state.cp_rank + 1) * (state.tp_rank + 1))
    result = None
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        prepare_mcore_data_parallel_backward(model)
        output = model(torch.tensor([local_contribution], device=device))
        collect_parity = step == args.steps - 1
        result = backward_mcore_update(
            loss=output.sum(),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            usp_size=1,
            parity=None,
            precision_context=nullcontext,
            reduction=LogicalReduction.for_model(model, state),
            diagnostics_config=DiagnosticsConfig(level="parity" if collect_parity else "basic"),
            parameter_bucket=((lambda name: name.rsplit(".", 1)[-1]) if collect_parity else None),
            backward_scale=args.cp_size,
            finish_gradient_sync=lambda: finish_mcore_data_parallel_gradient_sync(model),
        )
    assert result is not None

    dp_mean = (args.dp_size + 1) / 2.0
    cp_sum = args.cp_size * (args.cp_size + 1) / 2.0
    expected_grads = [dp_mean * cp_sum * float(tp_rank + 1) for tp_rank in range(args.tp_size)]
    expected_norm = sum(gradient**2 for gradient in expected_grads) ** 0.5
    expected_coefficient = min(1.0, args.max_norm / (expected_norm + 1e-6))
    # Every AdamW step bias-corrects the unchanged positive gradient to one.
    expected_weight = initial_weight
    for _ in range(args.steps):
        expected_update = 1.0 + 0.5 * expected_weight
        expected_weight -= 0.1 * expected_update
    logical_model = getattr(model, "module", model)

    torch.testing.assert_close(
        logical_model.weight,
        torch.tensor([expected_weight], device=device),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        logical_model.unused,
        torch.tensor([7.0], device=device),
        atol=0.0,
        rtol=0.0,
    )
    if logical_model.unused.grad is not None:
        raise AssertionError("unused parameter unexpectedly received an optimizer gradient")
    actual_norm = result.diagnostics["diag/grad_norm_post_reduce"]
    if abs(actual_norm - expected_norm) > 1e-6:
        raise AssertionError(f"logical norm mismatch: {actual_norm} != {expected_norm}")
    bucket_norm = result.diagnostics["probe/bucket_weight/grad_norm_post_reduce"]
    if abs(bucket_norm - expected_norm) > 1e-6:
        raise AssertionError(f"bucket norm mismatch: {bucket_norm} != {expected_norm}")
    adam_numel = result.diagnostics["probe/adam_exp_avg_numel"]
    # TE FusedAdam eagerly allocates zero state for the unused parameter, but
    # parity diagnostics expose logical optimizer state. Match torch AdamW's
    # state-dict semantics by counting only parameters that participated in the
    # step: one sharded weight scalar per TP rank.
    expected_adam_numel = float(args.tp_size)
    if adam_numel != expected_adam_numel:
        raise AssertionError(f"Adam shard numel mismatch: {adam_numel} != {expected_adam_numel}")
    coefficients = [torch.empty((), device=device) for _ in range(world_size)]
    dist.all_gather(
        coefficients,
        torch.tensor(result.diagnostics["diag/clip_coefficient"], device=device),
    )
    coefficient_values = [float(value.item()) for value in coefficients]
    if any(abs(value - expected_coefficient) > 1e-6 for value in coefficient_values):
        raise AssertionError(
            f"clip coefficients differ: {coefficient_values}, expected {expected_coefficient}"
        )

    report: dict[str, object] = {
        "passed": True,
        "gpu": torch.cuda.get_device_name(device),
        "dp_size": args.dp_size,
        "cp_size": args.cp_size,
        "tp_size": args.tp_size,
        "steps": args.steps,
        "overlap_grad_reduce": model.ddp_config.overlap_grad_reduce,
        "overlap_param_gather": model.ddp_config.overlap_param_gather,
        "bucket_count": sum(len(buffer.buckets) for buffer in model.buffers),
        "expected_logical_grad_norm": expected_norm,
        "actual_logical_grad_norm": result.diagnostics["diag/grad_norm_post_reduce"],
        "clip_coefficient": result.diagnostics["diag/clip_coefficient"],
        "weights": [],
    }
    gathered_weights = [torch.empty((), device=device) for _ in range(world_size)]
    dist.all_gather(gathered_weights, logical_model.weight.detach().reshape(()))
    report["weights"] = [float(value.item()) for value in gathered_weights]
    if rank == 0:
        print("VFORGE_MCORE_DDP " + json.dumps(report, sort_keys=True), flush=True)

    destroy_parallel_state()
    return report


def main() -> None:
    args = _parser().parse_args()
    try:
        run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
