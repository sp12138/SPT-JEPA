"""Distributed parameter, gradient, and optimizer diagnostic metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn

from vforge.distributed.parallel_state import ParallelState
from vforge.distributed.tensor_parallel import tensor_parallel_sharded_parameter_names
from vforge.training.parity.config import ParityStepConfig
from vforge.training.parity.optimizer_step import (
    manual_adamw_bf16_step,
    quantize_params_after_step,
    resolve_grad_scale,
    scale_gradients,
)
from vforge.training.parity.probe_metrics import legacy_probe_aliases
from vforge.training.recipe import GradientClipper
from vforge.utils.gradients import parameter_gradient
from vforge.utils.tensor import to_local_tensor as _local_tensor

ParameterBucket = Callable[[str], str | None]


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Control the cost of per-step distributed training diagnostics."""

    level: Literal["basic", "parity"] = "basic"
    log_local_parameter_stats: bool = False

    def __post_init__(self) -> None:
        if self.level not in {"basic", "parity"}:
            raise ValueError("diagnostics level must be either 'basic' or 'parity'")

    @property
    def collect_parity_metrics(self) -> bool:
        return self.level == "parity"


@dataclass(frozen=True)
class LogicalReduction:
    """Rank-local parameters that represent one logical, unsharded model.

    Collectives still run over the world so every rank receives the same result.
    What changes is which physical copies contribute to that result: replicated
    DP models contribute from one replica, TP shards all contribute, and TP
    replicas contribute from one TP rank.

    Membership goes by parameter identity, not by name, because the caller that
    builds this and the caller that consumes it need not share a name scope. A
    multi-model algorithm builds it once from the bundle root, where a parameter
    is ``generator.wan_2.blocks.0.attn1.to_k.weight``, then clips one role at a
    time, where the same parameter is ``wan_2.blocks.0.attn1.to_k.weight``.
    Matching those by name excludes every parameter, and an empty contribution
    set is indistinguishable from a legitimately silent replica: the norm comes
    back as zero and clipping quietly stops happening.
    """

    contributing_parameters: frozenset[int] | None = None

    @classmethod
    def for_model(cls, model: nn.Module, state: ParallelState) -> LogicalReduction:
        if not state.initialized:
            return cls()
        contributing = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        replicated_dp = (
            state.config.dp_mode == "ddp" and state.dp_cp_size > 1 and (state.dp_cp_rank != 0)
        )
        replicated_hsdp = (
            state.config.dp_mode == "fsdp2"
            and state.config.dp_replicate_size > 1
            and state.dp_replicate_rank != 0
        )
        if replicated_dp or replicated_hsdp:
            contributing.clear()
        if state.tp_enabled and state.tp_rank != 0:
            sharded = tensor_parallel_sharded_parameter_names(model)
            contributing.intersection_update(
                id(parameter) for name, parameter in model.named_parameters() if name in sharded
            )
        return cls(frozenset(contributing))

    def includes(self, parameter: nn.Parameter) -> bool:
        contributing = self.contributing_parameters
        return contributing is None or id(parameter) in contributing


@torch.no_grad()
def log_local_parameter_stats(
    model: nn.Module,
    *,
    phase: str,
    rank: int,
) -> None:
    """Print one mean/std record per local parameter shard without collectives."""
    for name, parameter in model.named_parameters():
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        values = local.detach().float()
        if values.numel() == 0:
            mean = std = float("nan")
        else:
            mean = float(values.mean().cpu())
            std = float(values.std(unbiased=False).cpu())
        print(
            "VFORGE_LOCAL_PARAMETER_STATS "
            f"phase={phase} rank={rank} name={name} "
            f"shape={tuple(values.shape)} numel={values.numel()} "
            f"mean={mean:.9g} std={std:.9g}",
            flush=True,
        )


@dataclass(frozen=True)
class BackwardUpdateResult:
    diagnostics: dict[str, float]
    backward_at: float
    optimizer_at: float


def _named_local_params(
    model: nn.Module,
    reduction: LogicalReduction | None = None,
) -> list[tuple[str, nn.Parameter, torch.Tensor]]:
    return [
        (name, param, _local_tensor(param.detach()))
        for name, param in model.named_parameters()
        if param.requires_grad and (reduction is None or reduction.includes(param))
    ]


def _local_params(
    model: nn.Module,
    reduction: LogicalReduction | None = None,
) -> list[tuple[nn.Parameter, torch.Tensor]]:
    return [
        (param, _local_tensor(param.detach()))
        for _name, param in model.named_parameters()
        if param.requires_grad and (reduction is None or reduction.includes(param))
    ]


def _global_sum_and_max(
    local_sum: torch.Tensor,
    local_max: torch.Tensor,
) -> tuple[float, float]:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
    return float(local_sum.sqrt().item()), float(local_max.item())


def global_grad_norm(
    model: nn.Module,
    reduction: LogicalReduction | None = None,
) -> float:
    device = next(model.parameters()).device
    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    max_abs = torch.zeros((), device=device, dtype=torch.float32)
    for _name, param in model.named_parameters():
        if reduction is not None and not reduction.includes(param):
            continue
        grad = parameter_gradient(param)
        if grad is None:
            continue
        local_grad = _local_tensor(grad.detach()).to(torch.float32)
        if local_grad.numel() == 0:
            continue
        total_sq += local_grad.pow(2).sum()
        max_abs = torch.maximum(max_abs, local_grad.abs().max())
    norm, _ = _global_sum_and_max(total_sq, max_abs)
    return norm


def global_grad_bucket_norms(
    model: nn.Module,
    parameter_bucket: ParameterBucket,
    reduction: LogicalReduction | None = None,
) -> dict[str, float]:
    """Return pre-clipping logical gradient norms grouped by parameter role."""
    device = next(model.parameters()).device
    bucket_squares: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        bucket = parameter_bucket(name) or "other"
        total_sq = bucket_squares.setdefault(
            bucket,
            torch.zeros((), device=device, dtype=torch.float32),
        )
        if reduction is not None and not reduction.includes(parameter):
            continue
        grad = parameter_gradient(parameter)
        if grad is None:
            continue
        local_grad = _local_tensor(grad.detach()).to(torch.float32)
        if local_grad.numel() != 0:
            total_sq += local_grad.square().sum()
    if dist.is_available() and dist.is_initialized():
        for bucket in sorted(bucket_squares):
            dist.all_reduce(bucket_squares[bucket], op=dist.ReduceOp.SUM)
    return {
        f"probe/bucket_{bucket}/grad_norm_post_reduce": float(total_sq.sqrt().item())
        for bucket, total_sq in bucket_squares.items()
    }


@torch.no_grad()
def clip_grad_norm_logical_(
    model: nn.Module,
    max_norm: float,
    reduction: LogicalReduction | None = None,
) -> float:
    """Clip every local gradient by one logical-model L2 norm.

    Contribution filtering affects only the norm calculation.  Once its scalar
    coefficient has been reduced to every rank, all physical shards and replicas
    must be scaled so they remain identical where the layout requires it.
    """
    total_norm = global_grad_norm(model, reduction)
    coefficient = min(1.0, float(max_norm) / (total_norm + 1e-6))
    if coefficient < 1.0:
        for parameter in model.parameters():
            gradient = parameter_gradient(parameter)
            if gradient is None:
                continue
            _local_tensor(gradient).mul_(coefficient)
    return total_norm


def validate_gradient_norm_dtype(norm_dtype: str, *, engine: str, tp_size: int):
    """Validate precision and topology before allocating the runtime."""
    if norm_dtype not in {"float32", "float64"}:
        raise ValueError("gradient_norm_dtype must be float32 or float64")
    if norm_dtype == "float64" and (engine != "fsdp2" or tp_size != 1):
        raise ValueError("FP64 gradient norm requires FSDP2 and TP=1")


def torch_gradient_clipper(norm_dtype: str):
    """Build the torch clipping callback selected by a training recipe."""
    if norm_dtype not in {"float32", "float64"}:
        raise ValueError("gradient_norm_dtype must be float32 or float64")
    return partial(clip_grad_norm_torch_, norm_dtype=norm_dtype)


def gradient_clipping_fingerprint(clipper, norm_dtype: str) -> dict[str, str]:
    """Preserve existing checkpoint keys for the resolved numerical contract.

    Only the built-in torch factory supports an explicit FP64 norm. Existing
    custom clippers keep ownership of their precision and the FP32 default.
    """
    if isinstance(clipper, partial) and clipper.func is clip_grad_norm_torch_:
        if clipper.keywords.get("norm_dtype") != norm_dtype:
            raise ValueError("gradient clipper precision does not match gradient_norm_dtype")
        return {"gradient_clip_impl": "torch", "gradient_norm_dtype": norm_dtype}
    if norm_dtype != "float32":
        raise ValueError("FP64 gradient norm requires a recipe with a precision-aware clipper")
    return {}


@torch.no_grad()
def clip_grad_norm_torch_(
    model, optimizer, max_norm, reduction=None, *, norm_dtype="float32"
) -> float:
    """Clip a complete TP=1 update target, preserving DTensor reductions.

    The engine passes the independently optimized tower, not the forward bundle.
    Unlike logical clipping, PyTorch derives replication/sharding from DTensors.
    """
    if norm_dtype not in {"float32", "float64"}:
        raise ValueError("gradient norm dtype must be float32 or float64")
    parameters = [p for p in model.parameters() if p.grad is not None]
    dtype = getattr(torch, norm_dtype)
    if all(p.grad.dtype == dtype for p in parameters):
        # Preserve the established PyTorch FP32 path for LTX trainable towers.
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    else:
        # Cast one gradient at a time; keep only per-parameter scalar norms.
        norms = [torch.linalg.vector_norm(p.grad.detach().to(dtype), 2) for p in parameters]
        norm = torch.nn.utils.get_total_norm(norms, 2)
        torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, norm)
    if hasattr(norm, "full_tensor"):
        norm = norm.full_tensor()
    return float(norm)


def _post_reduce_grad_norm(
    model: nn.Module,
    reduction: LogicalReduction | None = None,
) -> float:
    """Return the optimizer-visible norm without scaling gradients."""
    return global_grad_norm(model, reduction)


def global_grad_value_stats(model: nn.Module) -> dict[str, float]:
    """Return deterministic distributed moments of optimizer-visible gradients."""
    device = next(model.parameters()).device
    stats = torch.zeros(5, device=device, dtype=torch.float64)
    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None:
            continue
        value = _local_tensor(grad.detach()).to(torch.float64)
        if value.numel() == 0:
            continue
        stats[0] += value.numel()
        stats[1] += value.sum()
        stats[2] += value.abs().sum()
        stats[3] += value.square().sum()
        stats[4] = torch.maximum(stats[4], value.abs().max())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats[:4], op=dist.ReduceOp.SUM)
        dist.all_reduce(stats[4], op=dist.ReduceOp.MAX)
    return {
        "numel": float(stats[0].item()),
        "sum": float(stats[1].item()),
        "abs_sum": float(stats[2].item()),
        "square_sum": float(stats[3].item()),
        "max_abs": float(stats[4].item()),
    }


def global_param_norm(
    model: nn.Module,
    reduction: LogicalReduction | None = None,
) -> float:
    device = next(model.parameters()).device
    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    max_abs = torch.zeros((), device=device, dtype=torch.float32)
    for _, local_param in _local_params(model, reduction):
        value = local_param.to(torch.float32)
        if value.numel() == 0:
            continue
        total_sq += value.pow(2).sum()
        max_abs = torch.maximum(max_abs, value.abs().max())
    norm, _ = _global_sum_and_max(total_sq, max_abs)
    return norm


def global_param_numel(
    model: nn.Module,
    reduction: LogicalReduction | None = None,
) -> float:
    device = next(model.parameters()).device
    total = torch.zeros((), device=device, dtype=torch.int64)
    for _, local_param in _local_params(model, reduction):
        total += local_param.numel()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return float(total.item())


def global_param_value_stats(
    model: nn.Module,
    reduction: LogicalReduction | None = None,
) -> dict[str, float]:
    """Return distributed moments of trainable parameter values."""
    device = next(model.parameters()).device
    stats = torch.zeros(5, device=device, dtype=torch.float64)
    for _, local_param in _local_params(model, reduction):
        value = local_param.to(torch.float64)
        if value.numel() == 0:
            continue
        stats[0] += value.numel()
        stats[1] += value.sum()
        stats[2] += value.abs().sum()
        stats[3] += value.square().sum()
        stats[4] = torch.maximum(stats[4], value.abs().max())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats[:4], op=dist.ReduceOp.SUM)
        dist.all_reduce(stats[4], op=dist.ReduceOp.MAX)
    return {
        "numel": float(stats[0].item()),
        "sum": float(stats[1].item()),
        "abs_sum": float(stats[2].item()),
        "square_sum": float(stats[3].item()),
        "max_abs": float(stats[4].item()),
    }


def snapshot_trainable_params(
    model: nn.Module,
) -> list[tuple[str, torch.Tensor]]:
    return [
        (name, local_param.detach().clone()) for name, _, local_param in _named_local_params(model)
    ]


def global_param_update_stats(
    model: nn.Module,
    snapshot: list[tuple[str, torch.Tensor]],
    parameter_bucket: ParameterBucket | None = None,
    reduction: LogicalReduction | None = None,
) -> dict[str, float]:
    device = next(model.parameters()).device
    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    max_abs = torch.zeros((), device=device, dtype=torch.float32)
    param_sq = torch.zeros((), device=device, dtype=torch.float32)
    update_numel = torch.zeros((), device=device, dtype=torch.int64)
    update_nonzero = torch.zeros((), device=device, dtype=torch.int64)
    update_abs_sum = torch.zeros((), device=device, dtype=torch.float32)
    update_signed_sum = torch.zeros((), device=device, dtype=torch.float32)
    update_param_dot = torch.zeros((), device=device, dtype=torch.float32)
    update_param_abs_dot = torch.zeros((), device=device, dtype=torch.float32)
    buckets: dict[str, dict[str, torch.Tensor]] = {}
    for (name, parameter, local_param), (before_name, before) in zip(
        _named_local_params(model),
        snapshot,
        strict=True,
    ):
        if name != before_name:
            raise RuntimeError(f"parameter snapshot mismatch: {name} != {before_name}")
        bucket = parameter_bucket(name) if parameter_bucket is not None else None
        if bucket is not None:
            stats = buckets.setdefault(
                bucket,
                {
                    key: torch.zeros((), device=device, dtype=torch.float32)
                    for key in (
                        "sq",
                        "max",
                        "numel",
                        "nonzero",
                        "abs_sum",
                        "signed_sum",
                        "param_dot",
                        "param_abs_dot",
                    )
                },
            )
        else:
            stats = None
        if reduction is not None and not reduction.includes(parameter):
            continue
        after_f = local_param.detach().to(torch.float32)
        before_f = before.to(device=after_f.device, dtype=torch.float32)
        if after_f.numel() == 0:
            continue
        delta = after_f - before_f
        total_sq += delta.pow(2).sum()
        max_abs = torch.maximum(max_abs, delta.abs().max())
        param_sq += after_f.pow(2).sum()
        update_numel += delta.numel()
        update_nonzero += torch.count_nonzero(delta)
        update_abs_sum += delta.abs().sum()
        update_signed_sum += delta.sum()
        update_param_dot += (delta * before_f).sum()
        update_param_abs_dot += (delta.abs() * before_f.abs()).sum()
        if stats is not None:
            stats["sq"] += delta.pow(2).sum()
            stats["max"] = torch.maximum(stats["max"], delta.abs().max())
            stats["numel"] += float(delta.numel())
            stats["nonzero"] += float(torch.count_nonzero(delta).item())
            stats["abs_sum"] += delta.abs().sum()
            stats["signed_sum"] += delta.sum()
            stats["param_dot"] += (delta * before_f).sum()
            stats["param_abs_dot"] += (delta.abs() * before_f.abs()).sum()
    update_norm, update_max = _global_sum_and_max(total_sq, max_abs)
    param_norm, _ = _global_sum_and_max(
        param_sq,
        torch.zeros((), device=device, dtype=torch.float32),
    )
    if dist.is_available() and dist.is_initialized():
        for value in (
            update_numel,
            update_nonzero,
            update_abs_sum,
            update_signed_sum,
            update_param_dot,
            update_param_abs_dot,
        ):
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        for stats in buckets.values():
            for key in (
                "sq",
                "numel",
                "nonzero",
                "abs_sum",
                "signed_sum",
                "param_dot",
                "param_abs_dot",
            ):
                dist.all_reduce(stats[key], op=dist.ReduceOp.SUM)
            dist.all_reduce(stats["max"], op=dist.ReduceOp.MAX)
    metrics = {
        "probe/param_norm_after_update": param_norm,
        "probe/update_norm": update_norm,
        "probe/update_max_abs": update_max,
        "probe/update_ratio": update_norm / (param_norm + 1e-12),
        "probe/update_numel": float(update_numel.item()),
        "probe/update_nonzero_count": float(update_nonzero.item()),
        "probe/update_nonzero_ratio": float(update_nonzero.item() / (update_numel.item() + 1e-12)),
        "probe/update_abs_sum": float(update_abs_sum.item()),
        "probe/update_signed_sum": float(update_signed_sum.item()),
        "probe/update_param_dot": float(update_param_dot.item()),
        "probe/update_param_abs_dot": float(update_param_abs_dot.item()),
    }
    for bucket, stats in buckets.items():
        prefix = f"probe/bucket_{bucket}"
        numel = float(stats["numel"].item())
        nonzero = float(stats["nonzero"].item())
        metrics[f"{prefix}/update_norm"] = float(stats["sq"].sqrt().item())
        metrics[f"{prefix}/update_max_abs"] = float(stats["max"].item())
        metrics[f"{prefix}/update_numel"] = numel
        metrics[f"{prefix}/update_nonzero_count"] = nonzero
        metrics[f"{prefix}/update_nonzero_ratio"] = nonzero / (numel + 1e-12)
        for key in (
            "abs_sum",
            "signed_sum",
            "param_dot",
            "param_abs_dot",
        ):
            metrics[f"{prefix}/update_{key}"] = float(stats[key].item())
    return metrics


def global_optimizer_state_stats(
    optimizer: Any,
    model: nn.Module,
    reduction: LogicalReduction | None = None,
) -> dict[str, float]:
    device = next(model.parameters()).device
    exp_avg_sq_sum = torch.zeros((), device=device, dtype=torch.float32)
    exp_avg_sq_max = torch.zeros((), device=device, dtype=torch.float32)
    exp_avg_sq_sq_sum = torch.zeros((), device=device, dtype=torch.float32)
    exp_avg_sq_sq_max = torch.zeros((), device=device, dtype=torch.float32)
    # Element counts exceed float32's exact-integer range on 14B models.  Keep
    # them integral through both local accumulation and the distributed sum;
    # otherwise different sharding layouts can appear to have different Adam
    # state sizes solely because their additions round in a different order.
    exp_avg_numel = torch.zeros((), device=device, dtype=torch.int64)
    exp_avg_sq_numel = torch.zeros((), device=device, dtype=torch.int64)
    step_min = torch.full((), float("inf"), device=device, dtype=torch.float32)
    step_max = torch.full((), -float("inf"), device=device, dtype=torch.float32)
    state_count = torch.zeros((), device=device, dtype=torch.int64)

    for parameter, state in optimizer.state.items():
        if not state:
            continue
        if reduction is not None and not reduction.includes(parameter):
            continue
        state_count += 1
        step = state.get("step")
        if step is not None:
            step_value = (
                step.detach().to(device=device, dtype=torch.float32)
                if isinstance(step, torch.Tensor)
                else torch.tensor(float(step), device=device, dtype=torch.float32)
            )
            step_min = torch.minimum(step_min, step_value)
            step_max = torch.maximum(step_max, step_value)
        exp_avg = state.get("exp_avg")
        if isinstance(exp_avg, torch.Tensor):
            local = _local_tensor(exp_avg.detach()).to(torch.float32)
            if local.numel() > 0:
                exp_avg_sq_sum += local.pow(2).sum()
                exp_avg_sq_max = torch.maximum(
                    exp_avg_sq_max,
                    local.abs().max(),
                )
                exp_avg_numel += local.numel()
        exp_avg_sq = state.get("exp_avg_sq")
        if isinstance(exp_avg_sq, torch.Tensor):
            local_sq = _local_tensor(exp_avg_sq.detach()).to(torch.float32)
            if local_sq.numel() > 0:
                exp_avg_sq_sq_sum += local_sq.pow(2).sum()
                exp_avg_sq_sq_max = torch.maximum(
                    exp_avg_sq_sq_max,
                    local_sq.abs().max(),
                )
                exp_avg_sq_numel += local_sq.numel()

    if dist.is_available() and dist.is_initialized():
        for value, operation in (
            (exp_avg_sq_sum, dist.ReduceOp.SUM),
            (exp_avg_sq_max, dist.ReduceOp.MAX),
            (exp_avg_sq_sq_sum, dist.ReduceOp.SUM),
            (exp_avg_sq_sq_max, dist.ReduceOp.MAX),
            (exp_avg_numel, dist.ReduceOp.SUM),
            (exp_avg_sq_numel, dist.ReduceOp.SUM),
            (step_min, dist.ReduceOp.MIN),
            (step_max, dist.ReduceOp.MAX),
            (state_count, dist.ReduceOp.SUM),
        ):
            dist.all_reduce(value, op=operation)

    has_step = bool(torch.isfinite(step_min).item())
    return {
        "probe/adam_state_count": float(state_count.item()),
        "probe/adam_step_min": float(step_min.item()) if has_step else 0.0,
        "probe/adam_step_max": float(step_max.item()) if has_step else 0.0,
        "probe/adam_exp_avg_norm": float(exp_avg_sq_sum.sqrt().item()),
        "probe/adam_exp_avg_max_abs": float(exp_avg_sq_max.item()),
        "probe/adam_exp_avg_numel": float(exp_avg_numel.item()),
        "probe/adam_exp_avg_sq_norm": float(exp_avg_sq_sq_sum.sqrt().item()),
        "probe/adam_exp_avg_sq_max_abs": float(exp_avg_sq_sq_max.item()),
        "probe/adam_exp_avg_sq_numel": float(exp_avg_sq_numel.item()),
    }


def run_backward(
    *,
    loss: torch.Tensor,
    backward_scale: float = 1.0,
    offload_manager: Any = None,
    backward_losses: Callable[[], Iterable[torch.Tensor]] | None = None,
    backward_forward_context: Callable = nullcontext,
) -> None:
    """Backpropagate one micro-batch, including its offload lifecycle."""
    backward_scale = float(backward_scale)
    if backward_scale <= 0:
        raise ValueError(f"backward_scale must be positive, got {backward_scale}")
    if backward_losses is not None and offload_manager is not None:
        raise ValueError("lazy backward losses do not support activation offload")
    if offload_manager is not None:
        offload_manager.start_backward_prefetch()
    losses = iter((loss,)) if backward_losses is None else iter(backward_losses())
    count = 0
    try:
        while True:
            with backward_forward_context():
                try:
                    item = next(losses)
                except StopIteration:
                    break
            backward_loss = item if backward_scale == 1.0 else item * backward_scale
            backward_loss.backward()
            count += 1
            del item, backward_loss
        if count == 0:
            raise ValueError("backward losses must contain at least one scalar loss")
    finally:
        close = getattr(losses, "close", None)
        if close is not None:
            close()
        if offload_manager is not None:
            offload_manager.finish_backward_prefetch()


def clip_and_step(
    *,
    model: nn.Module,
    optimizer: Any,
    scheduler: Any,
    usp_size: int,
    gradient_clip_val: float,
    parity: ParityStepConfig | None,
    parameter_bucket: ParameterBucket | None = None,
    diagnostics_config: DiagnosticsConfig | None = None,
    reduction: LogicalReduction | None = None,
    gradient_clipper: GradientClipper | None = None,
) -> dict[str, float]:
    """Diagnose and clip accumulated gradients, then update exactly once."""
    usp_size = max(1, int(usp_size))
    collect_parity_metrics = bool(
        diagnostics_config is not None and diagnostics_config.collect_parity_metrics
    )
    gradient_bucket_metrics = (
        global_grad_bucket_norms(model, parameter_bucket, reduction)
        if collect_parity_metrics and parameter_bucket is not None
        else {}
    )
    if gradient_clip_val > 0:
        post_reduce_norm = (
            gradient_clipper(model, optimizer, gradient_clip_val, reduction)
            if gradient_clipper is not None
            else clip_grad_norm_logical_(
                model,
                gradient_clip_val,
                reduction,
            )
        )
        clip_coefficient = min(
            1.0,
            float(gradient_clip_val) / (post_reduce_norm + 1e-6),
        )
    else:
        post_reduce_norm = _post_reduce_grad_norm(model, reduction)
        clip_coefficient = 1.0
    post_clip_norm = post_reduce_norm * clip_coefficient
    clip_active = float(clip_coefficient < 1.0)
    diagnostics: dict[str, float] = {
        "diag/grad_norm_post_reduce": post_reduce_norm,
        "diag/grad_norm_post_clip": post_clip_norm,
        "diag/clip_coefficient": clip_coefficient,
        "diag/clip_active": clip_active,
    }
    diagnostics.update(gradient_bucket_metrics)
    param_snapshot = snapshot_trainable_params(model) if collect_parity_metrics else None
    if collect_parity_metrics:
        diagnostics.update(
            {
                "probe/usp_size": float(usp_size),
                "probe/grad_norm_before_clip": post_reduce_norm,
                "probe/grad_norm_backward": post_reduce_norm,
                "probe/grad_norm_full_loss_equiv": (post_reduce_norm * usp_size),
                "probe/clip_return_norm_local": post_reduce_norm,
                "probe/clip_return_norm": post_reduce_norm,
                "probe/clip_coef": clip_coefficient,
                "probe/clip_active": clip_active,
                "probe/grad_norm_after_clip": post_clip_norm,
                "probe/grad_norm_after_clip_backward": post_clip_norm,
                "probe/grad_norm_after_clip_full_loss_equiv": (post_clip_norm * usp_size),
                "probe/param_norm_before_update": global_param_norm(model, reduction),
                "probe/param_numel": global_param_numel(model, reduction),
            }
        )

    grad_scale = resolve_grad_scale(
        parity.scale_grads_before_optimizer_step if parity is not None else None,
        usp_size,
    )
    if grad_scale != 1.0:
        scaled_grads = scale_gradients(model, grad_scale)
        if collect_parity_metrics:
            diagnostics["probe/scale_grads_before_optimizer_step"] = grad_scale
            diagnostics["probe/scaled_grad_tensors"] = float(scaled_grads)
    if collect_parity_metrics:
        diagnostics["probe/grad_norm_optimizer_input"] = (
            global_grad_norm(model, reduction) if grad_scale != 1.0 else post_clip_norm
        )
        diagnostics.update(
            {f"probe/grad_{key}": value for key, value in global_grad_value_stats(model).items()}
        )

    optimizer_step_mode = parity.optimizer_step_mode if parity is not None else "torch"
    if optimizer_step_mode == "manual_adamw_bf16":
        manual_updates = manual_adamw_bf16_step(optimizer)
        diagnostics["probe/manual_adamw_bf16_step"] = 1.0
        diagnostics["probe/manual_adamw_updated_tensors"] = float(manual_updates)
    else:
        optimizer.step()
    quantized_params = quantize_params_after_step(
        model,
        parity.quantize_after_optimizer_step if parity is not None else None,
    )
    if quantized_params:
        diagnostics["probe/quantize_after_optimizer_step"] = 1.0
        diagnostics["probe/quantized_param_tensors"] = float(quantized_params)
    if collect_parity_metrics:
        assert param_snapshot is not None
        diagnostics.update(
            global_param_update_stats(
                model,
                param_snapshot,
                parameter_bucket=parameter_bucket,
                reduction=reduction,
            )
        )
        diagnostics.update(global_optimizer_state_stats(optimizer, model, reduction))
        if parity is not None and parity.emit_legacy_probe_aliases:
            diagnostics.update(legacy_probe_aliases(diagnostics))
    if scheduler is not None:
        scheduler.step()
    return diagnostics


__all__ = [
    "BackwardUpdateResult",
    "DiagnosticsConfig",
    "LogicalReduction",
    "ParameterBucket",
    "clip_and_step",
    "clip_grad_norm_logical_",
    "global_grad_bucket_norms",
    "global_grad_norm",
    "global_grad_value_stats",
    "global_optimizer_state_stats",
    "global_param_norm",
    "global_param_numel",
    "global_param_value_stats",
    "global_param_update_stats",
    "run_backward",
    "snapshot_trainable_params",
]
