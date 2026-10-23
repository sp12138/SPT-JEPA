"""vForge lifecycle adapter for MCore's distributed optimizer."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from vforge.distributed.mcore_data_parallel import (
    finish_mcore_data_parallel_parameter_sync,
    is_mcore_data_parallel,
    mcore_process_group_collection,
)
from vforge.training.parity.optimizer_step import resolve_grad_scale
from vforge.training.recipe import OptimizerSpec, SchedulerSpec
from vforge.utils.gradients import parameter_gradient


@dataclass(frozen=True)
class MCoreOptimizerStepResult:
    """Numerical outcome of one MCore optimizer step."""

    success: bool
    grad_norm: float
    clip_coefficient: float
    gradient_bucket_norms: dict[str, float]


class MCoreDistributedOptimizer:
    """Expose a small engine-facing surface over MCore DistributedOptimizer.

    MCore deliberately splits ``step`` into gradient preparation, clipping and
    parameter synchronization. Keeping those phases visible here lets vForge
    preserve unused-parameter semantics and collect parity diagnostics after
    reduce-scatter without reaching into them from the generic Trainer.
    """

    def __init__(self, model: nn.Module, optimizer: Any, *, clip_grad: float) -> None:
        self.model = model
        self.inner = optimizer
        self.clip_grad = float(clip_grad)
        # MCore may preallocate optimizer state for every shard. Track which
        # shards have actually participated so diagnostics preserve torch's
        # lazy-state semantics without coupling that behavior to any model's
        # conditional execution policy.
        self._initialized_optimizer_shards: set[int] = set()
        self._mark_existing_optimizer_shards_initialized()

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.inner.param_groups

    @property
    def state(self) -> dict[Any, Any]:
        base = getattr(self.inner, "optimizer", None)
        return getattr(base, "state", {})

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return self.inner.scale_loss(loss)

    @torch.no_grad()
    def reload_main_params_from_pretrained(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Initialize sharded FP32 masters from the original pretrained tensors."""
        self.inner.reload_model_params(state_dict=state_dict)

    @torch.no_grad()
    def step(
        self,
        *,
        parameter_bucket=None,
        grad_scale: float = 1.0,
    ) -> MCoreOptimizerStepResult:
        found_inf = bool(self.inner.prepare_grads())
        self._preserve_unused_parameter_semantics()
        if grad_scale != 1.0:
            self._scale_ready_grads(float(grad_scale))

        bucket_norms = self._logical_gradient_bucket_norms(parameter_bucket)
        if found_inf:
            return MCoreOptimizerStepResult(
                success=False,
                grad_norm=float("inf"),
                clip_coefficient=0.0,
                gradient_bucket_norms=bucket_norms,
            )

        if self.clip_grad > 0:
            grad_norm = float(self.inner.clip_grad_norm(self.clip_grad))
            clip_coefficient = min(1.0, self.clip_grad / (grad_norm + 1e-6))
        else:
            grad_norm = float(self.inner.get_grad_norm())
            clip_coefficient = 1.0
        success = bool(self.inner.step_with_ready_grads())
        return MCoreOptimizerStepResult(
            success=success,
            grad_norm=grad_norm,
            clip_coefficient=clip_coefficient,
            gradient_bucket_norms=bucket_norms,
        )

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError(
            "MCore distributed optimizer state is only exposed through its native "
            "distributed-checkpoint sharded state"
        )

    def checkpoint_sharded_state_dict(
        self,
        model_sharded_state_dict: dict[str, Any],
        *,
        is_loading: bool,
    ) -> dict[str, Any]:
        """Describe native MCore optimizer shards without translating their layout."""
        return self.inner.sharded_state_dict(
            model_sharded_state_dict,
            is_loading=is_loading,
            metadata={"distrib_optim_sharding_type": "dp_reshardable"},
        )

    def load_checkpoint_sharded_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Apply state populated by MCore distributed checkpoint loading."""
        # MCore copies loaded master parameters into views backed by its flat
        # optimizer buffers. Those views may require grad on recent PyTorch
        # versions, so the restore boundary needs optimizer-style no-grad
        # semantics. Keep the compatibility shim here instead of patching the
        # vendored implementation.
        with torch.no_grad():
            self.inner.load_state_dict(state_dict)
        self._mark_existing_optimizer_shards_initialized()

    def _mark_existing_optimizer_shards_initialized(self) -> None:
        base = getattr(self.inner, "optimizer", None)
        for group in getattr(base, "param_groups", ()):
            if int(group.get("step", 0)) <= 0:
                continue
            self._initialized_optimizer_shards.update(id(param) for param in group["params"])
        for parameter, state in getattr(base, "state", {}).items():
            step = state.get("step")
            if isinstance(step, torch.Tensor):
                step = step.item()
            if step is not None and int(step) > 0:
                self._initialized_optimizer_shards.add(id(parameter))

    @torch.no_grad()
    def optimizer_state_stats(self) -> dict[str, float]:
        """Return logical Adam-state metrics over distributed optimizer shards."""
        device = next(self.model.parameters()).device
        exp_avg_sq_sum = torch.zeros((), device=device, dtype=torch.float64)
        exp_avg_max = torch.zeros((), device=device, dtype=torch.float32)
        exp_avg_numel = torch.zeros((), device=device, dtype=torch.float64)
        exp_avg_sq_sq_sum = torch.zeros((), device=device, dtype=torch.float64)
        exp_avg_sq_max = torch.zeros((), device=device, dtype=torch.float32)
        exp_avg_sq_numel = torch.zeros((), device=device, dtype=torch.float64)
        step_min = torch.full((), float("inf"), device=device)
        step_max = torch.zeros((), device=device)
        state_count = torch.zeros((), device=device, dtype=torch.float64)
        tp_group = getattr(self.inner, "tp_group", None)
        base = getattr(self.inner, "optimizer", None)
        optimizer_shard_to_model_parameter = {
            id(optimizer_shard): model_parameter
            for model_parameter, optimizer_shard in self._model_and_optimizer_shards()
        }
        group_step_by_parameter = {
            id(parameter): group.get("step")
            for group in getattr(base, "param_groups", ())
            for parameter in group.get("params", ())
        }
        for parameter, state in getattr(base, "state", {}).items():
            model_parameter = optimizer_shard_to_model_parameter.get(id(parameter))
            # DistributedOptimizer may preallocate zero Adam state for every
            # shard. Match torch's logical state_dict semantics: a parameter
            # unused on every DP x CP replica has no optimizer state this step.
            if (
                model_parameter is not None
                and id(parameter) not in self._initialized_optimizer_shards
            ):
                continue
            source_parameter = parameter if model_parameter is None else model_parameter
            is_tp_shard = bool(getattr(source_parameter, "tensor_model_parallel", False))
            if not is_tp_shard and tp_group is not None and tp_group.rank() != 0:
                continue
            state_count += 1
            step = state.get("step", group_step_by_parameter.get(id(parameter)))
            if isinstance(step, torch.Tensor):
                step = float(step.item())
            if step is not None:
                step_min.copy_(torch.minimum(step_min, torch.tensor(float(step), device=device)))
                step_max.copy_(torch.maximum(step_max, torch.tensor(float(step), device=device)))
            exp_avg = state.get("exp_avg")
            if isinstance(exp_avg, torch.Tensor):
                values = exp_avg.detach().float()
                exp_avg_sq_sum += values.double().pow(2).sum()
                if values.numel():
                    exp_avg_max.copy_(torch.maximum(exp_avg_max, values.abs().max()))
                exp_avg_numel += values.numel()
            exp_avg_sq = state.get("exp_avg_sq")
            if isinstance(exp_avg_sq, torch.Tensor):
                values = exp_avg_sq.detach().float()
                exp_avg_sq_sq_sum += values.double().pow(2).sum()
                if values.numel():
                    exp_avg_sq_max.copy_(torch.maximum(exp_avg_sq_max, values.abs().max()))
                exp_avg_sq_numel += values.numel()
        if dist.is_initialized():
            for value, operation in (
                (exp_avg_sq_sum, dist.ReduceOp.SUM),
                (exp_avg_max, dist.ReduceOp.MAX),
                (exp_avg_numel, dist.ReduceOp.SUM),
                (exp_avg_sq_sq_sum, dist.ReduceOp.SUM),
                (exp_avg_sq_max, dist.ReduceOp.MAX),
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
            "probe/adam_exp_avg_max_abs": float(exp_avg_max.item()),
            "probe/adam_exp_avg_numel": float(exp_avg_numel.item()),
            "probe/adam_exp_avg_sq_norm": float(exp_avg_sq_sq_sum.sqrt().item()),
            "probe/adam_exp_avg_sq_max_abs": float(exp_avg_sq_max.item()),
            "probe/adam_exp_avg_sq_numel": float(exp_avg_sq_numel.item()),
        }

    def load_state_dict(self, _state_dict: dict[str, Any]) -> None:
        raise NotImplementedError(
            "MCore distributed optimizer state requires the vForge DCP adapter"
        )

    def _model_and_optimizer_shards(self):
        inner = self.inner
        if inner.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            shard_float16_groups = inner.shard_float16_groups
        else:
            shard_float16_groups = inner.shard_fp32_from_float16_groups
        for model_groups, shard_groups in (
            (inner.model_float16_groups, shard_float16_groups),
            (inner.model_fp32_groups, inner.shard_fp32_groups),
        ):
            for model_group, shard_group in zip(model_groups, shard_groups, strict=True):
                yield from zip(model_group, shard_group, strict=True)

    def _preserve_unused_parameter_semantics(self) -> None:
        """Keep globally unused parameters out of AdamW and its state."""
        use_decoupled_grad = self.inner.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        for model_parameter, optimizer_shard in self._model_and_optimizer_shards():
            if parameter_gradient(model_parameter) is not None:
                self._initialized_optimizer_shards.add(id(optimizer_shard))
                continue
            if use_decoupled_grad:
                optimizer_shard.decoupled_grad = None
            else:
                optimizer_shard.grad = None

    def _ready_grad(self, parameter: nn.Parameter) -> torch.Tensor | None:
        if self.inner.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            return getattr(parameter, "decoupled_grad", None)
        return parameter.grad

    def _scale_ready_grads(self, factor: float) -> None:
        for parameter in self.inner.get_parameters():
            gradient = self._ready_grad(parameter)
            if gradient is not None:
                gradient.mul_(factor)

    def _logical_gradient_bucket_norms(self, parameter_bucket) -> dict[str, float]:
        if parameter_bucket is None:
            return {}
        names = {id(parameter): name for name, parameter in self.model.named_parameters()}
        buckets = sorted({parameter_bucket(name) or "other" for name in names.values()})
        tp_group = getattr(self.inner, "tp_group", None)
        local: dict[str, torch.Tensor] = {}
        for model_parameter, optimizer_shard in self._model_and_optimizer_shards():
            gradient = self._ready_grad(optimizer_shard)
            if gradient is None:
                continue
            is_tp_shard = bool(getattr(model_parameter, "tensor_model_parallel", False))
            if not is_tp_shard and tp_group is not None and tp_group.rank() != 0:
                continue
            name = names.get(id(model_parameter))
            if name is None:
                continue
            bucket = parameter_bucket(name) or "other"
            value = gradient.detach().float().pow(2).sum()
            local[bucket] = local.get(bucket, torch.zeros_like(value)) + value

        result: dict[str, float] = {}
        device = next(self.model.parameters()).device
        for bucket in buckets:
            squared = local.get(bucket, torch.zeros((), device=device, dtype=torch.float32))
            if dist.is_initialized():
                dist.all_reduce(squared, op=dist.ReduceOp.SUM)
            result[f"probe/bucket_{bucket}/grad_norm_post_reduce"] = float(squared.sqrt().item())
        return result


def backward_mcore_update(
    *,
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: MCoreDistributedOptimizer,
    scheduler: Any,
    usp_size: int,
    parity: Any,
    precision_context: Callable[[], AbstractContextManager[Any]],
    finish_gradient_sync: Callable[[], None],
    offload_manager: Any = None,
    parameter_bucket=None,
    diagnostics_config=None,
    reduction=None,
    backward_scale: float = 1.0,
):
    """Backward and update through MCore-owned gradient and optimizer shards."""
    from vforge.training.diagnostics import (
        BackwardUpdateResult,
        global_param_norm,
        global_param_numel,
        global_param_update_stats,
        snapshot_trainable_params,
    )

    if parity is not None:
        if parity.optimizer_step_mode != "torch":
            raise NotImplementedError("MCore optimizer does not support manual optimizer parity")
        if parity.quantize_after_optimizer_step is not None:
            raise NotImplementedError("MCore optimizer does not support post-step quantization")
    grad_scale = resolve_grad_scale(
        parity.scale_grads_before_optimizer_step if parity is not None else None,
        max(1, int(usp_size)),
    )
    if grad_scale != 1.0:
        raise NotImplementedError("MCore optimizer parity gradient scaling is not supported yet")

    if offload_manager is not None:
        offload_manager.start_backward_prefetch()
    try:
        with precision_context():
            backward_loss = loss if backward_scale == 1.0 else loss * float(backward_scale)
            optimizer.scale_loss(backward_loss).backward()
    finally:
        if offload_manager is not None:
            offload_manager.finish_backward_prefetch()
    finish_gradient_sync()
    backward_at = time.perf_counter()

    collect_parity = bool(
        diagnostics_config is not None and diagnostics_config.collect_parity_metrics
    )
    snapshot = snapshot_trainable_params(model) if collect_parity else None
    param_norm_before = global_param_norm(model, reduction) if collect_parity else 0.0
    param_numel = global_param_numel(model, reduction) if collect_parity else 0.0
    step_result = optimizer.step(parameter_bucket=parameter_bucket if collect_parity else None)
    if not step_result.success:
        raise FloatingPointError(
            "MCore optimizer skipped the update after detecting non-finite grads"
        )

    post_clip_norm = step_result.grad_norm * step_result.clip_coefficient
    diagnostics = {
        "diag/grad_norm_post_reduce": step_result.grad_norm,
        "diag/grad_norm_post_clip": post_clip_norm,
        "diag/clip_coefficient": step_result.clip_coefficient,
        "diag/clip_active": float(step_result.clip_coefficient < 1.0),
        "diag/mcore_overlap_grad_reduce": float(model.ddp_config.overlap_grad_reduce),
        "diag/mcore_overlap_param_gather": float(model.ddp_config.overlap_param_gather),
        "diag/mcore_grad_bucket_count": float(
            sum(len(buffer.buckets) for buffer in model.buffers + model.expert_parallel_buffers)
        ),
    }
    diagnostics.update(step_result.gradient_bucket_norms)
    if collect_parity:
        diagnostics.update(
            {
                "probe/usp_size": float(max(1, int(usp_size))),
                "probe/grad_norm_before_clip": step_result.grad_norm,
                "probe/grad_norm_backward": step_result.grad_norm,
                "probe/grad_norm_full_loss_equiv": (step_result.grad_norm * max(1, int(usp_size))),
                "probe/clip_return_norm_local": step_result.grad_norm,
                "probe/clip_return_norm": step_result.grad_norm,
                "probe/clip_coef": step_result.clip_coefficient,
                "probe/clip_active": float(step_result.clip_coefficient < 1.0),
                "probe/grad_norm_after_clip": post_clip_norm,
                "probe/grad_norm_after_clip_backward": post_clip_norm,
                "probe/grad_norm_after_clip_full_loss_equiv": (
                    post_clip_norm * max(1, int(usp_size))
                ),
                "probe/grad_norm_optimizer_input": post_clip_norm,
                "probe/param_norm_before_update": param_norm_before,
                "probe/param_numel": param_numel,
            }
        )
        assert snapshot is not None
        # DistributedOptimizer normally leaves parameter all-gathers to the
        # next forward so they overlap with compute. Parity diagnostics inspect
        # every updated parameter immediately, so finish the gather only on
        # this diagnostic path.
        finish_mcore_data_parallel_parameter_sync(model)
        diagnostics.update(
            global_param_update_stats(
                model,
                snapshot,
                parameter_bucket=parameter_bucket,
                reduction=reduction,
            )
        )
        diagnostics.update(optimizer.optimizer_state_stats())
    if scheduler is not None:
        scheduler.step()
    return BackwardUpdateResult(
        diagnostics=diagnostics,
        backward_at=backward_at,
        optimizer_at=time.perf_counter(),
    )


class ParamGroupScheduler:
    """Minimal scheduler that works with torch and MCore optimizer wrappers."""

    def __init__(self, optimizer: Any, spec: SchedulerSpec) -> None:
        self.optimizer = optimizer
        self.spec = spec
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if spec.name == "linear":
            self.start_factor = spec.start_factor
            self.end_factor = spec.end_factor
            self.total_iters = spec.total_iters
            self.last_epoch = 0
            self._step_count = 1
            self._is_initial = False
            self._get_lr_called_within_step = False
            for group, base_lr in zip(
                self.optimizer.param_groups,
                self.base_lrs,
                strict=True,
            ):
                group.setdefault("initial_lr", base_lr)
                group["lr"] *= self.start_factor
            self._last_lr = [group["lr"] for group in optimizer.param_groups]
        else:
            self.last_epoch = -1
            self._apply(self._factor(0))

    def _factor(self, step: int) -> float:
        if self.spec.name == "constant_with_warmup":
            return min(float(step) / float(self.spec.warmup_steps), 1.0)
        if self.spec.name == "constant" and step < 5:
            return self.spec.factor
        return 1.0

    def _apply(self, factor: float) -> None:
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor

    def step(self) -> None:
        if self.spec.name == "linear":
            self.last_epoch += 1
            self._step_count += 1
            if self.last_epoch <= self.total_iters:
                multiplier = 1.0 + (self.end_factor - self.start_factor) / (
                    self.total_iters * self.start_factor
                    + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
                )
                for group in self.optimizer.param_groups:
                    group["lr"] *= multiplier
            self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
            return
        self.last_epoch += 1
        self._apply(self._factor(self.last_epoch + 1))

    def state_dict(self) -> dict[str, Any]:
        if self.spec.name == "linear":
            return {
                "start_factor": self.start_factor,
                "end_factor": self.end_factor,
                "total_iters": self.total_iters,
                "base_lrs": self.base_lrs,
                "last_epoch": self.last_epoch,
                "_step_count": self._step_count,
                "_is_initial": self._is_initial,
                "_get_lr_called_within_step": self._get_lr_called_within_step,
                "_last_lr": self._last_lr,
            }
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.spec.name == "linear":
            self.start_factor = float(state_dict["start_factor"])
            self.end_factor = float(state_dict["end_factor"])
            self.total_iters = int(state_dict["total_iters"])
            self.base_lrs = [float(value) for value in state_dict["base_lrs"]]
            self.last_epoch = int(state_dict["last_epoch"])
            self._step_count = int(state_dict["_step_count"])
            self._is_initial = bool(state_dict["_is_initial"])
            self._get_lr_called_within_step = bool(state_dict["_get_lr_called_within_step"])
            self._last_lr = [float(value) for value in state_dict["_last_lr"]]
            return
        self.base_lrs = [float(value) for value in state_dict["base_lrs"]]
        self.last_epoch = int(state_dict["last_epoch"])
        self._apply(self._factor(self.last_epoch + 1))


def build_mcore_distributed_optimizer(
    model: nn.Module,
    spec: OptimizerSpec,
    scheduler_spec: SchedulerSpec | None,
    *,
    clip_grad: float,
) -> tuple[MCoreDistributedOptimizer, ParamGroupScheduler | None]:
    """Construct MCore DistributedOptimizer over vForge-owned groups."""
    if not is_mcore_data_parallel(model):
        raise TypeError("MCore distributed optimizer requires an MCore DDP model")

    from vforge.mcore.optimizer import get_megatron_optimizer
    from vforge.mcore.optimizer.optimizer_config import OptimizerConfig

    parameter = next(model.parameters(), None)
    parameter_dtype = parameter.dtype if parameter is not None else torch.float32
    config = OptimizerConfig(
        optimizer="adam",
        lr=spec.learning_rate,
        min_lr=0.0,
        weight_decay=spec.weight_decay,
        adam_beta1=spec.betas[0],
        adam_beta2=spec.betas[1],
        adam_eps=spec.eps,
        decoupled_weight_decay=True,
        params_dtype=parameter_dtype,
        bf16=parameter_dtype == torch.bfloat16,
        fp16=parameter_dtype == torch.float16,
        use_distributed_optimizer=True,
        overlap_param_gather=model.ddp_config.overlap_param_gather,
        clip_grad=float(clip_grad),
    )
    inner = get_megatron_optimizer(
        config,
        [model],
        # The existing Wan torch AdamW factory applies weight decay to every
        # parameter. MCore interprets None as its own bias/norm no-decay policy;
        # an explicit empty override preserves the recipe's current semantics.
        config_overrides={},
        use_gloo_process_groups=False,
        pg_collection=mcore_process_group_collection(model),
    )
    chained = getattr(inner, "chained_optimizers", None)
    if chained is not None:
        if len(chained) != 1:
            raise NotImplementedError(
                "the first vForge MCore optimizer milestone supports one dense optimizer shard"
            )
        inner = chained[0]
    optimizer = MCoreDistributedOptimizer(model, inner, clip_grad=clip_grad)
    scheduler = (
        None
        if scheduler_spec is None or scheduler_spec.name == "none"
        else ParamGroupScheduler(optimizer, scheduler_spec)
    )
    return optimizer, scheduler


__all__ = [
    "MCoreDistributedOptimizer",
    "MCoreOptimizerStepResult",
    "ParamGroupScheduler",
    "backward_mcore_update",
    "build_mcore_distributed_optimizer",
]
