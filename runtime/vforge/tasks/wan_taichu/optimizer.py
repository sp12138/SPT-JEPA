"""Task-local optimizer adapters used to reproduce Taichu/DeepSpeed numerics.

These classes are not a new vforge optimizer abstraction.  Wan Taichu keeps
them locally because its migration gate compares parameter updates bit for bit
with the existing Taichu Application, whose DeepSpeed BF16 optimizer has two
observable behaviours that ordinary ``torch.optim.AdamW`` does not reproduce:

* without FSDP, it updates one flat FP32 master parameter while the model works
  in BF16, then copies the updated values back to the BF16 model;
* with sharded training, it computes clipping norms in DeepSpeed's per-parameter
  FP32 accumulation order instead of the default FSDP shard-local order.

The training task selects the matching adapter in ``build_train_components``.
Other Native models should continue to use vforge's normal optimizer factory.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from vforge.utils.gradients import parameter_gradient
from vforge.utils.tensor import to_local_tensor


class WanTaichuAdamW(torch.optim.AdamW):
    """FSDP AdamW with DeepSpeed-BF16-compatible gradient clipping.

    FSDP stores one shard of each reduced gradient on every rank.  Summing the
    shard-local squares and reducing that scalar is mathematically equivalent
    to DeepSpeed's norm, but not bitwise equivalent: DeepSpeed first exposes a
    complete reduced gradient and computes one norm per parameter.  Preserve
    that order here, including the device-side FP32 clipping coefficient.
    """

    @torch.no_grad()
    def clip_grad_norm_(
        self,
        model: nn.Module,
        max_norm: float,
        reduction: Any | None = None,
    ) -> float:
        device = next(model.parameters()).device
        total_sq = torch.zeros((), device=device, dtype=torch.float32)
        for _name, parameter in model.named_parameters():
            if reduction is not None and not reduction.includes(parameter):
                continue
            gradient = parameter_gradient(parameter)
            if gradient is None:
                continue
            materialize = getattr(gradient, "full_tensor", None)
            full_gradient = materialize() if callable(materialize) else gradient
            norm = full_gradient.detach().float().norm(2)
            total_sq.add_(norm.square())

        total_norm = total_sq.sqrt()
        coefficient = float(max_norm) / (total_norm + 1e-6)
        if coefficient < 1.0:
            for parameter in model.parameters():
                gradient = parameter_gradient(parameter)
                if gradient is not None:
                    to_local_tensor(gradient).mul_(coefficient)
        return float(total_norm.item())


class FlatFp32MasterAdamW(torch.optim.AdamW):
    """Match DeepSpeed's non-FSDP BF16 working/FP32 master parameter layout.

    This is selected only for the DP=1, USP=1, BF16 parity path.  It is needed
    because ordinary AdamW would update the BF16 model parameters directly,
    changing both optimizer rounding and the values copied into the next step.
    """

    def __init__(self, model: nn.Module, **kwargs: Any) -> None:
        working_parameters = tuple(
            parameter for parameter in model.parameters() if parameter.requires_grad
        )
        if not working_parameters:
            raise ValueError("FP32-master AdamW requires trainable parameters")
        if any(parameter.dtype != torch.bfloat16 for parameter in working_parameters):
            dtypes = sorted({str(parameter.dtype) for parameter in working_parameters})
            raise TypeError(f"FP32-master AdamW expects BF16 working parameters; found {dtypes}")

        self._working_parameters = working_parameters
        self._parameter_numels = tuple(parameter.numel() for parameter in working_parameters)
        self._pending_clip: tuple[torch.Tensor, float] | None = None
        master = nn.Parameter(
            torch.cat([parameter.detach().reshape(-1) for parameter in working_parameters]).float(),
            requires_grad=True,
        )
        self._master_parameter = master
        super().__init__([master], **kwargs)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self._working_parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()
        super().zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def clip_grad_norm_(
        self,
        model: nn.Module,
        max_norm: float,
        reduction: Any | None = None,
    ) -> float:
        """Match DeepSpeed BF16 norm order without scaling BF16 gradients in place."""
        total_sq = torch.zeros(
            (),
            device=next(model.parameters()).device,
            dtype=torch.float32,
        )
        for _name, parameter in model.named_parameters():
            if reduction is not None and not reduction.includes(parameter):
                continue
            gradient = parameter_gradient(parameter)
            if gradient is None:
                continue
            local_gradient = to_local_tensor(gradient.detach())
            if local_gradient.numel() == 0:
                continue
            # DeepSpeed BF16_Optimizer computes each parameter norm after an
            # explicit FP32 cast, then accumulates the squared norms in FP32.
            # Using FP64 here is mathematically close but changes the rounding
            # point and therefore the clipping coefficient.
            norm = torch.linalg.vector_norm(local_gradient.float(), ord=2.0)
            total_sq.add_(norm.square())

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)
        total_norm = total_sq.sqrt()
        self._pending_clip = (total_norm.detach(), float(max_norm))
        return float(total_norm.item())

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None):
        gradients = []
        for parameter in self._working_parameters:
            gradient = parameter.grad
            gradients.append(
                torch.zeros_like(parameter).reshape(-1)
                if gradient is None
                else gradient.detach().reshape(-1)
            )
        master_gradient = torch.cat(gradients).float()
        if self._pending_clip is not None:
            total_norm, max_norm = self._pending_clip
            scale = torch.clamp((total_norm + 1e-6) / max_norm, min=1.0)
            master_gradient.mul_(scale.reciprocal())
            self._pending_clip = None

        self._master_parameter.grad = master_gradient
        output = super().step(closure=closure)
        offset = 0
        for parameter, numel in zip(
            self._working_parameters,
            self._parameter_numels,
            strict=True,
        ):
            parameter.copy_(
                self._master_parameter[offset : offset + numel]
                .view_as(parameter)
                .to(dtype=parameter.dtype)
            )
            offset += numel
        self._master_parameter.grad = None
        return output


def build_fp32_master_optimizer_factory(
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> Callable[[nn.Module], torch.optim.Optimizer]:
    """Build the DP=1/USP=1 DeepSpeed-parity FP32-master optimizer."""

    def factory(model: nn.Module) -> torch.optim.Optimizer:
        return FlatFp32MasterAdamW(
            model,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    return factory


def build_wan_taichu_optimizer_factory(
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
) -> Callable[[nn.Module], torch.optim.Optimizer]:
    """Build the FSDP optimizer with Taichu-compatible clipping order."""

    def factory(model: nn.Module) -> torch.optim.Optimizer:
        return WanTaichuAdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    return factory


def clip_wan_taichu_gradients(
    model: nn.Module,
    optimizer: Any,
    max_norm: float,
    reduction: Any | None = None,
) -> float:
    """Dispatch the explicit recipe hook to a Wan Taichu parity optimizer."""
    if not isinstance(optimizer, (FlatFp32MasterAdamW, WanTaichuAdamW)):
        raise TypeError(
            "Wan Taichu gradient clipping requires FlatFp32MasterAdamW or "
            f"WanTaichuAdamW; got {type(optimizer).__name__}"
        )
    return optimizer.clip_grad_norm_(model, max_norm, reduction)


__all__ = [
    "FlatFp32MasterAdamW",
    "WanTaichuAdamW",
    "build_fp32_master_optimizer_factory",
    "build_wan_taichu_optimizer_factory",
    "clip_wan_taichu_gradients",
]
