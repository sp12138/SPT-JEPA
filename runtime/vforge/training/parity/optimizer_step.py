"""Optional optimizer-step controls for numerical parity experiments."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vforge.training.parity.config import GRAD_SCALE_NAMES
from vforge.utils.gradients import parameter_gradient
from vforge.utils.tensor import to_local_tensor


def resolve_grad_scale(scale: str | float | None, usp_size: int) -> float:
    if scale is None:
        return 1.0
    if scale in GRAD_SCALE_NAMES:
        return float(usp_size)
    return float(scale)


def scale_gradients(model: nn.Module, scale: float) -> int:
    count = 0
    with torch.no_grad():
        for param in model.parameters():
            gradient = parameter_gradient(param)
            if gradient is None:
                continue
            gradient.mul_(scale)
            count += 1
    return count


def manual_adamw_bf16_step(optimizer: Any) -> int:
    """Apply an explicit BF16-rounded AdamW update."""
    updated = 0
    with torch.no_grad():
        for group in optimizer.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = (float(value) for value in group.get("betas", (0.9, 0.999)))
            eps = float(group.get("eps", 1e-8))
            weight_decay = float(group.get("weight_decay", 0.0))
            for param in group["params"]:
                gradient = parameter_gradient(param)
                if gradient is None:
                    continue
                local_param = to_local_tensor(param.detach())
                local_grad = to_local_tensor(gradient.detach()).to(torch.float32)
                if local_param.numel() == 0 or local_grad.numel() == 0:
                    continue
                state = optimizer.state[param]
                if not state:
                    state["step"] = torch.zeros(
                        (),
                        device=local_param.device,
                        dtype=torch.float32,
                    )
                    state["exp_avg"] = torch.zeros_like(
                        local_param,
                        dtype=torch.float32,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        local_param,
                        dtype=torch.float32,
                    )
                step = state["step"]
                if isinstance(step, torch.Tensor):
                    step.add_(1.0)
                    step_value = float(step.item())
                else:
                    step_value = float(step) + 1.0
                    state["step"] = step_value
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                param_f = local_param.to(torch.bfloat16).to(torch.float32)
                if weight_decay:
                    param_f.mul_(1.0 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(local_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    local_grad,
                    local_grad,
                    value=1.0 - beta2,
                )
                bias_correction1 = 1.0 - beta1**step_value
                bias_correction2 = 1.0 - beta2**step_value
                step_size = lr / bias_correction1
                denom = exp_avg_sq.sqrt()
                denom.div_(bias_correction2**0.5).add_(eps)
                param_f.addcdiv_(exp_avg, denom, value=-step_size)
                local_param.copy_(param_f.to(torch.bfloat16).to(local_param.dtype))
                updated += 1
    return updated


def quantize_params_after_step(model: nn.Module, mode: str | None) -> int:
    if mode is None:
        return 0
    if mode != "bf16":
        raise ValueError(f"unsupported optimizer-step quantization mode: {mode}")
    count = 0
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            local_param = to_local_tensor(param.detach())
            if local_param.numel() == 0 or not local_param.is_floating_point():
                continue
            local_param.copy_(local_param.to(torch.bfloat16).to(local_param.dtype))
            count += 1
    return count


__all__ = [
    "manual_adamw_bf16_step",
    "quantize_params_after_step",
    "resolve_grad_scale",
    "scale_gradients",
]
