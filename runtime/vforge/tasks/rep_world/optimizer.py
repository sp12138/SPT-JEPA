"""Task-local Muon preserving RepMiniWorld's matrix and auxiliary Adam updates.

Orthogonalization must see a complete matrix, never an FSDP-local row shard.
DTensor parameters retain sharded momentum; only one update at a time is
gathered for the Newton--Schulz calculation. Checkpoint ownership stays Native.
"""

from __future__ import annotations

import math

import torch
from torch.distributed.tensor import DTensor, Replicate


def orthogonalize(gradient, steps=5):
    value = gradient.bfloat16()
    transpose = value.shape[0] > value.shape[1]
    if transpose:
        value = value.mT
    value = value / (value.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = value @ value.mT
        polynomial = -4.7750 * gram + 2.0315 * (gram @ gram)
        value = 3.4445 * value + polynomial @ value
    if transpose:
        value = value.mT
    return value.to(gradient.dtype)


def _restore_layout(value, reference):
    if not isinstance(reference, DTensor):
        return value
    replicated = DTensor.from_local(
        value,
        device_mesh=reference.device_mesh,
        placements=[Replicate()] * reference.device_mesh.ndim,
        run_check=False,
    )
    return replicated.redistribute(placements=reference.placements)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Native state dicts use parameter-shaped momentum, including convolutions.

    Values/update equations match the source; old optimizer payloads are NOT a
    supported resume format. Missing gradients receive zero updates as in the
    source (including momentum evolution and matrix weight decay).
    """

    def __init__(self, model, *, lr, weight_decay, betas=(0.9, 0.95), eps=1e-8):
        if not math.isfinite(lr) or lr < 0 or not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError("Muon learning rate and weight decay must be finite and nonnegative")
        if len(betas) != 2 or any(not 0 <= beta < 1 for beta in betas) or eps <= 0:
            raise ValueError("Invalid auxiliary Adam settings")
        matrix, auxiliary = [], []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                destination = auxiliary if parameter.ndim < 2 or name.endswith(".bias") else matrix
                destination.append(parameter)
        matrix.sort(key=lambda parameter: tuple(parameter.shape), reverse=True)
        super().__init__(
            [
                dict(params=auxiliary, use_muon=False, weight_decay=0.0, betas=betas, eps=eps),
                dict(
                    params=matrix,
                    use_muon=True,
                    weight_decay=weight_decay,
                    momentum=0.95,
                    ns_steps=5,
                ),
            ],
            dict(lr=lr),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    gradient = torch.zeros_like(parameter)
                if gradient.is_sparse:
                    raise ValueError("RepWorld Muon does not support sparse gradients")
                state = self.state[parameter]
                if group["use_muon"]:
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    momentum = state["momentum_buffer"]
                    momentum.lerp_(gradient, 1 - group["momentum"])
                    # Match the source Nesterov interpolation and flatten rule.
                    direction = gradient.lerp(momentum, group["momentum"])
                    if isinstance(direction, DTensor):
                        direction = direction.full_tensor()
                    matrix = direction.reshape(direction.shape[0], -1)
                    with torch.autocast(parameter.device.type, enabled=False):
                        update = orthogonalize(matrix, group["ns_steps"])
                        update = update * max(1, matrix.shape[0] / matrix.shape[1]) ** 0.5
                    update = _restore_layout(update.reshape(parameter.shape), parameter)
                    learning_rate = group["lr"] * 0.2 * math.sqrt(max(parameter.shape[:2]))
                else:
                    if not state:
                        state.update(
                            exp_avg=torch.zeros_like(parameter),
                            exp_avg_sq=torch.zeros_like(parameter),
                            step=0,
                        )
                    state["step"] += 1
                    beta1, beta2 = group["betas"]
                    state["exp_avg"].lerp_(gradient, 1 - beta1)
                    state["exp_avg_sq"].lerp_(gradient.square(), 1 - beta2)
                    first = state["exp_avg"] / (1 - beta1 ** state["step"] + 1e-16)
                    second = state["exp_avg_sq"] / (1 - beta2 ** state["step"] + 1e-16)
                    update = first / (second.sqrt() + group["eps"])
                    learning_rate = group["lr"]
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-learning_rate)
        return loss
