"""Stable hashes and snapshots for old-vs-new comparisons."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from vforge.utils.tensor import to_local_tensor


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Return a stable hash for tensor value comparisons."""
    cpu = to_local_tensor(tensor).detach().contiguous().cpu()
    if cpu.dtype is torch.bfloat16:
        cpu = cpu.view(torch.int16)
    return hashlib.sha256(cpu.numpy().tobytes()).hexdigest()


def batch_sha256(batch: Any) -> str:
    """Hash a nested batch by tensor contents, types, and stable paths."""
    hasher = hashlib.sha256()

    def visit(value: Any, path: str) -> None:
        hasher.update(path.encode())
        hasher.update(type(value).__name__.encode())
        if isinstance(value, torch.Tensor):
            hasher.update(str(tuple(value.shape)).encode())
            hasher.update(str(value.dtype).encode())
            hasher.update(tensor_sha256(value).encode())
        elif isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}")
        else:
            hasher.update(repr(value).encode())

    visit(batch, "batch")
    return hasher.hexdigest()


def rng_sha256() -> str:
    """Hash CPU and CUDA RNG states for parity debugging."""
    hasher = hashlib.sha256()
    hasher.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            hasher.update(state.cpu().numpy().tobytes())
    return hasher.hexdigest()


def local_grad_norm(
    parameters: nn.Module | Iterable[torch.nn.Parameter],
) -> float:
    """Return the local L2 norm of gradients from a module or iterator."""
    if isinstance(parameters, nn.Module):
        parameters = parameters.parameters()
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += float(grad.square().sum())
    return total**0.5


def module_param_hash(module: nn.Module) -> str:
    """Hash all trainable parameter values in module order."""
    hasher = tensor_sha256
    parts: list[str] = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        parts.append(f"{name}:{hasher(param.data)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def step_snapshot_json(
    model: nn.Module,
    loss: torch.Tensor,
    *,
    extra: dict[str, Any] | None = None,
) -> str:
    """Return one stable JSON snapshot after forward/backward/update."""
    payload: dict[str, Any] = {
        "loss": float(loss.detach().float().cpu()),
        "loss_hash": tensor_sha256(loss.detach().float().cpu()),
        "grad_norm": local_grad_norm(model),
        "param_hash": module_param_hash(model),
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


__all__ = [
    "batch_sha256",
    "local_grad_norm",
    "module_param_hash",
    "rng_sha256",
    "step_snapshot_json",
    "tensor_sha256",
]
