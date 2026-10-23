"""Backend-neutral access to optimizer-visible parameter gradients.

PyTorch DDP and FSDP leave the reduced gradient in ``Parameter.grad``.
MCore DDP owns a contiguous buffer instead and exposes each parameter's view as
``Parameter.main_grad``. Training code can use this module without branching on
the selected engine or data-parallel mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_MAIN_GRAD_ACTIVE = "_vforge_main_grad_active"


def parameter_gradient(parameter: nn.Parameter) -> torch.Tensor | None:
    """Return the gradient that the selected data-parallel backend reduced."""
    main_grad = getattr(parameter, "main_grad", None)
    if main_grad is None:
        return parameter.grad
    if not getattr(parameter, _MAIN_GRAD_ACTIVE, True):
        return None
    return main_grad


def set_main_gradient_active(parameter: nn.Parameter, active: bool) -> None:
    """Record whether an MCore buffer view represents a gradient this step."""
    setattr(parameter, _MAIN_GRAD_ACTIVE, bool(active))


__all__ = [
    "parameter_gradient",
    "set_main_gradient_active",
]
