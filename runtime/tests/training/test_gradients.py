from __future__ import annotations

import torch

from vforge.utils.gradients import (
    parameter_gradient,
    set_main_gradient_active,
)


def test_parameter_gradient_prefers_an_active_main_grad() -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.full_like(parameter, 2.0)
    parameter.main_grad = torch.full_like(parameter, 3.0)

    assert parameter_gradient(parameter) is parameter.main_grad

    set_main_gradient_active(parameter, False)
    assert parameter_gradient(parameter) is None


def test_inactive_main_grad_preserves_unused_parameter_semantics() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    parameter.main_grad = torch.zeros(1)
    set_main_gradient_active(parameter, False)

    assert parameter_gradient(parameter) is None
