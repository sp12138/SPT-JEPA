"""Verify clipped likelihood as a probability measure, not just its code path."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def run(output):
    from as_jepa_lewm.censored_action import CensoredActionBridge
    torch.set_num_threads(2)
    torch.manual_seed(700)
    model = CensoredActionBridge(visual_source='local', objective='mse', chunk=1, window=2,
        action_dim=4, temporal_aux=False, condition_horizon='none', history_context=False,
        goal_representation='directed', action_lower=[-1]*4, action_upper=[1]*4)
    # Independent quadrature: interior continuous mass + two boundary atoms =1.
    grid = torch.linspace(-1, 1, 100001, dtype=torch.float64)
    quadrature = []
    for location, scale in [(-2., .3), (0., .7), (1.5, 2.)]:
        z = (grid-location)/scale
        density = torch.exp(F.logsigmoid(z)+F.logsigmoid(-z))/scale
        mass = torch.trapezoid(density, grid)+torch.sigmoid(torch.tensor((-1-location)/scale))+torch.sigmoid(torch.tensor((location-1)/scale))
        assert abs(float(mass)-1) < 1e-7
        quadrature.append(float(mass))
    # Analytic tails/point density for a single independent logistic component.
    logits = torch.zeros(1, 1, 4)
    location = torch.zeros(1, 1, 4, 4, requires_grad=True)
    scale = torch.ones_like(location)
    target = torch.tensor([[[-1., 0., .2, 1.]]])
    actual = model.log_probability(target, logits, location, scale)
    expected = 2*F.logsigmoid(torch.tensor(-1.))-2*torch.log(torch.tensor(2.))+F.logsigmoid(torch.tensor(.2))+F.logsigmoid(torch.tensor(-.2))
    assert torch.allclose(actual.squeeze(), expected)
    (-actual.mean()).backward()
    assert torch.isfinite(location.grad).all()
    assert bool((location.grad[..., 0] > 0).all()) and bool((location.grad[..., 3] < 0).all())
    # Empirical draw/clip agrees with tail probabilities; no learned parameters.
    generator = torch.Generator().manual_seed(701)
    u = torch.rand(200000, generator=generator).clamp(1e-7, 1-1e-7)
    clipped = (u.log()-torch.log1p(-u)).clamp(-1, 1)
    left = float((clipped == -1).float().mean()); right = float((clipped == 1).float().mean())
    tail = float(torch.sigmoid(torch.tensor(-1.)))
    assert abs(left-tail) < .004 and abs(right-tail) < .004
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(passed=True, new_updates=0, quadrature_total_mass=quadrature,
        analytic_logp=float(expected), empirical_boundary_mass=[left, right], expected_tail_mass=tail,
        tail_gradient_sign_correct=True, scope='Probability normalization/analytic boundary likelihood and clipped sampling. No performance result.'), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output)
