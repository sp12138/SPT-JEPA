"""Factual forward prediction when an issued command can have no visible effect."""

import torch


def fit_visible_posterior(log_changes):
    """Fit two normals to ordinary RGB changes, returning the posterior log odds.

    Each observation contributes its own marginal negative log likelihood.
    A common variance keeps the posterior monotone and prevents either
    component from fitting an individual transition with zero variance.
    """
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit, logsumexp

    x = np.asarray(log_changes, dtype=np.float64).reshape(-1)
    if len(x) < 2 or not np.isfinite(x).all() or np.ptp(x) == 0:
        raise ValueError("Expected finite, varying log RGB changes")

    def objective(parameters):
        silent, visible, log_sigma, logit = parameters
        residual = x - np.array([[silent], [visible]])
        precision = np.exp(-2 * log_sigma)
        log_prior = -np.logaddexp(0., [logit, -logit])
        component = log_prior[:, None] - log_sigma - .5 * precision * residual**2
        marginal = logsumexp(component, axis=0)
        posterior = np.exp(component - marginal)
        location_grad = -(posterior * residual * precision).mean(axis=1)
        width_grad = (posterior * (1 - precision * residual**2)).sum(axis=0).mean()
        prior_grad = expit(logit) - posterior[1].mean()
        return -marginal.mean(), np.r_[location_grad, width_grad, prior_grad]

    fit = minimize(objective, [-4., 3., 0., 0.], jac=True, method='L-BFGS-B',
        bounds=[(None, None), (None, None), (-8., 8.), (None, None)],
        options={'maxiter': 160, 'ftol': 1e-12, 'gtol': 1e-7})
    if not fit.success:
        raise ValueError(f"RGB change calibration failed: {fit.message}")
    first, second, log_sigma, logit = fit.x
    if first > second:
        first, second, logit = second, first, -logit
    precision = np.exp(-2 * log_sigma)
    return dict(slope=float((second - first) * precision),
                bias=float(logit + .5 * (first**2 - second**2) * precision))


def visible_probability(pixels, slope, bias):
    """Posterior from a two-component equal-variance fit to log RGB change.

    Input is uint8 in physical sample, synchronized camera, time, C, H, W
    order. One digital count over the entire observation gives a finite log
    for two identical observations. Calibration uses ordinary training RGB.
    """
    if pixels.ndim != 6 or pixels.dtype != torch.uint8:
        raise ValueError("Expected synchronized uint8 RGB windows")
    _, views, _, channels, height, width = pixels.shape
    values = pixels.float()
    change = (values[:, :, 1:] - values[:, :, :-1]).abs().mean((1, 3, 4, 5))
    log_change = (change + 1 / (views * channels * height * width)).log()
    return torch.sigmoid(slope * log_change + bias)


def visible_effect_rate(codes, action_effect, probability, sigma):
    """Expected factual rate under the RGB-only posterior on visible motion."""
    dtype = torch.float64 if codes.dtype == torch.float64 else torch.float32
    codes = codes.to(dtype)
    action_effect = action_effect.to(dtype)
    still = (codes[:, 1:] - codes[:, :-1]).square().sum(-1)
    moved = (codes[:, 1:] - (codes[:, :-1] + action_effect)).square().sum(-1)
    magnitude = action_effect.detach().square().sum(-1)
    variance = sigma**2 * torch.where(magnitude > 0, magnitude, 1.)
    p = probability.detach().to(dtype)
    rate = torch.lerp(torch.log1p(still / variance), torch.log1p(moved / variance), p) * .5
    return rate, torch.lerp(still, moved, p)
