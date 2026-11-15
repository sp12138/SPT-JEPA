"""Learn observation precision from factual future RGB, separately from code rate.

For scalar k = log(variance / .5), Gaussian pixel NLL up to a fixed constant
is exp(-k) * MSE + k/2. At k=0 this is exactly the existing mean objective.
Only a global model parameter is learned; no batch feature statistics, action
targets, evaluation feedback, or external visual teachers are introduced.
The original code-rate normalization and beta are retained, so this is not a
claim that the full forecasting objective is an exact variational bound.
"""
import types

import torch
from torch import nn


def gaussian_pixel_nll(mse, log_variance_ratio):
    assert log_variance_ratio.ndim == 0
    value = log_variance_ratio.float()
    return (-value).exp() * mse + .5 * value


def haar_bands(image):
    """Orthonormal, critically sampled 2D Haar bands; never mix images."""
    assert image.ndim == 4
    low = image
    while low.shape[-2] % 2 == 0 and low.shape[-1] % 2 == 0:
        a, b = low[..., 0::2, 0::2], low[..., 0::2, 1::2]
        c, d = low[..., 1::2, 0::2], low[..., 1::2, 1::2]
        yield torch.cat(((a-b+c-d)/2, (a+b-c-d)/2, (a-b-c+d)/2), dim=1)
        low = (a+b+c+d)/2
    yield low


def profiled_pixel_nll(error, mode):
    """Profile independent Gaussian variances within each physical image.

    Variance is the ML residual variance, bounded by the quantization variance
    of uint8 RGB normalized to [-1,1]. Haar bands get their exact coefficient
    count weights, with no tunable scale weights or cross-image statistics.
    This is a profile likelihood, not a normalized marginal image density.
    """
    assert error.ndim == 4 and mode in ("profile-global", "profile-haar")
    error = error.float()
    floor = (2/255)**2/12
    bands = haar_bands(error) if mode == "profile-haar" else (error,)
    count = error[0].numel()
    loss = error.new_zeros(error.shape[0])
    quadratic = torch.zeros_like(loss)
    variance = torch.zeros_like(loss)
    for band in bands:
        weight = band[0].numel()/count
        mse = band.square().flatten(1).mean(1)
        var = mse.clamp_min(floor)
        fit = .5*mse/var
        loss = loss+weight*(fit+.5*var.log())
        quadratic = quadratic+weight*fit
        variance = variance+weight*var
    value = loss.mean()
    return value, dict(pixel_mse=error.square().mean().detach(),
        pixel_gaussian_nll=value.detach(), pixel_variance=variance.mean().detach(),
        pixel_gaussian_quadratic=quadratic.mean().detach())


def profiled_loss(self, batch, step=0, ramp_steps=0):
    assert self._profiled_pixel_metrics is None
    try:
        loss, metrics = self._pixel_likelihood_base_loss(batch, step, ramp_steps)
        stats = self._profiled_pixel_metrics
        assert stats is not None
    finally:
        self._profiled_pixel_metrics = None
    beta = self.beta * min(1., step / max(1, ramp_steps)) if ramp_steps else self.beta
    return loss, dict(metrics, **stats,
        rate_share=beta*metrics["rate"]/stats["pixel_gaussian_quadratic"].clamp_min(1e-8))


def calibrated_loss(self, batch, step=0, ramp_steps=0):
    _, metrics = self._pixel_likelihood_base_loss(batch, step, ramp_steps)
    mse = metrics["fm"]
    value = self.renderer.pixel_log_variance_ratio.float()
    distortion = gaussian_pixel_nll(mse, value)
    beta = self.beta * min(1., step / max(1, ramp_steps)) if ramp_steps else self.beta
    loss = distortion + beta * metrics["rate"]
    return loss, dict(metrics, loss=loss, pixel_mse=mse,
        pixel_gaussian_nll=distortion, pixel_variance=.5 * value.exp(),
        pixel_log_variance_ratio=value,
        rate_share=beta * metrics["rate"] / ((-value).exp() * mse.detach()).clamp_min(1e-8))


def install_pixel_likelihood(model, mode):
    assert mode in ("fixed", "learned", "profile-global", "profile-haar")
    assert model.pixel_objective == "future-mean"
    if mode.startswith("profile-"):
        model.pixel_profile = mode
        model._profiled_pixel_metrics = None
        model._pixel_likelihood_base_loss = model.loss
        model.loss = types.MethodType(profiled_loss, model)
        return
    assert not hasattr(model.renderer, "pixel_log_variance_ratio")
    # The existing renderer AdamW group has zero weight decay. No new LR is used.
    model.renderer.register_parameter("pixel_log_variance_ratio",
        nn.Parameter(torch.zeros(()), requires_grad=mode == "learned"))
    model._pixel_likelihood_base_loss = model.loss
    model.loss = types.MethodType(calibrated_loss, model)
