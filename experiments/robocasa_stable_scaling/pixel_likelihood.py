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
    assert mode in ("fixed", "learned")
    assert model.pixel_objective == "future-mean"
    assert not hasattr(model.renderer, "pixel_log_variance_ratio")
    # The existing renderer AdamW group has zero weight decay. No new LR is used.
    model.renderer.register_parameter("pixel_log_variance_ratio",
        nn.Parameter(torch.zeros(()), requires_grad=mode == "learned"))
    model._pixel_likelihood_base_loss = model.loss
    model.loss = types.MethodType(calibrated_loss, model)
