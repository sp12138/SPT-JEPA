"""LTX-2 sigma grids and the FP32 velocity-to-clean conversion."""

import math

import torch


def ltx_sigmas(steps: int) -> torch.Tensor:
    if steps < 2:
        raise ValueError("LTX sigma grid requires at least two steps")
    values = torch.linspace(1.0, 0.0, steps + 1)
    shift = math.exp(2.05)  # Reference uses the default 4096-token anchor.
    values = torch.where(values != 0, shift / (shift + (1 / values - 1)), 0)
    nonzero = values != 0
    complement = 1.0 - values[nonzero]
    values[nonzero] = 1.0 - complement / (complement[-1] / 0.9)
    return values.float()


def to_denoised(sample, velocity, sigma):
    return (sample.float() - velocity.float() * sigma.float()).to(sample.dtype)
