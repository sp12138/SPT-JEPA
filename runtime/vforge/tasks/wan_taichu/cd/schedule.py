"""Wan CD schedule and reference arithmetic, supplied through family hooks."""

from __future__ import annotations

import torch

from vforge.algorithms.cd.model import CdNoisePair, CdSchedule
from vforge.tasks.requests import TrainRequest


def create_wan_cd_schedule(request: TrainRequest) -> CdSchedule:
    """Bind validated Wan configuration without owning or resetting RNG state."""
    levels = request.cd["discrete_levels"]
    shift = request.cd["consistency_shift"]
    boundary_ratio = request.cd.get("boundary_ratio")

    def sample_pair(generator: torch.Generator, high_noise: bool | None) -> CdNoisePair:
        sigmas, timesteps = wan_sigma_grid(levels, shift)
        if high_noise is None:
            return sample_adjacent_pair(sigmas, timesteps, generator=generator)
        if boundary_ratio is None:
            raise ValueError("Wan expert-band sampling requires cd.boundary_ratio")
        return sample_adjacent_pair_in_band(
            sigmas,
            timesteps,
            boundary_timestep=boundary_ratio * 1000.0,
            high_noise=high_noise,
            generator=generator,
        )

    return CdSchedule(
        sample_pair=sample_pair,
        add_noise=add_flow_noise,
        teacher_step=teacher_euler_step,
        velocity_to_x0=velocity_to_x0,
    )


def wan_sigma_grid(
    discrete_levels: int,
    shift: float,
    *,
    num_train_timesteps: int = 1000,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Wan's single-shift grid without a terminal-zero sentinel."""
    if discrete_levels < 1:
        raise ValueError("discrete_levels must be positive")
    if shift <= 0:
        raise ValueError("shift must be positive")
    raw = torch.linspace(
        1.0,
        0.0,
        int(discrete_levels) + 1,
        dtype=torch.float32,
        device=device,
    )[:-1]
    sigmas = float(shift) * raw / (1.0 + (float(shift) - 1.0) * raw)
    return sigmas, sigmas * float(num_train_timesteps)


def sample_adjacent_pair(
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    generator: torch.Generator,
) -> CdNoisePair:
    """Uniformly sample one adjacent descending pair."""
    if sigmas.ndim != 1 or timesteps.ndim != 1:
        raise ValueError("CD schedule tensors must be rank one")
    if sigmas.shape != timesteps.shape or sigmas.numel() < 2:
        raise ValueError("CD sigmas and timesteps must have the same length >= 2")
    index = int(
        torch.randint(
            0,
            int(sigmas.numel()) - 1,
            (1,),
            device=generator.device,
            generator=generator,
        ).item()
    )
    return CdNoisePair(
        index=index,
        timestep=timesteps[index].to(generator.device),
        next_timestep=timesteps[index + 1].to(generator.device),
        sigma=sigmas[index].to(generator.device),
        next_sigma=sigmas[index + 1].to(generator.device),
    )


def valid_adjacent_indices_for_band(
    timesteps: torch.Tensor,
    *,
    boundary_timestep: float,
    high_noise: bool,
) -> tuple[int, ...]:
    """Return adjacent-pair starts whose two endpoints stay in one noise band."""
    if timesteps.ndim != 1 or timesteps.numel() < 2:
        raise ValueError("CD timesteps must be rank one with length >= 2")
    if boundary_timestep <= 0:
        raise ValueError("CD expert boundary must be positive")
    starts: list[int] = []
    for index in range(int(timesteps.numel()) - 1):
        timestep = float(timesteps[index])
        next_timestep = float(timesteps[index + 1])
        current_in_band = (
            timestep >= boundary_timestep if high_noise else timestep < boundary_timestep
        )
        next_in_band = (
            next_timestep >= boundary_timestep if high_noise else next_timestep < boundary_timestep
        )
        if current_in_band and next_in_band:
            starts.append(index)
    return tuple(starts)


def sample_adjacent_pair_in_band(
    sigmas: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    boundary_timestep: float,
    high_noise: bool,
    generator: torch.Generator,
) -> CdNoisePair:
    """Uniformly sample an adjacent pair fully contained in one expert band."""
    if sigmas.ndim != 1 or sigmas.shape != timesteps.shape:
        raise ValueError("CD sigmas and timesteps must be matching rank-one tensors")
    indices = valid_adjacent_indices_for_band(
        timesteps,
        boundary_timestep=boundary_timestep,
        high_noise=high_noise,
    )
    if not indices:
        band = "high" if high_noise else "low"
        raise ValueError(f"CD schedule has no adjacent pair in the {band}-noise band")
    pick = int(
        torch.randint(
            0,
            len(indices),
            (1,),
            device=generator.device,
            generator=generator,
        ).item()
    )
    index = indices[pick]
    return CdNoisePair(
        index=index,
        timestep=timesteps[index].to(generator.device),
        next_timestep=timesteps[index + 1].to(generator.device),
        sigma=sigmas[index].to(generator.device),
        next_sigma=sigmas[index + 1].to(generator.device),
    )


def add_flow_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    if clean.shape != noise.shape:
        raise ValueError("clean and noise tensors must have identical shapes")
    # A non-scalar FP32 sigma promotes BF16 operands to FP32, matching
    # Taichu's scheduler arithmetic before the result is cast back to BF16.
    value = sigma.to(device=noise.device).reshape((1,) * noise.ndim)
    return ((1.0 - value) * clean + value * noise).to(dtype=noise.dtype)


def teacher_euler_step(
    noisy: torch.Tensor,
    velocity: torch.Tensor,
    timestep: torch.Tensor,
    next_timestep: torch.Tensor,
) -> torch.Tensor:
    if noisy.shape != velocity.shape:
        raise ValueError("noisy and velocity tensors must have identical shapes")
    dt = (timestep - next_timestep).to(device=noisy.device, dtype=noisy.dtype) / 1000.0
    return noisy - dt * velocity


def velocity_to_x0(
    velocity: torch.Tensor,
    noisy: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Convert flow velocity to x0 using Taichu's fp64 intermediate."""
    if velocity.shape != noisy.shape:
        raise ValueError("velocity and noisy tensors must have identical shapes")
    original_dtype = velocity.dtype
    prediction = noisy.double() - sigma.to(device=noisy.device).double() * velocity.double()
    return prediction.to(dtype=original_dtype)


__all__ = ["create_wan_cd_schedule"]
