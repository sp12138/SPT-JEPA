"""Future-state targets, quadrant mosaics and parameter-free gap conditioning."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from vforge.models.rep_world.sequence import share_random


def sample_target_offsets(
    shape, probability, device, *, choices=3, uniform=None, sequence_parallel=True
):
    """Sample offsets: 0/1 with probabilities 1-p/p, or 0/1/2 with 1-p/p/2/p/2.

    Offset zero is ordinary next-state prediction; source = origin + 1 + offset.

    With p=0 no RNG is consumed. Explicit uniforms are useful for controlled
    comparisons and must already agree across ranks sharing one sample.
    """
    if type(choices) is not int or choices not in {2, 3}:
        raise ValueError("future_target_choices must be 2 or 3")
    if isinstance(probability, bool) or not 0 <= probability <= 1:
        raise ValueError("distant_target_prob must be in [0,1]")
    if uniform is not None:
        if (
            uniform.shape != shape
            or uniform.device != torch.device(device)
            or not uniform.is_floating_point()
            or not bool(torch.isfinite(uniform).all())
            or bool(((uniform < 0) | (uniform >= 1)).any())
        ):
            raise ValueError("Target uniforms must be finite [0,1) values of shape [B,origins]")
    if probability == 0:
        return torch.zeros(shape, device=device, dtype=torch.long)
    if uniform is None:
        uniform = share_random(torch.rand(shape, device=device), enabled=sequence_parallel)
    if choices == 2:
        return (uniform >= 1 - probability).long()
    return (uniform >= 1 - probability).long() + (uniform >= 1 - probability / 2).long()


def sample_available_target_offsets(
    shape, total_states, choices, device, *, uniform=None, sequence_parallel=True
):
    """Sample uniformly among future targets that exist at each origin.

    Shape is [B,S-1]. Zero-based offset d selects origin+1+d.
    K=1 consumes no RNG, matching the original next-state baseline.
    """
    if type(choices) is not int or choices not in {1, 2, 3, 4}:
        raise ValueError("available future choices must be 1, 2, 3 or 4")
    if (
        len(shape) != 2
        or type(total_states) is not int
        or total_states < 2
        or shape[1] != total_states - 1
    ):
        raise ValueError("Available future shape must be [B,S-1] for S >= 2")
    if uniform is not None and (
        uniform.shape != shape
        or uniform.device != torch.device(device)
        or not uniform.is_floating_point()
        or not bool(torch.isfinite(uniform).all())
        or bool(((uniform < 0) | (uniform >= 1)).any())
    ):
        raise ValueError("Target uniforms must be finite [0,1) values of shape [B,S-1]")
    if choices == 1:
        return torch.zeros(shape, device=device, dtype=torch.long)
    if uniform is None:
        uniform = share_random(torch.rand(shape, device=device), enabled=sequence_parallel)
    origins = torch.arange(shape[1], device=device)
    available = (total_states - 1 - origins).clamp(max=choices)
    return (uniform * available).long().minimum(available - 1)


def validate_target_offsets(offsets, batch, device, *, tokens=None, max_offset=2):
    tokenwise = offsets.ndim == 2 and tokens is not None
    shape = (batch, tokens) if tokenwise else (batch,)
    if max_offset not in {2, 3}:
        raise ValueError("max_offset must be 2 or 3")
    maximum = 3 if tokenwise else max_offset
    if (
        offsets.dtype != torch.long
        or offsets.shape != shape
        or offsets.device != torch.device(device)
        or bool(((offsets < 0) | (offsets > maximum)).any())
    ):
        raise ValueError(
            "Target offsets must be int64 [B] or [B,N] in {0,1,2,3} on the input device"
        )


def target_gap_embedding(offsets, width, dtype):
    """DuET sin/cos(offset+1) - sin/cos(1); zero for ordinary offset=0."""
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=offsets.device, dtype=torch.float32)
        / max(half - 1, 1)
    )

    def embedding(positions):
        angles = positions.float()[..., None] * frequencies
        result = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return F.pad(result, (0, width - result.shape[-1])).to(dtype)

    return embedding(offsets + 1) - embedding(torch.ones_like(offsets))


def validate_quadrant_geometry(geometry, patch_size):
    if not geometry.spatially_ordered or geometry.grid % (2 * patch_size):
        raise ValueError(
            "Quadrant targets require a spatially ordered codec and an even DiT patch grid"
        )


def sample_quadrant_offsets(shape, device, *, offsets=None, sequence_parallel=True):
    """Assign a permutation of 0..3 to TL/TR/BL/BR, independently per clip/origin."""
    expected = (*shape, 4)
    if offsets is None:
        scores = share_random(torch.rand(expected, device=device), enabled=sequence_parallel)
        return scores.argsort(dim=-1)
    if (
        offsets.dtype != torch.long
        or offsets.shape != expected
        or offsets.device != torch.device(device)
        or not torch.equal(
            offsets.sort(dim=-1).values, torch.arange(4, device=device).expand(expected)
        )
    ):
        raise ValueError("Quadrant offsets must be int64 [B,S-4,4] permutations of 0..3")
    return offsets


def quadrant_future_targets(states, offsets, patch_size=1):
    """Keep each quadrant's spatial coordinates; select only its source state.

    states: [B,S,C,P,H,W]; offsets: [B,S-4,4] in TL/TR/BL/BR order, a permutation of 0..3.
    Offset zero selects the ordinary next state (origin+1).
    Returns full-resolution targets, token offsets [B*(S-4),P*h*w],
    and pixel offsets [B*(S-4),H,W]. All planes share the quadrant assignment.
    """
    if states.ndim != 6:
        raise ValueError("Quadrant sources must be [B,S,C,P,H,W]")
    batch, length, channels, planes, height, width = states.shape
    count = length - 4
    if count < 1:
        raise ValueError("Quadrant targets need at least 5 states")
    if patch_size < 1 or height % (2 * patch_size) or width % (2 * patch_size):
        raise ValueError("Quadrant boundaries must align with an even DiT patch grid")
    offsets = sample_quadrant_offsets((batch, count), states.device, offsets=offsets)
    row = torch.arange(height, device=states.device) >= height // 2
    col = torch.arange(width, device=states.device) >= width // 2
    quadrant = 2 * row.long()[:, None] + col.long()[None, :]
    pixel_offsets = offsets[..., quadrant]
    source = torch.arange(count, device=states.device)[None, :, None, None] + 1 + pixel_offsets
    index = source[:, :, None, None].expand(batch, count, channels, planes, height, width)
    targets = states.gather(1, index)
    patch_offsets = pixel_offsets[..., ::patch_size, ::patch_size]
    token_offsets = (
        patch_offsets[:, :, None]
        .expand(batch, count, planes, height // patch_size, width // patch_size)
        .reshape(batch * count, -1)
    )
    return targets, token_offsets, pixel_offsets.reshape(batch * count, height, width)


def sample_half_offsets(shape, device, *, offsets=None, sequence_parallel=True):
    """Assign offsets 0/1 to left/right, independently and equiprobably per origin."""
    expected = (*shape, 2)
    if offsets is None:
        left = (
            share_random(torch.rand(shape, device=device), enabled=sequence_parallel).ge(0.5).long()
        )
        return torch.stack((left, 1 - left), dim=-1)
    if (
        offsets.dtype != torch.long
        or offsets.shape != expected
        or offsets.device != torch.device(device)
        or not torch.equal(
            offsets.sort(dim=-1).values, torch.arange(2, device=device).expand(expected)
        )
    ):
        raise ValueError("Half offsets must be int64 [B,origins,2] permutations of 0/1")
    return offsets


def half_future_targets(states, offsets, patch_size=1):
    """Select each half from its assigned future state without moving spatial positions."""
    if states.ndim != 6:
        raise ValueError("Half sources must be [B,S,C,P,H,W]")
    batch, length, channels, planes, height, width = states.shape
    count = offsets.shape[1] if offsets.ndim == 3 else 0
    if count < 1 or count > length - 2:
        raise ValueError("Half targets require between 1 and S-2 prediction origins")
    if patch_size < 1 or height % patch_size or width % (2 * patch_size):
        raise ValueError("Half boundaries must align with an even DiT width")
    offsets = sample_half_offsets((batch, count), states.device, offsets=offsets)
    half = (torch.arange(width, device=states.device) >= width // 2).long()
    pixel_offsets = offsets[..., half][:, :, None].expand(batch, count, height, width)
    source = torch.arange(count, device=states.device)[None, :, None, None] + 1 + pixel_offsets
    index = source[:, :, None, None].expand(batch, count, channels, planes, height, width)
    targets = states.gather(1, index)
    token_offsets = (
        pixel_offsets[..., ::patch_size, ::patch_size][:, :, None]
        .expand(batch, count, planes, height // patch_size, width // patch_size)
        .reshape(batch * count, -1)
    )
    return targets, token_offsets, pixel_offsets.reshape(batch * count, height, width)


def validate_token_mosaic_geometry(geometry, patch_size):
    if (
        not geometry.spatially_ordered
        or patch_size < 1
        or geometry.grid % patch_size
        or (geometry.planes * (geometry.grid // patch_size) ** 2) % 2
    ):
        raise ValueError(
            "Random token targets require a spatially ordered codec and even token count"
        )


def sample_token_offsets(shape, tokens, device, *, offsets=None, sequence_parallel=True):
    """Choose exactly half of each origin's P*h*w tokens for offsets 0 and 1."""
    if type(tokens) is not int or tokens < 2 or tokens % 2:
        raise ValueError("Random token targets require a positive even token count")
    expected = (*shape, tokens)
    if offsets is None:
        scores = share_random(torch.rand(expected, device=device), enabled=sequence_parallel)
        chosen = scores.argsort(dim=-1)[..., : tokens // 2]
        return torch.zeros(expected, dtype=torch.long, device=device).scatter_(-1, chosen, 1)
    if (
        offsets.dtype != torch.long
        or offsets.shape != expected
        or offsets.device != torch.device(device)
        or bool(((offsets < 0) | (offsets > 1)).any())
        or not bool(offsets.sum(dim=-1).eq(tokens // 2).all())
    ):
        raise ValueError("Token assignments must be int64 [B,origins,N] with exactly half 0/1")
    return offsets


def token_future_targets(states, offsets, patch_size=1):
    """Pick a source time independently per DiT token; preserve its P/H/W location.

    All channels and all pixels inside one DiT patch use the same source state.
    Unlike spatial quadrants/halves, the latent planes have independent assignments.
    Returns targets [B,origins,C,P,H,W], token offsets [B*origins,P*h*w],
    and full-resolution offsets [B*origins,P,H,W].
    """
    if states.ndim != 6:
        raise ValueError("Token sources must be [B,S,C,P,H,W]")
    batch, length, channels, planes, height, width = states.shape
    if patch_size < 1 or height % patch_size or width % patch_size:
        raise ValueError("Token patches must divide the latent height and width")
    count = offsets.shape[1] if offsets.ndim == 3 else 0
    if count < 1 or count > length - 2:
        raise ValueError("Token targets require between 1 and S-2 prediction origins")
    tokens = planes * (height // patch_size) * (width // patch_size)
    offsets = sample_token_offsets((batch, count), tokens, states.device, offsets=offsets)
    pixels = offsets.reshape(batch, count, planes, height // patch_size, width // patch_size)
    pixels = pixels.repeat_interleave(patch_size, dim=-2).repeat_interleave(patch_size, dim=-1)
    origins = torch.arange(count, device=states.device)[None, :, None, None, None]
    source = (origins + 1 + pixels)[:, :, None].expand(
        batch, count, channels, planes, height, width
    )
    return (
        states.gather(1, source),
        offsets.reshape(batch * count, tokens),
        pixels.reshape(batch * count, planes, height, width),
    )
