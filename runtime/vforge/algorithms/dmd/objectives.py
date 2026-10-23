"""Composable DMD loss kernels for tensor and multi-component latents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F

from vforge.algorithms.dmd.model import LatentBundle


def dmd_surrogate_loss(
    generated: torch.Tensor,
    score_gradient: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inject a detached DMD direction into a differentiable generated tensor."""

    _require_same_shape(generated, score_gradient, "generated", "score_gradient")
    generated_double = generated.double()
    target = (generated_double - score_gradient.double()).detach()
    return 0.5 * _masked_mse(generated_double, target, valid_mask)


def flow_critic_loss(
    flow_prediction: torch.Tensor,
    clean: torch.Tensor,
    noise: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Regress canonical flow velocity ``noise - clean``."""

    _require_same_shape(flow_prediction, clean, "flow_prediction", "clean")
    _require_same_shape(clean, noise, "clean", "noise")
    return _masked_mse(
        flow_prediction,
        noise - clean,
        valid_mask,
    )


def dmd_surrogate_loss_bundle(
    generated: LatentBundle,
    score_gradient: LatentBundle,
    *,
    valid_masks: Mapping[str, torch.Tensor | None] | None = None,
    component_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Combine per-component generator surrogate losses."""

    return _weighted_bundle_loss(
        generated,
        score_gradient,
        valid_masks=valid_masks,
        component_weights=component_weights,
        loss_fn=dmd_surrogate_loss,
    )


def flow_critic_loss_bundle(
    flow_prediction: LatentBundle,
    clean: LatentBundle,
    noise: LatentBundle,
    *,
    valid_masks: Mapping[str, torch.Tensor | None] | None = None,
    component_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Combine per-component critic flow-regression losses."""

    keys = matching_keys(flow_prediction, clean, noise)
    weights = resolve_component_weights(keys, component_weights)
    masks = component_masks(keys, valid_masks)
    result: torch.Tensor | None = None
    for key in keys:
        component = weights[key] * flow_critic_loss(
            flow_prediction[key],
            clean[key],
            noise[key],
            valid_mask=masks[key],
        )
        result = component if result is None else result + component
    assert result is not None
    return result


def data_regression_loss_bundle(
    generated: LatentBundle,
    clean: LatentBundle,
    *,
    valid_masks: Mapping[str, torch.Tensor | None] | None = None,
    component_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Regress generated values directly toward clean data."""

    return _weighted_bundle_loss(
        generated,
        clean,
        valid_masks=valid_masks,
        component_weights=component_weights,
        loss_fn=lambda prediction, target, *, valid_mask: _masked_mse(
            prediction.double(),
            target.double(),
            valid_mask,
        ),
    )


def _weighted_bundle_loss(
    left: LatentBundle,
    right: LatentBundle,
    *,
    valid_masks: Mapping[str, torch.Tensor | None] | None,
    component_weights: Mapping[str, float] | None,
    loss_fn,
) -> torch.Tensor:
    keys = matching_keys(left, right)
    weights = resolve_component_weights(keys, component_weights)
    masks = component_masks(keys, valid_masks)
    result: torch.Tensor | None = None
    for key in keys:
        component = weights[key] * loss_fn(
            left[key],
            right[key],
            valid_mask=masks[key],
        )
        result = component if result is None else result + component
    assert result is not None
    return result


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    if valid_mask is None:
        return F.mse_loss(prediction, target, reduction="mean")
    if valid_mask.dtype != torch.bool:
        raise TypeError("loss valid_mask must be boolean")
    try:
        expanded = torch.broadcast_to(valid_mask, prediction.shape)
    except RuntimeError as error:
        raise ValueError(
            f"loss valid_mask shape {tuple(valid_mask.shape)} cannot broadcast to "
            f"{tuple(prediction.shape)}"
        ) from error
    if not bool(expanded.any()):
        raise ValueError("loss valid_mask must select at least one value")
    return F.mse_loss(prediction[expanded], target[expanded], reduction="mean")


def matching_keys(*bundles: LatentBundle) -> tuple[str, ...]:
    if not bundles or not bundles[0]:
        raise ValueError("latent bundles must contain at least one component")
    keys = tuple(bundles[0])
    expected = set(keys)
    for bundle in bundles[1:]:
        if set(bundle) != expected:
            raise ValueError("latent bundle component keys must match")
    return keys


def resolve_component_weights(
    keys: tuple[str, ...],
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    if weights is None:
        return dict.fromkeys(keys, 1.0)
    if set(weights) != set(keys):
        raise ValueError("component weight keys must match latent bundle keys")
    resolved = {key: float(weights[key]) for key in keys}
    if any(weight < 0 for weight in resolved.values()):
        raise ValueError("component weights must be non-negative")
    if not any(resolved.values()):
        raise ValueError("at least one component weight must be positive")
    return resolved


def component_masks(
    keys: tuple[str, ...],
    masks: Mapping[str, torch.Tensor | None] | None,
) -> dict[str, torch.Tensor | None]:
    if masks is None:
        return dict.fromkeys(keys, None)
    if set(masks) != set(keys):
        raise ValueError("loss mask keys must match latent bundle keys")
    return {key: masks[key] for key in keys}


def _require_same_shape(
    left: torch.Tensor,
    right: torch.Tensor,
    left_name: str,
    right_name: str,
) -> None:
    if left.shape != right.shape:
        raise ValueError(
            f"{left_name} and {right_name} must have matching shapes; "
            f"got {tuple(left.shape)} and {tuple(right.shape)}"
        )


TensorScoreCallable = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
BundleScoreCallable = Callable[[LatentBundle, Any], LatentBundle]


def remap_dmd_timesteps(
    timestep: torch.Tensor,
    scheduler_timesteps: torch.Tensor,
    *,
    timestep_shift: float,
    min_timestep: int,
    max_timestep: int,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """Apply the optional shifted-schedule lookup and effective clamp."""

    if timestep_shift > 1:
        indices = num_train_timesteps - timestep.clamp(1, num_train_timesteps)
        remapped = scheduler_timesteps.to(timestep.device)[indices.long()]
    else:
        remapped = timestep
    return remapped.clamp(min_timestep, max_timestep)


def dmd_score_gradient(
    noisy_latents: torch.Tensor,
    clean_latents: torch.Tensor,
    timestep: torch.Tensor,
    *,
    fake_score: TensorScoreCallable,
    real_cond_score: TensorScoreCallable,
    real_uncond_score: TensorScoreCallable,
    real_guidance_scale: float,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute one detached DMD gradient using cond-centric real CFG."""

    with torch.no_grad():
        pred_fake = fake_score(noisy_latents, timestep)
        pred_real_cond = real_cond_score(noisy_latents, timestep)
        pred_real_uncond = real_uncond_score(noisy_latents, timestep)
        require_tensor_shape(pred_fake, noisy_latents, "fake score")
        require_tensor_shape(pred_real_cond, noisy_latents, "conditional real score")
        require_tensor_shape(pred_real_uncond, noisy_latents, "unconditional real score")
        pred_real = pred_real_cond + real_guidance_scale * (pred_real_cond - pred_real_uncond)
        gradient = pred_fake - pred_real
        if normalize:
            reduce_dims = tuple(range(1, clean_latents.ndim))
            normalizer = (
                (clean_latents - pred_real)
                .abs()
                .mean(
                    dim=reduce_dims,
                    keepdim=True,
                )
            )
            gradient = gradient / normalizer
        return torch.nan_to_num(gradient).detach()


def dmd_score_gradient_bundle(
    noisy: LatentBundle,
    clean: LatentBundle,
    noise_level: Any,
    *,
    fake_score: BundleScoreCallable,
    real_cond_score: BundleScoreCallable,
    real_uncond_score: BundleScoreCallable,
    real_guidance_scale: float,
    normalize: bool = True,
) -> dict[str, torch.Tensor]:
    """Compute detached per-component DMD gradients with three score calls."""

    with torch.no_grad():
        keys, fake, real_cond, real_uncond = bundle_score_predictions(
            noisy,
            clean,
            noise_level,
            fake_score=fake_score,
            real_cond_score=real_cond_score,
            real_uncond_score=real_uncond_score,
        )
        gradients: dict[str, torch.Tensor] = {}
        for key in keys:
            guided_real = real_cond[key] + real_guidance_scale * (real_cond[key] - real_uncond[key])
            gradient = fake[key] - guided_real
            if normalize:
                normalizer = (
                    (clean[key] - guided_real)
                    .abs()
                    .mean(
                        dim=tuple(range(1, clean[key].ndim)),
                        keepdim=True,
                    )
                )
                gradient = gradient / normalizer
            gradients[key] = torch.nan_to_num(gradient).detach()
        return gradients


def bundle_score_predictions(
    noisy, clean, noise_level, *, fake_score, real_cond_score, real_uncond_score
):
    """Call each tower once and validate the shared bundle/shape contract."""
    keys = validated_bundle_keys(noisy, clean)
    fake = fake_score(noisy, noise_level)
    real_cond = real_cond_score(noisy, noise_level)
    real_uncond = real_uncond_score(noisy, noise_level)
    validated_bundle_keys(noisy, fake, real_cond, real_uncond)
    for key in keys:
        require_tensor_shape(fake[key], noisy[key], f"fake score component {key!r}")
        require_tensor_shape(
            real_cond[key], noisy[key], f"conditional real score component {key!r}"
        )
        require_tensor_shape(
            real_uncond[key], noisy[key], f"unconditional real score component {key!r}"
        )
    return keys, fake, real_cond, real_uncond


def validated_bundle_keys(*bundles: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    if not bundles or not bundles[0]:
        raise ValueError("latent bundles must contain at least one component")
    keys = tuple(bundles[0])
    expected = set(keys)
    for bundle in bundles[1:]:
        if set(bundle) != expected:
            raise ValueError(
                "latent bundle component keys must match; "
                f"expected={sorted(expected)}, got={sorted(bundle)}"
            )
    return keys


def require_tensor_shape(
    value: torch.Tensor,
    reference: torch.Tensor,
    name: str,
) -> None:
    if value.shape != reference.shape:
        raise ValueError(
            f"{name} shape must match its noisy input; "
            f"got {tuple(value.shape)} and {tuple(reference.shape)}"
        )


__all__ = [
    "BundleScoreCallable",
    "TensorScoreCallable",
    "data_regression_loss_bundle",
    "dmd_score_gradient",
    "dmd_score_gradient_bundle",
    "dmd_surrogate_loss",
    "dmd_surrogate_loss_bundle",
    "flow_critic_loss",
    "flow_critic_loss_bundle",
    "remap_dmd_timesteps",
]
