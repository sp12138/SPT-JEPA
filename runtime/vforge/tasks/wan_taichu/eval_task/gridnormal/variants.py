"""Declarative Wan Taichu grid/normal ablations and pure bundle transforms."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch

GridSource = Literal["real", "fallback"]
NormalSource = Literal["full", "foreground_only", "background_only", "fallback"]

GRID_LATENTS_FIELD = "condition_latents_grid"
NORMAL_LATENTS_FIELD = "condition_latents_normal"
GRID_FALLBACK_LATENTS_FIELD = "condition_fallback_latents_grid"
NORMAL_FALLBACK_LATENTS_FIELD = "condition_fallback_latents_normal"
NORMAL_FG_BODY_FIELD = "normal_fg_body"
NORMAL_BG_BODY_FIELD = "normal_bg_body"
REF_FRAME_LATENTS_FIELD = "ref_frame_latents"
GRID_FALLBACK_PIXEL_VALUE = 0.0
NORMAL_FALLBACK_PIXEL_VALUE = 0.5


class GridNormalCapabilityError(ValueError):
    """A requested ablation needs a capability unavailable to the model or sample."""


class ConstantConditionEncoder(Protocol):
    """Stable fallback-encoding surface supplied by Wan Taichu shared assembly."""

    def encode_constant_condition(
        self,
        *,
        pixel_value: float,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class GridNormalVariant:
    """One declarative grid/normal/ref condition selection."""

    name: str
    grid: GridSource
    normal: NormalSource
    ref: bool

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("gridnormal variant name must not be empty")
        if self.grid not in {"real", "fallback"}:
            raise ValueError(f"invalid gridnormal grid source {self.grid!r}")
        if self.normal not in {
            "full",
            "foreground_only",
            "background_only",
            "fallback",
        }:
            raise ValueError(f"invalid gridnormal normal source {self.normal!r}")


GRIDNORMAL_VARIANTS: tuple[GridNormalVariant, ...] = (
    GridNormalVariant("full", "real", "full", True),
    GridNormalVariant("normal_to_video", "fallback", "full", False),
    GridNormalVariant("normal_fg_to_video", "fallback", "foreground_only", False),
    GridNormalVariant("normal_bg_to_video", "fallback", "background_only", False),
    GridNormalVariant("grid_to_video", "real", "fallback", False),
    GridNormalVariant("ref_normal_to_video", "fallback", "full", True),
    GridNormalVariant("ref_grid_to_video", "real", "fallback", True),
    GridNormalVariant("ti2v", "fallback", "fallback", False),
)
TAICHU_GRIDNORMAL_VARIANTS: tuple[GridNormalVariant, ...] = GRIDNORMAL_VARIANTS[:5]
RUNNABLE_GRIDNORMAL_VARIANTS: tuple[str, ...] = tuple(
    variant.name for variant in GRIDNORMAL_VARIANTS if not variant.ref
)
_VARIANTS_BY_NAME = {variant.name: variant for variant in GRIDNORMAL_VARIANTS}
A14B_GRIDNORMAL_VARIANTS: tuple[GridNormalVariant, ...] = (
    GridNormalVariant("full", "real", "full", False),
    *GRIDNORMAL_VARIANTS[1:5],
)
_A14B_VARIANTS_BY_NAME = {variant.name: variant for variant in A14B_GRIDNORMAL_VARIANTS}


def resolve_gridnormal_variants(
    extras: Mapping[str, Any],
    configured_variants: Sequence[str] | None = None,
) -> tuple[GridNormalVariant, ...]:
    """Resolve configured variants and enforce the ref-projection capability gate."""
    variant_name = str(extras.get("variant") or "")
    has_ref_token = variant_name == "wan22_ti2v_5b_ref"
    available = _A14B_VARIANTS_BY_NAME if variant_name == "wan22_a14b" else _VARIANTS_BY_NAME
    names: Sequence[str] = (
        tuple(variant.name for variant in TAICHU_GRIDNORMAL_VARIANTS)
        if configured_variants is None and has_ref_token
        else (
            tuple(available)
            if configured_variants is None and variant_name == "wan22_a14b"
            else RUNNABLE_GRIDNORMAL_VARIANTS
            if configured_variants is None
            else tuple(str(name) for name in configured_variants)
        )
    )

    if not names:
        raise ValueError("evaluation.gridnormal.variants must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"gridnormal variant names must be unique, got {list(names)}")
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"unknown gridnormal variant(s) {unknown}; expected {list(available)}")

    variants = tuple(available[name] for name in names)
    requires_ref = [variant.name for variant in variants if variant.ref]
    if requires_ref and not has_ref_token:
        raise GridNormalCapabilityError(
            "gridnormal ref variants require "
            "model.extras.variant=wan22_ti2v_5b_ref; "
            f"unsupported variant(s): {requires_ref}"
        )
    return variants


def prepare_gridnormal_encoded(
    encoded: Mapping[str, Any],
    encoder: ConstantConditionEncoder,
    height: int,
    width: int,
    num_frames: int,
) -> dict[str, Any]:
    """Attach frozen-VAE black-grid and neutral-normal fallback latents."""
    grid = encoded.get(GRID_LATENTS_FIELD)
    normal = encoded.get(NORMAL_LATENTS_FIELD)
    if not isinstance(grid, torch.Tensor) or not isinstance(normal, torch.Tensor):
        raise ValueError(
            "gridnormal eval requires condition_latents_grid and condition_latents_normal tensors"
        )
    if grid.shape != normal.shape:
        raise ValueError(
            "grid and normal latent geometry must match, got "
            f"{tuple(grid.shape)} and {tuple(normal.shape)}"
        )
    encode_constant = getattr(encoder, "encode_constant_condition", None)
    if not callable(encode_constant):
        raise RuntimeError(
            "gridnormal fallback preparation requires an encoder implementing "
            "encode_constant_condition()"
        )
    fallback_args = {
        "batch_size": int(grid.shape[0]),
        "num_frames": int(num_frames),
        "height": int(height),
        "width": int(width),
        "dtype": grid.dtype,
        "device": grid.device,
    }
    grid_fallback = encode_constant(
        pixel_value=GRID_FALLBACK_PIXEL_VALUE,
        **fallback_args,
    )
    normal_fallback = encode_constant(
        pixel_value=NORMAL_FALLBACK_PIXEL_VALUE,
        **fallback_args,
    )
    if not isinstance(grid_fallback, torch.Tensor) or not isinstance(
        normal_fallback,
        torch.Tensor,
    ):
        raise TypeError("encode_constant_condition() must return a tensor")
    if grid_fallback.shape != grid.shape or normal_fallback.shape != normal.shape:
        raise ValueError(
            "frozen-VAE fallback geometry does not match encoded condition streams: "
            f"expected={tuple(grid.shape)}, grid={tuple(grid_fallback.shape)}, "
            f"normal={tuple(normal_fallback.shape)}"
        )
    return {
        **encoded,
        GRID_FALLBACK_LATENTS_FIELD: grid_fallback,
        NORMAL_FALLBACK_LATENTS_FIELD: normal_fallback,
    }


def _require_tensor(encoded: Mapping[str, Any], field: str, variant: str) -> torch.Tensor:
    value = encoded.get(field)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"gridnormal variant {variant!r} requires tensor field {field!r}")
    return value


def resolve_effective_normal_source(
    encoded: Mapping[str, Any],
    variant: GridNormalVariant,
) -> NormalSource:
    """Resolve Taichu's neutral fallback for unavailable partial-normal inputs."""
    partial_fields = {
        "foreground_only": NORMAL_FG_BODY_FIELD,
        "background_only": NORMAL_BG_BODY_FIELD,
    }
    field = partial_fields.get(variant.normal)
    if field is not None and encoded.get(field) is None:
        return "fallback"
    return variant.normal


def _normal_variant_tensor(
    encoded: Mapping[str, Any],
    *,
    field: str,
    variant: str,
    partial: bool,
) -> torch.Tensor:
    selected = _require_tensor(encoded, field, variant)
    if not partial:
        return selected
    full = _require_tensor(encoded, NORMAL_LATENTS_FIELD, variant)
    if selected.shape[1] == full.shape[1]:
        return selected
    mask_channels = int(full.shape[1]) - int(selected.shape[1])
    if mask_channels <= 0 or selected.shape[2:] != full.shape[2:]:
        raise ValueError(
            f"gridnormal partial normal geometry {tuple(selected.shape)} cannot replace "
            f"full normal block {tuple(full.shape)}"
        )
    return torch.cat([full[:, :mask_channels], selected], dim=1)


def apply_gridnormal_variant(
    encoded: Mapping[str, Any],
    variant: GridNormalVariant,
) -> dict[str, Any]:
    """Return a transformed condition bundle without mutating the input.

    A present ``ref_frame_latents=None`` denotes the intentional online-ref
    ``K=0`` case. Ref variants then run without ref tokens, while an omitted
    field remains an error so malformed explicit-ref samples fail early.
    """
    out = dict(encoded)
    out[GRID_LATENTS_FIELD] = _require_tensor(
        encoded,
        GRID_LATENTS_FIELD if variant.grid == "real" else GRID_FALLBACK_LATENTS_FIELD,
        variant.name,
    )

    normal_fields: dict[NormalSource, str] = {
        "full": NORMAL_LATENTS_FIELD,
        "foreground_only": NORMAL_FG_BODY_FIELD,
        "background_only": NORMAL_BG_BODY_FIELD,
        "fallback": NORMAL_FALLBACK_LATENTS_FIELD,
    }
    effective_normal = resolve_effective_normal_source(encoded, variant)
    out[NORMAL_LATENTS_FIELD] = _normal_variant_tensor(
        encoded,
        field=normal_fields[effective_normal],
        variant=variant.name,
        partial=effective_normal in {"foreground_only", "background_only"},
    )

    if not variant.ref:
        out.pop(REF_FRAME_LATENTS_FIELD, None)
    elif REF_FRAME_LATENTS_FIELD not in encoded:
        raise GridNormalCapabilityError(
            f"gridnormal ref variant {variant.name!r} requires ref_frame_latents"
        )
    else:
        ref_latents = encoded[REF_FRAME_LATENTS_FIELD]
        if ref_latents is None:
            out.pop(REF_FRAME_LATENTS_FIELD, None)
        elif not isinstance(ref_latents, torch.Tensor):
            raise ValueError(
                f"gridnormal ref variant {variant.name!r} requires tensor "
                "ref_frame_latents or None for online K=0"
            )
    return out


__all__ = [
    "A14B_GRIDNORMAL_VARIANTS",
    "GRIDNORMAL_VARIANTS",
    "GRID_FALLBACK_LATENTS_FIELD",
    "GRID_FALLBACK_PIXEL_VALUE",
    "GRID_LATENTS_FIELD",
    "NORMAL_BG_BODY_FIELD",
    "NORMAL_FALLBACK_LATENTS_FIELD",
    "NORMAL_FALLBACK_PIXEL_VALUE",
    "NORMAL_FG_BODY_FIELD",
    "NORMAL_LATENTS_FIELD",
    "REF_FRAME_LATENTS_FIELD",
    "RUNNABLE_GRIDNORMAL_VARIANTS",
    "TAICHU_GRIDNORMAL_VARIANTS",
    "ConstantConditionEncoder",
    "GridNormalCapabilityError",
    "GridNormalVariant",
    "apply_gridnormal_variant",
    "prepare_gridnormal_encoded",
    "resolve_effective_normal_source",
    "resolve_gridnormal_variants",
]
