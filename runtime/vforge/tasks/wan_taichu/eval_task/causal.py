"""Wan Taichu causal evaluation assembly."""

from __future__ import annotations

from typing import Any

from vforge.evaluation.runner import EvalComponents
from vforge.models.wan_taichu.inference.causal import (
    DEFAULT_NEGATIVE_PROMPT,
    FULL_RECOMPUTE,
    INCREMENTAL_KV,
    build_inference_pipeline,
)
from vforge.tasks.requests import EvalRequest
from vforge.tasks.wan_taichu.conditioning import ConditionDropoutPolicy
from vforge.tasks.wan_taichu.eval_task.base import build_base_eval_components
from vforge.tasks.wan_taichu.variants import require_variant
from vforge.utils.training import resolve_dtype

CAUSAL_VARIANT = "wan22_ti2v_5b_causal_teacher"
A14B_CAUSAL_VARIANT = "wan22_a14b_causal_teacher"
CAUSAL_VARIANTS = (CAUSAL_VARIANT, A14B_CAUSAL_VARIANT)


def causal_pipeline_kwargs(request: EvalRequest) -> dict[str, Any]:
    """Resolve and validate causal-only pipeline options."""
    extras = dict(request.extras)
    variant = str(extras.get("variant", CAUSAL_VARIANT))
    if variant not in CAUSAL_VARIANTS:
        raise ValueError(f"causal eval requires one of {CAUSAL_VARIANTS!r}, got {variant!r}")
    rollout_mode = str(request.rollout_mode or FULL_RECOMPUTE)
    if rollout_mode not in {FULL_RECOMPUTE, INCREMENTAL_KV}:
        raise ValueError(
            f"unsupported rollout_mode={rollout_mode!r}; expected "
            f"{FULL_RECOMPUTE!r} or {INCREMENTAL_KV!r}"
        )
    is_a14b = variant == A14B_CAUSAL_VARIANT
    frame_per_block = int(extras.get("frame_per_block", 3))
    if frame_per_block < 1:
        raise ValueError(f"frame_per_block must be >= 1, got {frame_per_block}")
    first_frame_own_block = bool(extras.get("first_frame_own_block", not is_a14b))
    return {
        "frame_per_block": frame_per_block,
        "first_frame_own_block": first_frame_own_block,
        "rollout_mode": rollout_mode,
        "usp_size": int(request.usp_size),
        "vae_scale_factor_spatial": 8 if is_a14b else 16,
        "execution_dtype": resolve_dtype(request.precision),
    }


def build_causal_eval_components(
    request: EvalRequest,
) -> EvalComponents:
    """Build causal evaluation with the package-local base assembler."""
    extras = dict(request.extras)
    variant = str(extras.get("variant", CAUSAL_VARIANT))
    prepare_encoded = None
    if variant == A14B_CAUSAL_VARIANT:
        spec = require_variant(variant)
        streams = spec.resolve_condition_streams(extras.get("condition_streams"))
        dropout = ConditionDropoutPolicy(spec=spec, streams=streams, extras=extras)

        def prepare_encoded(encoded, encoder, _height, _width, _num_frames):
            return dropout.apply(encoded, encoder=encoder)

    return build_base_eval_components(
        request,
        build_pipeline=build_inference_pipeline,
        default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        pipeline_kwargs=causal_pipeline_kwargs(request),
        prepare_encoded=prepare_encoded,
        supported_variants=CAUSAL_VARIANTS,
    )


__all__ = [
    "A14B_CAUSAL_VARIANT",
    "CAUSAL_VARIANT",
    "CAUSAL_VARIANTS",
    "build_causal_eval_components",
    "causal_pipeline_kwargs",
]
