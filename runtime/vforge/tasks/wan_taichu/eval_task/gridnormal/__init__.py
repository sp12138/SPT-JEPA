"""Wan Taichu gridnormal evaluation assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch

from vforge.evaluation.multi_variant import MultiVariantInference
from vforge.evaluation.runner import EngineForward, EvalComponents, EvalInference
from vforge.models.wan_taichu.inference import WAN_TAICHU_GENERATOR_STATE_FIELD
from vforge.tasks.requests import EvalRequest
from vforge.tasks.wan_taichu.eval_task.gridnormal.artifacts import (
    GRIDNORMAL_SHARED_SOURCES_FIELD,
    GridNormalOutputWriter,
    gridnormal_completion_artifact_name,
    gridnormal_sample_context,
)
from vforge.tasks.wan_taichu.eval_task.gridnormal.variants import (
    GRID_FALLBACK_LATENTS_FIELD,
    GRID_FALLBACK_PIXEL_VALUE,
    GRID_LATENTS_FIELD,
    GRIDNORMAL_VARIANTS,
    NORMAL_BG_BODY_FIELD,
    NORMAL_FALLBACK_LATENTS_FIELD,
    NORMAL_FALLBACK_PIXEL_VALUE,
    NORMAL_FG_BODY_FIELD,
    NORMAL_LATENTS_FIELD,
    REF_FRAME_LATENTS_FIELD,
    RUNNABLE_GRIDNORMAL_VARIANTS,
    ConstantConditionEncoder,
    GridNormalCapabilityError,
    GridNormalVariant,
    apply_gridnormal_variant,
    prepare_gridnormal_encoded,
    resolve_effective_normal_source,
    resolve_gridnormal_variants,
)
from vforge.tasks.wan_taichu.eval_task.reference import REF_SELECTION_FIELD
from vforge.tasks.wan_taichu.variants import require_variant

DEFAULT_ENCODED_FIELD = "__encoded__"


@dataclass
class GridNormalInferencePipeline:
    """Apply all selected condition transforms around one inference pipeline."""

    base: EvalInference
    variants: tuple[GridNormalVariant, ...]
    encoded_field: str = DEFAULT_ENCODED_FIELD
    ref_selection_field: str = REF_SELECTION_FIELD

    def resolve_sample_seed(self, sample: Mapping[str, Any], base_seed: int) -> int:
        """Use Taichu's ``base seed + dataset index`` formula when preserved."""
        seed_offset = sample.get("__vforge_eval_seed_offset__")
        if seed_offset is None:
            seed_offset = sample.get("__vforge_eval_index__")
        if seed_offset is None:
            raise ValueError("gridnormal eval sample is missing its seed offset")
        return int(base_seed) + int(seed_offset)

    @torch.no_grad()
    def infer_sample(self, sample: dict[str, Any], *, seed: int) -> dict[str, Any]:
        encoded = sample.get(self.encoded_field)
        if not isinstance(encoded, Mapping):
            raise ValueError(
                f"gridnormal eval requires an encoded bundle in {self.encoded_field!r}"
            )
        ref_selection = sample.get(self.ref_selection_field)
        normalized_encoded = dict(encoded)
        if _is_online_ref_k_zero(ref_selection):
            normalized_encoded.setdefault("ref_frame_latents", None)

        def prepare_variant(
            source: dict[str, Any],
            variant: GridNormalVariant,
        ) -> dict[str, Any]:
            prepared = {
                **source,
                self.encoded_field: apply_gridnormal_variant(
                    normalized_encoded,
                    variant,
                ),
            }
            if not variant.ref:
                prepared.pop(WAN_TAICHU_GENERATOR_STATE_FIELD, None)
            return prepared

        result = MultiVariantInference(
            base=self.base,
            variants=self.variants,
            prepare_variant=prepare_variant,
        ).infer_sample(sample, seed=seed)
        context = sample.get(GRIDNORMAL_SHARED_SOURCES_FIELD)
        if not isinstance(context, Mapping):
            context = gridnormal_sample_context(sample)[GRIDNORMAL_SHARED_SOURCES_FIELD]
        variant_results = result.get("variants")
        if not isinstance(variant_results, Mapping):
            raise TypeError("gridnormal inference returned no variant result mapping")
        for variant in self.variants:
            payload = variant_results.get(variant.name)
            if not isinstance(payload, dict):
                raise TypeError(
                    f"gridnormal inference returned invalid result for {variant.name!r}"
                )
            payload["metadata"] = {
                **dict(payload.get("metadata") or {}),
                "gridnormal_condition": _variant_condition_provenance(
                    variant,
                    context,
                    encoded=normalized_encoded,
                    ref_selection=ref_selection,
                ),
            }
        result["shared_sources"] = dict(context)
        result["metadata"] = {
            **dict(result.get("metadata") or {}),
            "global_manifest_index": int(sample["__vforge_eval_index__"]),
            "seed_offset": int(
                sample.get(
                    "__vforge_eval_seed_offset__",
                    sample["__vforge_eval_index__"],
                )
            ),
            "input_sources": dict(context),
            "ref_selection": ref_selection,
        }
        return result


def _is_online_ref_k_zero(selection: Any) -> bool:
    return (
        isinstance(selection, Mapping)
        and selection.get("source") == "target_video"
        and int(selection.get("count", 0)) == 0
    )


def _variant_condition_provenance(
    variant: GridNormalVariant,
    sources: Mapping[str, Any],
    *,
    encoded: Mapping[str, Any],
    ref_selection: Any,
) -> dict[str, Any]:
    effective_normal = resolve_effective_normal_source(encoded, variant)
    normal_stream = {
        "full": "normal",
        "foreground_only": "normal_fg",
        "background_only": "normal_bg",
        "fallback": None,
    }[effective_normal]
    normal_provenance = {
        "selection": effective_normal,
        "input": sources.get(normal_stream) if normal_stream is not None else None,
        "fallback_pixel_value": (
            NORMAL_FALLBACK_PIXEL_VALUE if effective_normal == "fallback" else None
        ),
    }
    if effective_normal != variant.normal:
        normal_provenance["requested_selection"] = variant.normal
    return {
        "grid": {
            "selection": variant.grid,
            "input": sources.get("grid") if variant.grid == "real" else None,
            "fallback_pixel_value": (
                GRID_FALLBACK_PIXEL_VALUE if variant.grid == "fallback" else None
            ),
        },
        "normal": normal_provenance,
        "ref": {
            "enabled": variant.ref,
            "input": sources.get("ref") if variant.ref else None,
            "selection": ref_selection if variant.ref else None,
        },
    }


def wrap_gridnormal_eval_components(
    request: EvalRequest,
    components: EvalComponents,
    *,
    encoded_field: str = DEFAULT_ENCODED_FIELD,
    ref_selection_field: str = REF_SELECTION_FIELD,
) -> EvalComponents:
    """Wrap model-ready base components with gridnormal iteration and output."""
    variants = resolve_gridnormal_variants(
        request.extras,
        request.gridnormal_variants,
    )
    base_build_inference = components.build_inference

    def build_inference(model: torch.nn.Module, engine_forward: EngineForward) -> EvalInference:
        return GridNormalInferencePipeline(
            base=base_build_inference(model, engine_forward),
            variants=variants,
            encoded_field=encoded_field,
            ref_selection_field=ref_selection_field,
        )

    extras = dict(request.extras)
    input_manifest = request.manifest
    if not input_manifest and request.sources and request.sources[0].index_files:
        input_manifest = request.sources[0].index_files[0]
    return replace(
        components,
        build_inference=build_inference,
        output_writer=GridNormalOutputWriter(
            fps=float(extras.get("fps") or 24.0),
            variants=variants,
            data_source=request.data_source,
            manifest=input_manifest,
            max_frames=extras.get("max_frames"),
        ),
    )


def build_gridnormal_eval_components(
    request: EvalRequest,
    *,
    encoded_field: str = DEFAULT_ENCODED_FIELD,
    ref_selection_field: str = REF_SELECTION_FIELD,
) -> EvalComponents:
    """Build and wrap model-ready gridnormal evaluation components."""
    spec = require_variant(str(request.extras.get("variant") or ""))
    streams = spec.resolve_condition_streams(request.extras.get("condition_streams"))
    expected_streams = (
        ("rgb", "grid", "normal") if spec.architecture == "wan22_a14b" else ("grid", "normal")
    )
    if streams != expected_streams:
        raise ValueError(
            f"Wan Taichu gridnormal eval for {spec.name} requires "
            f"model.extras.condition_streams={list(expected_streams)} in that order"
        )
    from vforge.tasks.wan_taichu.eval_task.base import build_base_eval_components

    return wrap_gridnormal_eval_components(
        request,
        build_base_eval_components(
            request,
            prepare_encoded=prepare_gridnormal_encoded,
            prepare_sample_context=gridnormal_sample_context,
        ),
        encoded_field=encoded_field,
        ref_selection_field=ref_selection_field,
    )


__all__ = [
    "DEFAULT_ENCODED_FIELD",
    "GRIDNORMAL_SHARED_SOURCES_FIELD",
    "GRIDNORMAL_VARIANTS",
    "GRID_FALLBACK_LATENTS_FIELD",
    "GRID_LATENTS_FIELD",
    "NORMAL_BG_BODY_FIELD",
    "NORMAL_FALLBACK_LATENTS_FIELD",
    "NORMAL_FG_BODY_FIELD",
    "NORMAL_LATENTS_FIELD",
    "REF_FRAME_LATENTS_FIELD",
    "RUNNABLE_GRIDNORMAL_VARIANTS",
    "ConstantConditionEncoder",
    "GridNormalCapabilityError",
    "GridNormalInferencePipeline",
    "GridNormalOutputWriter",
    "GridNormalVariant",
    "apply_gridnormal_variant",
    "build_gridnormal_eval_components",
    "gridnormal_completion_artifact_name",
    "gridnormal_sample_context",
    "prepare_gridnormal_encoded",
    "resolve_gridnormal_variants",
    "wrap_gridnormal_eval_components",
]
