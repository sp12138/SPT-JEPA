"""Wan 2.2 A14B evaluation: the shared recipe with the expert pair.

Sampling walks the whole timestep schedule, so it crosses the expert boundary
mid-loop. The model root routes each denoising call on its own, which is why
the Wan2.1 pipeline can drive the expert pair unchanged.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from vforge.evaluation.runner import EvalComponents
from vforge.tasks.requests import EvalRequest
from vforge.tasks.wan.eval import WanEvalSpec, assemble_eval_components
from vforge.tasks.wan.v21.shared import _collate_encoded_bundles
from vforge.tasks.wan.v22.a14b.shared import (
    WAN22_A14B_ENCODED_BATCH_CODEC,
    _build_a14b_model,
    _build_condition_encoder,
    _resolve_expert_configs,
    _validate,
)
from vforge.tasks.wan.v22.a14b.train import _a14b_pretrained_sources
from vforge.training.checkpoint import TrainingCheckpointWeightSource

logger = logging.getLogger(__name__)


def build_eval_components(request: EvalRequest) -> EvalComponents:
    _validate(
        extras=request.extras,
        latent_cache_mode=request.latent_cache_mode,
        usp_size=request.usp_size,
    )
    extras = dict(request.extras)
    expert_configs, layouts = _resolve_expert_configs(extras, request.pretrained_path)
    model = _build_a14b_model(
        expert_configs=expert_configs,
        extras=extras,
        init_device=request.init_device,
    )

    from vforge.models.wan.v21.inference import (
        DEFAULT_NEGATIVE_PROMPT,
        build_inference_pipeline,
    )

    components = assemble_eval_components(
        request,
        extras=extras,
        spec=WanEvalSpec(
            model=model,
            codec=WAN22_A14B_ENCODED_BATCH_CODEC,
            build_condition_encoder=_build_condition_encoder,
            build_pipeline=build_inference_pipeline,
            default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            collate_encoded=_collate_encoded_bundles,
            pretrained_weights=_a14b_pretrained_sources(
                layouts=layouts,
                pretrained_path=request.pretrained_path,
                weight_loading="auto",
            ),
            noise_shift=3.0,
            fps=16.0,
            allow_checkpoint_mapping=True,
        ),
    )
    pair = _checkpoint_pair(request)
    if pair is None:
        return components

    _validate_checkpoint_pair(pair, request.extras)

    return replace(
        components,
        initial_weights=tuple(
            TrainingCheckpointWeightSource(
                path=pair[expert],
                model_path=model_path,
                format=request.checkpoint_format,
            )
            for expert, model_path in (("high", "wan"), ("low", "wan_2"))
        ),
    )


def _checkpoint_pair(request: EvalRequest) -> dict[str, str] | None:
    source = request.checkpoint_path
    if isinstance(source, str):
        return None
    if not isinstance(source, Mapping):
        raise TypeError("Wan2.2 evaluation.checkpoint_path must be a string or mapping")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in source.items()):
        raise TypeError("Wan2.2 evaluation.checkpoint_path mapping must contain strings")
    paths = dict(source)
    if set(paths) != {"high", "low"} or any(not path for path in paths.values()):
        raise ValueError(
            "Wan2.2 paired evaluation requires exactly non-empty "
            "evaluation.checkpoint_path.high and .low"
        )
    if os.path.realpath(paths["high"]) == os.path.realpath(paths["low"]):
        raise ValueError("Wan2.2 high and low checkpoint paths must be different")
    return paths


def _validate_checkpoint_pair(paths: dict[str, str], current_extras) -> dict[str, Any]:
    from vforge.training.checkpoint import read_manifest

    manifests: dict[str, Any] = {}
    structures: dict[str, dict[str, Any]] = {}
    for expert, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Wan2.2 {expert} checkpoint does not exist: {path}")
        if not os.path.isdir(path):
            manifests[expert] = None
            continue
        manifest = read_manifest(path)
        manifests[expert] = manifest
        semantic = manifest.fingerprints.get("semantic_config")
        if not isinstance(semantic, dict) or not isinstance(semantic.get("model"), dict):
            raise ValueError(f"Wan2.2 {expert} checkpoint manifest has no semantic model identity")
        model = semantic["model"]
        if model.get("name") != "wan22_a14b":
            raise ValueError(f"Wan2.2 {expert} checkpoint has model.name={model.get('name')!r}")
        extras = dict(model.get("extras") or {})
        if extras.get("train_expert", "both") != expert:
            raise ValueError(
                f"Wan2.2 {expert} checkpoint identifies train_expert={extras.get('train_expert')!r}"
            )
        structures[expert] = _structural_extras(extras)

    if structures:
        current = _structural_extras(dict(current_extras))
        for expert, saved in structures.items():
            if saved != current:
                raise ValueError(
                    f"Wan2.2 {expert} checkpoint structure does not match eval: "
                    f"saved={saved!r}, current={current!r}"
                )
        if len(structures) == 2 and structures["high"] != structures["low"]:
            raise ValueError("Wan2.2 high and low checkpoints have incompatible structures")
    return manifests


def _structural_extras(extras: dict[str, Any]) -> dict[str, Any]:
    from vforge.tasks.wan.v22.a14b.shared import _condition_streams

    return {
        "boundary_ratio": float(extras.get("boundary_ratio", 0.9)),
        "num_train_timesteps": int(extras.get("num_train_timesteps", 1000)),
        "condition_streams": list(_condition_streams(extras)),
        "wan_overrides": {
            key: value for key, value in sorted(extras.items()) if key.startswith("wan_")
        },
    }


__all__ = ["build_eval_components"]
