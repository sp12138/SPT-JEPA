"""Wan 2.1 evaluation: the shared recipe with 2.1's model and pipeline."""

from __future__ import annotations

import logging

from vforge.evaluation.runner import EvalComponents
from vforge.tasks.requests import EvalRequest
from vforge.tasks.wan.eval import WanEvalSpec, assemble_eval_components
from vforge.tasks.wan.shared import build_state_dict_mapper, validate_shared
from vforge.tasks.wan.v21.shared import (
    WAN21_ENCODED_BATCH_CODEC,
    _build_condition_encoder,
    _build_wan21_model,
    _collate_encoded_bundles,
)
from vforge.tasks.wan.v21.train import _wan_pretrained_source

logger = logging.getLogger(__name__)


def build_eval_components(request: EvalRequest) -> EvalComponents:
    validate_shared(
        latent_cache_mode=request.latent_cache_mode,
        usp_size=request.usp_size,
    )
    extras = dict(request.extras)
    model = _build_wan21_model(
        extras=extras,
        init_device=request.init_device,
        pretrained_path=request.pretrained_path,
    )

    from vforge.models.wan.v21.inference import (
        DEFAULT_NEGATIVE_PROMPT,
        build_inference_pipeline,
    )

    return assemble_eval_components(
        request,
        extras=extras,
        spec=WanEvalSpec(
            model=model,
            codec=WAN21_ENCODED_BATCH_CODEC,
            build_condition_encoder=_build_condition_encoder,
            build_pipeline=build_inference_pipeline,
            default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            collate_encoded=_collate_encoded_bundles,
            pretrained_weights=_wan_pretrained_source(
                pretrained_path=request.pretrained_path,
                weight_loading="auto",
            ),
            noise_shift=5.0,
            fps=16.0,
            state_dict_mapper=build_state_dict_mapper(),
        ),
    )


__all__ = ["build_eval_components"]
