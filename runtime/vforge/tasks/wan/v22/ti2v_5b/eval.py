"""Wan 2.2 TI2V evaluation: the shared recipe with TI2V's model and pipeline."""

from __future__ import annotations

import logging
from typing import Any

from vforge.evaluation.runner import EvalComponents
from vforge.tasks.requests import EvalRequest
from vforge.tasks.wan.eval import WanEvalSpec, assemble_eval_components
from vforge.tasks.wan.shared import build_state_dict_mapper
from vforge.tasks.wan.v22.ti2v_5b.ct.train import _wan22_pretrained_source
from vforge.tasks.wan.v22.ti2v_5b.shared import (
    _build_condition_encoder,
    _build_wan22_model,
    _collate_encoded_bundles,
    _encoder_codec,
    _resolve_model_config,
    _stream_fields,
    _validate,
)

logger = logging.getLogger(__name__)


def build_eval_components(request: EvalRequest) -> EvalComponents:
    _validate(
        extras=request.extras,
        latent_cache_mode=request.latent_cache_mode,
        usp_size=request.usp_size,
    )
    extras = dict(request.extras)
    codec = _encoder_codec(extras)
    config, layout = _resolve_model_config(extras, request.pretrained_path)
    model = _build_wan22_model(config=config, init_device=request.init_device)

    from vforge.models.wan.v22.ti2v_5b.ct.inference import (
        DEFAULT_NEGATIVE_PROMPT,
        build_inference_pipeline,
    )

    def collate_encoded(batches: list[dict[str, Any]]) -> dict[str, Any]:
        return _collate_encoded_bundles(codec, batches)

    return assemble_eval_components(
        request,
        extras=extras,
        spec=WanEvalSpec(
            model=model,
            codec=codec,
            build_condition_encoder=_build_condition_encoder,
            build_pipeline=build_inference_pipeline,
            default_negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            collate_encoded=collate_encoded,
            pretrained_weights=_wan22_pretrained_source(
                layout=layout,
                pretrained_path=request.pretrained_path,
                weight_loading="auto",
            ),
            noise_shift=5.0,
            fps=24.0,
            state_dict_mapper=build_state_dict_mapper(),
            # TI2V in-paints the reference frame into latent frame 0, so the
            # sampler needs to know which streams the latent carries.
            pipeline_kwargs={"stream_fields": _stream_fields(extras)},
        ),
    )


__all__ = ["build_eval_components"]
