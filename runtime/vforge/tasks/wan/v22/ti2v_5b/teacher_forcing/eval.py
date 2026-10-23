"""Wan 2.2 TI2V teacher-forcing causal evaluation recipe."""

from __future__ import annotations

from typing import Any

from vforge.evaluation.runner import EvalComponents
from vforge.tasks.requests import EvalRequest
from vforge.tasks.wan.eval import WanEvalSpec, assemble_eval_components
from vforge.tasks.wan.shared import build_state_dict_mapper
from vforge.tasks.wan.v22.ti2v_5b.ct.train import _wan22_pretrained_source
from vforge.tasks.wan.v22.ti2v_5b.shared import (
    _build_condition_encoder,
    _collate_encoded_bundles,
    _encoder_codec,
    _resolve_model_config,
    _stream_fields,
    _validate,
)
from vforge.tasks.wan.v22.ti2v_5b.teacher_forcing.train import (
    _build_tf_model,
    _first_frame_own_block,
    _frame_per_block,
    _validate_tf,
)


def build_eval_components(request: EvalRequest) -> EvalComponents:
    """Build the TF model and causal sampler while reusing shared Wan eval I/O."""
    from vforge.models.wan.v22.ti2v_5b.teacher_forcing.inference import (
        DEFAULT_NEGATIVE_PROMPT,
        FULL_RECOMPUTE,
        INCREMENTAL_KV,
        build_inference_pipeline,
    )

    rollout_mode = str(request.rollout_mode or FULL_RECOMPUTE)
    if rollout_mode == INCREMENTAL_KV and request.usp_size != 1:
        raise NotImplementedError(
            f"rollout_mode='incremental_kv' requires parallel.usp_size=1; got {request.usp_size}"
        )
    _validate(
        extras=request.extras,
        latent_cache_mode=request.latent_cache_mode,
        usp_size=request.usp_size,
    )
    _validate_tf(extras=request.extras)
    extras = dict(request.extras)
    codec = _encoder_codec(extras)
    config, layout = _resolve_model_config(extras, request.pretrained_path)
    model = _build_tf_model(
        config=config,
        extras=extras,
        init_device=request.init_device,
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
            pipeline_kwargs={
                "stream_fields": _stream_fields(extras),
                "frame_per_block": _frame_per_block(extras),
                "first_frame_own_block": _first_frame_own_block(extras),
                "rollout_mode": str(request.rollout_mode or FULL_RECOMPUTE),
                "usp_size": request.usp_size,
            },
        ),
    )


__all__ = ["build_eval_components"]
