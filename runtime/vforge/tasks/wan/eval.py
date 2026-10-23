"""Shared Wan evaluation recipe."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
import torch.nn as nn

from vforge.data.cache import EncodedBatchCodec
from vforge.data.collate import build_inference_collate_fn
from vforge.distributed.parallel_state import get_parallel_state
from vforge.evaluation.runner import (
    EngineForward,
    EvalComponents,
    EvalOutputWriter,
    ModelWeightSource,
    sample_artifact_name,
    save_video_artifact,
)
from vforge.models.loading import CompatibleLoadCallbacks, PretrainedWeightSource
from vforge.tasks.eval_source import open_eval_dataset
from vforge.tasks.prepared_eval import (
    PreparedEvalBatch,
    PreparedEvalShape,
    build_prepared_eval_loader,
    move_prepared_eval_tensors,
)
from vforge.tasks.requests import EvalRequest
from vforge.tasks.wan_flow_match_eval import (
    WanFlowMatchEvalSettings,
    build_wan_flow_match_inference,
    prepare_wan_negative_prompt,
    resolve_wan_flow_match_eval_settings,
    resolve_wan_prepared_eval_shape,
)
from vforge.training.checkpoint import StateDictMapper, TrainingCheckpointWeightSource


class _ConditionEncoderBuilder(Protocol):
    def __call__(
        self,
        *,
        extras: Mapping[str, Any],
        pretrained_path: str,
        precision: str,
        require_assets: bool,
    ) -> Any: ...


@dataclass(frozen=True)
class WanEvalSpec:
    """Version-owned pieces consumed by the shared Wan eval recipe."""

    model: nn.Module
    codec: EncodedBatchCodec
    build_condition_encoder: _ConditionEncoderBuilder
    build_pipeline: Callable[..., Any]
    default_negative_prompt: str
    collate_encoded: Callable[[list[dict[str, Any]]], dict[str, Any]]
    pretrained_weights: PretrainedWeightSource | Sequence[PretrainedWeightSource]
    noise_shift: float
    fps: float
    state_dict_mapper: StateDictMapper | None = None
    pipeline_kwargs: Mapping[str, Any] = field(default_factory=dict)
    prepare_encoded: Callable[[dict[str, Any], Any, int, int, int], dict[str, Any]] | None = None
    wrap_inference: Callable[[Any, Any], Any] | None = None
    allow_checkpoint_mapping: bool = False
    checkpoint_load_callbacks: CompatibleLoadCallbacks | None = None
    prepare_raw_batch: (
        Callable[[dict[str, Any], Mapping[str, Any], int], Mapping[str, Any]] | None
    ) = None
    prepare_sample_context: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None


class _PrepareWanEvalBatch:
    def __init__(
        self,
        *,
        request: EvalRequest,
        spec: WanEvalSpec,
        encoder: Any | None,
        raw_collate: Callable[[list[dict[str, Any]]], dict[str, Any]] | None,
        settings: WanFlowMatchEvalSettings,
    ) -> None:
        self._request = request
        self._spec = spec
        self._encoder = encoder
        self._raw_collate = raw_collate
        self._settings = settings

    def __call__(
        self,
        samples: list[dict[str, Any]],
        source_kind: str,
        device: torch.device,
    ) -> PreparedEvalBatch:
        if self._encoder is None:
            raise RuntimeError("only the batch leader may prepare eval samples")
        sample = samples[0]
        declared_shape: PreparedEvalShape | None = None
        if source_kind == "cache":
            encoded = dict(self._spec.collate_encoded(samples))
            context = self._sample_context(sample)
        else:
            if self._raw_collate is None:
                raise RuntimeError("only the batch leader may encode raw eval samples")
            self._encoder.eval()
            batch = sample if "image_condition" in sample else self._raw_collate(samples)
            context: dict[str, Any] = {}
            if self._spec.prepare_raw_batch is not None:
                context.update(self._spec.prepare_raw_batch(batch, sample, self._request.seed))
            context.update(self._sample_context(sample, existing=context))
            encoded = dict(self._encoder.encode_inference_batch(batch))
            declared_shape = PreparedEvalShape(
                height=int(batch["height"]),
                width=int(batch["width"]),
                num_frames=int(batch["num_frames"]),
            )

        shape = resolve_wan_prepared_eval_shape(
            encoded,
            declared=declared_shape,
            vae=self._encoder.vae,
        )
        if self._spec.prepare_encoded is not None:
            encoded = self._spec.prepare_encoded(
                encoded,
                self._encoder,
                shape.height,
                shape.width,
                shape.num_frames,
            )
        encoded = move_prepared_eval_tensors(encoded, device)
        context["__negative_prompt_embeds__"] = prepare_wan_negative_prompt(
            encoded,
            tokenizer=self._encoder.tokenizer,
            text_encoder=self._encoder.text_encoder,
            settings=self._settings,
            device=device,
        )
        return PreparedEvalBatch(encoded=encoded, shape=shape, context=context)

    def _sample_context(
        self,
        sample: Mapping[str, Any],
        *,
        existing: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._spec.prepare_sample_context is None:
            return {}
        context = dict(self._spec.prepare_sample_context(sample))
        collisions = sorted(set(existing or {}).intersection(context))
        if collisions:
            raise ValueError(f"Wan eval sample context contains duplicate fields: {collisions}")
        return context


def assemble_eval_components(
    request: EvalRequest,
    *,
    extras: Mapping[str, Any],
    spec: WanEvalSpec,
) -> EvalComponents:
    """Build leader-owned input work and defer inference until model preparation."""
    state = get_parallel_state()
    encoder = (
        spec.build_condition_encoder(
            extras=extras,
            pretrained_path=request.pretrained_path,
            precision=request.precision,
            require_assets=True,
        )
        if state.is_batch_leader
        else None
    )
    dataset = open_eval_dataset(request, codec=spec.codec)
    source_kind = request.data_source
    raw_collate = None
    if source_kind != "cache" and encoder is not None:
        raw_collate = build_inference_collate_fn(
            extras=extras,
            pretrained_path=request.pretrained_path,
            allow_target_video=source_kind == "validation",
        )

    settings = resolve_wan_flow_match_eval_settings(
        request,
        extras=extras,
        default_negative_prompt=spec.default_negative_prompt,
        default_noise_shift=spec.noise_shift,
    )
    dataloader = build_prepared_eval_loader(
        request,
        dataset=dataset,
        parallel_state=state,
        label="Wan",
        prepare_batch=_PrepareWanEvalBatch(
            request=request,
            spec=spec,
            encoder=encoder,
            raw_collate=raw_collate,
            settings=settings,
        ),
    )

    def build_inference(model: nn.Module, engine_forward: EngineForward):
        pipeline = build_wan_flow_match_inference(
            model=model,
            engine_forward=engine_forward,
            pipeline_factory=spec.build_pipeline,
            tokenizer=encoder.tokenizer if encoder is not None else None,
            text_encoder=encoder.text_encoder if encoder is not None else None,
            vae=encoder.vae if encoder is not None else None,
            settings=settings,
            pipeline_kwargs=spec.pipeline_kwargs,
        )
        if spec.wrap_inference is not None:
            return spec.wrap_inference(pipeline, encoder)
        return pipeline

    checkpoint_path = request.checkpoint_path
    initial_weights: ModelWeightSource | Sequence[ModelWeightSource] | None
    if isinstance(checkpoint_path, str) and checkpoint_path:
        initial_weights = TrainingCheckpointWeightSource(
            path=checkpoint_path,
            format=request.checkpoint_format,
            mapper=spec.state_dict_mapper,
            compatible_load_callbacks=spec.checkpoint_load_callbacks,
        )
    elif isinstance(checkpoint_path, Mapping):
        if not spec.allow_checkpoint_mapping:
            raise TypeError("this Wan evaluation expects checkpoint_path to be a string")
        initial_weights = None
    else:
        initial_weights = spec.pretrained_weights

    def write_output(result: dict[str, Any], output_dir: str) -> str:
        return save_video_artifact(
            result,
            output_dir,
            fps=float(extras.get("fps") or spec.fps),
        )

    return EvalComponents(
        model=spec.model,
        dataloader=dataloader,
        initial_weights=initial_weights,
        build_inference=build_inference,
        output_writer=EvalOutputWriter(
            artifact_name=eval_artifact_name,
            write_output=write_output,
        ),
    )


def eval_artifact_name(sample_id: str) -> str:
    return sample_artifact_name(sample_id, suffix=".mp4")


__all__ = ["WanEvalSpec", "assemble_eval_components", "eval_artifact_name"]
