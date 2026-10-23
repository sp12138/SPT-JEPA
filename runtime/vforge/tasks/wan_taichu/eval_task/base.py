"""Wan Taichu-only evaluation component assembly."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import partial
from types import MappingProxyType
from typing import Any

import torch

from vforge.data.mixed_dataset import wrap_latent_bounds
from vforge.distributed.parallel_state import get_parallel_state
from vforge.evaluation.runner import (
    EngineForward,
    EvalComponents,
    EvalOutputWriter,
    sample_artifact_name,
    save_video_artifact,
)
from vforge.models.wan_taichu.inference import (
    DEFAULT_NEGATIVE_PROMPT,
    encode_inference_batch,
)
from vforge.models.wan_taichu.inference import (
    build_inference_pipeline as build_bidirectional_pipeline,
)
from vforge.tasks.eval_source import open_eval_dataset
from vforge.tasks.prepared_eval import (
    PreparedEvalBatch,
    PreparedEvalShape,
    build_prepared_eval_loader,
    move_prepared_eval_tensors,
)
from vforge.tasks.requests import EvalRequest
from vforge.tasks.wan_flow_match_eval import (
    build_wan_flow_match_inference,
    prepare_wan_negative_prompt,
    resolve_wan_flow_match_eval_settings,
    resolve_wan_prepared_eval_shape,
)
from vforge.tasks.wan_taichu.batch import (
    build_encoded_batch_codec,
    canonicalize_encoded_batch,
    collate_encoded_bundles,
    stream_field,
)
from vforge.tasks.wan_taichu.cache import open_taichu_encoded_cache_dataset
from vforge.tasks.wan_taichu.encoder import (
    build_condition_encoder,
    build_inference_collate_fn,
)
from vforge.tasks.wan_taichu.eval_task.checkpoint import (
    a14b_checkpoint_mapper,
    checkpoint_load_callbacks,
)
from vforge.tasks.wan_taichu.eval_task.reference import _prepare_reference_batch
from vforge.tasks.wan_taichu.model import build_model
from vforge.tasks.wan_taichu.temporal import truncate_encoded_latents
from vforge.tasks.wan_taichu.train import _pretrained_sources
from vforge.tasks.wan_taichu.variants import WanTaichuVariant, require_variant
from vforge.training.checkpoint import StateDictMapper, TrainingCheckpointWeightSource

_RAW_STREAM_KEYS = MappingProxyType(
    {
        "normal": "normal_condition",
        "grid": "grid_condition",
    }
)
_BIDIRECTIONAL_VARIANTS = ("wan22_a14b", "wan22_ti2v_5b", "wan22_ti2v_5b_ref")


def _open_wan_taichu_eval_cache(
    source: Any,
    codec: Any,
    *,
    load_ref_sidecar: bool = False,
):
    dataset = open_taichu_encoded_cache_dataset(
        source.cache_dir,
        source.manifest,
        codec=codec,
        load_ref_sidecar=load_ref_sidecar,
        indexed_sample_count=source.extras.get("indexed_sample_count"),
    )
    return wrap_latent_bounds(
        dataset,
        min_latents=source.min_latents,
        max_latents=source.max_latents,
    )


def resolve_bidirectional_request(
    request: EvalRequest,
) -> tuple[WanTaichuVariant, tuple[str, ...], dict[str, Any]]:
    """Resolve the supported standalone bidirectional variants."""
    return resolve_eval_request(request, supported_variants=_BIDIRECTIONAL_VARIANTS)


def resolve_eval_request(
    request: EvalRequest,
    *,
    supported_variants: Sequence[str],
) -> tuple[WanTaichuVariant, tuple[str, ...], dict[str, Any]]:
    """Resolve one package-local workflow and its condition streams."""
    if request.usp_size < 1:
        raise ValueError("parallel.usp_size must be >= 1")
    if request.latent_cache_mode not in {"online", "offline"}:
        raise ValueError(f"unsupported latent_cache.mode={request.latent_cache_mode!r}")
    name = str(request.extras.get("variant") or "")
    if not name:
        raise ValueError("model.extras.variant is required for model.name=wan_taichu")
    spec = require_variant(name)
    accepted = tuple(str(value) for value in supported_variants)
    if spec.name not in accepted:
        raise NotImplementedError(
            f"Wan Taichu workflow currently supports {list(accepted)}, got {spec.name!r}"
        )
    streams = spec.resolve_condition_streams(request.extras.get("condition_streams"))
    extras = dict(request.extras)
    extras["condition_streams"] = list(streams)
    extras.setdefault("noise_shift", spec.noise_shift)
    extras["grid_random_rot_deg"] = 0.0
    if spec.boundary_ratio is not None:
        extras.setdefault("boundary_ratio", spec.boundary_ratio)
    if spec.attention == "block_causal_teacher":
        extras.setdefault("frame_per_block", spec.frame_per_block)
        extras.setdefault("first_frame_own_block", spec.first_frame_own_block)
        extras.setdefault("local_attn_size", -1)
        extras.setdefault("sink_size", 0)
    online_refs = int(extras.get("ref_num_frames_online", 0))
    if online_refs < 0:
        raise ValueError("model.extras.ref_num_frames_online must be non-negative")
    if online_refs and not spec.ref_token:
        raise ValueError("ref_num_frames_online is only valid for wan22_ti2v_5b_ref")
    online_ref_override = (
        request.online_ref_num_frames is not None or request.online_ref_frame_indices is not None
    )
    if online_ref_override and not spec.ref_token:
        raise ValueError("evaluation.online_ref requires a ref-token variant")
    if online_ref_override and request.data_source != "validation":
        raise ValueError("evaluation.online_ref requires evaluation.data_source=validation")
    return spec, streams, extras


def build_base_eval_components(
    request: EvalRequest,
    *,
    build_pipeline: Callable[..., Any] | None = None,
    default_negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    pipeline_kwargs: Mapping[str, Any] | None = None,
    prepare_encoded: (Callable[[dict[str, Any], Any, int, int, int], dict[str, Any]] | None) = None,
    prepare_sample_context: (Callable[[Mapping[str, Any]], Mapping[str, Any]] | None) = None,
    supported_variants: Sequence[str] = _BIDIRECTIONAL_VARIANTS,
) -> EvalComponents:
    """Build the model-ready components for one Wan Taichu eval workflow."""
    spec, streams, extras = resolve_eval_request(
        request,
        supported_variants=supported_variants,
    )
    uses_bidirectional_pipeline = build_pipeline is None
    if build_pipeline is None:
        build_pipeline = build_bidirectional_pipeline
    codec = build_encoded_batch_codec(spec, streams)
    model, layouts = build_model(
        spec=spec,
        streams=streams,
        extras=extras,
        pretrained_path=request.pretrained_path,
        init_device=request.init_device,
    )
    state = get_parallel_state()
    encoder = (
        build_condition_encoder(
            architecture=spec.architecture,
            streams=streams,
            mask_channels=spec.mask_channels,
            raw_stream_keys=_RAW_STREAM_KEYS,
            extras=extras,
            pretrained_path=request.pretrained_path,
            precision=request.precision,
        )
        if state.is_batch_leader
        else None
    )
    dataset = open_eval_dataset(
        request,
        codec=codec,
        open_encoded_source=partial(
            _open_wan_taichu_eval_cache,
            load_ref_sidecar=spec.ref_token,
        ),
    )
    raw_collate = (
        build_inference_collate_fn(
            tokenizer=encoder.tokenizer,
            extras=extras,
            streams=streams,
            allow_target_video=request.data_source == "validation",
        )
        if request.data_source != "cache" and encoder is not None
        else None
    )
    stream_fields = {name: stream_field(name) for name in streams}
    settings = resolve_wan_flow_match_eval_settings(
        request,
        extras=extras,
        default_negative_prompt=default_negative_prompt,
        default_noise_shift=spec.noise_shift,
    )

    def sample_context(
        sample: Mapping[str, Any],
        existing: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if prepare_sample_context is None:
            return {}
        context = dict(prepare_sample_context(sample))
        collisions = sorted(set(existing or {}).intersection(context))
        if collisions:
            raise ValueError(
                f"Wan Taichu eval sample context contains duplicate fields: {collisions}"
            )
        return context

    def prepare_batch(
        samples: list[dict[str, Any]],
        source_kind: str,
        device: torch.device,
    ) -> PreparedEvalBatch:
        if encoder is None:
            raise RuntimeError("only the batch leader may prepare Wan Taichu eval samples")
        sample = samples[0]
        declared_shape: PreparedEvalShape | None = None
        if source_kind == "cache":
            encoded = canonicalize_encoded_batch(
                collate_encoded_bundles(codec, samples),
                variant=spec,
                streams=streams,
            )
            encoded = truncate_encoded_latents(
                encoded,
                step=0,
                device=device,
                generator=torch.Generator(device=device).manual_seed(
                    request.seed + int(sample.get("__vforge_eval_seed_offset__", 0))
                ),
            )
            minimum = encoded.get("min_latents")
            maximum = encoded.get("max_latents")
            if minimum is not None and minimum == maximum:
                actual = int(encoded["frame_latents"].shape[2])
                if actual != int(minimum):
                    raise ValueError(
                        "Wan Taichu eval cache does not satisfy its fixed latent length: "
                        f"expected {minimum}, got {actual}"
                    )
            context = sample_context(sample)
        else:
            if raw_collate is None:
                raise RuntimeError("only the batch leader may encode raw Wan Taichu eval samples")
            encoder.eval()
            batch = raw_collate(samples)
            context = _prepare_reference_batch(
                batch,
                sample,
                base_seed=request.seed,
                spec=spec,
                extras=extras,
                device=device,
                num_frames_override=request.online_ref_num_frames,
                frame_indices_override=request.online_ref_frame_indices,
            )
            context.update(sample_context(sample, context))
            encoded = encode_inference_batch(
                encoder,
                batch,
                streams=streams,
                stream_fields=stream_fields,
            )
            declared_shape = PreparedEvalShape(
                height=int(batch["height"]),
                width=int(batch["width"]),
                num_frames=int(batch["num_frames"]),
            )

        encoded = dict(encoded)
        shape = resolve_wan_prepared_eval_shape(
            encoded,
            declared=declared_shape,
            vae=encoder.vae,
        )
        encoded = move_prepared_eval_tensors(encoded, device)
        if prepare_encoded is not None:
            encoded = prepare_encoded(
                encoded,
                encoder,
                shape.height,
                shape.width,
                shape.num_frames,
            )
        context["__negative_prompt_embeds__"] = prepare_wan_negative_prompt(
            encoded,
            tokenizer=encoder.tokenizer,
            text_encoder=encoder.text_encoder,
            settings=settings,
            device=device,
        )
        return PreparedEvalBatch(encoded=encoded, shape=shape, context=context)

    dataloader = build_prepared_eval_loader(
        request,
        dataset=dataset,
        parallel_state=state,
        label="Wan Taichu",
        prepare_batch=prepare_batch,
    )

    def build_eval_inference(model: torch.nn.Module, engine_forward: EngineForward):
        return build_wan_flow_match_inference(
            model=model,
            engine_forward=engine_forward,
            pipeline_factory=build_pipeline,
            tokenizer=encoder.tokenizer if encoder is not None else None,
            text_encoder=encoder.text_encoder if encoder is not None else None,
            vae=encoder.vae if encoder is not None else None,
            settings=settings,
            pipeline_kwargs={
                "stream_fields": stream_fields,
                **(
                    {
                        "first_frame_inpaint": spec.first_frame_inpaint,
                        "vae_scale_factor_spatial": (
                            8 if spec.architecture == "wan22_a14b" else 16
                        ),
                    }
                    if uses_bidirectional_pipeline
                    else {}
                ),
                **dict(pipeline_kwargs or {}),
            },
        )

    if isinstance(request.checkpoint_path, str) and request.checkpoint_path:
        initial_weights = TrainingCheckpointWeightSource(
            path=request.checkpoint_path,
            format=request.checkpoint_format,
            mapper=(
                a14b_checkpoint_mapper()
                if spec.architecture == "wan22_a14b"
                else StateDictMapper.strip_prefix("transformer.")
            ),
            compatible_load_callbacks=checkpoint_load_callbacks(spec),
        )
    elif isinstance(request.checkpoint_path, Mapping):
        raise TypeError("Wan Taichu eval expects checkpoint_path to be a string")
    else:
        initial_weights = _pretrained_sources(
            spec=spec,
            model=model,
            layouts=layouts,
            pretrained_path=request.pretrained_path,
            weight_loading="auto",
        )

    return EvalComponents(
        model=model,
        dataloader=dataloader,
        initial_weights=initial_weights,
        build_inference=build_eval_inference,
        output_writer=EvalOutputWriter(
            artifact_name=lambda sample_id: sample_artifact_name(sample_id, suffix=".mp4"),
            write_output=lambda result, output_dir: save_video_artifact(
                result,
                output_dir,
                fps=float(extras.get("fps") or 24.0),
            ),
        ),
    )


__all__ = [
    "build_base_eval_components",
    "resolve_bidirectional_request",
    "resolve_eval_request",
]
