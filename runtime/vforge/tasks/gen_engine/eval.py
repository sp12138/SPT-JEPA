"""Latent-only evaluation for precomputed multimodal generation manifests."""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
import torch.nn as nn

from vforge.data.loader import build_eval_loader
from vforge.distributed.parallel_state import get_parallel_state
from vforge.evaluation.runner import (
    EngineForward,
    EvalComponents,
    EvalOutputWriter,
    sample_artifact_name,
)
from vforge.models.factory import ModelSpec
from vforge.models.gen_engine.packing import RuntimePacker
from vforge.models.gen_engine.preparation import prepare_inference_sample
from vforge.tasks.eval_source import open_eval_dataset
from vforge.tasks.gen_engine.dataset import (
    MMGEN_MME_BUNDLE_CODEC,
    MMGenDataset,
    dataset_input_options,
)
from vforge.tasks.gen_engine.shared import (
    _build_gen_engine_model,
    _gen_engine_state_dict_mapper,
    _validate_shared,
    target_modalities,
)
from vforge.tasks.requests import EvalRequest, validate_eval_source
from vforge.training.checkpoint import TrainingCheckpointWeightSource


def _validate_eval_request(request: EvalRequest) -> str:
    checkpoint = request.checkpoint_path
    if not isinstance(checkpoint, str):
        raise TypeError("gen_engine evaluation.checkpoint_path must be a string")
    checkpoint = checkpoint.strip()
    if not checkpoint:
        raise ValueError("gen_engine evaluation.checkpoint_path is required")
    validate_eval_source(request)
    if request.data_source not in {"inference", "validation"}:
        raise ValueError(
            "gen_engine evaluation.data_source must be inference or validation; "
            "cache/prepare sources are not supported"
        )
    if request.sources:
        raise ValueError(
            "gen_engine evaluation reads a precomputed MMGenDataset manifest; "
            "evaluation.sources is not supported"
        )
    if not request.manifest.strip():
        raise ValueError("gen_engine evaluation.manifest is required")
    return checkpoint


def _open_mmgen_dataset(
    request: EvalRequest,
    *,
    transformer_config: Mapping[str, Any] | None = None,
) -> Any:
    expected_field = "target_spec" if request.data_source == "inference" else "target"

    def factory(manifest: str) -> MMGenDataset:
        return MMGenDataset(
            manifest,
            target_modalities=target_modalities(request.extras),
            require_digests=bool(request.extras.get("require_digests", True)),
            sample_kind=expected_field,
            max_reference_index=(
                None
                if transformer_config is None
                else int(transformer_config.get("max_reference_index", 32))
            ),
            **dataset_input_options(request.extras),
        )

    return open_eval_dataset(
        request,
        codec=MMGEN_MME_BUNDLE_CODEC,
        manifest_dataset_factory=factory,
    )


def _collate_mmgen_samples(
    samples: list[dict[str, Any]],
    *,
    packer: RuntimePacker,
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    if len(samples) != 1:
        raise ValueError(f"gen_engine evaluation requires one sample per batch, got {len(samples)}")
    return prepare_inference_sample(samples[0], packer=packer, generation_config=generation_config)


def _cpu_latent(result: Mapping[str, Any], key: str) -> torch.Tensor | None:
    value = result.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"gen_engine eval result {key!r} must be a tensor")
    return value.detach().cpu()


def write_latent_output(result: dict[str, Any], output_dir: str) -> str:
    """Atomically save generated latents and their geometry as one ``.pt``."""

    os.makedirs(output_dir, exist_ok=True)
    sample_id = str(result.get("sample_id") or "sample")
    path = os.path.join(output_dir, sample_artifact_name(sample_id, suffix=".pt"))
    video = _cpu_latent(result, "video_latent")
    audio = _cpu_latent(result, "audio_latent")
    if video is None and audio is None:
        raise ValueError("gen_engine eval result must contain a video or audio latent")
    geometry = result.get("geometry")
    metadata = result.get("metadata") or {}
    if not isinstance(geometry, Mapping):
        raise TypeError("gen_engine eval result geometry must be a mapping")
    if not isinstance(metadata, Mapping):
        raise TypeError("gen_engine eval result metadata must be a mapping")
    payload: dict[str, Any] = {
        "sample_id": sample_id,
        "seed": int(result.get("seed", 0)),
        "geometry": dict(geometry),
        "metadata": dict(metadata),
    }
    if video is not None:
        payload["video_latent"] = video
    if audio is not None:
        payload["audio_latent"] = audio

    temporary = path + ".tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return path


def build_output_writer() -> EvalOutputWriter:
    return EvalOutputWriter(
        artifact_name=lambda sample_id: sample_artifact_name(sample_id, suffix=".pt"),
        write_output=write_latent_output,
    )


def build_eval_components(request: EvalRequest) -> EvalComponents:
    state = get_parallel_state()
    if state.tp_enabled or state.cp_enabled:
        raise ValueError(
            "gen_engine evaluation requires parallel.tp_size=1 and parallel.cp_size=1; "
            "tensor/context parallelism is not implemented"
        )
    checkpoint = _validate_eval_request(request)
    config = _validate_shared(
        extras=request.extras,
        pretrained_path=request.pretrained_path,
        init_device=request.init_device,
        latent_cache_mode=request.latent_cache_mode,
        batch_size=1,
    )
    transformer_config = config.transformer
    latent_codec = config.codec
    patch_size = config.patch_size
    generation_config = config.transformer
    video_width, audio_width = config.row_widths
    packer = RuntimePacker(
        patch_size,
        text_width=config.text_width,
        visual_row_width=video_width,
        audio_row_width=audio_width,
        latent_codec=latent_codec,
    )
    dataset = _open_mmgen_dataset(
        request,
        transformer_config=transformer_config,
    )
    dataloader = build_eval_loader(
        dataset,
        parallel_state=state,
        collate_fn=partial(
            _collate_mmgen_samples,
            packer=packer,
            generation_config=generation_config,
        ),
        num_samples=request.num_samples,
        sample_selection=request.sample_selection,
    )
    model = ModelSpec(
        lambda: _build_gen_engine_model(
            extras=request.extras,
            init_device=request.init_device,
            pretrained_path=request.pretrained_path,
            config=config,
        )
    )

    def build_inference(live_model: nn.Module, engine_forward: EngineForward):
        from vforge.models.gen_engine.inference import GenEngineInference

        return GenEngineInference(
            model=live_model,
            engine_forward=engine_forward,
            generation_config=generation_config,
            num_inference_steps=(
                30 if request.num_inference_steps is None else int(request.num_inference_steps)
            ),
            video_shift=config.extras["video_flow_shift"],
            audio_shift=config.extras["audio_flow_shift"],
            patch_size=patch_size,
            latent_codec=latent_codec,
            reference_timestep=config.extras["reference_timestep"],
            text_timestep=config.extras["text_timestep"],
        )

    initial_weights = TrainingCheckpointWeightSource(
        path=checkpoint,
        format=request.checkpoint_format,
        mapper=_gen_engine_state_dict_mapper(),
    )
    return EvalComponents(
        model=model,
        dataloader=dataloader,
        initial_weights=initial_weights,
        build_inference=build_inference,
        output_writer=build_output_writer(),
    )


__all__ = [
    "build_eval_components",
    "build_output_writer",
    "write_latent_output",
]
