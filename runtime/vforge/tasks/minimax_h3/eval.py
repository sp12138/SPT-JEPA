"""MiniMax-H3 FL2VA fixed-seed evaluation."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
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
from vforge.tasks.eval_source import open_eval_dataset
from vforge.tasks.minimax_h3.shared import (
    H3_ENCODED_BATCH_CODEC,
    _build_h3_model,
    _collate_encoded_bundles,
    _h3_state_dict_mapper,
    _torch_dtype,
    _validate_shared,
)
from vforge.tasks.minimax_h3.train import _h3_pretrained_source
from vforge.tasks.requests import EvalRequest
from vforge.training.checkpoint import TrainingCheckpointWeightSource


def build_eval_components(request: EvalRequest) -> EvalComponents:
    _validate_shared(
        extras=request.extras,
        latent_cache_mode=request.latent_cache_mode,
        usp_size=request.usp_size,
    )
    dataset = open_eval_dataset(request, codec=H3_ENCODED_BATCH_CODEC)
    source_kind = request.data_source
    model = _build_h3_model(
        extras=request.extras,
        init_device=request.init_device,
        pretrained_path=request.pretrained_path,
    )
    _prepare_model_for_eval(model)

    from vforge.distributed.runtime import get_runtime

    runtime = get_runtime()
    if runtime is None:
        raise RuntimeError("H3 evaluation requires an initialized runtime")
    state = get_parallel_state()
    vae = None
    audio_vae = None
    encoder = None
    if state.is_batch_leader:
        from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

        vae = (
            AutoencoderKLMiniMaxH3.from_pretrained(
                request.pretrained_path,
                subfolder=str(request.extras.get("vae_subfolder", "vae")),
                torch_dtype=torch.float32,
            )
            .eval()
            .to(runtime.device)
        )
        audio_vae = (
            AutoencoderKLMiniMaxH3Audio.from_pretrained(
                request.pretrained_path,
                subfolder=str(request.extras.get("audio_vae_subfolder", "audio_vae")),
                torch_dtype=torch.float32,
            )
            .eval()
            .to(runtime.device)
        )
        if source_kind != "cache":
            from vforge.models.minimax_h3.encoder import MiniMaxH3ConditionEncoder

            encoder = MiniMaxH3ConditionEncoder.from_pretrained(
                request.pretrained_path,
                extras=request.extras,
                device=runtime.device,
                dtype=_torch_dtype(request.precision),
                vae=vae,
                audio_vae=audio_vae,
            )

    def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
        if len(samples) != 1:
            raise ValueError(f"H3 evaluation requires batch_size=1, got {len(samples)}")
        sample = samples[0]
        if source_kind == "cache":
            encoded = _collate_encoded_bundles(samples)
        else:
            if encoder is None:
                raise RuntimeError("only the batch leader may encode H3 eval samples")
            encoded = encoder.encode_inference_sample(
                sample,
                allow_video_fallback=source_kind == "validation",
            )
        encoded["sample_id"] = str(sample.get("sample_id") or "sample")
        encoded["__vforge_eval_padding__"] = bool(sample.get("__vforge_eval_padding__", False))
        return encoded

    dataloader = build_eval_loader(
        dataset,
        parallel_state=state,
        collate_fn=collate,
        num_samples=request.num_samples,
        sample_selection=request.sample_selection,
    )

    def build_inference(live_model: nn.Module, engine_forward: EngineForward):
        from vforge.models.minimax_h3.inference import MiniMaxH3Inference

        return MiniMaxH3Inference(
            model=live_model,
            engine_forward=engine_forward,
            vae=vae,
            audio_vae=audio_vae,
            num_inference_steps=(
                30 if request.num_inference_steps is None else int(request.num_inference_steps)
            ),
            video_shift=float(request.extras.get("video_flow_shift", 12.0)),
            audio_shift=float(request.extras.get("audio_flow_shift", 3.0)),
            keyframe_noise_aug=float(request.extras.get("keyframe_noise_aug", 0.999)),
            pixel_mean=tuple(request.extras.get("pixel_mean", (0.485, 0.456, 0.406))),
            pixel_std=tuple(request.extras.get("pixel_std", (0.229, 0.224, 0.225))),
            audio_sample_rate=int(request.extras.get("audio_sample_rate", 32000)),
            patch_size=tuple(request.extras.get("patch_size", (1, 2, 2))),
            trace_path=request.extras.get("trace_path"),
            fixed_seed=(
                int(request.extras["inference_seed"])
                if request.extras.get("inference_seed") is not None
                else None
            ),
        )

    if request.checkpoint_path:
        if not isinstance(request.checkpoint_path, str):
            raise TypeError("MiniMax H3 evaluation expects checkpoint_path to be a string")
        initial_weights = TrainingCheckpointWeightSource(
            path=request.checkpoint_path,
            format=request.checkpoint_format,
            mapper=_h3_state_dict_mapper(),
        )
    else:
        initial_weights = _h3_pretrained_source(request.pretrained_path, "auto")

    def write_output(result: dict[str, Any], output_dir: str) -> str:
        from diffusers.utils.export_utils import encode_video

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(
            output_dir,
            _artifact_name(str(result.get("sample_id") or "sample")),
        )
        temporary = path + ".tmp.mp4"
        encode_video(
            result["frames"],
            fps=24,
            output_path=temporary,
            audio=result["audio"],
            audio_sample_rate=int(result["sampling_rate"]),
        )
        os.replace(temporary, path)
        if bool(request.extras.get("save_tensor_artifacts", False)):
            tensor_path = os.path.splitext(path)[0] + ".pt"
            temporary_tensor = tensor_path + ".tmp"
            torch.save(
                {
                    "video": torch.from_numpy(
                        np.stack([np.asarray(frame) for frame in result["frames"]])
                    ),
                    "audio": result["audio"].detach().cpu(),
                    "sampling_rate": int(result["sampling_rate"]),
                },
                temporary_tensor,
            )
            os.replace(temporary_tensor, tensor_path)
        return path

    return EvalComponents(
        model=model,
        dataloader=dataloader,
        initial_weights=initial_weights,
        build_inference=build_inference,
        output_writer=EvalOutputWriter(
            artifact_name=_artifact_name,
            write_output=write_output,
        ),
    )


def _artifact_name(sample_id: str) -> str:
    return sample_artifact_name(sample_id, suffix=".mp4")


def _prepare_model_for_eval(model: nn.Module) -> None:
    """Freeze mixed-dtype H3 parameters before FSDP inference setup."""
    model.requires_grad_(False)
    model.eval()


__all__ = ["build_eval_components"]
