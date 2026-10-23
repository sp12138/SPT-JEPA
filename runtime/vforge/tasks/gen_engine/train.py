"""Scratch training recipe for precomputed multimodal generation samples."""

from __future__ import annotations

import math
from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
import torch.nn as nn

from vforge.data.loader import build_resumable_loader
from vforge.distributed.parallel_state import get_parallel_state
from vforge.models.factory import ModelSpec
from vforge.models.gen_engine.packing import RuntimePacker
from vforge.models.gen_engine.timesteps import build_row_timesteps
from vforge.tasks.gen_engine.data import build_data_spec
from vforge.tasks.gen_engine.shared import (
    LAYOUT_VERSION,
    PACKING_VERSION,
    GenEngineConfig,
    _build_gen_engine_model,
    _gen_engine_state_dict_mapper,
    _initialize_gen_engine_model,
    _model_dimensions,
    _validate_shared,
    validate_execution,
)
from vforge.tasks.requests import TrainRequest
from vforge.training.recipe import (
    OptimizerSpec,
    SchedulerSpec,
    StepContext,
    StepOutput,
    TrainComponents,
)
from vforge.utils.training import build_optimizer_factory, build_scheduler_factory


def shifted_sigma(u: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the exponential flow shift to a shared base sample."""
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError(f"flow shift must be positive, got {shift}")
    return shift * u / (1.0 + (shift - 1.0) * u)


def flow_noisy_and_target(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a flow-matching input and its data-ward velocity target."""
    timestep = 1.0 - sigma
    noisy = timestep.to(clean.dtype) * clean + sigma.to(clean.dtype) * noise
    return noisy, clean - noise


def build_train_components(request: TrainRequest) -> TrainComponents:
    validate_execution(request, "fsdp2")
    config = _validate_shared(
        extras=request.extras,
        pretrained_path=request.pretrained_path,
        init_device=request.init_device,
        latent_cache_mode=request.latent_cache_mode,
        batch_size=request.batch_size,
    )
    model = ModelSpec(
        lambda: _build_gen_engine_model(
            extras=request.extras,
            init_device=request.init_device,
            pretrained_path=request.pretrained_path,
            config=config,
        )
    )
    dataloader, resume_contract = _build_train_dataloader(request, config=config)

    def step_fn(
        live_model: nn.Module,
        batch: dict[str, Any],
        step: int,
        context: StepContext,
    ) -> StepOutput:
        return _gen_engine_train_step(
            live_model,
            batch,
            step,
            context,
            seed=request.seed,
            extras=config.extras,
        )

    optimizer_spec = OptimizerSpec(
        learning_rate=request.learning_rate,
        weight_decay=request.weight_decay,
        betas=request.betas,
        eps=request.eps,
    )
    scheduler_spec = SchedulerSpec(
        name=request.scheduler_name,
        factor=request.scheduler_factor,
        warmup_steps=request.warmup_steps,
        start_factor=request.scheduler_start_factor,
        end_factor=request.scheduler_end_factor,
        total_iters=request.scheduler_total_iters,
    )
    return TrainComponents(
        model=model,
        dataloader=dataloader,
        optimizer_factory=build_optimizer_factory(
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            betas=request.betas,
            eps=request.eps,
        ),
        step_fn=step_fn,
        scheduler_factory=build_scheduler_factory(
            name=request.scheduler_name,
            factor=request.scheduler_factor,
            warmup_steps=request.warmup_steps,
            start_factor=request.scheduler_start_factor,
            end_factor=request.scheduler_end_factor,
            total_iters=request.scheduler_total_iters,
        ),
        optimizer_spec=optimizer_spec,
        scheduler_spec=scheduler_spec,
        pretrained_weights=None,
        state_dict_mapper=_gen_engine_state_dict_mapper(),
        dataloader_resume_contract=resume_contract,
        model_initializer=partial(_initialize_gen_engine_model, seed=request.seed),
        post_optimizer_step=partial(
            _gen_engine_post_optimizer_step,
            total_steps=request.max_steps,
        ),
    )


def _gen_engine_post_optimizer_step(
    model: nn.Module,
    step: int,
    context: StepContext,
    output: StepOutput,
    *,
    total_steps: int,
) -> Mapping[str, torch.Tensor | float | int]:
    del output
    candidate = model
    callback = getattr(candidate, "post_optimizer_step", None)
    if not callable(callback):
        candidate = getattr(model, "module", model)
        callback = getattr(candidate, "post_optimizer_step", None)
    if not callable(callback):
        raise TypeError("gen_engine model must define post_optimizer_step()")
    return callback(
        completed_step=int(step) + 1,
        total_steps=int(total_steps),
        process_group=context.parallel_state.fsdp_group,
    )


def _build_train_dataloader(
    request: TrainRequest,
    *,
    config: GenEngineConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    if config is None:
        config = _validate_shared(
            extras=request.extras,
            pretrained_path=request.pretrained_path,
            init_device=request.init_device,
            latent_cache_mode=request.latent_cache_mode,
            batch_size=request.batch_size,
        )
    transformer_config = config.transformer
    latent_codec = config.codec
    patch_size = config.patch_size
    video_row_width, audio_row_width = config.row_widths
    text_width = config.text_width
    state = get_parallel_state()
    collate_fn = partial(
        _collate_runtime_samples,
        packer=RuntimePacker(
            patch_size,
            text_width=text_width,
            visual_row_width=video_row_width,
            audio_row_width=audio_row_width,
            latent_codec=latent_codec,
        ),
        video_row_width=video_row_width,
        audio_row_width=audio_row_width,
    )
    total_samples = None
    if request.sources:
        if type(request.max_steps) is not int or request.max_steps < 1:
            raise ValueError("gen_engine multi-source training requires runtime.max_steps > 0")
        # The current training loop uses one local batch per optimizer step.
        # Fix the global budget before sampling so the final block can be corrected.
        total_samples = request.max_steps * request.batch_size * int(state.config.dp_size)
    data_spec, data_fingerprint = build_data_spec(
        request,
        collate_fn=collate_fn,
        max_reference_index=int(transformer_config.get("max_reference_index", 32)),
        total_samples=total_samples,
    )
    loader = build_resumable_loader(
        data_spec,
        parallel_state=state,
        batch_size=request.batch_size,
        num_workers=request.num_workers,
        drop_last=request.drop_last,
        seed=request.sampler_seed,
        shuffle=request.shuffle,
        pin_memory=request.pin_memory,
        persistent_workers=request.persistent_workers,
        prefetch_factor=request.prefetch_factor,
    )
    fingerprint = {
        **data_fingerprint,
        "packing": {
            "version": PACKING_VERSION,
            "patch_size": list(patch_size),
        },
        "layout": {"version": LAYOUT_VERSION},
        "latent_codecs": {
            "video": {
                "digest": config.video_codec_digest,
                "config": latent_codec.video_config_dict(),
            },
            "audio": {
                "digest": config.audio_codec_digest,
                "config": latent_codec.audio_config_dict(),
            },
        },
        "transformer_config_digest": config.transformer_digest,
        "model_dimensions": _model_dimensions(request.extras, transformer_config),
        "loader": {
            "batch_size": request.batch_size,
            "num_workers": request.num_workers,
            "shuffle": request.shuffle,
            "drop_last": request.drop_last,
            "base_seed": request.base_seed,
            "sampler_seed": request.sampler_seed,
            "pin_memory": request.pin_memory,
            "persistent_workers": request.persistent_workers,
            "prefetch_factor": request.prefetch_factor,
        },
    }
    return loader, fingerprint


def _normalize_empty_rows(
    rows: torch.Tensor,
    *,
    width: int,
    label: str,
) -> torch.Tensor:
    if rows.ndim != 2:
        raise ValueError(f"{label} must be rank 2, got {tuple(rows.shape)}")
    if rows.shape[0] == 0 and rows.shape[1] == 0:
        return rows.new_empty((0, width))
    if rows.shape[1] != width:
        raise ValueError(f"{label} width {rows.shape[1]} does not match transformer width {width}")
    return rows


def _collate_runtime_samples(
    samples: list[dict[str, Any]],
    *,
    packer: RuntimePacker,
    video_row_width: int,
    audio_row_width: int,
) -> dict[str, Any]:
    if len(samples) != 1:
        raise ValueError(f"gen_engine requires one packed sample per batch, got {len(samples)}")
    packed = packer(samples[0])
    if packed["is_inference"]:
        raise ValueError("gen_engine training requires target latents, not target_spec")

    for key in ("reference_video_rows", "target_video_rows"):
        packed[key] = _normalize_empty_rows(
            packed[key],
            width=video_row_width,
            label=key,
        )
    for key in ("reference_audio_rows", "target_audio_rows"):
        packed[key] = _normalize_empty_rows(
            packed[key],
            width=audio_row_width,
            label=key,
        )

    return {
        "sample_id": [str(samples[0]["sample_id"])],
        "mme_embedding": packed["mme_embedding"].unsqueeze(0),
        "reference_video_rows": packed["reference_video_rows"].unsqueeze(0),
        "target_video_rows": packed["target_video_rows"].unsqueeze(0),
        "reference_audio_rows": packed["reference_audio_rows"].unsqueeze(0),
        "target_audio_rows": packed["target_audio_rows"].unsqueeze(0),
        "audio_plane_ids": packed["audio_plane_ids"],
        "token_tags": packed["token_tags"],
        "position_ids": packed["position_ids"],
        "video_indices": packed["video_indices"],
        "audio_indices": packed["audio_indices"],
        "text_indices": packed["text_indices"],
        "reference_modality_ids": packed["reference_modality_ids"],
        "reference_index_ids": packed["reference_index_ids"],
        "video_loss_mask": packed["video_loss_mask"],
        "audio_loss_mask": packed["audio_loss_mask"],
        "num_reference_video_rows": packed["num_reference_video_rows"],
        "num_reference_audio_rows": packed["num_reference_audio_rows"],
        "num_target_video_rows": packed["num_target_video_rows"],
        "num_target_audio_rows": packed["num_target_audio_rows"],
    }


def _gen_engine_train_step(
    model: nn.Module,
    batch: dict[str, Any],
    step: int,
    context: StepContext,
    *,
    seed: int,
    extras: Mapping[str, Any],
) -> StepOutput:
    device = context.device
    parameter = next(
        (candidate for candidate in model.parameters() if candidate.requires_grad),
        None,
    )
    dtype = parameter.dtype if parameter is not None else torch.float32
    generator = context.random_streams.generator("gen_engine.input", seed=seed, device=device)

    mme_embedding = batch["mme_embedding"].to(device=device, dtype=dtype)
    reference_video = batch["reference_video_rows"].to(device=device, dtype=dtype)
    reference_audio = batch["reference_audio_rows"].to(device=device, dtype=dtype)
    target_video_clean = batch["target_video_rows"].to(device=device, dtype=dtype)
    target_audio_clean = batch["target_audio_rows"].to(device=device, dtype=dtype)
    structural = {
        key: batch[key].to(device)
        for key in (
            "token_tags",
            "position_ids",
            "video_indices",
            "audio_indices",
            "text_indices",
            "reference_modality_ids",
            "reference_index_ids",
            "audio_plane_ids",
        )
    }

    reference_timestep = float(extras.get("reference_timestep", 1.0))
    text_timestep = float(extras.get("text_timestep", 1.0))
    if not 0.0 <= reference_timestep <= 1.0:
        raise ValueError("model.extras.reference_timestep must be in [0, 1]")
    if not 0.0 <= text_timestep <= 1.0:
        raise ValueError("model.extras.text_timestep must be in [0, 1]")
    reference_video = _noise_reference_rows(
        reference_video,
        timestep=reference_timestep,
        generator=generator,
    )
    reference_audio = _noise_reference_rows(
        reference_audio,
        timestep=reference_timestep,
        generator=generator,
    )

    u = torch.rand((), generator=generator, device=device, dtype=torch.float32)
    sigma_video = shifted_sigma(u, float(extras.get("video_flow_shift", 12.0)))
    sigma_audio = shifted_sigma(u, float(extras.get("audio_flow_shift", 3.0)))
    video_timestep = 1.0 - sigma_video
    audio_timestep = 1.0 - sigma_audio

    target_video_noisy, target_video = _noisy_target(
        target_video_clean,
        sigma=sigma_video,
        generator=generator,
    )
    target_audio_noisy, target_audio = _noisy_target(
        target_audio_clean,
        sigma=sigma_audio,
        generator=generator,
    )
    video_hidden_states = torch.cat((reference_video, target_video_noisy), dim=1)
    audio_hidden_states = torch.cat((reference_audio, target_audio_noisy), dim=1)

    num_reference_video = int(batch["num_reference_video_rows"])
    num_reference_audio = int(batch["num_reference_audio_rows"])
    timesteps, timestep_indices = build_row_timesteps(
        video_indices=structural["video_indices"],
        audio_indices=structural["audio_indices"],
        text_indices=structural["text_indices"],
        num_reference_video_rows=num_reference_video,
        num_reference_audio_rows=num_reference_audio,
        video_timestep=float(video_timestep),
        audio_timestep=float(audio_timestep),
        reference_timestep=reference_timestep,
        text_timestep=text_timestep,
    )
    output = model(
        hidden_states=video_hidden_states,
        audio_hidden_states=audio_hidden_states,
        encoder_hidden_states=mme_embedding,
        timestep=timesteps,
        timestep_indices=timestep_indices,
        **structural,
    )
    video_output, audio_output = _stream_outputs(output)
    predicted_video = _target_rows(
        video_output,
        reference_rows=num_reference_video,
        target_rows=int(batch["num_target_video_rows"]),
        label="video",
    )
    predicted_audio = _target_rows(
        audio_output,
        reference_rows=num_reference_audio,
        target_rows=int(batch["num_target_audio_rows"]),
        label="audio",
    )
    video_loss = _stream_loss(predicted_video, target_video)
    audio_loss = _stream_loss(predicted_audio, target_audio)
    video_weight = float(extras.get("video_loss_weight", 1.0))
    audio_weight = float(extras.get("audio_loss_weight", 1.0))
    if (
        not math.isfinite(video_weight)
        or not math.isfinite(audio_weight)
        or video_weight < 0
        or audio_weight < 0
    ):
        raise ValueError("video_loss_weight and audio_loss_weight must be finite and non-negative")
    loss = video_weight * video_loss + audio_weight * audio_loss
    return StepOutput(
        loss=loss,
        metrics={
            "video_loss": video_loss.detach(),
            "audio_loss": audio_loss.detach(),
            "video_output_rms": _tensor_rms(predicted_video),
            "audio_output_rms": _tensor_rms(predicted_audio),
            "sigma_video": sigma_video.detach(),
            "sigma_audio": sigma_audio.detach(),
        },
        workload={
            "local_tokens": int(structural["position_ids"].shape[0]),
            "local_batch": 1,
        },
        metadata={"sample_ids": list(batch.get("sample_id", ()))},
    )


def _noise_reference_rows(
    rows: torch.Tensor,
    *,
    timestep: float,
    generator: torch.Generator,
) -> torch.Tensor:
    if timestep == 1.0 or rows.numel() == 0:
        return rows
    noise = torch.randn(
        rows.shape,
        generator=generator,
        device=rows.device,
        dtype=rows.dtype,
    )
    return timestep * rows + (1.0 - timestep) * noise


def _noisy_target(
    clean: torch.Tensor,
    *,
    sigma: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.randn(
        clean.shape,
        generator=generator,
        device=clean.device,
        dtype=clean.dtype,
    )
    return flow_noisy_and_target(clean, noise, sigma)


def _stream_outputs(output: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, tuple) and len(output) == 2:
        video, audio = output
    else:
        video = getattr(output, "sample", None)
        audio = getattr(output, "audio_sample", None)
    if not isinstance(video, torch.Tensor) or not isinstance(audio, torch.Tensor):
        raise TypeError("gen_engine model must return sample and audio_sample tensors")
    return video, audio


def _target_rows(
    output: torch.Tensor,
    *,
    reference_rows: int,
    target_rows: int,
    label: str,
) -> torch.Tensor:
    expected = reference_rows + target_rows
    if output.ndim != 3 or output.shape[1] != expected:
        raise ValueError(
            f"model {label} output must contain {expected} packed rows, got {tuple(output.shape)}"
        )
    return output[:, reference_rows:]


def _stream_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match "
            f"target shape {tuple(target.shape)}"
        )
    if target.numel() == 0:
        return prediction.float().sum() * 0.0
    return (prediction.float() - target.float()).square().mean()


def _tensor_rms(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return value.new_zeros((), dtype=torch.float32)
    return value.detach().float().square().mean().sqrt()


__all__ = [
    "build_row_timesteps",
    "build_train_components",
    "flow_noisy_and_target",
    "shifted_sigma",
]
