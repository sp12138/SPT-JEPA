"""MiniMax-H3 FL2VA dual-flow training recipe."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from vforge.data.cache import read_cache_manifest
from vforge.data.dataset import EncodedCacheDataset, RawSampleDataset
from vforge.data.loader import build_resumable_sp_loader
from vforge.distributed.parallel_state import get_parallel_state
from vforge.models.loading import PretrainedWeightSource
from vforge.tasks.minimax_h3.shared import (
    H3_ENCODED_BATCH_CODEC,
    _build_h3_model,
    _collate_encoded_bundles,
    _h3_state_dict_mapper,
    _torch_dtype,
    _validate_shared,
)
from vforge.tasks.requests import TrainRequest, single_offline_source
from vforge.training.recipe import StepContext, StepOutput, TrainComponents


def shifted_sigma(u: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply H3's exponential flow shift to a shared base sample."""
    if shift <= 0:
        raise ValueError(f"flow shift must be positive, got {shift}")
    return shift * u / (1.0 + (shift - 1.0) * u)


def flow_noisy_and_target(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """H3 forward process and its data-ward velocity target."""
    timestep = 1.0 - sigma
    noisy = timestep.to(clean.dtype) * clean + sigma.to(clean.dtype) * noise
    return noisy, clean - noise


def build_train_components(request: TrainRequest) -> TrainComponents:
    _validate_shared(
        extras=request.extras,
        latent_cache_mode=request.latent_cache_mode,
        usp_size=request.usp_size,
    )
    if request.batch_size != 1:
        raise ValueError("MiniMax-H3 packed layout currently requires data.batch_size=1")
    model = _build_h3_model(
        extras=request.extras,
        init_device=request.init_device,
        pretrained_path=request.pretrained_path,
    )
    dataloader, resume_contract = _build_train_dataloader(request)

    def step_fn(
        live_model: nn.Module,
        batch: dict[str, Any],
        step: int,
        context: StepContext,
    ) -> StepOutput:
        return _h3_train_step(
            live_model,
            batch,
            step,
            context,
            seed=request.seed,
            extras=request.extras,
        )

    return TrainComponents(
        model=model,
        dataloader=dataloader,
        optimizer_factory=_optimizer_factory(
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            betas=request.betas,
            eps=request.eps,
        ),
        step_fn=step_fn,
        scheduler_factory=_scheduler_factory(
            name=request.scheduler_name,
            factor=request.scheduler_factor,
            warmup_steps=request.warmup_steps,
            start_factor=request.scheduler_start_factor,
            end_factor=request.scheduler_end_factor,
            total_iters=request.scheduler_total_iters,
        ),
        pretrained_weights=_h3_pretrained_source(
            request.pretrained_path,
            request.weight_loading,
        ),
        state_dict_mapper=_h3_state_dict_mapper(),
        dataloader_resume_contract=resume_contract,
    )


def _build_train_dataloader(request: TrainRequest):
    state = get_parallel_state()
    source = single_offline_source(request, recipe="MiniMax-H3")
    if request.latent_cache_mode == "offline":
        cache_dir = source.cache_dir if source is not None else ""
        manifest = (
            read_cache_manifest(cache_dir, codec=H3_ENCODED_BATCH_CODEC) if cache_dir else None
        )
        if manifest is None:
            if request.max_steps > 0:
                raise ValueError("H3 offline training requires a prepared cache manifest")
            dataset = EncodedCacheDataset(
                cache_dir or ".",
                [],
                H3_ENCODED_BATCH_CODEC,
            )
            fingerprint = {"source_kind": "encoded_cache", "empty": True}
        else:
            dataset = EncodedCacheDataset(
                cache_dir,
                manifest.sample_ids,
                H3_ENCODED_BATCH_CODEC,
            )
            fingerprint = {
                "source_kind": "encoded_cache",
                "sample_ids_digest": manifest.sample_ids_digest,
                "encoding_fingerprint": manifest.encoding_fingerprint,
                "codec_name": manifest.codec_name,
                "codec_schema_version": manifest.codec_schema_version,
            }
        collate_fn = _collate_encoded_bundles
    else:
        if request.num_workers:
            raise ValueError("H3 online encoding currently requires data.num_workers=0")
        if not request.train_manifest:
            if request.max_steps > 0:
                raise ValueError("H3 online training requires data.train_manifest")
            dataset = RawSampleDataset([])

            def collate_fn(samples):
                del samples
                return {}

            fingerprint = {"source_kind": "raw_video", "empty": True}
        else:
            dataset = RawSampleDataset.from_manifest(request.train_manifest)
            encoder = None
            if state.config.usp_size <= 1 or state.usp_rank == 0:
                from vforge.models.minimax_h3.encoder import MiniMaxH3ConditionEncoder

                runtime_device = (
                    torch.device("cuda", torch.cuda.current_device())
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
                encoder = MiniMaxH3ConditionEncoder.from_pretrained(
                    request.pretrained_path,
                    extras=request.extras,
                    device=runtime_device,
                    dtype=_torch_dtype(request.precision),
                )

            def collate_fn(samples):
                if encoder is None:
                    raise RuntimeError("only the SP leader may encode H3 online samples")
                return _collate_encoded_bundles(
                    [
                        {
                            **H3_ENCODED_BATCH_CODEC.canonicalize(encoder.encode_sample(sample)),
                            "sample_id": sample["sample_id"],
                        }
                        for sample in samples
                    ]
                )

            fingerprint = {
                "source_kind": "raw_video",
                "manifest": request.train_manifest,
            }

    loader = build_resumable_sp_loader(
        dataset,
        parallel_state=state,
        batch_size=request.batch_size,
        num_workers=request.num_workers,
        drop_last=request.drop_last,
        seed=request.sampler_seed,
        collate_fn=collate_fn,
        pin_memory=request.pin_memory,
        persistent_workers=request.persistent_workers,
        prefetch_factor=request.prefetch_factor,
    )
    return loader, fingerprint


def _h3_train_step(
    model: nn.Module,
    batch: dict[str, Any],
    step: int,
    context: StepContext,
    *,
    seed: int,
    extras: Mapping[str, Any],
) -> StepOutput:
    device = context.device
    dtype = next(parameter for parameter in model.parameters() if parameter.requires_grad).dtype
    generator = context.random_streams.generator("h3.input", seed=seed, device=device)

    video_clean = batch["target_video_rows"].to(device=device, dtype=dtype)
    audio_clean = batch["target_audio_rows"].to(device=device, dtype=dtype)
    condition_video = batch["condition_video_rows"].to(
        device=device,
        dtype=torch.float32,
    )
    prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=dtype)
    audio_loss_mask = batch["audio_loss_mask"].to(device=device, dtype=torch.float32)
    structural = {
        key: batch[key].to(device)
        for key in (
            "token_tags",
            "position_ids",
            "video_indices",
            "audio_indices",
            "text_indices",
        )
    }
    num_condition_video = int(batch["num_condition_video_rows"])
    num_condition_audio = int(batch["num_condition_audio_rows"])

    condition_t = float(extras.get("keyframe_noise_aug", 0.999))
    if condition_video.numel():
        condition_noise = torch.randn(
            condition_video.shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        condition_video = condition_t * condition_video + (1.0 - condition_t) * condition_noise
    u = torch.rand((), generator=generator, device=device, dtype=torch.float32)
    sigma_video = shifted_sigma(u, float(extras.get("video_flow_shift", 12.0)))
    sigma_audio = shifted_sigma(u, float(extras.get("audio_flow_shift", 3.0)))
    t_video = 1.0 - sigma_video
    t_audio = 1.0 - sigma_audio

    eps_video = torch.randn(
        video_clean.shape,
        generator=generator,
        device=device,
        dtype=video_clean.dtype,
    )
    eps_audio = torch.randn(
        audio_clean.shape,
        generator=generator,
        device=device,
        dtype=audio_clean.dtype,
    )
    noisy_video, target_video = flow_noisy_and_target(
        video_clean,
        eps_video,
        sigma_video,
    )
    noisy_audio, target_audio = flow_noisy_and_target(
        audio_clean,
        eps_audio,
        sigma_audio,
    )

    model_video = torch.cat([condition_video, noisy_video], dim=1)

    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3SetTimestepsStep,
    )

    timesteps, timestep_indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        structural["video_indices"],
        structural["audio_indices"],
        num_condition_video,
        num_condition_audio,
        int(structural["text_indices"].numel()),
        float(t_video),
        float(t_audio),
        max(float(t_video), condition_t),
        1.0,
    )
    output = model(
        hidden_states=model_video,
        audio_hidden_states=noisy_audio,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps.to(device),
        timestep_indices=timestep_indices.to(device),
        **structural,
    )
    pred_video = output.sample[:, num_condition_video:]
    pred_audio = output.audio_sample[:, num_condition_audio:]
    video_loss = (pred_video.float() - target_video.float()).square().mean()
    audio_per_sample = (pred_audio.float() - target_audio.float()).square().flatten(1).mean(dim=1)
    audio_loss = (audio_per_sample * audio_loss_mask.flatten()).mean()
    loss = video_loss + float(extras.get("audio_loss_weight", 1.0)) * audio_loss
    return StepOutput(
        loss=loss,
        metrics={
            "video_loss": video_loss.detach(),
            "audio_loss": audio_loss.detach(),
            "sigma_video": sigma_video.detach(),
            "sigma_audio": sigma_audio.detach(),
        },
    )


def _optimizer_factory(
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
):
    def build(model: nn.Module):
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            return None
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    return build


def _scheduler_factory(
    *,
    name: str,
    factor: float = 1.0,
    warmup_steps: int,
    start_factor: float = 1.0,
    end_factor: float = 0.1,
    total_iters: int = 0,
):
    if name == "none":
        return None

    def build(optimizer):
        if name == "constant":
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
        if name == "constant_with_warmup":
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: min(1.0, float(step + 1) / max(1, warmup_steps)),
            )
        if name == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=end_factor,
                total_iters=total_iters,
            )
        raise ValueError(f"unsupported scheduler {name!r}")

    return build


def _h3_pretrained_source(path: str, loading_mode: str) -> PretrainedWeightSource:
    def fallback_factory():
        from diffusers import MiniMaxH3Transformer3DModel

        return MiniMaxH3Transformer3DModel.from_pretrained(
            path,
            subfolder="transformer",
            torch_dtype=torch.float32,
        )

    return PretrainedWeightSource(
        path=os.path.join(path, "transformer"),
        subfolder="",
        model_prefix="transformer.",
        loading_mode=loading_mode,
        fallback_model_factory=fallback_factory,
        stage_to_local=True,
    )


__all__ = [
    "build_train_components",
    "flow_noisy_and_target",
    "shifted_sigma",
]
