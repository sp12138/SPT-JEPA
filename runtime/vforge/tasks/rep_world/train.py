"""World-DiT training components; Native owns the complete training lifecycle."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch

from vforge.data.loader import build_resumable_loader
from vforge.distributed.parallel_state import get_parallel_state
from vforge.tasks.rep_world.data import VideoClipDataset
from vforge.tasks.rep_world.ema import initialize_ema, initialize_tuning_from_ema, update_ema
from vforge.tasks.rep_world.latent_cache import CachedConsecutiveVideoLatents, CachedVideoLatents
from vforge.tasks.rep_world.optimizer import MuonWithAuxAdam
from vforge.tasks.rep_world.shared import (
    CodecProvider,
    RepaTeacherProvider,
    initialize_model,
    model_spec,
    parse_config,
    pretrained_source,
    validate_train_request,
)
from vforge.training.recipe import StepOutput, TrainComponents
from vforge.utils.training import build_optimizer_factory, build_scheduler_factory


def random_nine_state_view(latents, starts=None):
    """Choose a consecutive 9-state view of a committed 10-state latent window.

    The draw runs in the training process, so Native's saved torch RNG state
    controls the view across full checkpoint/resume; data workers only read the
    immutable cached 10-state sample.
    """
    if latents.ndim != 5 or latents.shape[2] != 10:
        raise ValueError("random_9_from_10 requires [B,C,10,H,W] cached latents")
    batch, channels, _, height, width = latents.shape
    supplied_starts = starts is not None
    if starts is None:
        starts = torch.randint(0, 2, (batch,), device=latents.device)
    if (
        starts.shape != (batch,)
        or starts.device != latents.device
        or starts.dtype != torch.long
        or (supplied_starts and bool(((starts < 0) | (starts > 1)).any()))
    ):
        raise ValueError("Each cached window start must be 0 or 1")
    indices = starts[:, None] + torch.arange(9, device=latents.device)[None, :]
    indices = indices[:, None, :, None, None].expand(batch, channels, 9, height, width)
    return latents.gather(2, indices), starts


def build_train_components(request):
    validate_train_request(request, "fsdp2")
    config = parse_config(request.extras)
    offline = request.latent_cache_mode == "offline"
    if offline:
        source = request.sources[0]
        cache_type = (
            CachedVideoLatents
            if (Path(source.cache_dir) / "contract.json").is_file()
            else CachedConsecutiveVideoLatents
        )
        dataset = cache_type(
            source.cache_dir,
            request.train_manifest,
            config=config,
            precision=request.precision,
        )
    else:
        dataset = VideoClipDataset(
            request.train_manifest,
            frames=config.states * config.geometry.rgb_frames,
            frame_interval=config.frame_interval,
            random_clip=config.random_clip,
        )
    loader = build_resumable_loader(
        dataset,
        parallel_state=get_parallel_state(),
        batch_size=request.batch_size,
        num_workers=request.num_workers,
        drop_last=request.drop_last,
        seed=request.sampler_seed,
        shuffle=request.shuffle,
        pin_memory=request.pin_memory,
        persistent_workers=request.persistent_workers,
        prefetch_factor=request.prefetch_factor,
    )
    provider = None if offline else CodecProvider(config, request.precision)
    teacher_provider = (
        RepaTeacherProvider(config.repa, request.precision) if config.repa.enabled else None
    )

    lifecycle = {"tuning_initialized": False}

    def step_fn(model, batch, step, context):
        tuning = (
            config.next_state_tuning_start_step is not None
            and step >= config.next_state_tuning_start_step
        )
        phase_metrics = {}
        if (
            config.next_state_tuning_init == "ema_reset_optimizer"
            and step == config.next_state_tuning_start_step
            and not lifecycle["tuning_initialized"]
        ):
            phase_metrics = initialize_tuning_from_ema(
                lifecycle["model"], lifecycle["optimizer"], config.ema, step
            )
            lifecycle["tuning_initialized"] = True
        window_starts = None
        if offline:
            rgb = batch["rgb"].to(context.device, dtype=torch.float32) if "rgb" in batch else None
            latents = batch["latents"].to(context.device, dtype=torch.float32).detach()
            if config.cached_state_view == "random_9_from_10":
                latents, window_starts = random_nine_state_view(latents)
        else:
            rgb = batch["rgb"].to(context.device, dtype=torch.float32)
            codec = provider.get(context.device)
            # Frozen codec AMP is independent of the trainable engine policy.
            # Packed latents stay FP32 in both online and cached training.
            with (
                torch.no_grad(),
                torch.autocast(
                    context.device.type,
                    dtype=provider.dtype,
                    enabled=provider.dtype != torch.float32,
                ),
            ):
                latents = codec.encode(rgb).detach()
        metrics = {}
        teacher_features = None
        if teacher_provider is not None:
            if rgb is None:
                raise ValueError("Video REPA requires clean RGB alongside packed states")
            teacher = teacher_provider.get(context.device)
            with torch.no_grad(), torch.autocast(
                context.device.type,
                dtype=teacher_provider.dtype,
                enabled=teacher_provider.dtype != torch.float32,
            ):
                teacher_features = teacher.encode_states(rgb, config.geometry.rgb_frames)
        if config.flow.uses_target_selection or config.repa.enabled:
            result = model(
                latents=latents, rgb=rgb, teacher_features=teacher_features,
                return_metrics=True, next_state_only=tuning,
            )
            loss, metrics = result["loss"], result["metrics"]
        else:
            loss = model(latents=latents, rgb=rgb)
        metrics.update(phase_metrics)
        clips = latents.shape[0]
        targets = clips * config.flow.target_count(config.states)
        metadata = {"sample_ids": list(batch["sample_id"]), "prediction_positions": targets}
        if window_starts is not None:
            metadata["cached_state_window_starts"] = window_starts.detach().cpu().tolist()
            metrics["cached_state_window_shifted_fraction"] = window_starts.float().mean()
        if config.flow.uses_target_selection:
            metadata["target_phase"] = "next_state_tuning" if tuning else "main"
        if config.flow.target_layout == "random_token_mosaic" and not tuning:
            metadata.update(target_layout="random_token_mosaic", token_horizon_fraction=0.5)
        if config.flow.target_layout == "left_right_mosaic" and not tuning:
            metadata.update(target_layout="left_right_mosaic", prediction_halves=targets * 2)
        if config.flow.target_layout == "quadrant_mosaic" and not tuning:
            metadata.update(target_layout="quadrant_mosaic", prediction_quadrants=targets * 4)
        return StepOutput(
            loss=loss,
            metrics={"loss_fm": loss.detach(), **metrics},
            workload={
                "local_batch": clips,
                "local_tokens": targets * config.geometry.tokens_per_state,
            },
            metadata=metadata,
        )

    if config.use_muon:

        def optimizer_factory(model):
            return MuonWithAuxAdam(
                model,
                lr=request.learning_rate,
                weight_decay=request.weight_decay,
                betas=request.betas,
                eps=request.eps,
            )
    else:
        optimizer_factory = build_optimizer_factory(
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            betas=request.betas,
            eps=request.eps,
        )
    base_optimizer_factory = optimizer_factory

    def optimizer_with_ema(model):
        initialize_ema(model, config.ema)
        optimizer = base_optimizer_factory(model)
        lifecycle.update(model=model, optimizer=optimizer)
        return optimizer

    return TrainComponents(
        model=model_spec(config, request.init_device),
        dataloader=loader,
        optimizer_factory=optimizer_with_ema if config.ema.enabled else optimizer_factory,
        step_fn=step_fn,
        scheduler_factory=build_scheduler_factory(
            name=request.scheduler_name,
            factor=request.scheduler_factor,
            warmup_steps=request.warmup_steps,
            start_factor=request.scheduler_start_factor,
            end_factor=request.scheduler_end_factor,
            total_iters=request.scheduler_total_iters,
        ),
        pretrained_weights=pretrained_source(
            request.pretrained_path, config, request.weight_loading
        ),
        model_initializer=initialize_model,
        post_optimizer_step=partial(update_ema, config=config.ema) if config.ema.enabled else None,
        dataloader_resume_contract=dataset.resume_contract,
    )
