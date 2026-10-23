"""LTX-2.3 Stage 1 causal autoregressive teacher-forcing recipe."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend

import vforge.ops as vforge_ops
from vforge.distributed.ulysses import gather_sequence
from vforge.models.factory import ModelSpec
from vforge.models.ltx.causal import CausalLTXModelConfigurator, CausalMasks
from vforge.models.ltx.modality import Modality
from vforge.models.ltx.transformer import TransformerOpsConfig
from vforge.tasks.ltx.causal import (
    PackedSequence,
    build_packed_sequence,
    causal_block_ranges,
    compile_cross_segmented_sparsity,
    compile_self_segmented_sparsity,
    noisy_only_mse,
    tail_pad_to_multiple,
    time_aligned_audio_ranges,
    unpad_sequence,
)
from vforge.tasks.ltx.train import (
    DEFAULT_FPS,
    _batch_float,
    _batch_scalar,
    _build_scheduler_spec,
    _build_train_dataloader,
    _prepare_conditions,
    _pretrained_source,
    _read_checkpoint_config,
    audio_positions,
    patchify_audio,
    patchify_video,
    sample_shifted_logit_normal,
    video_positions,
)
from vforge.tasks.requests import TrainRequest, single_offline_source
from vforge.training.recipe import OptimizerSpec, StepContext, StepOutput, TrainComponents
from vforge.utils.training import build_optimizer_factory, build_scheduler_factory

AUDIO_LATENT_FPS = 25
REF_AUDIO_SECONDS = 1.5
REF_AUDIO_TOKENS = int(REF_AUDIO_SECONDS * AUDIO_LATENT_FPS + 0.5)
REF_POSITION_GAP_SECONDS = 5.0


@dataclass(frozen=True)
class CausalTrainingInputs:
    video: Modality
    audio: Modality
    video_targets: torch.Tensor
    audio_targets: torch.Tensor
    video_loss_mask: torch.Tensor
    audio_loss_mask: torch.Tensor
    masks: CausalMasks
    video_logical_length: int
    audio_logical_length: int


def _causal_transformer_ops(attention_adapter, *, usp_size: int) -> TransformerOpsConfig:
    ops_config = vforge_ops.get_ops_config()
    if ops_config is not None and ops_config.backend == "sdpa_math":
        priority = (SDPBackend.MATH,)
    elif usp_size > 1:
        # cuDNN SDPA backward crashes inside cudnn_frontend plan construction
        # for the Stage-1 text cross-attention shapes when USP is enabled.
        # Keep fused PyTorch kernels available, but exclude only that backend.
        priority = (
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        )
    else:
        priority = (
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        )
    masked_attention = partial(
        vforge_ops.sdpa_attention,
        priority=priority,
        contiguous=False,
    )
    connector_masked_attention = partial(
        vforge_ops.sdpa_attention,
        priority=(
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ),
        contiguous=False,
    )
    return TransformerOpsConfig.from_functions(
        attention=attention_adapter,
        masked_attention=masked_attention,
        connector_attention=vforge_ops.attention,
        connector_masked_attention=connector_masked_attention,
    )


def validate_train_request(request: TrainRequest) -> None:
    """Validate Stage 1 without relaxing the independent Stage 0 recipe."""
    source = single_offline_source(request, recipe="LTX-2.3 causal teacher forcing")
    if request.latent_cache_mode != "offline":
        raise ValueError("LTX-2.3 causal teacher forcing requires latent_cache.mode=offline")
    if request.batch_size != 1:
        raise ValueError("LTX-2.3 causal teacher forcing requires data.batch_size=1")
    if request.usp_size < 1:
        raise ValueError("parallel.usp_size must be >= 1")
    if str(request.extras.get("training_mode", "causal_ar_tf")) != "causal_ar_tf":
        raise ValueError("LTX-2.3 Stage 1 requires model.extras.training_mode=causal_ar_tf")
    if not request.pretrained_path:
        raise ValueError("model.pretrained_path is required")
    if source is None and request.max_steps > 0:
        raise ValueError("data.cache_dir must point to an LTX .precomputed dataset")


def _shift_ref_positions(positions: torch.Tensor) -> torch.Tensor:
    shifted = positions.clone()
    shifted[:, 0] -= REF_POSITION_GAP_SECONDS + shifted[:, 0].max()
    return shifted


def _paired_block_indices(
    video_ranges: tuple[tuple[int, int], ...],
    audio_ranges: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    paired = tuple(
        index
        for index, ((video_start, video_end), (audio_start, audio_end)) in enumerate(
            zip(video_ranges, audio_ranges, strict=True)
        )
        if video_end > video_start and audio_end > audio_start
    )
    if not paired:
        raise ValueError("sample has no block with both video and audio tokens")
    return paired


def _modality_from_packed(
    packed: PackedSequence,
    *,
    prompt_sigma: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> Modality:
    return Modality(
        enabled=True,
        latent=packed.latents,
        sigma=prompt_sigma,
        timesteps=packed.timesteps,
        positions=packed.positions,
        context=context,
        context_mask=context_mask,
        attention_mask=attention_mask,
    )


def _localize_modality(
    modality: Modality,
    *,
    usp_size: int,
    usp_rank: int,
) -> Modality:
    """Tail-pad a full logical modality and return this USP rank's contiguous shard."""
    if usp_size == 1:
        return modality

    def local(tensor: torch.Tensor, *, dim: int) -> torch.Tensor:
        padded, _ = tail_pad_to_multiple(tensor, multiple=usp_size, dim=dim)
        local_length = padded.shape[dim] // usp_size
        return padded.narrow(dim, usp_rank * local_length, local_length).contiguous()

    return Modality(
        enabled=modality.enabled,
        latent=local(modality.latent, dim=1),
        sigma=modality.sigma,
        timesteps=local(modality.timesteps, dim=1),
        positions=local(modality.positions, dim=2),
        context=modality.context,
        context_mask=modality.context_mask,
        attention_mask=None,
    )


def _local_range(
    global_start: int,
    global_end: int,
    *,
    local_length: int,
    usp_rank: int,
) -> tuple[int, int]:
    shard_start = usp_rank * local_length
    start = min(max(global_start - shard_start, 0), local_length)
    end = min(max(global_end - shard_start, 0), local_length)
    return start, end


def prepare_causal_training_inputs(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    device: torch.device,
    usp_size: int = 1,
    usp_rank: int = 0,
) -> CausalTrainingInputs:
    """Build the exact upstream logical Stage 1 packed batch on one device."""
    if usp_size < 1 or not 0 <= usp_rank < usp_size:
        raise ValueError("usp_rank must be within usp_size")
    video_data = batch["video_latents"]
    audio_data = batch["audio_latents"]
    video_clean = patchify_video(video_data["latents"].to(device))
    audio_clean = patchify_audio(audio_data["latents"].to(device))
    conditions = _prepare_conditions(model, batch["conditions"], device)

    batch_size = video_clean.shape[0]
    num_frames = _batch_scalar(video_data["num_frames"])
    height = _batch_scalar(video_data["height"])
    width = _batch_scalar(video_data["width"])
    fps = _batch_float(video_data.get("fps"), DEFAULT_FPS)
    tokens_per_frame = height * width

    video_frame_ranges = causal_block_ranges(num_frames)
    video_token_ranges = tuple(
        (start * tokens_per_frame, end * tokens_per_frame) for start, end in video_frame_ranges
    )
    audio_token_ranges = time_aligned_audio_ranges(
        video_frame_ranges,
        audio_length=audio_clean.shape[1],
        video_fps=fps,
    )
    block_indices = _paired_block_indices(video_token_ranges, audio_token_ranges)
    sigmas = (
        sample_shifted_logit_normal(
            batch_size * len(block_indices),
            video_clean.shape[1],
            device,
        )
        .to(dtype=video_clean.dtype)
        .reshape(batch_size, len(block_indices))
    )
    prompt_sigma = sigmas.mean(dim=1)

    video_clean_positions = video_positions(
        num_frames=num_frames,
        height=height,
        width=width,
        batch_size=batch_size,
        fps=fps,
        device=device,
    )
    audio_clean_positions = audio_positions(
        num_time_steps=audio_clean.shape[1],
        batch_size=batch_size,
        device=device,
    )
    video_ref = video_clean[:, :tokens_per_frame]
    video_ref_positions = _shift_ref_positions(
        video_positions(
            num_frames=1,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
        )
    )
    audio_ref = audio_clean.new_zeros((batch_size, REF_AUDIO_TOKENS, audio_clean.shape[2]))
    audio_ref_positions = _shift_ref_positions(
        audio_positions(
            num_time_steps=REF_AUDIO_TOKENS,
            batch_size=batch_size,
            device=device,
        )
    )
    video_packed = build_packed_sequence(
        clean=video_clean,
        positions=video_clean_positions,
        token_ranges=video_token_ranges,
        block_indices=block_indices,
        sigmas=sigmas,
        ref_latents=video_ref,
        ref_positions=video_ref_positions,
    )
    audio_packed = build_packed_sequence(
        clean=audio_clean,
        positions=audio_clean_positions,
        token_ranges=audio_token_ranges,
        block_indices=block_indices,
        sigmas=sigmas,
        ref_latents=audio_ref,
        ref_positions=audio_ref_positions,
    )

    video_full = _modality_from_packed(
        video_packed,
        prompt_sigma=prompt_sigma,
        context=conditions["video_prompt_embeds"],
        context_mask=conditions["prompt_attention_mask"],
    )
    audio_full = _modality_from_packed(
        audio_packed,
        prompt_sigma=prompt_sigma,
        context=conditions["audio_prompt_embeds"],
        context_mask=conditions["prompt_attention_mask"],
    )
    video = _localize_modality(video_full, usp_size=usp_size, usp_rank=usp_rank)
    audio = _localize_modality(audio_full, usp_size=usp_size, usp_rank=usp_rank)
    video_start, video_end = _local_range(
        video_ref.shape[1] + video_clean.shape[1],
        video_packed.layout.logical_length,
        local_length=video.latent.shape[1],
        usp_rank=usp_rank,
    )
    audio_start, audio_end = _local_range(
        audio_ref.shape[1] + audio_clean.shape[1],
        audio_packed.layout.logical_length,
        local_length=audio.latent.shape[1],
        usp_rank=usp_rank,
    )
    masks = CausalMasks(
        video_self_sparsity=compile_self_segmented_sparsity(video_packed.layout, multiple=usp_size),
        audio_self_sparsity=compile_self_segmented_sparsity(audio_packed.layout, multiple=usp_size),
        a2v_sparsity=compile_cross_segmented_sparsity(
            query=video_packed.layout,
            key_value=audio_packed.layout,
            multiple=usp_size,
        ),
        v2a_sparsity=compile_cross_segmented_sparsity(
            query=audio_packed.layout,
            key_value=video_packed.layout,
            multiple=usp_size,
        ),
        video_text_start=video_start,
        audio_text_start=audio_start,
        video_text_end=video_end,
        audio_text_end=audio_end,
    )
    return CausalTrainingInputs(
        video=video,
        audio=audio,
        video_targets=video_packed.targets,
        audio_targets=audio_packed.targets,
        video_loss_mask=video_packed.layout.noisy_mask.unsqueeze(0).expand(batch_size, -1),
        audio_loss_mask=audio_packed.layout.noisy_mask.unsqueeze(0).expand(batch_size, -1),
        masks=masks,
        video_logical_length=video_packed.layout.logical_length,
        audio_logical_length=audio_packed.layout.logical_length,
    )


def _build_causal_model(request: TrainRequest) -> nn.Module:
    config = _read_checkpoint_config(request.pretrained_path)

    def attention_adapter(q, k, v, attn_mask=None, is_causal=False, sparsity=None):
        return vforge_ops.attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            ulysses=request.usp_size > 1,
            sparsity=sparsity,
        )

    model = CausalLTXModelConfigurator.from_config(
        config,
        device=request.init_device,
        ops=_causal_transformer_ops(attention_adapter, usp_size=request.usp_size),
    )
    model.to(dtype=torch.float32)
    model.video_embeddings_connector.to(dtype=torch.bfloat16)
    model.audio_embeddings_connector.to(dtype=torch.bfloat16)
    return model


def _causal_train_step(
    model: nn.Module,
    batch: dict[str, Any],
    step: int,
    context: StepContext,
    *,
    seed: int,
) -> StepOutput:
    usp_size = int(context.parallel_state.config.usp_size)
    inputs = prepare_causal_training_inputs(
        model,
        batch,
        device=context.device,
        usp_size=usp_size,
        usp_rank=int(context.parallel_state.usp_rank),
    )
    video_prediction, audio_prediction = model(
        video=inputs.video,
        audio=inputs.audio,
        masks=inputs.masks,
    )
    if video_prediction is None or audio_prediction is None:
        raise ValueError("LTX causal model must return video and audio predictions")
    if usp_size > 1:
        video_prediction = gather_sequence(
            video_prediction,
            dim=1,
            group=context.parallel_state.usp_group,
        )
        audio_prediction = gather_sequence(
            audio_prediction,
            dim=1,
            group=context.parallel_state.usp_group,
        )
    video_prediction = unpad_sequence(video_prediction, inputs.video_logical_length, dim=1)
    audio_prediction = unpad_sequence(audio_prediction, inputs.audio_logical_length, dim=1)
    video_loss = noisy_only_mse(
        video_prediction.float(), inputs.video_targets, inputs.video_loss_mask
    )
    audio_loss = noisy_only_mse(
        audio_prediction.float(), inputs.audio_targets, inputs.audio_loss_mask
    )
    reported_loss = video_loss + audio_loss
    loss_for_backward = reported_loss / usp_size if usp_size > 1 else reported_loss
    return StepOutput(
        loss=loss_for_backward,
        metrics={
            "loss_scalar": reported_loss.detach(),
            "video_loss": video_loss.detach(),
            "audio_loss": audio_loss.detach(),
            "sigma": inputs.video.sigma.detach().mean(),
        },
        workload={
            "local_batch": int(inputs.video.latent.shape[0]),
            "local_tokens": int(inputs.video.latent.shape[1] + inputs.audio.latent.shape[1]),
        },
        metadata={
            "global_step": int(step),
            "video_logical_length": inputs.video_logical_length,
            "audio_logical_length": inputs.audio_logical_length,
            "usp_size": usp_size,
        },
    )


def build_train_components(request: TrainRequest) -> TrainComponents:
    validate_train_request(request)
    model = ModelSpec(lambda: _build_causal_model(request))
    dataloader, resume_contract = _build_train_dataloader(request)

    def step_fn(
        live_model: nn.Module,
        batch: dict[str, Any],
        step: int,
        context: StepContext,
    ) -> StepOutput:
        return _causal_train_step(
            live_model,
            batch,
            step,
            context,
            seed=request.seed,
        )

    scheduler_factory = build_scheduler_factory(
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
        scheduler_factory=scheduler_factory,
        optimizer_spec=OptimizerSpec(
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            betas=request.betas,
            eps=request.eps,
        ),
        scheduler_spec=_build_scheduler_spec(request),
        pretrained_weights=_pretrained_source(request.pretrained_path, request.weight_loading),
        dataloader_resume_contract=resume_contract,
    )


__all__ = [
    "CausalTrainingInputs",
    "build_train_components",
    "prepare_causal_training_inputs",
    "validate_train_request",
]
