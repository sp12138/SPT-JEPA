"""LTX-2.3 TI2AV full-SFT recipe.

The task owns data loading, stochastic input construction, targets, and loss.
The native transformer remains behind a lazy ``vforge.models.ltx`` import so
task discovery does not race or shadow model-package installation.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend

import vforge.ops as vforge_ops
from vforge.data.cache import digest_sample_ids
from vforge.data.loader import build_resumable_loader
from vforge.distributed.parallel_state import get_parallel_state
from vforge.distributed.ulysses import gather_sequence
from vforge.models.factory import ModelSpec
from vforge.models.loading import PretrainedWeightSource, stage_pretrained_weights
from vforge.ops._attention.segmented import INVALID_SEGMENT, SegmentedSparsity
from vforge.tasks.ltx.causal import tail_pad_to_multiple, unpad_sequence
from vforge.tasks.ltx.data import LtxPrecomputedDataset
from vforge.tasks.requests import TrainRequest, single_offline_source
from vforge.training.recipe import (
    OptimizerSpec,
    SchedulerSpec,
    StepContext,
    StepOutput,
    TrainComponents,
)
from vforge.utils.training import build_optimizer_factory, build_scheduler_factory

DEFAULT_FPS = 24.0
VIDEO_SCALE_FACTORS = (8, 32, 32)


def patchify_video(latents: torch.Tensor) -> torch.Tensor:
    """Match upstream ``VideoLatentPatchifier(patch_size=1)``."""
    if latents.ndim != 5:
        raise ValueError(f"video latents must be [B,C,F,H,W], got {tuple(latents.shape)}")
    return latents.permute(0, 2, 3, 4, 1).reshape(
        latents.shape[0],
        latents.shape[2] * latents.shape[3] * latents.shape[4],
        latents.shape[1],
    )


def patchify_audio(latents: torch.Tensor) -> torch.Tensor:
    """Match upstream ``AudioPatchifier(patch_size=1)``."""
    if latents.ndim != 4:
        raise ValueError(f"audio latents must be [B,C,T,F], got {tuple(latents.shape)}")
    return latents.permute(0, 2, 1, 3).reshape(
        latents.shape[0],
        latents.shape[2],
        latents.shape[1] * latents.shape[3],
    )


def video_positions(
    *,
    num_frames: int,
    height: int,
    width: int,
    batch_size: int,
    fps: float,
    device: torch.device,
) -> torch.Tensor:
    """Return upstream-compatible pixel-space ``[start, end)`` video bounds."""
    frame, y, x = torch.meshgrid(
        torch.arange(num_frames, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    starts = torch.stack((frame, y, x), dim=0)
    ends = starts + 1
    bounds = torch.stack((starts, ends), dim=-1)
    bounds = bounds.reshape(3, num_frames * height * width, 2)
    bounds = bounds.unsqueeze(0).expand(batch_size, -1, -1, -1)
    scale = torch.tensor(VIDEO_SCALE_FACTORS, device=device).reshape(1, 3, 1, 1)
    positions = bounds * scale
    positions = positions.clone()
    positions[:, 0] = (positions[:, 0] + 1 - VIDEO_SCALE_FACTORS[0]).clamp(min=0)
    positions = positions.float()
    positions[:, 0] = positions[:, 0] / float(fps)
    return positions


def audio_positions(
    *,
    num_time_steps: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return upstream-compatible causal audio time bounds in seconds."""
    starts = torch.arange(num_time_steps, device=device, dtype=torch.float32) * 4
    starts = (starts + 1 - 4).clamp(min=0) * 160 / 16000
    ends = torch.arange(1, num_time_steps + 1, device=device, dtype=torch.float32) * 4
    ends = (ends + 1 - 4).clamp(min=0) * 160 / 16000
    return (
        torch.stack((starts, ends), dim=-1)
        .reshape(1, 1, num_time_steps, 2)
        .expand(
            batch_size,
            -1,
            -1,
            -1,
        )
    )


def sample_shifted_logit_normal(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    *,
    std: float = 1.0,
    eps: float = 1e-3,
    uniform_prob: float = 0.1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample sigma with the upstream operation and RNG-call order."""
    slope = (2.05 - 0.95) / (4096 - 1024)
    mu = slope * sequence_length + (0.95 - slope * 1024)
    normal_samples = torch.randn((batch_size,), device=device, generator=generator) * std + mu
    logitnormal_samples = torch.sigmoid(normal_samples)
    upper = torch.sigmoid(torch.tensor(mu + 3.0902 * std, device=device))
    lower = torch.sigmoid(torch.tensor(mu - 2.5758 * std, device=device))
    zero_terminal_raw = (logitnormal_samples - lower) / (upper - lower)
    stretched = torch.where(
        zero_terminal_raw >= eps,
        zero_terminal_raw,
        2 * eps - zero_terminal_raw,
    ).clamp(0, 1)
    uniform = (1 - eps) * torch.rand((batch_size,), device=device, generator=generator) + eps
    probability = torch.rand((batch_size,), device=device, generator=generator)
    return torch.where(probability > uniform_prob, stretched, uniform)


def first_frame_conditioning_mask(
    *,
    batch_size: int,
    sequence_length: int,
    height: int,
    width: int,
    probability: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw the per-sample first-frame mask, including for probability one."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"first-frame probability must be in [0, 1], got {probability}")
    mask = torch.zeros(batch_size, sequence_length, device=device)
    first_frame_tokens = height * width
    # FlexibleStrategy always makes the Bernoulli call for a configured
    # first-frame condition, including probability zero or one and one-frame
    # samples where the first frame spans the full sequence.
    selected = torch.rand((batch_size,), device=device, generator=generator) < probability
    if first_frame_tokens > 0:
        mask[selected, :first_frame_tokens] = 1.0
    return mask


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute upstream's per-sample masked MSE in FP32."""
    target_length = target.shape[1]
    prediction = prediction[:, -target_length:, :]
    mask = loss_mask[:, -target_length:].unsqueeze(-1).float()
    squared = (prediction - target).pow(2)
    return (squared * mask).mean(dim=(-2, -1)) / mask.mean(dim=(-2, -1)).clamp(min=1e-8)


def build_train_components(request: TrainRequest) -> TrainComponents:
    """Build the vForge-owned LTX TI2AV training recipe."""
    _validate_request(request)
    model = ModelSpec(
        lambda: _build_ltx_model(
            pretrained_path=request.pretrained_path,
            init_device=request.init_device,
            extras=request.extras,
            usp_size=request.usp_size,
        )
    )
    dataloader, resume_contract = _build_train_dataloader(request)
    first_frame_probability = float(request.extras.get("first_frame_conditioning_probability", 1.0))

    def step_fn(
        live_model: nn.Module,
        batch: dict[str, Any],
        step: int,
        context: StepContext,
    ) -> StepOutput:
        return _ltx_train_step(
            live_model,
            batch,
            step,
            context,
            seed=request.seed,
            first_frame_probability=first_frame_probability,
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
        optimizer_spec=OptimizerSpec(
            learning_rate=request.learning_rate,
            weight_decay=request.weight_decay,
            betas=request.betas,
            eps=request.eps,
        ),
        scheduler_spec=_build_scheduler_spec(request),
        pretrained_weights=_pretrained_source(
            request.pretrained_path,
            request.weight_loading,
        ),
        dataloader_resume_contract=resume_contract,
    )


def _validate_request(request: TrainRequest) -> None:
    source = single_offline_source(request, recipe="LTX-2.3 TI2AV SFT")
    if request.latent_cache_mode != "offline":
        raise ValueError("LTX-2.3 TI2AV SFT requires latent_cache.mode=offline")
    if request.batch_size != 1:
        raise ValueError("LTX-2.3 TI2AV SFT currently requires data.batch_size=1")
    if request.usp_size < 1:
        raise ValueError("parallel.usp_size must be >= 1")
    if str(request.extras.get("training_mode", "full")) != "full":
        raise ValueError("LTX-2.3 TI2AV task supports full SFT only")
    if str(request.extras.get("timestep_sampling_mode", "shifted_logit_normal")) != (
        "shifted_logit_normal"
    ):
        raise ValueError("LTX-2.3 TI2AV task requires shifted_logit_normal sampling")
    if not request.pretrained_path:
        raise ValueError("model.pretrained_path is required")
    if source is None and request.max_steps > 0:
        raise ValueError("data.cache_dir must point to an LTX .precomputed dataset")


def _build_ltx_model(
    *,
    pretrained_path: str,
    init_device: str,
    extras: Mapping[str, Any],
    usp_size: int,
) -> nn.Module:
    ltx = importlib.import_module("vforge.models.ltx")
    builder = getattr(ltx, "build_model", None)
    if builder is None:
        builder = getattr(ltx, "build_ltx23_model", None)
    if builder is not None:
        return builder(
            pretrained_path=pretrained_path,
            init_device=init_device,
            extras=extras,
            dtype=torch.float32,
            connector_dtype=torch.bfloat16,
            ops=_stage0_transformer_ops(usp_size),
        )

    configurator = getattr(ltx, "LTXModelConfigurator", None)
    if configurator is None:
        raise AttributeError(
            "vforge.models.ltx must expose build_model, build_ltx23_model, or LTXModelConfigurator"
        )
    config = _read_checkpoint_config(pretrained_path)
    return configurator.from_config(
        config,
        device=init_device,
        ops=_stage0_transformer_ops(usp_size),
    )


def _stage0_transformer_ops(usp_size: int):
    TransformerOpsConfig = importlib.import_module("vforge.models.ltx").TransformerOpsConfig

    def attention_adapter(q, k, v, attn_mask=None, is_causal=False, sparsity=None):
        q = q.to(v.dtype)
        k = k.to(v.dtype)
        return vforge_ops.attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            ulysses=usp_size > 1,
            sparsity=sparsity,
        )

    ops_config = vforge_ops.get_ops_config()
    if ops_config is not None and ops_config.backend == "sdpa_math":
        priority = (SDPBackend.MATH,)
    elif usp_size > 1:
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
    return TransformerOpsConfig.from_functions(
        attention=attention_adapter,
        masked_attention=masked_attention,
        connector_attention=vforge_ops.attention,
        connector_masked_attention=vforge_ops.sdpa_attention,
    )


def _stage0_attention_sparsity(
    *,
    video_length: int,
    audio_length: int,
    multiple: int,
) -> Any | None:
    """Mask physical USP padding in Stage 0 self- and cross-attention."""
    if multiple < 1:
        raise ValueError(f"multiple must be >= 1, got {multiple}")

    def segments(real_length: int) -> tuple[int, ...]:
        padded_length = ((real_length + multiple - 1) // multiple) * multiple
        return (0,) * real_length + (INVALID_SEGMENT,) * (padded_length - real_length)

    video_segments = segments(video_length)
    audio_segments = segments(audio_length)
    video_padded = len(video_segments) != video_length
    audio_padded = len(audio_segments) != audio_length
    if not video_padded and not audio_padded:
        return None

    def sparsity(query: tuple[int, ...], key_value: tuple[int, ...]) -> SegmentedSparsity:
        return SegmentedSparsity(
            q_segment_ids=query,
            kv_segment_ids=key_value,
            allowed_pairs=((0, 0),),
        )

    AVAttentionSparsity = importlib.import_module(
        "vforge.models.ltx.transformer"
    ).AVAttentionSparsity
    return AVAttentionSparsity(
        video_self_sparsity=sparsity(video_segments, video_segments) if video_padded else None,
        audio_self_sparsity=sparsity(audio_segments, audio_segments) if audio_padded else None,
        a2v_sparsity=sparsity(video_segments, audio_segments),
        v2a_sparsity=sparsity(audio_segments, video_segments),
    )


def _build_train_dataloader(request: TrainRequest):
    state = get_parallel_state()
    source = single_offline_source(request, recipe="LTX-2.3 TI2AV SFT")
    if source is not None:
        dataset = LtxPrecomputedDataset(source.cache_dir)
        fingerprint = {
            "source_kind": "ltx_precomputed",
            "sample_ids_digest": digest_sample_ids(list(dataset.sample_ids)),
            "sources": ("conditions", "latents", "audio_latents"),
        }
    else:
        dataset = _EmptyDataset()
        fingerprint = {"source_kind": "ltx_precomputed", "empty": True}
    loader = build_resumable_loader(
        dataset,
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
    return loader, fingerprint


class _EmptyDataset(torch.utils.data.Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> Any:
        raise IndexError(index)


def _ltx_train_step(
    model: nn.Module,
    batch: dict[str, Any],
    step: int,
    context: StepContext,
    *,
    seed: int,
    first_frame_probability: float,
) -> StepOutput:
    device = context.device
    video_data = batch["video_latents"]
    audio_data = batch["audio_latents"]
    video_clean = patchify_video(video_data["latents"].to(device))
    audio_clean = patchify_audio(audio_data["latents"].to(device))
    conditions = _prepare_conditions(model, batch["conditions"], device)
    dp_rank = int(context.parallel_state.dp_rank)
    # One stochastic stream per DP replica, shared by its USP peers.
    generator = context.random_streams.generator(
        "ltx.input", seed=seed + dp_rank * 1_000_003, device=device
    )

    video_sigma, audio_sigma = (
        sample_shifted_logit_normal(
            video_clean.shape[0],
            video_clean.shape[1],
            device,
            generator=generator,
        ),
        sample_shifted_logit_normal(
            audio_clean.shape[0],
            audio_clean.shape[1],
            device,
            generator=generator,
        ),
    )
    video_noise = torch.randn(
        video_clean.shape,
        dtype=video_clean.dtype,
        device=device,
        generator=generator,
    )
    audio_noise = torch.randn(
        audio_clean.shape,
        dtype=audio_clean.dtype,
        device=device,
        generator=generator,
    )
    video_noisy, video_target = _noisy_and_velocity(video_clean, video_noise, video_sigma)
    video_mask = first_frame_conditioning_mask(
        batch_size=video_clean.shape[0],
        sequence_length=video_clean.shape[1],
        height=_batch_scalar(video_data["height"]),
        width=_batch_scalar(video_data["width"]),
        probability=first_frame_probability,
        device=device,
        generator=generator,
    )
    mask_expanded = video_mask.unsqueeze(-1)
    video_noisy = mask_expanded * video_clean + (1 - mask_expanded) * video_noisy
    video_timesteps = (1 - video_mask) * video_sigma[:, None].expand_as(video_mask)
    video_loss_mask = video_mask == 0

    audio_noisy, audio_target = _noisy_and_velocity(audio_clean, audio_noise, audio_sigma)
    audio_timesteps = (
        audio_sigma[:, None].expand(audio_clean.shape[0], audio_clean.shape[1]).clone()
    )
    audio_mask = torch.ones(
        audio_clean.shape[0],
        audio_clean.shape[1],
        dtype=torch.bool,
        device=device,
    )

    Modality = importlib.import_module("vforge.models.ltx").Modality
    context_mask = conditions["prompt_attention_mask"]
    video = Modality(
        enabled=True,
        latent=video_noisy,
        sigma=video_sigma,
        timesteps=video_timesteps,
        positions=video_positions(
            num_frames=_batch_scalar(video_data["num_frames"]),
            height=_batch_scalar(video_data["height"]),
            width=_batch_scalar(video_data["width"]),
            batch_size=video_clean.shape[0],
            fps=_batch_float(video_data.get("fps"), DEFAULT_FPS),
            device=device,
        ),
        context=conditions["video_prompt_embeds"],
        context_mask=context_mask,
    )
    audio = Modality(
        enabled=True,
        latent=audio_noisy,
        sigma=audio_sigma,
        timesteps=audio_timesteps,
        positions=audio_positions(
            num_time_steps=audio_clean.shape[1],
            batch_size=audio_clean.shape[0],
            device=device,
        ),
        context=conditions["audio_prompt_embeds"],
        context_mask=context_mask,
    )
    usp_size = int(context.parallel_state.config.usp_size)
    video = _localize_stage0_modality(
        video,
        usp_size=usp_size,
        usp_rank=int(context.parallel_state.usp_rank),
    )
    audio = _localize_stage0_modality(
        audio,
        usp_size=usp_size,
        usp_rank=int(context.parallel_state.usp_rank),
    )
    masks = _stage0_attention_sparsity(
        video_length=video_clean.shape[1],
        audio_length=audio_clean.shape[1],
        multiple=usp_size,
    )
    video_prediction, audio_prediction = model(video=video, audio=audio, masks=masks)
    if video_prediction is None or audio_prediction is None:
        raise ValueError("LTX TI2AV model must return both video and audio predictions")
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
    video_prediction = unpad_sequence(video_prediction, video_clean.shape[1], dim=1)
    audio_prediction = unpad_sequence(audio_prediction, audio_clean.shape[1], dim=1)
    # Accelerate wraps mixed-precision model forwards with
    # ``convert_outputs_to_fp32`` before the upstream trainer computes loss.
    video_prediction = video_prediction.float()
    audio_prediction = audio_prediction.float()

    video_per_sample = masked_mse(video_prediction, video_target, video_loss_mask)
    audio_per_sample = masked_mse(audio_prediction, audio_target, audio_mask)
    video_loss = video_per_sample.mean()
    audio_loss = audio_per_sample.mean()
    reported_loss = video_loss + audio_loss
    loss = reported_loss / usp_size if usp_size > 1 else reported_loss
    sample_ids = batch.get("sample_id", [])
    if isinstance(sample_ids, str):
        sample_ids = [sample_ids]
    normalized_sample_ids = [str(sample_id) for sample_id in sample_ids]
    return StepOutput(
        loss=loss,
        metrics={
            "loss_scalar": reported_loss.detach(),
            "video_loss": video_loss.detach(),
            "audio_loss": audio_loss.detach(),
            "video_sigma": video_sigma.detach().mean(),
            "audio_sigma": audio_sigma.detach().mean(),
        },
        workload={
            "local_batch": int(video_clean.shape[0]),
            "local_tokens": int(video_clean.shape[1] + audio_clean.shape[1]),
        },
        metadata={
            "sample_ids": normalized_sample_ids,
            "timestep_sampling": "shifted_logit_normal",
            "first_frame_conditioning_probability": float(first_frame_probability),
            "global_step": int(step),
            "video_sequence_length": int(video_clean.shape[1]),
            "audio_sequence_length": int(audio_clean.shape[1]),
            "usp_size": usp_size,
        },
    )


def _localize_stage0_modality(
    modality: Any,
    *,
    usp_size: int,
    usp_rank: int,
) -> Any:
    if usp_size == 1:
        return modality

    def local(tensor: torch.Tensor, *, dim: int) -> tuple[torch.Tensor, int]:
        padded, real_length = tail_pad_to_multiple(tensor, multiple=usp_size, dim=dim)
        local_length = padded.shape[dim] // usp_size
        shard = padded.narrow(dim, usp_rank * local_length, local_length).contiguous()
        return shard, real_length

    latent, _ = local(modality.latent, dim=1)
    timesteps, _ = local(modality.timesteps, dim=1)
    positions, _ = local(modality.positions, dim=2)
    Modality = importlib.import_module("vforge.models.ltx").Modality
    return Modality(
        enabled=modality.enabled,
        latent=latent,
        sigma=modality.sigma,
        timesteps=timesteps,
        positions=positions,
        context=modality.context,
        context_mask=modality.context_mask,
        attention_mask=None,
    )


def _prepare_conditions(
    model: nn.Module,
    conditions: Mapping[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    moved = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in conditions.items()
    }
    prepare = getattr(model, "prepare_text_conditions", None)
    if not callable(prepare):
        raise RuntimeError(
            "the native LTX training model must expose prepare_text_conditions() "
            "for upstream .precomputed feature tensors"
        )
    moved = prepare(moved)
    if "video_prompt_embeds" not in moved:
        moved["video_prompt_embeds"] = moved["prompt_embeds"]
    if "audio_prompt_embeds" not in moved:
        moved["audio_prompt_embeds"] = moved["prompt_embeds"]
    return moved


def _noisy_and_velocity(
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expanded = sigma.view(-1, 1, 1)
    return (1 - expanded) * clean + expanded * noise, noise - clean


def _batch_scalar(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def _batch_float(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def _build_scheduler_spec(request: TrainRequest) -> SchedulerSpec:
    kwargs = {
        "name": request.scheduler_name,
        "factor": request.scheduler_factor,
        "warmup_steps": request.warmup_steps,
        "start_factor": request.scheduler_start_factor,
        "end_factor": request.scheduler_end_factor,
        "total_iters": request.scheduler_total_iters,
    }
    try:
        return SchedulerSpec(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "LTX-2.3 TI2AV requires Task B's SchedulerSpec linear fields "
            "(start_factor, end_factor, total_iters)"
        ) from exc


def _pretrained_source(path: str, loading_mode: str) -> PretrainedWeightSource:
    def fallback_factory() -> nn.Module:
        # The trainer stages source.path, but this closure retains the original
        # path. Resolve the same node cache so fallback also reads the SSD copy.
        staged = stage_pretrained_weights(PretrainedWeightSource(path=path, stage_to_local=True))
        ltx = importlib.import_module("vforge.models.ltx")
        loader = getattr(ltx, "load_pretrained_model", None)
        if loader is None:
            raise AttributeError("vforge.models.ltx must expose load_pretrained_model")
        return loader(
            pretrained_path=staged.path,
            dtype=torch.float32,
            connector_dtype=torch.bfloat16,
        )

    return PretrainedWeightSource(
        path=path,
        subfolder="",
        loading_mode=loading_mode,
        fallback_model_factory=fallback_factory,
        stage_to_local=True,
    )


def _read_checkpoint_config(path: str) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("building LTX from a checkpoint requires safetensors") from exc
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    if metadata is None or "config" not in metadata:
        raise ValueError(f"LTX checkpoint metadata has no 'config' entry: {path}")
    import json

    return json.loads(metadata["config"])


__all__ = [
    "audio_positions",
    "build_train_components",
    "first_frame_conditioning_mask",
    "masked_mse",
    "patchify_audio",
    "patchify_video",
    "sample_shifted_logit_normal",
    "video_positions",
]
