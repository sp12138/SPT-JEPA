"""Generated-history causal evaluation for Wan Taichu teacher models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from vforge.models.causal import (
    IncrementalKVTransformer,
    allocate_incremental_kv_guidance_caches,
    use_incremental_kv_cache,
)
from vforge.models.video_tokens import FrameMajorPatchTokenLayout
from vforge.models.wan_taichu.inference.incremental_kv import (
    build_incremental_kv_executor,
)

FULL_RECOMPUTE = "full_recompute"
INCREMENTAL_KV = "incremental_kv"

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, "
    "works, paintings, images, static, overall gray, worst quality, low "
    "quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, "
    "three legs, many people in the background, walking backwards"
)


def build_tf_per_token_timestep(
    latent: torch.Tensor,
    scalar_timestep: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """Build doubled ``[clean | noisy]`` per-token timesteps."""
    layout = FrameMajorPatchTokenLayout.from_latent(latent, patch_size)
    value = scalar_timestep.to(latent.device, dtype=torch.float32).reshape(())
    clean = torch.zeros(layout.frames_per_sequence, device=latent.device)
    noisy = torch.full_like(clean, float(value))
    noisy[0] = 0.0
    return layout.expand_frame_values(torch.cat([clean, noisy]))


def build_active_block_tf_per_token_timestep(
    latent: torch.Tensor,
    scalar_timestep: torch.Tensor,
    patch_size: tuple[int, int, int],
    *,
    active_block: tuple[int, int],
) -> torch.Tensor:
    """Activate only the noisy tokens belonging to the block being denoised."""
    layout = FrameMajorPatchTokenLayout.from_latent(latent, patch_size)
    start, end = (int(value) for value in active_block)
    if not 0 <= start < end <= layout.frames_per_sequence:
        raise ValueError(
            "active_block must satisfy "
            f"0 <= start < end <= {layout.frames_per_sequence}, got {active_block}"
        )
    value = scalar_timestep.to(latent.device, dtype=torch.float32).reshape(())
    clean = torch.zeros(layout.frames_per_sequence, device=latent.device)
    noisy = torch.zeros_like(clean)
    noisy[start:end] = value
    return layout.expand_frame_values(torch.cat([clean, noisy]))


def build_causal_block_layout(
    num_frames: int,
    *,
    frame_per_block: int,
    first_frame_own_block: bool,
) -> tuple[tuple[int, int], ...]:
    """Return half-open ranges in post-patch latent-frame units."""
    num_frames = int(num_frames)
    frame_per_block = int(frame_per_block)
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    if frame_per_block < 1:
        raise ValueError(f"frame_per_block must be >= 1, got {frame_per_block}")
    blocks: list[tuple[int, int]] = []
    cursor = 0
    if first_frame_own_block:
        blocks.append((0, 1))
        cursor = 1
    while cursor < num_frames:
        end = min(cursor + frame_per_block, num_frames)
        blocks.append((cursor, end))
        cursor = end
    return tuple(blocks)


class WanTaichuCausalPipeline:
    """Roll out one causal block at a time from generated clean history."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        text_encoder: Any,
        vae: Any,
        transformer: nn.Module,
        engine_forward: Callable[..., Any],
        scheduler: Any,
        frame_per_block: int,
        first_frame_own_block: bool,
        rollout_mode: str = FULL_RECOMPUTE,
        usp_size: int = 1,
        incremental_transformer: IncrementalKVTransformer | None = None,
        vae_scale_factor_temporal: int = 4,
        vae_scale_factor_spatial: int = 16,
        execution_dtype: torch.dtype | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.vae = vae
        self.transformer = transformer
        self.engine_forward = engine_forward
        self.scheduler = scheduler
        self.frame_per_block = int(frame_per_block)
        self.first_frame_own_block = bool(first_frame_own_block)
        self.rollout_mode = str(rollout_mode)
        self.usp_size = int(usp_size)
        self.execution_dtype = execution_dtype
        self.vae_scale_factor_temporal = int(
            getattr(
                getattr(vae, "config", None),
                "scale_factor_temporal",
                vae_scale_factor_temporal,
            )
        )
        self.vae_scale_factor_spatial = int(
            getattr(
                getattr(vae, "config", None),
                "scale_factor_spatial",
                vae_scale_factor_spatial,
            )
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        if self.rollout_mode not in {FULL_RECOMPUTE, INCREMENTAL_KV}:
            raise ValueError(
                f"unsupported rollout_mode={self.rollout_mode!r}; expected "
                f"{FULL_RECOMPUTE!r} or {INCREMENTAL_KV!r}"
            )
        if self.frame_per_block < 1:
            raise ValueError(f"frame_per_block must be >= 1, got {self.frame_per_block}")
        patch_size = tuple(int(value) for value in self.transformer.config.patch_size)
        if patch_size[0] != 1:
            raise ValueError(
                f"causal TI2V eval requires temporal patch size p_t=1; got patch_size={patch_size}"
            )
        if self.rollout_mode == INCREMENTAL_KV:
            base = (
                self.transformer.primary_transformer()
                if hasattr(self.transformer, "primary_transformer")
                else self.transformer
            )
            self.incremental_transformer = incremental_transformer or (
                build_incremental_kv_executor(base, usp_size=self.usp_size)
            )
        else:
            self.incremental_transformer = incremental_transformer

    @property
    def dtype(self) -> torch.dtype:
        if self.execution_dtype is not None:
            return self.execution_dtype
        declared = getattr(self.transformer, "dtype", None)
        if isinstance(declared, torch.dtype):
            return declared
        parameter = next(self.transformer.parameters(), None)
        if parameter is None:
            raise RuntimeError("Wan Taichu transformer must declare dtype or own parameters")
        return parameter.dtype

    def _cache_context(self, name: str):
        context = getattr(self.transformer, "cache_context", None)
        return context(name) if callable(context) else nullcontext()

    @torch.no_grad()
    def encode_negative_prompt(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        text: str = "",
        max_sequence_length: int = 512,
    ) -> torch.Tensor:
        if self.tokenizer is None or self.text_encoder is None:
            raise RuntimeError("negative prompt encoding is only available on Ulysses leaders")
        max_len = getattr(self.tokenizer, "model_max_length", max_sequence_length)
        if not isinstance(max_len, int) or max_len > 8192:
            max_len = max_sequence_length
        encoded = self.tokenizer(
            [text] * batch_size,
            padding="max_length",
            max_length=max_len,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(device)
        mask = encoded.attention_mask.to(device)
        lengths = mask.gt(0).sum(dim=1).long()
        embeds = self.text_encoder(input_ids, mask).last_hidden_state.to(
            device=device,
            dtype=dtype,
        )
        rows = [row[:length] for row, length in zip(embeds, lengths, strict=True)]
        return torch.stack(
            [torch.cat([row, row.new_zeros(max_len - row.size(0), row.size(1))]) for row in rows]
        )

    def prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        num_frames: int,
        dtype: torch.dtype,
        device: torch.device,
        *,
        generator: torch.Generator | None,
        latents: torch.Tensor | None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels,
            latent_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def align_num_frames(self, num_frames: int) -> int:
        stride = self.vae_scale_factor_temporal
        if (num_frames - 1) % stride == 0:
            return num_frames
        return ((num_frames - 1) // stride) * stride + 1

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("VAE decode is only available on Ulysses leaders")
        declared = getattr(self.vae, "dtype", None)
        vae_dtype = (
            declared if isinstance(declared, torch.dtype) else next(self.vae.parameters()).dtype
        )
        latents = latents.to(vae_dtype)
        z_dim = int(self.vae.config.z_dim)
        mean = torch.tensor(
            self.vae.config.latents_mean,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, z_dim, 1, 1, 1)
        inverse_std = 1.0 / torch.tensor(
            self.vae.config.latents_std,
            device=latents.device,
            dtype=latents.dtype,
        ).view(1, z_dim, 1, 1, 1)
        return self.vae.decode(latents / inverse_std + mean, return_dict=False)[0]

    def block_layout(self, num_latent_frames: int) -> tuple[tuple[int, int], ...]:
        return build_causal_block_layout(
            num_latent_frames,
            frame_per_block=self.frame_per_block,
            first_frame_own_block=self.first_frame_own_block,
        )

    def _denoise_full(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cache_name: str,
    ) -> torch.Tensor:
        with self._cache_context(cache_name):
            output = self.engine_forward(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                attention_kwargs=None,
                return_dict=False,
            )
        return output[0] if isinstance(output, tuple) else output

    def _denoise_incremental(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cache_name: str,
        **kwargs: Any,
    ) -> torch.Tensor:
        kv_cache = kwargs.pop("kv_cache")
        with self._cache_context(cache_name), use_incremental_kv_cache(kv_cache):
            output = self.engine_forward(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                attention_kwargs=None,
                return_dict=False,
                incremental_kv=True,
                **kwargs,
            )
        return output[0] if isinstance(output, tuple) else output

    def _run_incremental(
        self,
        *,
        clean_full: torch.Tensor,
        initial_noise: torch.Tensor,
        condition: torch.Tensor | None,
        prompt: torch.Tensor,
        negative_prompt: torch.Tensor | None,
        num_inference_steps: int,
        guidance_scale: float,
        transformer_dtype: torch.dtype,
    ) -> torch.Tensor:
        executor = self.incremental_transformer
        assert executor is not None
        patch_size = tuple(int(value) for value in self.transformer.config.patch_size)
        _, p_h, p_w = patch_size
        tokens_per_frame = (clean_full.shape[3] // p_h) * (clean_full.shape[4] // p_w)
        total_tokens = clean_full.shape[2] * tokens_per_frame
        base_transformer = (
            self.transformer.primary_transformer()
            if hasattr(self.transformer, "primary_transformer")
            else self.transformer
        )
        local_attn_size = int(getattr(base_transformer, "local_attn_size", -1))
        sink_size = int(getattr(base_transformer, "sink_size", 0))
        num_cache_tokens = (
            total_tokens
            if local_attn_size == -1
            else min(clean_full.shape[2], local_attn_size) * tokens_per_frame
        )
        is_moe = bool(getattr(self.transformer, "wan_2", None) is not None)
        boundary_timestep = float(self.transformer.boundary_timestep) if is_moe else None
        full_input = clean_full.to(transformer_dtype)
        if condition is not None:
            full_input = torch.cat([full_input, condition], dim=1)
        full_rope = executor.full_clip_rotary_emb(full_input)
        cache_dtype = prompt.dtype if prompt.is_floating_point() else transformer_dtype
        use_cfg = guidance_scale > 1 and negative_prompt is not None
        guidance_caches = allocate_incremental_kv_guidance_caches(
            executor,
            batch_size=clean_full.shape[0],
            num_cache_tokens=num_cache_tokens,
            device=clean_full.device,
            dtype=cache_dtype,
            use_cfg=use_cfg,
        )
        cond_cache = guidance_caches.conditional
        uncond_cache = guidance_caches.unconditional

        def run_block(
            block_latents: torch.Tensor,
            *,
            start: int,
            end: int,
            scheduler_timestep: torch.Tensor | float,
            text: torch.Tensor,
            cache_name: str,
            cache: Any,
            commit: bool,
            expert_name: str | None = None,
        ) -> torch.Tensor:
            block_input = block_latents.to(transformer_dtype)
            if condition is not None:
                block_input = torch.cat([block_input, condition[:, :, start:end]], dim=1)
            block_tokens = (end - start) * tokens_per_frame
            timestep_value = torch.as_tensor(
                scheduler_timestep,
                device=clean_full.device,
            ).reshape(())
            timestep = torch.full(
                (clean_full.shape[0], block_tokens),
                float(timestep_value),
                device=clean_full.device,
                dtype=torch.float32,
            )
            return self._denoise_incremental(
                block_input,
                timestep,
                text,
                cache_name,
                kv_cache=cache,
                full_rotary_emb=full_rope,
                kv_write_start=start * tokens_per_frame,
                commit_kv=commit,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                usp_size=self.usp_size,
                expert_name=expert_name,
            )

        blocks = self.block_layout(clean_full.shape[2])
        generated_blocks = blocks
        committed_blocks: list[tuple[int, int]] = []
        owner_expert: str | None = None

        def expert_for_timestep(value: torch.Tensor | float) -> str | None:
            if not is_moe:
                return None
            timestep = float(torch.as_tensor(value).reshape(()))
            assert boundary_timestep is not None
            return "wan" if timestep >= boundary_timestep else "wan_2"

        def reset_cache(cache: Any) -> None:
            reset = getattr(cache, "reset", None)
            if not callable(reset):
                raise TypeError("A14B MoE incremental KV cache must support reset()")
            reset()

        def ensure_expert(expert_name: str | None) -> None:
            nonlocal owner_expert
            if not is_moe or owner_expert == expert_name:
                return
            reset_cache(cond_cache)
            if uncond_cache is not None:
                reset_cache(uncond_cache)
            for committed_start, committed_end in committed_blocks:
                committed_latents = clean_full[:, :, committed_start:committed_end]
                run_block(
                    committed_latents,
                    start=committed_start,
                    end=committed_end,
                    scheduler_timestep=0.0,
                    text=prompt,
                    cache_name="cond",
                    cache=cond_cache,
                    commit=True,
                    expert_name=expert_name,
                )
                if uncond_cache is not None:
                    run_block(
                        committed_latents,
                        start=committed_start,
                        end=committed_end,
                        scheduler_timestep=0.0,
                        text=negative_prompt,
                        cache_name="uncond",
                        cache=uncond_cache,
                        commit=True,
                        expert_name=expert_name,
                    )
            owner_expert = expert_name

        if self.first_frame_own_block:
            first = blocks[0]
            if first != (0, 1):
                raise RuntimeError(f"incremental_kv requires first block (0, 1), got {first}")
            run_block(
                clean_full[:, :, :1],
                start=0,
                end=1,
                scheduler_timestep=0.0,
                text=prompt,
                cache_name="cond",
                cache=cond_cache,
                commit=True,
                expert_name=expert_for_timestep(0.0),
            )
            if use_cfg:
                run_block(
                    clean_full[:, :, :1],
                    start=0,
                    end=1,
                    scheduler_timestep=0.0,
                    text=negative_prompt,
                    cache_name="uncond",
                    cache=uncond_cache,
                    commit=True,
                    expert_name=expert_for_timestep(0.0),
                )
            committed_blocks.append(first)
            generated_blocks = blocks[1:]

        for start, end in generated_blocks:
            block_latents = initial_noise[:, :, start:end].clone()
            self.scheduler.set_timesteps(num_inference_steps, device=clean_full.device)
            active_expert = expert_for_timestep(0.0)
            for scheduler_timestep in self.scheduler.timesteps:
                active_expert = expert_for_timestep(scheduler_timestep)
                ensure_expert(active_expert)
                noise_pred = run_block(
                    block_latents,
                    start=start,
                    end=end,
                    scheduler_timestep=scheduler_timestep,
                    text=prompt,
                    cache_name="cond",
                    cache=cond_cache,
                    commit=False,
                    expert_name=active_expert,
                )
                if use_cfg:
                    noise_uncond = run_block(
                        block_latents,
                        start=start,
                        end=end,
                        scheduler_timestep=scheduler_timestep,
                        text=negative_prompt,
                        cache_name="uncond",
                        cache=uncond_cache,
                        commit=False,
                        expert_name=active_expert,
                    )
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                block_latents = self.scheduler.step(
                    noise_pred.to(block_latents.dtype),
                    scheduler_timestep,
                    block_latents,
                    return_dict=False,
                )[0]
            run_block(
                block_latents,
                start=start,
                end=end,
                scheduler_timestep=0.0,
                text=prompt,
                cache_name="cond",
                cache=cond_cache,
                commit=True,
                expert_name=active_expert,
            )
            if use_cfg:
                run_block(
                    block_latents,
                    start=start,
                    end=end,
                    scheduler_timestep=0.0,
                    text=negative_prompt,
                    cache_name="uncond",
                    cache=uncond_cache,
                    commit=True,
                    expert_name=active_expert,
                )
            clean_full[:, :, start:end] = block_latents
            committed_blocks.append((start, end))
        return clean_full

    @torch.no_grad()
    def call_with_latent(
        self,
        encoder_hidden_state: torch.Tensor,
        *,
        y: torch.Tensor | None,
        first_frame_latents: torch.Tensor | None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 121,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        latents: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        decode_output: bool = True,
    ) -> torch.Tensor:
        """Generate from frame 0 without reading target or future clean latents."""
        device = encoder_hidden_state.device
        batch_size = int(encoder_hidden_state.shape[0])
        transformer_dtype = self.dtype
        num_frames = self.align_num_frames(num_frames)
        condition_channels = y.shape[1] if y is not None else 0
        noise_channels = int(self.transformer.config.in_channels) - condition_channels
        initial_noise = self.prepare_latents(
            batch_size,
            noise_channels,
            height,
            width,
            num_frames,
            self.dtype,
            device,
            generator=generator,
            latents=latents,
        ).clone()
        clean_full = torch.zeros_like(initial_noise)
        if self.first_frame_own_block:
            if first_frame_latents is None:
                raise ValueError("first_frame_latents is required when first_frame_own_block=true")
            first_frame = first_frame_latents.to(device=device, dtype=clean_full.dtype)
            if first_frame.shape[2] != 1:
                raise ValueError(
                    "first_frame_latents must contain exactly one frame, "
                    f"got shape {tuple(first_frame.shape)}"
                )
            clean_full[:, :, :1] = first_frame
        if y is not None:
            y = y.to(device=device, dtype=transformer_dtype)
            if y.shape[2] != clean_full.shape[2]:
                raise ValueError(
                    "condition and generated latent must have the same frame count; "
                    f"got {y.shape[2]} and {clean_full.shape[2]}"
                )
        prompt = (
            encoder_hidden_state.to(device)
            if self.execution_dtype is None
            else encoder_hidden_state.to(device=device, dtype=transformer_dtype)
        )
        if guidance_scale > 1:
            if negative_prompt_embeds is None:
                negative_prompt_embeds = self.encode_negative_prompt(
                    batch_size,
                    device,
                    self.dtype,
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    device=device,
                    dtype=self.dtype,
                )
        else:
            negative_prompt_embeds = None

        if self.rollout_mode == INCREMENTAL_KV:
            clean_full = self._run_incremental(
                clean_full=clean_full,
                initial_noise=initial_noise,
                condition=y,
                prompt=prompt,
                negative_prompt=negative_prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                transformer_dtype=transformer_dtype,
            )
        else:
            patch_size = tuple(int(value) for value in self.transformer.config.patch_size)
            for start, end in self.block_layout(clean_full.shape[2]):
                active_start = max(start, 1) if self.first_frame_own_block else start
                if active_start >= end:
                    continue
                block_latents = initial_noise[:, :, active_start:end].clone()
                self.scheduler.set_timesteps(num_inference_steps, device=device)
                for scheduler_timestep in self.scheduler.timesteps:
                    noisy_full = clean_full.clone()
                    noisy_full[:, :, active_start:end] = block_latents
                    clean_input = clean_full.to(transformer_dtype)
                    noisy_input = noisy_full.to(transformer_dtype)
                    if y is not None:
                        clean_input = torch.cat([clean_input, y], dim=1)
                        noisy_input = torch.cat([noisy_input, y], dim=1)
                    doubled = torch.cat([clean_input, noisy_input], dim=2)
                    timestep = build_active_block_tf_per_token_timestep(
                        clean_full,
                        scheduler_timestep,
                        patch_size,
                        active_block=(active_start, end),
                    )
                    noise_pred = self._denoise_full(
                        doubled,
                        timestep,
                        prompt,
                        "cond",
                    )[:, :, active_start:end]
                    if negative_prompt_embeds is not None:
                        noise_uncond = self._denoise_full(
                            doubled,
                            timestep,
                            negative_prompt_embeds,
                            "uncond",
                        )[:, :, active_start:end]
                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                    block_latents = self.scheduler.step(
                        noise_pred.to(block_latents.dtype),
                        scheduler_timestep,
                        block_latents,
                        return_dict=False,
                    )[0]
                clean_full[:, :, active_start:end] = block_latents

        if not decode_output:
            return clean_full
        return self.decode_latents(clean_full)


def _concat_condition_streams(
    encoded: Mapping[str, Any],
    stream_fields: Mapping[str, str],
    device: torch.device,
) -> torch.Tensor | None:
    blocks: list[torch.Tensor] = []
    for name, field in stream_fields.items():
        block = encoded.get(field)
        if block is None:
            raise ValueError(
                f"Wan Taichu causal sampling needs {name!r}, but encoded data has no {field!r}"
            )
        blocks.append(block.to(device))
    if not blocks:
        return None
    return blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=1)


def _resolve_pixel_grid(
    sample: Mapping[str, Any],
    encoded: Mapping[str, Any],
    pipeline: WanTaichuCausalPipeline,
) -> tuple[int, int, int]:
    if "__height__" in sample:
        return (
            int(sample["__height__"]),
            int(sample["__width__"]),
            int(sample["__num_frames__"]),
        )
    frames = sample.get("video_frames")
    if isinstance(frames, torch.Tensor):
        num_frames = int(frames.shape[1] if frames.ndim == 5 else frames.shape[0])
        return int(frames.shape[-2]), int(frames.shape[-1]), num_frames
    frame_latents = encoded["frame_latents"]
    height = int(frame_latents.shape[-2] * pipeline.vae_scale_factor_spatial)
    width = int(frame_latents.shape[-1] * pipeline.vae_scale_factor_spatial)
    num_frames = (int(frame_latents.shape[-3]) - 1) * pipeline.vae_scale_factor_temporal + 1
    return height, width, num_frames


@dataclass
class WanTaichuCausalInferencePipeline:
    model: nn.Module
    encode_condition: Callable[[dict[str, Any]], dict[str, Any]]
    pipeline: WanTaichuCausalPipeline
    stream_fields: Mapping[str, str]
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT

    def resolve_sample_seed(self, sample: Mapping[str, Any], base_seed: int) -> int:
        """Match Taichu's ``base seed + dataset index`` inference contract."""
        seed_offset = sample.get("__vforge_eval_seed_offset__")
        if seed_offset is None:
            seed_offset = sample.get("__vforge_eval_index__")
        if seed_offset is None:
            raise ValueError("Wan Taichu causal sample is missing its seed offset")
        return int(base_seed) + int(seed_offset)

    @torch.no_grad()
    def infer_sample(self, sample: dict[str, Any], *, seed: int) -> dict[str, Any]:
        device = next(self.model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(int(seed))
        encoded = sample.get("__encoded__")
        if encoded is None:
            encoded = self.encode_condition(sample)
        prompt = encoded["prompt"].to(device)
        condition = _concat_condition_streams(encoded, self.stream_fields, device)
        first_frame = encoded.get("first_frame_latents")
        if self.pipeline.first_frame_own_block and not isinstance(first_frame, torch.Tensor):
            raise ValueError("Wan Taichu causal sampling requires first_frame_latents")
        height, width, num_frames = _resolve_pixel_grid(sample, encoded, self.pipeline)
        negative = sample.get("__negative_prompt_embeds__")
        if isinstance(negative, torch.Tensor):
            negative = negative.to(device)
        from vforge.distributed.parallel_state import get_parallel_state

        is_output_leader = get_parallel_state().is_batch_leader
        video = self.pipeline.call_with_latent(
            prompt,
            y=condition,
            first_frame_latents=(
                first_frame.to(device) if isinstance(first_frame, torch.Tensor) else None
            ),
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            negative_prompt_embeds=negative,
            decode_output=is_output_leader,
        )
        latent_frames = (self.pipeline.align_num_frames(num_frames) - 1) // (
            self.pipeline.vae_scale_factor_temporal
        ) + 1
        return {
            "video": video.detach().cpu() if is_output_leader else None,
            "sample_id": sample.get("sample_id"),
            "seed": int(seed),
            "metadata": {
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "rollout_mode": self.pipeline.rollout_mode,
                "block_layout": {
                    "unit": "post_patch_latent_frames",
                    "frame_per_block": self.pipeline.frame_per_block,
                    "first_frame_own_block": self.pipeline.first_frame_own_block,
                    "ranges": [list(block) for block in self.pipeline.block_layout(latent_frames)],
                },
            },
        }


def build_inference_pipeline(
    *,
    model: nn.Module,
    encode_condition: Callable[[dict[str, Any]], dict[str, Any]],
    tokenizer: Any,
    text_encoder: Any,
    vae: Any,
    engine_forward: Callable[..., Any],
    scheduler: Any,
    stream_fields: Mapping[str, str],
    frame_per_block: int,
    first_frame_own_block: bool,
    rollout_mode: str = FULL_RECOMPUTE,
    usp_size: int = 1,
    incremental_transformer: IncrementalKVTransformer | None = None,
    vae_scale_factor_spatial: int = 16,
    execution_dtype: torch.dtype | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> WanTaichuCausalInferencePipeline:
    pipeline = WanTaichuCausalPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=model,
        engine_forward=engine_forward,
        scheduler=scheduler,
        frame_per_block=frame_per_block,
        first_frame_own_block=first_frame_own_block,
        rollout_mode=rollout_mode,
        usp_size=usp_size,
        incremental_transformer=incremental_transformer,
        vae_scale_factor_spatial=vae_scale_factor_spatial,
        execution_dtype=execution_dtype,
    )
    return WanTaichuCausalInferencePipeline(
        model=model,
        encode_condition=encode_condition,
        pipeline=pipeline,
        stream_fields=stream_fields,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
    )


__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "FULL_RECOMPUTE",
    "INCREMENTAL_KV",
    "WanTaichuCausalInferencePipeline",
    "WanTaichuCausalPipeline",
    "build_active_block_tf_per_token_timestep",
    "build_causal_block_layout",
    "build_inference_pipeline",
    "build_tf_per_token_timestep",
]
