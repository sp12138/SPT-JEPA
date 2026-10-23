"""Block-causal full-recompute sampling for Wan 2.2 TI2V teacher forcing."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from vforge.models.causal import allocate_incremental_kv_guidance_caches
from vforge.models.video_tokens import FrameMajorPatchTokenLayout
from vforge.models.wan.inference import (
    DEFAULT_NEGATIVE_PROMPT,
    WanFlowMatchPipelineBase,
)

FULL_RECOMPUTE = "full_recompute"
INCREMENTAL_KV = "incremental_kv"


def build_tf_per_token_timestep(
    latent: torch.Tensor,
    scalar_timestep: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> torch.Tensor:
    """``[B, 2S]`` per-token timestep for a ``[clean | noisy]`` sequence.

    ``latent`` is one half, so the result is twice its token count. Token order
    is frame-major, matching ``hidden_states.flatten(2).transpose(1, 2)`` inside
    the transformer, so every spatial token of a latent frame shares a
    timestep.

    The clean half is entirely ``0``: it carries ground-truth context that was
    never noised, and telling the model otherwise would ask it to denoise its
    own answer. In the noisy half frame 0 is also ``0``, because it holds the
    in-painted reference frame rather than a prediction.
    """
    layout = FrameMajorPatchTokenLayout.from_latent(latent, patch_size)

    device = latent.device
    value = scalar_timestep.to(device).reshape(())
    clean = torch.zeros(layout.frames_per_sequence, device=device, dtype=torch.float32)
    noisy = torch.full(
        (layout.frames_per_sequence,),
        float(value),
        device=device,
        dtype=torch.float32,
    )
    noisy[0] = 0.0
    per_frame = torch.cat([clean, noisy], dim=0)
    return layout.expand_frame_values(per_frame)


def build_active_block_tf_per_token_timestep(
    latent: torch.Tensor,
    scalar_timestep: torch.Tensor,
    patch_size: tuple[int, int, int],
    *,
    active_block: tuple[int, int],
) -> torch.Tensor:
    """Build ``[B, 2S]`` timesteps with only one noisy block active.

    ``active_block`` is a half-open range in post-patch latent-frame units,
    which is also the unit used by the teacher-forcing attention sparsity.
    The clean half is always zero. The noisy half is zero outside the block,
    including generated history and ungenerated future frames.
    """
    layout = FrameMajorPatchTokenLayout.from_latent(latent, patch_size)
    t_post = layout.frames_per_sequence
    start, end = (int(active_block[0]), int(active_block[1]))
    if not 0 <= start < end <= t_post:
        raise ValueError(
            f"active_block must satisfy 0 <= start < end <= {t_post}, got {active_block}"
        )

    value = scalar_timestep.to(latent.device, dtype=torch.float32).reshape(())
    clean = torch.zeros(t_post, device=latent.device, dtype=torch.float32)
    noisy = torch.zeros_like(clean)
    noisy[start:end] = value
    return layout.expand_frame_values(torch.cat([clean, noisy]))


def build_causal_block_layout(
    num_frames: int,
    *,
    frame_per_block: int,
    first_frame_own_block: bool,
) -> tuple[tuple[int, int], ...]:
    """Return half-open block ranges in post-patch latent-frame units."""
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


class Wan22Ti2v5bCausalPipeline(WanFlowMatchPipelineBase):
    """Generate one causal block at a time with full transformer recomputation."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        text_encoder: Any,
        vae: Any,
        transformer: Any,
        engine_forward: Callable[..., Any],
        scheduler: FlowMatchEulerDiscreteScheduler,
        frame_per_block: int,
        first_frame_own_block: bool,
        rollout_mode: str = FULL_RECOMPUTE,
        usp_size: int = 1,
        vae_scale_factor_temporal: int = 4,
        vae_scale_factor_spatial: int = 16,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
        )
        self.engine_forward = engine_forward
        self.frame_per_block = int(frame_per_block)
        self.first_frame_own_block = bool(first_frame_own_block)
        self.rollout_mode = str(rollout_mode)
        self.usp_size = int(usp_size)
        if self.rollout_mode not in {FULL_RECOMPUTE, INCREMENTAL_KV}:
            raise ValueError(
                f"unsupported rollout_mode={self.rollout_mode!r}; "
                f"expected {FULL_RECOMPUTE!r} or {INCREMENTAL_KV!r}"
            )
        if self.rollout_mode == INCREMENTAL_KV and self.usp_size != 1:
            raise NotImplementedError(
                f"rollout_mode='incremental_kv' requires usp_size=1; got usp_size={self.usp_size}"
            )
        if self.frame_per_block < 1:
            raise ValueError(f"frame_per_block must be >= 1, got {self.frame_per_block}")
        if not self.first_frame_own_block:
            raise ValueError(
                "causal TI2V eval requires first_frame_own_block=true so generated "
                "blocks can attend the committed first-frame context"
            )
        patch_size = tuple(int(value) for value in self.transformer.config.patch_size)
        if patch_size[0] != 1:
            raise ValueError(
                "causal TI2V eval currently requires temporal patch size 1, "
                f"got patch_size={patch_size}"
            )

    def _denoise(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cache_name: str,
    ) -> torch.Tensor:
        with self.transformer.cache_context(cache_name):
            output = self.engine_forward(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                attention_kwargs=None,
                return_dict=False,
            )
        return output[0] if isinstance(output, tuple) else output

    def _denoise_kv(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        cache_name: str,
        *,
        kv_cache: Any,
        full_rotary_emb: tuple[torch.Tensor, torch.Tensor],
        kv_write_start: int,
        commit_kv: bool,
    ) -> torch.Tensor:
        """Keep incremental forwards inside the evaluation engine boundary."""
        with self.transformer.cache_context(cache_name):
            output = self.engine_forward(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                attention_kwargs=None,
                return_dict=False,
                kv_cache=kv_cache,
                full_rotary_emb=full_rotary_emb,
                kv_write_start=kv_write_start,
                commit_kv=commit_kv,
            )
        return output[0] if isinstance(output, tuple) else output

    def _call_incremental_kv(
        self,
        *,
        clean_full: torch.Tensor,
        initial_noise: torch.Tensor,
        condition: torch.Tensor | None,
        encoder_hidden_state: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        num_inference_steps: int,
        guidance_scale: float,
        transformer_dtype: torch.dtype,
        decode_output: bool,
    ) -> torch.Tensor:
        """Roll out one block at a time with temporary overwrite then clean commit."""
        patch_size = tuple(int(value) for value in self.transformer.config.patch_size)
        p_t, p_h, p_w = patch_size
        if p_t != 1:
            raise NotImplementedError(
                "rollout_mode='incremental_kv' requires temporal patch size p_t=1; "
                f"got patch_size={patch_size}"
            )
        if self.usp_size != 1:
            raise NotImplementedError(
                f"rollout_mode='incremental_kv' requires usp_size=1; got usp_size={self.usp_size}"
            )

        batch_size = int(clean_full.shape[0])
        device = clean_full.device
        tokens_per_frame = (clean_full.shape[3] // p_h) * (clean_full.shape[4] // p_w)
        total_tokens = clean_full.shape[2] * tokens_per_frame
        full_rotary_emb = self.transformer.full_clip_rotary_emb(clean_full.to(transformer_dtype))
        use_cfg = guidance_scale > 1 and negative_prompt_embeds is not None
        guidance_caches = allocate_incremental_kv_guidance_caches(
            self.transformer,
            batch_size=batch_size,
            num_cache_tokens=total_tokens,
            device=device,
            dtype=transformer_dtype,
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
            encoder_hidden_states: torch.Tensor,
            cache_name: str,
            cache: Any,
            commit: bool,
        ) -> torch.Tensor:
            hidden_states = block_latents.to(transformer_dtype)
            if condition is not None:
                hidden_states = torch.cat(
                    [hidden_states, condition[:, :, start:end]],
                    dim=1,
                )
            block_tokens = (end - start) * tokens_per_frame
            timestep_value = torch.as_tensor(scheduler_timestep, device=device).reshape(())
            timestep = torch.full(
                (batch_size, block_tokens),
                float(timestep_value),
                device=device,
                dtype=torch.float32,
            )
            return self._denoise_kv(
                hidden_states,
                timestep,
                encoder_hidden_states,
                cache_name,
                kv_cache=cache,
                full_rotary_emb=full_rotary_emb,
                kv_write_start=start * tokens_per_frame,
                commit_kv=commit,
            )

        layout = self.block_layout(clean_full.shape[2])
        first_start, first_end = layout[0]
        if (first_start, first_end) != (0, 1):
            raise RuntimeError(
                "incremental_kv requires the committed first frame to be block (0, 1); "
                f"got {(first_start, first_end)}"
            )
        run_block(
            clean_full[:, :, :1],
            start=0,
            end=1,
            scheduler_timestep=0.0,
            encoder_hidden_states=encoder_hidden_state,
            cache_name="cond",
            cache=cond_cache,
            commit=True,
        )
        if use_cfg:
            run_block(
                clean_full[:, :, :1],
                start=0,
                end=1,
                scheduler_timestep=0.0,
                encoder_hidden_states=negative_prompt_embeds,
                cache_name="uncond",
                cache=uncond_cache,
                commit=True,
            )

        for block_start, block_end in layout[1:]:
            block_latents = initial_noise[:, :, block_start:block_end].clone()
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            for scheduler_timestep in self.scheduler.timesteps:
                noise_pred = run_block(
                    block_latents,
                    start=block_start,
                    end=block_end,
                    scheduler_timestep=scheduler_timestep,
                    encoder_hidden_states=encoder_hidden_state,
                    cache_name="cond",
                    cache=cond_cache,
                    commit=False,
                )
                if use_cfg:
                    noise_uncond = run_block(
                        block_latents,
                        start=block_start,
                        end=block_end,
                        scheduler_timestep=scheduler_timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        cache_name="uncond",
                        cache=uncond_cache,
                        commit=False,
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
                start=block_start,
                end=block_end,
                scheduler_timestep=0.0,
                encoder_hidden_states=encoder_hidden_state,
                cache_name="cond",
                cache=cond_cache,
                commit=True,
            )
            if use_cfg:
                run_block(
                    block_latents,
                    start=block_start,
                    end=block_end,
                    scheduler_timestep=0.0,
                    encoder_hidden_states=negative_prompt_embeds,
                    cache_name="uncond",
                    cache=uncond_cache,
                    commit=True,
                )
            clean_full[:, :, block_start:block_end] = block_latents

        if not decode_output:
            return clean_full
        return self.decode_latents(clean_full)

    def block_layout(self, num_latent_frames: int) -> tuple[tuple[int, int], ...]:
        patch_size = tuple(int(value) for value in self.transformer.config.patch_size)
        p_t = patch_size[0]
        if num_latent_frames % p_t:
            raise ValueError(
                f"latent frame count {num_latent_frames} must divide by temporal patch size {p_t}"
            )
        return build_causal_block_layout(
            num_latent_frames // p_t,
            frame_per_block=self.frame_per_block,
            first_frame_own_block=self.first_frame_own_block,
        )

    @torch.no_grad()
    def call_with_latent(
        self,
        encoder_hidden_state: torch.Tensor,
        *,
        y: torch.Tensor | None,
        first_frame_latents: torch.Tensor,
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
        """Roll out generated history; target/validation latents are never inputs."""
        device = encoder_hidden_state.device
        batch_size = int(encoder_hidden_state.shape[0])
        transformer_dtype = self.transformer.dtype
        num_frames = self.align_num_frames(num_frames)

        condition_channels = y.shape[1] if y is not None else 0
        noise_channels = self.transformer.config.in_channels - condition_channels
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
        z0 = first_frame_latents.to(device=device, dtype=clean_full.dtype)
        if z0.shape[2] != 1:
            raise ValueError(
                f"first_frame_latents must contain exactly one frame, got shape {tuple(z0.shape)}"
            )
        clean_full[:, :, 0:1] = z0

        if y is not None:
            y = y.to(device=device, dtype=transformer_dtype)
            if y.shape[2] != clean_full.shape[2]:
                raise ValueError(
                    "condition and generated latent must have the same frame count; "
                    f"got {y.shape[2]} and {clean_full.shape[2]}"
                )
        encoder_hidden_state = encoder_hidden_state.to(device)
        if guidance_scale > 1:
            if negative_prompt_embeds is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    device=device,
                    dtype=self.dtype,
                )
            else:
                negative_prompt_embeds = self.encode_negative_prompt(
                    batch_size,
                    device,
                    self.dtype,
                    text="",
                )
        else:
            negative_prompt_embeds = None

        if self.rollout_mode == INCREMENTAL_KV:
            return self._call_incremental_kv(
                clean_full=clean_full,
                initial_noise=initial_noise,
                condition=y,
                encoder_hidden_state=encoder_hidden_state,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                transformer_dtype=transformer_dtype,
                decode_output=decode_output,
            )

        patch_size = tuple(int(value) for value in self.transformer.config.patch_size)
        p_t = patch_size[0]
        for block_start, block_end in self.block_layout(clean_full.shape[2]):
            # Frame 0 is committed from z0 before rollout. When it shares a
            # configured block, generate only the remaining frames.
            active_start = max(block_start, 1)
            if active_start >= block_end:
                continue
            latent_start = active_start * p_t
            latent_end = block_end * p_t
            block_latents = initial_noise[:, :, latent_start:latent_end].clone()

            # FlowMatchEulerDiscreteScheduler tracks a step index, so each new
            # block resets it and traverses the complete schedule independently.
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            for scheduler_timestep in self.scheduler.timesteps:
                noisy_full = clean_full.clone()
                noisy_full[:, :, latent_start:latent_end] = block_latents
                clean_input = clean_full.to(transformer_dtype)
                noisy_input = noisy_full.to(transformer_dtype)
                if y is not None:
                    clean_input = torch.cat([clean_input, y], dim=1)
                    noisy_input = torch.cat([noisy_input, y], dim=1)
                hidden_states = torch.cat([clean_input, noisy_input], dim=2)
                per_token_t = build_active_block_tf_per_token_timestep(
                    clean_full,
                    scheduler_timestep,
                    patch_size,
                    active_block=(active_start, block_end),
                )
                noise_pred = self._denoise(
                    hidden_states,
                    per_token_t,
                    encoder_hidden_state,
                    "cond",
                )[:, :, latent_start:latent_end]
                if guidance_scale > 1:
                    noise_uncond = self._denoise(
                        hidden_states,
                        per_token_t,
                        negative_prompt_embeds,
                        "uncond",
                    )[:, :, latent_start:latent_end]
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                block_latents = self.scheduler.step(
                    noise_pred,
                    scheduler_timestep,
                    block_latents,
                    return_dict=False,
                )[0]

            clean_full[:, :, latent_start:latent_end] = block_latents

        if not decode_output:
            return clean_full
        return self.decode_latents(clean_full)


@dataclass
class Wan22CausalInferencePipeline:
    """Frozen TI2V assets plus block-causal rollout metadata."""

    model: nn.Module
    encode_condition: Callable[[dict[str, Any]], dict[str, Any]]
    pipeline: Wan22Ti2v5bCausalPipeline
    stream_fields: Mapping[str, str]
    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT

    @torch.no_grad()
    def infer_sample(self, sample: dict[str, Any], *, seed: int) -> dict[str, Any]:
        from vforge.models.wan.v22.ti2v_5b.ct.inference import (
            _concat_condition_streams,
            _resolve_pixel_grid,
            _to_device,
        )

        device = next(self.model.parameters()).device
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        encoded = sample.get("__encoded__")
        if encoded is None:
            encoded = self.encode_condition(sample)
        prompt = encoded["prompt"].to(device)
        condition = _concat_condition_streams(encoded, self.stream_fields, device)
        first_frame_latents = _to_device(encoded.get("first_frame_latents"), device)
        if first_frame_latents is None:
            raise ValueError("Wan22 causal sampling requires first_frame_latents")
        height, width, num_frames = _resolve_pixel_grid(sample, encoded, self.pipeline)

        negative = None
        if self.guidance_scale > 1:
            negative = sample.get("__negative_prompt_embeds__")
            if negative is None:
                negative = self.pipeline.encode_negative_prompt(
                    batch_size=int(prompt.shape[0]),
                    device=device,
                    dtype=prompt.dtype,
                    text=self.negative_prompt,
                )
            else:
                negative = negative.to(device)

        from vforge.distributed.parallel_state import get_parallel_state

        is_output_leader = get_parallel_state().is_batch_leader
        video = self.pipeline.call_with_latent(
            prompt,
            y=condition,
            first_frame_latents=first_frame_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            negative_prompt_embeds=negative,
            decode_output=is_output_leader,
        )
        num_latent_frames = (self.pipeline.align_num_frames(num_frames) - 1) // (
            self.pipeline.vae_scale_factor_temporal
        ) + 1
        layout = self.pipeline.block_layout(num_latent_frames)
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
                    "ranges": [list(block) for block in layout],
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
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> Wan22CausalInferencePipeline:
    pipeline = Wan22Ti2v5bCausalPipeline(
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
    )
    return Wan22CausalInferencePipeline(
        model=model,
        encode_condition=encode_condition,
        pipeline=pipeline,
        stream_fields=stream_fields,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
    )


__all__ = [
    "FULL_RECOMPUTE",
    "INCREMENTAL_KV",
    "Wan22CausalInferencePipeline",
    "Wan22Ti2v5bCausalPipeline",
    "build_active_block_tf_per_token_timestep",
    "build_causal_block_layout",
    "build_inference_pipeline",
    "build_tf_per_token_timestep",
]
