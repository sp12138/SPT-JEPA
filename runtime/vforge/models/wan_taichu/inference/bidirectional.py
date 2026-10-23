"""Standalone bidirectional flow-matching inference for Wan Taichu TI2V."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from vforge.models.video_diffusion.flow_match import (
    build_first_frame_per_token_timestep,
    run_flow_match_denoising,
)
from vforge.models.video_diffusion.wan import (
    WAN_DEFAULT_NEGATIVE_PROMPT,
    WanFlowMatchPipelineBase,
)
from vforge.utils.inference_trace import (
    get_inference_trace,
    install_inference_module_trace,
)

DEFAULT_NEGATIVE_PROMPT = WAN_DEFAULT_NEGATIVE_PROMPT
# Taichu uses one generator for online-ref selection and initial latent noise.
# This private sample field carries the post-selection RNG state into inference.
WAN_TAICHU_GENERATOR_STATE_FIELD = "__wan_taichu_generator_state__"
build_per_token_timestep = build_first_frame_per_token_timestep


@torch.no_grad()
def encode_inference_batch(
    encoder: Any,
    batch: Mapping[str, Any],
    *,
    streams: Sequence[str],
    stream_fields: Mapping[str, str],
) -> dict[str, Any]:
    """Adapt the stable Wan Taichu encoder primitives to condition-only inference."""
    native_encode = getattr(encoder, "encode_inference_batch", None)
    if callable(native_encode):
        result = dict(native_encode(batch))
        for name in streams:
            expected = stream_fields[name]
            native = f"condition_latents_{name}"
            if expected != native and native in result:
                result[expected] = result.pop(native)
        return result
    vae = getattr(encoder, "vae", None)
    if vae is None:
        raise RuntimeError("Wan Taichu inference condition encoding requires a VAE")
    device = next(vae.parameters()).device
    dtype = encoder.training_dtype
    image = batch["image_condition"].to(device=device, dtype=dtype)
    if image.ndim != 5 or int(image.shape[1]) != 1:
        raise ValueError(
            f"Wan Taichu image_condition must be [B,1,C,H,W], got {tuple(image.shape)}"
        )
    text_inputs = batch["text_inputs"].to(device).long()
    text_mask = batch["text_mask"].to(device).long()
    prompt = encoder.encode_prompt(text_inputs, text_mask, int(image.shape[0]))
    first_pixels = encoder.scale_pixels(image).to(device=device, dtype=dtype)
    result: dict[str, Any] = {
        "prompt": prompt,
        "first_frame_latents": encoder.encode_image(first_pixels.permute(0, 2, 1, 3, 4)),
    }

    reference = batch.get("reference_condition")
    if reference is not None:
        result["ref_frame_latents"] = _encode_reference_frames(
            encoder,
            reference.to(device=device, dtype=dtype),
        )

    frame_count = int(batch["num_frames"])
    raw_keys = dict(getattr(encoder, "raw_stream_keys", {}))
    for name in streams:
        if name == "rgb":
            tail = first_pixels.new_zeros(
                (first_pixels.shape[0], frame_count - 1, *first_pixels.shape[2:])
            )
            pixels = torch.cat([first_pixels, tail], dim=1)
        else:
            raw_key = raw_keys.get(name)
            raw = batch.get(raw_key) if raw_key is not None else None
            if not isinstance(raw, torch.Tensor):
                raise ValueError(
                    f"Wan Taichu inference requires stream {name!r} in raw field {raw_key!r}"
                )
            pixels = encoder.scale_pixels(raw).to(device=device, dtype=dtype)
        result[stream_fields[name]] = encoder.encode_image(pixels.permute(0, 2, 1, 3, 4))

    for raw_field, latent_field in (
        ("normal_fg_condition", "normal_fg_body"),
        ("normal_bg_condition", "normal_bg_body"),
    ):
        raw = batch.get(raw_field)
        if isinstance(raw, torch.Tensor):
            pixels = encoder.scale_pixels(raw).to(device=device, dtype=dtype)
            result[latent_field] = encoder.encode_image(pixels.permute(0, 2, 1, 3, 4))
    return result


def _encode_reference_frames(encoder: Any, reference: torch.Tensor) -> torch.Tensor:
    """Encode every reference frame as an independent one-frame VAE sample."""
    if reference.ndim != 5:
        raise ValueError(f"reference_condition must be [B,K,C,H,W], got {tuple(reference.shape)}")
    batch, count, channels, height, width = reference.shape
    if count < 1:
        raise ValueError("reference_condition must contain at least one frame")
    scaled = encoder.scale_pixels(reference)
    flattened = scaled.reshape(batch * count, channels, 1, height, width)
    encoded = encoder.encode_image(flattened)
    if encoded.ndim != 5 or int(encoded.shape[0]) != batch * count or int(encoded.shape[2]) != 1:
        raise ValueError(
            f"independent reference encoding must return [B*K,C,1,H,W], got {tuple(encoded.shape)}"
        )
    return (
        encoded.reshape(batch, count, encoded.shape[1], encoded.shape[2], *encoded.shape[3:])
        .squeeze(3)
        .permute(0, 2, 1, 3, 4)
        .contiguous(memory_format=torch.channels_last_3d)
    )


class WanTaichuBidirectionalPipeline(WanFlowMatchPipelineBase):
    """Bidirectional scheduler loop over a single or dual-expert WanTaichuModel."""

    def __init__(
        self,
        *,
        tokenizer: Any,
        text_encoder: Any,
        vae: Any,
        transformer: nn.Module,
        engine_forward: Callable[..., Any],
        scheduler: Any,
        first_frame_inpaint: bool = True,
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
        self.first_frame_inpaint = bool(first_frame_inpaint)

    def _denoise(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        cache_name: str,
        ref_hidden_states: torch.Tensor | None,
        global_has_ref: bool,
        image_conditions: Mapping[str, torch.Tensor | None],
    ) -> torch.Tensor:
        cache_context = getattr(self.transformer, "cache_context", None)
        context = cache_context(cache_name) if callable(cache_context) else nullcontext()
        with context:
            kwargs = dict(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                ref_hidden_states=ref_hidden_states,
                global_has_ref=global_has_ref,
                attention_kwargs=None,
                return_dict=False,
            )
            if not self.first_frame_inpaint:
                kwargs.update(image_conditions)
            output = self.engine_forward(**kwargs)
        return output[0] if isinstance(output, tuple) else output

    @torch.no_grad()
    def call_with_latent(
        self,
        encoder_hidden_states: torch.Tensor,
        *,
        condition: torch.Tensor | None,
        first_frame_latents: torch.Tensor | None,
        encoder_hidden_states_image_vae: torch.Tensor | None = None,
        encoder_hidden_states_image_clip: torch.Tensor | None = None,
        encoder_hidden_states_image_dino: torch.Tensor | None = None,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        guidance_scale: float,
        generator: torch.Generator | None,
        negative_prompt_embeds: torch.Tensor | None,
        ref_hidden_states: torch.Tensor | None = None,
        latents: torch.Tensor | None = None,
        decode_output: bool = True,
    ) -> torch.Tensor:
        device = encoder_hidden_states.device
        model_dtype = encoder_hidden_states.dtype
        aligned_frames = ((num_frames - 1) // self.vae_scale_factor_temporal) * (
            self.vae_scale_factor_temporal
        ) + 1
        config = self.transformer.config
        condition_channels = 0 if condition is None else int(condition.shape[1])
        noise_channels = int(config.in_channels) - condition_channels
        latents = self.prepare_latents(
            int(encoder_hidden_states.shape[0]),
            noise_channels,
            height,
            width,
            aligned_frames,
            torch.float32,
            device,
            generator=generator,
            latents=latents,
        )
        first = None
        if self.first_frame_inpaint:
            if not isinstance(first_frame_latents, torch.Tensor):
                raise ValueError("Wan Taichu TI2V inference requires first_frame_latents")
            first = first_frame_latents.to(device=device, dtype=latents.dtype)
            latents[:, :, 0:1] = first
        condition = None if condition is None else condition.to(device=device, dtype=model_dtype)
        reference = (
            None
            if ref_hidden_states is None
            else ref_hidden_states.to(device=device, dtype=model_dtype)
        )
        global_has_ref = _global_ref_present(reference, device=device)
        patch_size = tuple(int(value) for value in config.patch_size)
        image_conditions = {
            "encoder_hidden_states_image_vae": _to_device(
                encoder_hidden_states_image_vae, device, model_dtype
            ),
            "encoder_hidden_states_image_clip": _to_device(
                encoder_hidden_states_image_clip, device, model_dtype
            ),
            "encoder_hidden_states_image_dino": _to_device(
                encoder_hidden_states_image_dino, device, model_dtype
            ),
        }
        trace = get_inference_trace()
        install_inference_module_trace(self.transformer)
        trace.tensor("input.prompt", encoder_hidden_states)
        trace.tensor("input.condition", condition)
        trace.tensor("input.negative_prompt", negative_prompt_embeds)
        trace.tensor("initial.latents", latents)
        if os.environ.get("WAN_TAICHU_INFERENCE_TRACE_INPUT_ONLY") == "1":
            return torch.zeros(
                (latents.shape[0], 3, 1, height, width),
                device=latents.device,
                dtype=model_dtype,
            )

        def trace_step(stage: str, step_index: int, value: torch.Tensor) -> None:
            metadata: dict[str, Any] = {}
            if (
                stage == "timestep"
                and getattr(self.transformer, "boundary_ratio", None) is not None
            ):
                boundary = float(self.transformer.boundary_timestep)
                metadata["expert"] = (
                    "wan" if float(value.detach().reshape(())) >= boundary else "wan_2"
                )
            trace.tensor(f"step.{step_index:03d}.{stage}", value, metadata)

        def predict(
            current_latents: torch.Tensor,
            timestep: torch.Tensor,
            text_condition: torch.Tensor,
            cache_name: str,
        ) -> torch.Tensor:
            latent_input = current_latents.to(dtype=model_dtype)
            hidden_states = (
                latent_input if condition is None else torch.cat([latent_input, condition], dim=1)
            )
            model_timestep = (
                build_per_token_timestep(current_latents, timestep, patch_size)
                if self.first_frame_inpaint
                else timestep.expand(current_latents.shape[0]).to(device)
            )
            return self._denoise(
                hidden_states,
                model_timestep,
                text_condition,
                cache_name=cache_name,
                ref_hidden_states=reference,
                global_has_ref=global_has_ref,
                image_conditions=image_conditions,
            )

        latents = run_flow_match_denoising(
            scheduler=self.scheduler,
            latents=latents,
            num_inference_steps=num_inference_steps,
            conditioning=encoder_hidden_states,
            predict=predict,
            guidance_scale=guidance_scale,
            negative_conditioning=negative_prompt_embeds,
            frozen_prefix_latents=first,
            trace=trace_step if trace.enabled else None,
        )
        trace.tensor("final.latents", latents)
        if not decode_output:
            return latents
        decoded = self._decode_latents_fp32(latents)
        trace.tensor("final.decoded", decoded)
        return decoded

    def _decode_latents_fp32(self, latents: torch.Tensor) -> torch.Tensor:
        """Match Taichu's FP32 Wan VAE decode contract."""
        if self.vae is None:
            raise RuntimeError("VAE decode is only available on output leaders")
        original_dtype = next(self.vae.parameters()).dtype
        if original_dtype != torch.float32:
            self.vae.to(dtype=torch.float32)
        try:
            return self.decode_latents(latents.to(torch.float32))
        finally:
            if original_dtype != torch.float32:
                self.vae.to(dtype=original_dtype)


@dataclass
class WanTaichuBidirectionalInference:
    """One-sample adapter consumed by the framework evaluation runner."""

    model: nn.Module
    encode_condition: Callable[[dict[str, Any]], dict[str, Any]]
    pipeline: WanTaichuBidirectionalPipeline
    stream_fields: Mapping[str, str]
    num_inference_steps: int
    guidance_scale: float
    negative_prompt: str

    def resolve_sample_seed(self, sample: Mapping[str, Any], base_seed: int) -> int:
        """Match Taichu's ``base seed + dataset index`` inference contract."""
        seed_offset = sample.get("__vforge_eval_seed_offset__")
        if seed_offset is None:
            seed_offset = sample.get("__vforge_eval_index__")
        if seed_offset is None:
            raise ValueError("Wan Taichu inference sample is missing its seed offset")
        return int(base_seed) + int(seed_offset)

    @torch.no_grad()
    def infer_sample(self, sample: dict[str, Any], *, seed: int) -> dict[str, Any]:
        device = next(self.model.parameters()).device
        generator = torch.Generator(device=device).manual_seed(int(seed))
        generator_state = sample.get(WAN_TAICHU_GENERATOR_STATE_FIELD)
        if generator_state is not None:
            if not isinstance(generator_state, torch.Tensor):
                raise TypeError(
                    f"{WAN_TAICHU_GENERATOR_STATE_FIELD} must be a generator-state tensor"
                )
            generator.set_state(generator_state.detach().cpu())
        encoded = sample.get("__encoded__")
        if encoded is None:
            encoded = self.encode_condition(sample)
        prompt = encoded["prompt"].to(device)
        condition = _concat_condition_streams(encoded, self.stream_fields, device)
        first = encoded.get("first_frame_latents")
        first_frame_inpaint = bool(getattr(self.pipeline, "first_frame_inpaint", True))
        if first_frame_inpaint and not isinstance(first, torch.Tensor):
            raise ValueError("Wan Taichu TI2V inference requires first_frame_latents")
        if first is not None and not isinstance(first, torch.Tensor):
            raise TypeError("first_frame_latents must be a tensor or None")
        reference = encoded.get("ref_frame_latents")
        if reference is not None and not isinstance(reference, torch.Tensor):
            raise TypeError("ref_frame_latents must be a tensor")
        negative = sample.get("__negative_prompt_embeds__")
        if self.guidance_scale > 1 and negative is None:
            negative = self.pipeline.encode_negative_prompt(
                int(prompt.shape[0]),
                device,
                prompt.dtype,
                text=self.negative_prompt,
            )
        if isinstance(negative, torch.Tensor):
            negative = negative.to(device)

        from vforge.distributed.parallel_state import get_parallel_state

        is_output_leader = get_parallel_state().is_batch_leader
        video = self.pipeline.call_with_latent(
            prompt,
            condition=condition,
            first_frame_latents=None if first is None else first.to(device),
            encoder_hidden_states_image_vae=_optional_tensor(
                encoded.get("image_condition_latent_vae"),
                "image_condition_latent_vae",
            ),
            encoder_hidden_states_image_clip=_optional_tensor(
                encoded.get("image_condition_latent_clip"),
                "image_condition_latent_clip",
            ),
            encoder_hidden_states_image_dino=_optional_tensor(
                encoded.get("image_condition_latent_dino"),
                "image_condition_latent_dino",
            ),
            height=int(sample["__height__"]),
            width=int(sample["__width__"]),
            num_frames=int(sample["__num_frames__"]),
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=generator,
            negative_prompt_embeds=negative,
            ref_hidden_states=None if reference is None else reference.to(device),
            decode_output=is_output_leader,
        )
        return {
            "video": video.detach().cpu() if is_output_leader else None,
            "sample_id": sample.get("sample_id"),
            "seed": int(seed),
            "metadata": {
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "height": int(sample["__height__"]),
                "width": int(sample["__width__"]),
                "num_frames": int(sample["__num_frames__"]),
            },
        }


def _concat_condition_streams(
    encoded: Mapping[str, Any],
    stream_fields: Mapping[str, str],
    device: torch.device,
) -> torch.Tensor | None:
    blocks = []
    for name, field in stream_fields.items():
        block = encoded.get(field)
        if not isinstance(block, torch.Tensor):
            raise ValueError(f"Wan Taichu inference requires {name!r} condition tensor {field!r}")
        blocks.append(block.to(device))
    if not blocks:
        return None
    return blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=1)


def _optional_tensor(value: Any, name: str) -> torch.Tensor | None:
    if value is None or isinstance(value, torch.Tensor):
        return value
    raise TypeError(f"{name} must be a tensor or None")


def _to_device(
    value: torch.Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    return None if value is None else value.to(device=device, dtype=dtype)


def _global_ref_present(
    ref_hidden_states: torch.Tensor | None,
    *,
    device: torch.device,
) -> bool:
    """OR reference presence across ranks that shard one FSDP model."""
    local_has_ref = ref_hidden_states is not None
    if not dist.is_available() or not dist.is_initialized():
        return local_has_ref
    from vforge.distributed.parallel_state import get_parallel_state

    flag = torch.tensor(int(local_has_ref), dtype=torch.int32, device=device)
    dist.all_reduce(
        flag,
        op=dist.ReduceOp.MAX,
        group=get_parallel_state().fsdp_group,
    )
    return bool(flag.item())


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
    first_frame_inpaint: bool = True,
    vae_scale_factor_spatial: int = 16,
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> WanTaichuBidirectionalInference:
    pipeline = WanTaichuBidirectionalPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=model,
        engine_forward=engine_forward,
        scheduler=scheduler,
        first_frame_inpaint=first_frame_inpaint,
        vae_scale_factor_spatial=vae_scale_factor_spatial,
    )
    return WanTaichuBidirectionalInference(
        model=model,
        encode_condition=encode_condition,
        pipeline=pipeline,
        stream_fields=dict(stream_fields),
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        negative_prompt=str(negative_prompt),
    )


__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "WAN_TAICHU_GENERATOR_STATE_FIELD",
    "WanTaichuBidirectionalInference",
    "WanTaichuBidirectionalPipeline",
    "build_inference_pipeline",
    "build_per_token_timestep",
    "encode_inference_batch",
]
