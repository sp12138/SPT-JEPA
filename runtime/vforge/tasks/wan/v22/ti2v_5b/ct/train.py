"""Wan 2.2 TI2V training recipe: dataloader, weight widening, step semantics."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from vforge.data.cache import EncodedBatchCodec
from vforge.data.collate import build_raw_collate_fn
from vforge.data.loader import broadcast_batch
from vforge.data.preprocess import preprocess_deferred_batch
from vforge.distributed.parallel_state import get_parallel_state
from vforge.models.loading import PretrainedWeightSource
from vforge.models.wan.model import WanChannelLayout
from vforge.tasks.requests import TrainRequest
from vforge.tasks.wan.pretrained import (
    condition_tail_initializer,
    patch_embedding_widener,
)
from vforge.tasks.wan.shared import (
    build_state_dict_mapper,
    build_train_dataloader,
    parameter_storage_dtype,
)
from vforge.tasks.wan.v22.ti2v_5b.shared import (
    CACHE_FLAVOR,
    PARTIAL_NORMAL_SOURCES,
    _build_condition_encoder,
    _build_wan22_model,
    _codec,
    _collate_encoded_bundles,
    _condition_streams,
    _partial_normal_aug,
    _resolve_model_config,
    _stream_drop_probs,
    _stream_fields,
    _validate,
    stream_field,
)
from vforge.training.recipe import StepContext, StepOutput, TrainComponents
from vforge.utils.training import (
    build_optimizer_factory,
    build_scheduler_factory,
    model_patch_size,
    patched_token_count,
    resolve_dtype,
)

logger = logging.getLogger(__name__)


def build_train_components(request: TrainRequest) -> TrainComponents:
    _validate(
        extras=request.extras,
        latent_cache_mode=request.latent_cache_mode,
        usp_size=request.usp_size,
    )
    codec = _codec(request.extras)
    config, layout = _resolve_model_config(request.extras, request.pretrained_path)
    model = _build_wan22_model(config=config, init_device=request.init_device)
    dataloader, dataloader_resume_contract = _build_train_dataloader(request, codec)
    processor = _ConditionProcessor(
        extras=request.extras,
        precision=request.precision,
    )
    encoder = None
    if request.latent_cache_mode == "online":
        state = get_parallel_state()
        if state.config.usp_size <= 1 or state.usp_rank == 0:
            encoder = _build_condition_encoder(
                extras=request.extras,
                pretrained_path=request.pretrained_path,
                precision=request.precision,
                require_assets=True,
            )

    def step_fn(
        live_model: nn.Module,
        batch: Any,
        step: int,
        context: StepContext,
    ) -> StepOutput:
        return _wan22_train_step(
            live_model,
            batch,
            step,
            context,
            latent_cache_mode=request.latent_cache_mode,
            seed=request.seed,
            codec=codec,
            processor=processor,
            encoder=encoder,
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
        pretrained_weights=_wan22_pretrained_source(
            layout=layout,
            pretrained_path=request.pretrained_path,
            weight_loading=request.weight_loading,
        ),
        state_dict_mapper=build_state_dict_mapper(),
        dataloader_resume_contract=dataloader_resume_contract,
    )


def _wan22_train_step(
    model: nn.Module,
    batch: Any,
    step: int,
    context: StepContext,
    *,
    latent_cache_mode: str,
    seed: int,
    codec: EncodedBatchCodec,
    processor: _ConditionProcessor,
    encoder: Any | None,
) -> StepOutput:
    usp_size = int(context.parallel_state.config.usp_size)
    usp_rank = int(context.parallel_state.usp_rank)
    usp_group = context.parallel_state.usp_group
    patch_size = model_patch_size(model)

    with torch.no_grad():
        if usp_size <= 1 or usp_rank == 0:
            if latent_cache_mode == "offline":
                encoded = codec.canonicalize(batch)
                for key in ("min_latents", "max_latents"):
                    if key in batch and batch[key] is not None:
                        encoded[key] = batch[key]
                if encoder is not None:
                    encoded = encoder.apply_training_dropouts(encoded)
            else:
                assert encoder is not None
                encoder.eval()
                # A no-op unless the source deferred its resize to this rank.
                preprocess_deferred_batch(batch, device=context.device)
                encoded = codec.canonicalize(encoder.encode_batch(batch))
                encoded = encoder.apply_training_dropouts(encoded)
            generator = context.random_streams.generator(
                "wan.input", seed=seed, device=context.device
            )
            processed = processor.process(
                encoded,
                step=step,
                device=context.device,
                generator=generator,
                apply_dropouts=encoder is None,
                patch_size=patch_size,
                length_generator=context.random_streams.generator(
                    "wan.length", seed=seed, device=context.device
                ),
            )
        else:
            processed = None

        if usp_size > 1:
            src = 0 if usp_group is None else int(torch.distributed.get_global_rank(usp_group, 0))
            processed = broadcast_batch(
                processed,
                src_rank=src,
                group=usp_group,
                group_rank=usp_rank,
                target_device=context.device,
            )

    assert processed is not None
    output = model(**processed["model_inputs"])
    if isinstance(output, tuple):
        output = output[0]
    loss = torch.mean(processed["loss_weights"] * (output - processed["targets"]) ** 2)
    loss_for_backward = loss / usp_size if usp_size > 1 else loss

    return StepOutput(
        loss=loss_for_backward,
        metrics={"loss_scalar": loss.detach()},
        workload={
            "local_tokens": patched_token_count(processed["targets"], patch_size),
            "local_batch": int(processed["targets"].shape[0]),
        },
    )


class _ConditionProcessor:
    """Per-step noising, first-frame in-painting and flow-matching targets."""

    def __init__(self, *, extras: Mapping[str, Any], precision: str) -> None:
        self._noise_shift = float(extras.get("noise_shift", 5.0))
        self._weight_mode = str(extras.get("weighting_scheme", "none"))
        self._dtype = resolve_dtype(precision)
        self._text_drop_prob = float(extras.get("text_drop_prob", 0.0))
        self._stream_fields = _stream_fields(extras)
        self._streams = tuple(self._stream_fields)
        self._stream_drop_probs = _stream_drop_probs(extras)
        self._partial_normal_aug = _partial_normal_aug(extras)
        self._exclude_first_frame_from_loss = bool(
            extras.get("exclude_first_frame_from_loss", True)
        )
        self._patch_size = tuple(int(value) for value in extras.get("patch_size", (1, 2, 2)))
        self._scheduler = None
        self._compute_loss_weighting = None

    def _ensure_diffusion_helpers(self) -> None:
        if self._scheduler is not None:
            return
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from diffusers.training_utils import compute_loss_weighting_for_sd3

        self._scheduler = FlowMatchEulerDiscreteScheduler(shift=self._noise_shift)
        self._compute_loss_weighting = compute_loss_weighting_for_sd3

    def process(
        self,
        encoded: dict[str, Any],
        *,
        step: int,
        device: torch.device,
        generator: torch.Generator | None = None,
        apply_dropouts: bool = True,
        patch_size: tuple[int, ...] | None = None,
        length_generator: torch.Generator | None = None,
    ) -> dict[str, Any]:
        self._ensure_diffusion_helpers()
        assert self._scheduler is not None
        assert self._compute_loss_weighting is not None
        enc = dict(encoded)
        if apply_dropouts:
            enc = self._apply_dropouts(enc)
        enc = self._maybe_random_truncate(enc, device=device, generator=length_generator)

        frame_latents = enc["frame_latents"].to(device=device)
        conditions = [
            self._require_stream(enc, name).to(device=device, dtype=self._dtype)
            for name in self._streams
        ]
        first_frame_latents = enc.get("first_frame_latents")
        if first_frame_latents is None:
            raise ValueError(
                "Wan22 TI2V training requires first_frame_latents; the batch "
                f"came from a {CACHE_FLAVOR!r}-incompatible cache"
            )
        first_frame_latents = first_frame_latents.to(device=device)

        time_step_u = torch.rand(1, device=device, generator=generator)
        indices = (time_step_u.cpu() * self._scheduler.config.num_train_timesteps).long()[0]
        timestep = self._scheduler.timesteps[indices].to(device)
        noise = torch.randn(
            frame_latents.shape,
            device=device,
            dtype=frame_latents.dtype,
            generator=generator,
        )
        sigmas = self._scheduler.sigmas[indices].to(device)
        noisy_latent = (1.0 - sigmas) * frame_latents + noise * sigmas
        # Temporal in-painting: latent frame 0 is the given reference frame.
        noisy_latent[:, :, 0:1] = first_frame_latents.to(noisy_latent.dtype)

        resolved_patch_size = tuple(int(value) for value in (patch_size or self._patch_size))
        per_token_timestep = self._assemble_timestep(
            noisy_latent=noisy_latent,
            timestep=timestep,
            patch_size=resolved_patch_size,
        )
        hidden = self._assemble_hidden_states(
            noisy_latent=noisy_latent,
            frame_latents=frame_latents,
            first_frame_latents=first_frame_latents,
            conditions=conditions,
        )
        loss_weights = self._build_loss_weights(
            frame_latents,
            self._compute_loss_weighting(self._weight_mode, sigmas),
        )
        return {
            "model_inputs": {
                "hidden_states": hidden,
                "timestep": per_token_timestep,
                "encoder_hidden_states": enc["prompt"].to(
                    device=device,
                    dtype=self._dtype,
                ),
            },
            "targets": (noise - frame_latents).to(dtype=self._dtype),
            "loss_weights": loss_weights,
        }

    def _maybe_random_truncate(
        self,
        enc: dict[str, Any],
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> dict[str, Any]:
        """Slice time-aligned latents to ``s`` in ``[min_latents, max_latents]``.

        Both bounds missing is a no-op. A separate persistent generator keeps
        length sampling independent of noise tensor sizes.
        ``first_frame_latents`` stays at a single frame.
        """

        def _scalar(value: Any) -> int | None:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return int(value.reshape(-1)[0].item())
            if isinstance(value, (list, tuple)):
                return int(value[0])
            return int(value)

        min_lat_cfg = _scalar(enc.get("min_latents"))
        max_lat_cfg = _scalar(enc.get("max_latents"))
        if min_lat_cfg is None and max_lat_cfg is None:
            return enc

        frame_latents = enc.get("frame_latents")
        if frame_latents is None or frame_latents.dim() < 3:
            return enc
        t_lat = int(frame_latents.shape[2])
        if t_lat <= 1:
            return enc

        min_lat = int(min_lat_cfg) if min_lat_cfg is not None else 1
        max_lat = int(max_lat_cfg) if max_lat_cfg is not None else t_lat
        if min_lat < 1:
            min_lat = 1
        if max_lat > t_lat:
            max_lat = t_lat
        if min_lat > max_lat:
            raise ValueError(
                f"batch min_latents ({min_lat_cfg}) must be <= "
                f"max_latents ({max_lat_cfg}) after clamping to T_lat={t_lat}."
            )
        if min_lat == t_lat and max_lat == t_lat:
            return enc

        length = int(
            torch.randint(
                min_lat,
                max_lat + 1,
                (1,),
                device=device,
                generator=generator,
            ).item()
        )
        if length >= t_lat:
            return enc

        truncated: dict[str, Any] = {}
        for key, value in enc.items():
            if key == "first_frame_latents":
                truncated[key] = value
                continue
            if isinstance(value, torch.Tensor) and value.dim() >= 3 and value.shape[2] == t_lat:
                truncated[key] = value[:, :, :length].contiguous()
            else:
                truncated[key] = value
        return truncated

    def _assemble_hidden_states(
        self,
        *,
        noisy_latent: torch.Tensor,
        frame_latents: torch.Tensor,
        first_frame_latents: torch.Tensor,
        conditions: list[torch.Tensor],
    ) -> torch.Tensor:
        """Channel-concatenate the conditions onto the noisy latent.

        Split out so the teacher-forcing recipe can build a doubled
        ``[clean | noisy]`` sequence off the same sampled noise.
        """
        del frame_latents, first_frame_latents  # the doubled recipe needs them
        return torch.cat([noisy_latent.to(dtype=self._dtype), *conditions], dim=1)

    def _assemble_timestep(
        self,
        *,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        patch_size: tuple[int, ...],
    ) -> torch.Tensor:
        from vforge.models.wan.v22.ti2v_5b.ct.inference import build_per_token_timestep

        return build_per_token_timestep(noisy_latent, timestep, patch_size)

    def _require_stream(self, encoded: dict[str, Any], name: str) -> torch.Tensor:
        block = encoded.get(stream_field(name))
        if block is None:
            raise ValueError(
                f"Wan22 TI2V training with condition_streams={list(self._streams)} "
                f"needs the {name!r} stream; the batch carries no "
                f"{stream_field(name)!r}"
            )
        return block

    def _build_loss_weights(
        self,
        frame_latents: torch.Tensor,
        base_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast the sigma weight to latent shape, zeroing the given frame.

        Frame 0 holds the in-painted reference, so it is not a denoising
        prediction and must not pull on the loss.
        """
        batch, channels, t_lat, h_lat, w_lat = frame_latents.shape
        weight = base_weight.to(frame_latents.device, dtype=torch.float32).reshape(-1)
        if weight.numel() == 1:
            weight = weight.expand(batch)
        weight = weight.view(batch, 1, 1, 1, 1).expand(batch, channels, t_lat, h_lat, w_lat).clone()
        if self._exclude_first_frame_from_loss and t_lat >= 1:
            weight[:, :, 0:1] = 0.0
        return weight

    def _apply_dropouts(self, bundle: dict[str, Any]) -> dict[str, Any]:
        from vforge.models.wan.v22.ti2v_5b.ct.encoder import apply_encoded_training_dropouts

        return apply_encoded_training_dropouts(
            bundle,
            stream_fields=self._stream_fields,
            stream_drop_probs=self._stream_drop_probs,
            text_drop_prob=self._text_drop_prob,
            partial_normal_aug=self._partial_normal_aug,
            partial_normal_sources=PARTIAL_NORMAL_SOURCES,
        )


def _build_train_dataloader(
    request: TrainRequest,
    codec: EncodedBatchCodec,
) -> tuple[Any, dict[str, Any]]:
    return build_train_dataloader(
        request,
        codec=codec,
        collate_encoded=lambda samples: _collate_encoded_bundles(codec, samples),
        make_raw_collate=lambda: build_raw_collate_fn(
            extras=request.extras,
            pretrained_path=request.pretrained_path,
        ),
        preprocess_fingerprint={
            "max_frames": request.extras.get("max_frames"),
            "max_sequence_length": request.extras.get("max_sequence_length", 512),
            "condition_streams": list(_condition_streams(request.extras)),
            "cache_flavor": CACHE_FLAVOR,
            "model_revision": request.pretrained_path,
        },
    )


def _widen_patch_embedding(layout: WanChannelLayout) -> Any:
    """Copy the published conv weights into the widened condition-concat conv.

    The published TI2V checkpoint only knows the noisy-latent channels. The new
    tail owns the condition streams and starts at zero so the pretrained
    generation prior is untouched at step 0 and the conditions are learned in.
    """
    return patch_embedding_widener(layout, log_label="wan22 ti2v widen")


def _zero_condition_input_channels(layout: WanChannelLayout):
    return condition_tail_initializer(
        layout,
        transformer_of=lambda model: getattr(model, "transformer", model),
    )


def _wan22_pretrained_source(
    *,
    layout: WanChannelLayout,
    pretrained_path: str,
    weight_loading: str,
) -> PretrainedWeightSource:
    from vforge.models.wan.v22.ti2v_5b.ct.transformer import WanTransformer3DModel

    dtype = parameter_storage_dtype()
    resolve = _widen_patch_embedding(layout)

    def fallback_factory() -> nn.Module:
        # Load at the checkpoint's own width. Widening happens in
        # ``resolve_shape_mismatch``, which keeps the published conv weights;
        # rebuilding here with ignore_mismatched_sizes would discard them.
        return WanTransformer3DModel.from_pretrained(
            pretrained_path,
            subfolder="transformer",
            torch_dtype=dtype,
        )

    return PretrainedWeightSource(
        path=pretrained_path,
        subfolder="transformer",
        model_prefix="transformer.",
        loading_mode=weight_loading,
        dtype=dtype,
        initialize_missing=_zero_condition_input_channels(layout),
        resolve_shape_mismatch=resolve,
        fallback_model_factory=fallback_factory,
    )


__all__ = ["build_train_components"]
