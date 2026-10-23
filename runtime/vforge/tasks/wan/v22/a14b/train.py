"""Wan 2.2 A14B training recipe: expert routing, timestep ranges, widening."""

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
from vforge.models.wan.v22.a14b.model import (
    EXPERT_SUBFOLDERS,
    HIGH_NOISE_EXPERT,
    LOW_NOISE_EXPERT,
)
from vforge.tasks.requests import TrainRequest
from vforge.tasks.wan.pretrained import (
    condition_tail_initializer,
    patch_embedding_widener,
)
from vforge.tasks.wan.shared import (
    build_train_dataloader,
    parameter_storage_dtype,
)
from vforge.tasks.wan.v21.shared import _collate_encoded_bundles
from vforge.tasks.wan.v21.train import _ConditionProcessor
from vforge.tasks.wan.v22.a14b.shared import (
    CACHE_FLAVOR,
    WAN22_A14B_ENCODED_BATCH_CODEC,
    _boundary_ratio,
    _build_a14b_model,
    _build_condition_encoder,
    _condition_spec,
    _condition_streams,
    _resolve_expert_configs,
    _train_expert,
    _training_expert_names,
    _validate,
)
from vforge.training.recipe import (
    OptimizerSpec,
    SchedulerSpec,
    StepContext,
    StepOutput,
    TrainComponents,
)
from vforge.utils.training import (
    build_optimizer_factory,
    build_scheduler_factory,
    model_patch_size,
    patched_token_count,
)

logger = logging.getLogger(__name__)


def build_optimizer_spec(request: TrainRequest) -> OptimizerSpec:
    """Describe the optimizer without constructing either tower."""
    return OptimizerSpec(
        learning_rate=request.learning_rate,
        weight_decay=request.weight_decay,
        betas=request.betas,
        eps=request.eps,
    )


def validate_train_request(request: TrainRequest, engine: str) -> None:
    """Reject model/engine combinations before distributed runtime startup."""
    if engine == "megatron" and _train_expert(request.extras) == "both":
        raise NotImplementedError(
            "the megatron engine does not support Wan2.2 A14B train_expert=both; "
            "set model.extras.train_expert=high or low"
        )


def build_train_components(request: TrainRequest) -> TrainComponents:
    _validate(
        extras=request.extras,
        latent_cache_mode=request.latent_cache_mode,
        usp_size=request.usp_size,
    )
    codec = WAN22_A14B_ENCODED_BATCH_CODEC
    expert_names = _training_expert_names(request.extras)
    expert_configs, layouts = _resolve_expert_configs(
        request.extras,
        request.pretrained_path,
        expert_names=expert_names,
    )
    model = _build_a14b_model(
        expert_configs=expert_configs,
        extras=request.extras,
        init_device=request.init_device,
        expert_names=expert_names,
    )
    dataloader, dataloader_resume_contract = _build_train_dataloader(request, codec)
    processor = _A14bConditionProcessor(extras=request.extras, precision=request.precision)
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
        return _a14b_train_step(
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
        optimizer_spec=build_optimizer_spec(request),
        step_fn=step_fn,
        scheduler_factory=build_scheduler_factory(
            name=request.scheduler_name,
            factor=request.scheduler_factor,
            warmup_steps=request.warmup_steps,
            start_factor=request.scheduler_start_factor,
            end_factor=request.scheduler_end_factor,
            total_iters=request.scheduler_total_iters,
        ),
        scheduler_spec=SchedulerSpec(
            name=request.scheduler_name,
            factor=request.scheduler_factor,
            warmup_steps=request.warmup_steps,
            start_factor=request.scheduler_start_factor,
            end_factor=request.scheduler_end_factor,
            total_iters=request.scheduler_total_iters,
        ),
        pretrained_weights=_a14b_pretrained_sources(
            layouts=layouts,
            pretrained_path=request.pretrained_path,
            weight_loading=request.weight_loading,
            expert_names=expert_names,
        ),
        # ``wan.`` / ``wan_2.`` is already the external key space, matching how
        # Taichu names the same two towers, so nothing needs rewriting.
        state_dict_mapper=None,
        dataloader_resume_contract=dataloader_resume_contract,
    )


def _a14b_train_step(
    model: nn.Module,
    batch: Any,
    step: int,
    context: StepContext,
    *,
    latent_cache_mode: str,
    seed: int,
    codec: EncodedBatchCodec,
    processor: _A14bConditionProcessor,
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

    metadata = {}
    if isinstance(batch, dict) and "sample_id" in batch:
        sample_ids = batch["sample_id"]
        metadata["sample_ids"] = (
            [sample_ids] if isinstance(sample_ids, str) else [str(v) for v in sample_ids]
        )
    return StepOutput(
        loss=loss_for_backward,
        metrics={"loss_scalar": loss.detach()},
        workload={
            "local_tokens": patched_token_count(processed["targets"], patch_size),
            "local_batch": int(processed["targets"].shape[0]),
        },
        metadata=metadata,
    )


class _A14bConditionProcessor(_ConditionProcessor):
    """Wan2.1 noising, restricted to the timestep range of one expert.

    A step trains exactly one tower, so the sampled timestep has to land on that
    tower's side of the boundary. With both experts trainable the two alternate
    by global step parity, which is identical on every rank and therefore keeps
    the ranks agreeing on which tower runs.
    """

    def __init__(self, *, extras: Mapping[str, Any], precision: str) -> None:
        super().__init__(
            extras=extras,
            precision=precision,
            condition=_condition_spec(extras),
        )
        self._boundary_ratio = _boundary_ratio(extras)
        self._train_expert = _train_expert(extras)

    def trains_high_noise_expert(self, step: int) -> bool:
        if self._train_expert == "both":
            return step % 2 == 0
        return self._train_expert == "high"

    def _sample_timestep_index(
        self,
        *,
        step: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        assert self._scheduler is not None
        timesteps = self._scheduler.timesteps
        boundary = self._boundary_ratio * self._scheduler.config.num_train_timesteps
        high_noise = self.trains_high_noise_expert(step)
        mask = timesteps >= boundary if high_noise else timesteps < boundary
        candidates = torch.nonzero(mask, as_tuple=False).flatten()
        if candidates.numel() == 0:
            side = "high" if high_noise else "low"
            raise ValueError(
                f"no {side}-noise timesteps exist for boundary_ratio="
                f"{self._boundary_ratio}; the expert has nothing to train on"
            )
        time_step_u = torch.rand(1, device=device, generator=generator)
        position = (
            (time_step_u.cpu() * candidates.numel()).long().clamp(max=candidates.numel() - 1)[0]
        )
        return candidates[position]


def _build_train_dataloader(
    request: TrainRequest,
    codec: EncodedBatchCodec,
) -> tuple[Any, dict[str, Any]]:
    return build_train_dataloader(
        request,
        codec=codec,
        collate_encoded=_collate_encoded_bundles,
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

    The published A14B checkpoint already knows the noisy latent plus its first
    condition stream. Any stream configured on top of that owns the new tail,
    which starts at zero so the pretrained prior is untouched at step 0.
    """
    return patch_embedding_widener(layout, log_label="wan22 a14b widen")


def _zero_condition_input_channels(layout: WanChannelLayout):
    return condition_tail_initializer(layout)


def _a14b_pretrained_sources(
    *,
    layouts: Mapping[str, WanChannelLayout],
    pretrained_path: str,
    weight_loading: str,
    expert_names: tuple[str, ...] = (HIGH_NOISE_EXPERT, LOW_NOISE_EXPERT),
) -> tuple[PretrainedWeightSource, ...]:
    """One source per expert, each narrowed to the tower it populates."""
    from vforge.models.wan.v21.transformer import WanTransformer3DModel

    dtype = parameter_storage_dtype()
    sources = []
    for name in expert_names:
        subfolder = EXPERT_SUBFOLDERS[name]
        layout = layouts[name]

        def fallback_factory(subfolder: str = subfolder) -> nn.Module:
            # Load at the checkpoint's own width. Widening happens in
            # ``resolve_shape_mismatch``, which keeps the published conv
            # weights; rebuilding here would discard them.
            return WanTransformer3DModel.from_pretrained(
                pretrained_path,
                subfolder=subfolder,
                torch_dtype=dtype,
                use_vae_condition=False,
            )

        sources.append(
            PretrainedWeightSource(
                path=pretrained_path,
                subfolder=subfolder,
                model_path=name,
                loading_mode=weight_loading,
                dtype=dtype,
                initialize_missing=_zero_condition_input_channels(layout),
                resolve_shape_mismatch=_widen_patch_embedding(layout),
                fallback_model_factory=fallback_factory,
            )
        )
    return tuple(sources)


__all__ = ["build_train_components"]
