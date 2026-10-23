"""Wan 2.2 TI2V teacher-forcing training recipe.

Data, conditioning, weight widening and loss are the bidirectional plugin's;
this module only changes what the transformer is asked to predict from. Two
things differ:

* ``hidden_states`` is a doubled ``[clean | noisy]`` sequence on the frame
  axis, both halves carrying the same condition streams;
* ``timestep`` is ``0`` across the clean half so the model is never asked to
  denoise the context it is being given.

The target and loss weights are untouched: the transformer still returns only
the noisy half, so a teacher-forcing step is scored exactly like a
bidirectional one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from vforge.distributed.parallel_state import get_parallel_state
from vforge.models.wan.v22.ti2v_5b.teacher_forcing.inference import build_tf_per_token_timestep
from vforge.tasks.requests import TrainRequest
from vforge.tasks.wan.shared import (
    build_state_dict_mapper,
    parameter_storage_dtype,
)
from vforge.tasks.wan.v22.ti2v_5b.ct.train import (
    _build_train_dataloader,
    _ConditionProcessor,
    _wan22_pretrained_source,
    _wan22_train_step,
)
from vforge.tasks.wan.v22.ti2v_5b.shared import (
    _build_condition_encoder,
    _codec,
    _require_meta_init,
    _resolve_model_config,
    _validate,
)
from vforge.training.recipe import StepContext, StepOutput, TrainComponents
from vforge.utils.training import build_optimizer_factory, build_scheduler_factory

logger = logging.getLogger(__name__)

DEFAULT_FRAME_PER_BLOCK = 3


def _frame_per_block(extras: Mapping[str, Any]) -> int:
    return int(extras.get("frame_per_block", DEFAULT_FRAME_PER_BLOCK))


def _first_frame_own_block(extras: Mapping[str, Any]) -> bool:
    return bool(extras.get("first_frame_own_block", True))


def build_train_components(request: TrainRequest) -> TrainComponents:
    _validate(
        extras=request.extras,
        latent_cache_mode=request.latent_cache_mode,
        usp_size=request.usp_size,
    )
    _validate_tf(extras=request.extras, sources=request.sources)
    codec = _codec(request.extras)
    config, layout = _resolve_model_config(request.extras, request.pretrained_path)
    model = _build_tf_model(
        config=config,
        extras=request.extras,
        init_device=request.init_device,
    )
    dataloader, dataloader_resume_contract = _build_train_dataloader(request, codec)
    processor = _TfConditionProcessor(
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


class _TfConditionProcessor(_ConditionProcessor):
    """Bidirectional noising, assembled into a doubled clean/noisy sequence."""

    def _assemble_hidden_states(
        self,
        *,
        noisy_latent: torch.Tensor,
        frame_latents: torch.Tensor,
        first_frame_latents: torch.Tensor,
        conditions: list[torch.Tensor],
    ) -> torch.Tensor:
        # The clean half is the ground truth with the same frame-0 in-painting
        # as the noisy half, so the two halves differ only by the noise.
        clean_latent = frame_latents.clone()
        clean_latent[:, :, 0:1] = first_frame_latents.to(clean_latent.dtype)
        clean = torch.cat([clean_latent.to(dtype=self._dtype), *conditions], dim=1)
        noisy = torch.cat([noisy_latent.to(dtype=self._dtype), *conditions], dim=1)
        return torch.cat([clean, noisy], dim=2)

    def _assemble_timestep(
        self,
        *,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        patch_size: tuple[int, ...],
    ) -> torch.Tensor:
        return build_tf_per_token_timestep(noisy_latent, timestep, patch_size)


def _validate_tf(*, extras: Mapping[str, Any], sources=()) -> None:
    frame_per_block = _frame_per_block(extras)
    if frame_per_block < 1:
        raise ValueError(f"model.extras.frame_per_block must be >= 1, got {frame_per_block}")
    for source in sources:
        if source.min_latents is not None or source.max_latents is not None:
            raise ValueError(
                "teacher forcing rejects per-source min_latents/max_latents; "
                "truncation would change T and break frame_per_block / SP divisibility"
            )


def _build_tf_model(
    *,
    config: Mapping[str, Any],
    extras: Mapping[str, Any],
    init_device: str,
) -> nn.Module:
    from vforge.models.wan.v22.ti2v_5b.teacher_forcing.model import Wan22Ti2v5bTfModel

    _require_meta_init(init_device)
    return Wan22Ti2v5bTfModel.build_empty(
        config,
        parameter_dtype=parameter_storage_dtype(),
        frame_per_block=_frame_per_block(extras),
        first_frame_own_block=_first_frame_own_block(extras),
    )


__all__ = ["build_train_components"]
