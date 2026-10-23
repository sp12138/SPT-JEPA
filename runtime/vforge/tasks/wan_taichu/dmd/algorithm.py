"""Taichu numerical policy over the shared distribution-matching phases."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import torch
import torch.nn as nn

from vforge.algorithms.dmd.algorithm import (
    CRITIC_RNG_SEED_OFFSET,
    DistributionMatchingAlgorithm,
    DistributionMatchingSteps,
    model_role,
)
from vforge.algorithms.dmd.config import (
    DistributionMatchingConfig,
    DmdRolloutConfig,
    DmdScoreConfig,
    config_section,
)
from vforge.algorithms.dmd.objectives import component_masks, matching_keys
from vforge.tasks.wan_taichu.dmd.runtime import build_wan_dmd_data
from vforge.training.diagnostics import torch_gradient_clipper
from vforge.training.recipe import StepContext, StepOutput


@dataclass(frozen=True)
class WanRolloutConfig(DmdRolloutConfig):
    context_full_trajectory: bool = False


@dataclass(frozen=True)
class WanScoreConfig(DmdScoreConfig):
    timestep_floor_at_rollout_exit: bool = False


class WanDistributionMatchingConfig(DistributionMatchingConfig):
    @classmethod
    def from_mapping(cls, value):
        rollout = config_section(value, "rollout")
        score = config_section(value, "score")
        full_context = rollout.pop("context_full_trajectory", False)
        floor_at_exit = score.pop("timestep_floor_at_rollout_exit", False)
        config = super().from_mapping({**value, "rollout": rollout, "score": score})
        return replace(
            config,
            rollout=WanRolloutConfig(**vars(config.rollout), context_full_trajectory=full_context),
            score=WanScoreConfig(
                **vars(config.score), timestep_floor_at_rollout_exit=floor_at_exit
            ),
        )


class WanDistributionMatchingAlgorithm(DistributionMatchingAlgorithm):
    """Keep phase/optimizer/EMA assembly shared; specialize only Taichu numerics."""

    config_type = WanDistributionMatchingConfig

    def build_components(self, request, *, integration):
        self.data_builder = partial(
            build_wan_dmd_data,
            random_seed=request.base_seed if request.num_workers == 0 else request.seed,
        )
        return replace(
            super().build_components(request, integration=integration),
            gradient_clipper_factory=torch_gradient_clipper,
        )

    @staticmethod
    def create_steps(**kwargs):
        return WanDistributionMatchingSteps(**kwargs)


class WanDistributionMatchingSteps(DistributionMatchingSteps):
    def generator_step(self, root, batch, step, context):
        with self.data.random_context(context.random_streams, context.device):
            return super().generator_step(root, batch, step, context)

    def critic_step(self, root, batch, step, context):
        with self.data.random_context(context.random_streams, context.device):
            return self._critic_step(root, batch, step, context)

    @staticmethod
    def predict_x0(adapter, tower, noisy, level, batch, conditioning):
        flow = adapter.predict_velocity(tower, noisy, level, batch, conditioning)
        # Taichu's score wrapper first narrows noisy latents to compute dtype,
        # then widens the x0 conversion and rounds its result to that dtype.
        return {
            key: (
                value.to(flow[key].dtype).double()
                - level.sigmas[key].to(value.device).double() * flow[key].double()
            ).to(flow[key].dtype)
            for key, value in noisy.items()
        }

    def _critic_step(
        self,
        root: nn.Module,
        batch: Any,
        step: int,
        context: StepContext,
    ) -> StepOutput:
        prepared = self.data.prepare_batch(batch, context.device)
        generator = model_role(root, "generator")
        generated = self.data.rollout(
            generator,
            self.generator_adapter,
            prepared,
            context.random_streams,
            "critic",
            CRITIC_RNG_SEED_OFFSET,
        )
        generated = {key: value.detach() for key, value in generated.items()}
        level, noisy, noise = self.data.sample_score_noise(
            generated,
            prepared,
            context.random_streams,
            CRITIC_RNG_SEED_OFFSET + 17,
        )
        fake_score = model_role(root, "fake_score")
        # The reference regresses velocity reconstructed from rounded x0.
        # Replacing this with the tower's direct velocity changes BF16 loss.
        x0 = self.predict_x0(
            self.fake_score_adapter, fake_score, noisy, level, prepared, "conditional"
        )
        prediction = {
            key: (
                (noisy[key].double() - value.double())
                / level.sigmas[key].to(value.device).double().clamp(min=1e-8)
            ).to(value.dtype)
            for key, value in x0.items()
        }
        loss = self.critic_loss(prediction, generated, noise, prepared)
        metrics = {"loss_scalar": loss.detach()}
        if self.data.metrics is not None:
            metrics.update(self.data.metrics(level, None))
        return StepOutput(
            loss=loss,
            metrics=metrics,
            workload=self._workload(generated, generator),
        )

    def critic_loss(self, prediction, generated, noise, prepared):
        keys = matching_keys(prediction, generated, noise)
        masks = component_masks(keys, self.data.valid_masks(prepared))
        result = None
        for key in keys:
            if (
                prediction[key].shape != generated[key].shape
                or noise[key].shape != generated[key].shape
            ):
                raise ValueError("critic prediction, generated and noise must have matching shapes")
            # Subtract and square at the tower dtype. Autocast mse_loss widens
            # before subtraction and would produce a different residual.
            residual = prediction[key] - (noise[key] - generated[key])
            mask = masks[key]
            if mask is not None:
                if mask.dtype != torch.bool:
                    raise TypeError("loss valid_mask must be boolean")
                try:
                    mask = torch.broadcast_to(mask, residual.shape)
                except RuntimeError as error:
                    raise ValueError(
                        "loss valid_mask cannot broadcast to prediction shape"
                    ) from error
                if not bool(mask.any()):
                    raise ValueError("loss valid_mask must select at least one value")
                residual = residual[mask]
            loss = (residual**2).mean()
            result = loss if result is None else result + loss
        return result
