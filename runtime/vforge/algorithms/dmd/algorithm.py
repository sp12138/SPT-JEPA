"""DMD assembly and shared generator/critic steps with explicit numerical hooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import torch
import torch.nn as nn

from vforge.algorithms.base import Algorithm
from vforge.algorithms.dmd.config import DistributionMatchingConfig
from vforge.algorithms.dmd.model import (
    DistributionMatchingData,
    DmdCriticAdapter,
    DmdFamilyComponents,
    DmdGeneratorAdapter,
    DmdModelFamily,
    DmdNoiseLevel,
    LatentBundle,
    PreparedDmdBatch,
    RoleModelBundle,
)
from vforge.algorithms.dmd.objectives import (
    data_regression_loss_bundle,
    dmd_score_gradient_bundle,
    dmd_surrogate_loss_bundle,
    flow_critic_loss_bundle,
)
from vforge.algorithms.dmd.runtime import build_distribution_matching_data
from vforge.models.loading import PretrainedWeightSource
from vforge.tasks.requests import TrainRequest
from vforge.training.ema import ShardedEmaState
from vforge.training.recipe import (
    OptimizerSpec,
    PhasedTrainComponents,
    PhaseSpec,
    StepContext,
    StepOutput,
)
from vforge.utils.training import build_optimizer_factory

CRITIC_RNG_SEED_OFFSET = 104729
_STREAM_SEED_OFFSETS = {"generator": 0, "critic": 7919}


class DistributionMatchingAlgorithm(Algorithm[DmdModelFamily]):
    """Shared assembly; each variant supplies config, rollout data and loss steps."""

    name = "distribution_matching"
    config_type = DistributionMatchingConfig
    data_builder = staticmethod(build_distribution_matching_data)

    required_roles = frozenset({"generator", "real_score", "fake_score"})
    role_capabilities = {
        "generator": "generator",
        "real_score": "critic",
        "fake_score": "critic",
    }
    frozen_roles = frozenset({"real_score"})

    @classmethod
    def _validate_definition(cls) -> None:
        super()._validate_definition()
        if not cls.required_roles:
            raise ValueError("distribution matching must require at least one model role")
        if set(cls.role_capabilities) != set(cls.required_roles):
            raise ValueError("role_capabilities must cover exactly required_roles")
        unknown = set(cls.role_capabilities.values()) - {"generator", "critic"}
        if unknown:
            raise ValueError(f"unknown DMD role capabilities: {sorted(unknown)}")
        generator_roles = [
            role for role, capability in cls.role_capabilities.items() if capability == "generator"
        ]
        if len(generator_roles) != 1:
            raise ValueError("distribution matching requires exactly one generator role")
        if not set(cls.frozen_roles) <= set(cls.required_roles):
            raise ValueError("frozen_roles must be a subset of required_roles")

    def validate_request(
        self,
        request: TrainRequest,
        *,
        engine: str,
        integration: DmdModelFamily,
    ) -> None:
        missing = sorted(self.required_roles - set(request.training_models))
        if missing:
            raise ValueError(f"DMD training.models is missing required roles: {missing}")
        if set(request.optimizers) != {"generator", "critic"}:
            raise ValueError("distribution_matching requires generator and critic optimizers")
        self.config_type.from_mapping(request.dmd)
        integration.validate_request(request, engine=engine)

    def build_components(
        self,
        request: TrainRequest,
        *,
        integration: DmdModelFamily,
    ) -> PhasedTrainComponents:
        config = self.config_type.from_mapping(request.dmd)
        family = integration.build(
            request,
            required_roles=self.required_roles,
            role_capabilities=self.role_capabilities,
        )
        if not isinstance(family, DmdFamilyComponents):
            raise TypeError(
                f"DMD model family must return DmdFamilyComponents, got {type(family).__name__}"
            )
        missing = sorted(self.required_roles - set(family.models))
        if missing:
            raise ValueError(f"DMD model family did not build required roles: {missing}")
        root = RoleModelBundle(family.models, frozen_roles=self.frozen_roles)
        data = self.data_builder(
            family,
            config=config,
            seed=request.seed,
            stream_seed_offsets=_STREAM_SEED_OFFSETS,
        )
        generator_adapter = family.generator_adapter
        real_score_adapter = family.critic_adapters.get("real_score")
        fake_score_adapter = family.critic_adapters.get("fake_score")
        if not isinstance(generator_adapter, DmdGeneratorAdapter):
            raise TypeError("DMD model family returned an invalid generator adapter")
        if not isinstance(real_score_adapter, DmdCriticAdapter):
            raise TypeError("DMD model family returned an invalid real-score adapter")
        if not isinstance(fake_score_adapter, DmdCriticAdapter):
            raise TypeError("DMD model family returned an invalid fake-score adapter")
        steps = self.create_steps(
            config=config,
            data=data,
            generator_adapter=generator_adapter,
            real_score_adapter=real_score_adapter,
            fake_score_adapter=fake_score_adapter,
        )
        if not all(
            isinstance(source, PretrainedWeightSource) for source in family.pretrained_weights
        ):
            raise TypeError("DMD model family returned invalid pretrained weights")
        return PhasedTrainComponents(
            model=root,
            dataloader=data.dataloader,
            phases=self._build_phases(request, config=config, steps=steps),
            phase_order=("generator", "critic"),
            batch_selector=data.batch_selector,
            pretrained_weights=family.pretrained_weights,
            dataloader_resume_contract=data.dataloader_resume_contract,
        )

    @staticmethod
    def create_steps(**kwargs):
        return DistributionMatchingSteps(**kwargs)

    @staticmethod
    def _build_phases(
        request: TrainRequest,
        *,
        config: Any,
        steps: DistributionMatchingSteps,
    ) -> Mapping[str, PhaseSpec]:
        """Build the executable phases owned by this algorithm."""

        def build_ema(active_root: nn.Module):
            if not config.ema.enabled:
                return lambda _root, _step: None
            generator = model_role(active_root, "generator")
            ema = ShardedEmaState(
                generator,
                decay=config.ema.decay,
                start_step=config.ema.start_step,
            )
            active_root.add_module("generator_ema", ema)
            return lambda live_root, step: live_root.generator_ema.update(
                model_role(live_root, "generator"),
                step,
            )

        return {
            "generator": PhaseSpec(
                step_fn=steps.generator_step,
                target=lambda live_root: model_role(live_root, "generator"),
                optimizer_factory=_optimizer_factory(request.optimizers["generator"]),
                every_n_steps=config.schedule.generator_every_n_steps,
                stream="generator",
                post_update_hook_factory=build_ema,
            ),
            "critic": PhaseSpec(
                step_fn=steps.critic_step,
                target=lambda live_root: model_role(live_root, "fake_score"),
                optimizer_factory=_optimizer_factory(request.optimizers["critic"]),
                stream="critic",
            ),
        }


class DistributionMatchingSteps:
    """One G/C workflow; subclasses specialize rollout and numerical objectives."""

    def __init__(
        self,
        *,
        config: Any,
        data: DistributionMatchingData,
        generator_adapter: DmdGeneratorAdapter,
        real_score_adapter: DmdCriticAdapter,
        fake_score_adapter: DmdCriticAdapter,
    ) -> None:
        self.config = config
        self.data = data
        self.generator_adapter = generator_adapter
        self.real_score_adapter = real_score_adapter
        self.fake_score_adapter = fake_score_adapter

    def generator_step(
        self,
        root: nn.Module,
        batch: Any,
        step: int,
        context: StepContext,
    ) -> StepOutput:
        prepared = self.data.prepare_batch(batch, context.device)
        generator = model_role(root, "generator")
        generated, trace = self.generator_rollout(generator, prepared, context.random_streams)
        self.retain_rollout(generated, prepared, "generator")
        score_generated, score_prepared = self.fake_score_adapter.score_window(generated, prepared)
        level, noisy, _noise = self.data.sample_score_noise(
            score_generated,
            score_prepared,
            context.random_streams,
            17,
        )
        fake_score = model_role(root, "fake_score")
        real_score = model_role(root, "real_score")
        gradient = self.score_gradient(
            noisy,
            score_generated,
            level,
            score_prepared,
            fake_score=lambda values, active_level: self.predict_x0(
                self.fake_score_adapter,
                fake_score,
                values,
                active_level,
                score_prepared,
                "conditional",
            ),
            real_cond_score=lambda values, active_level: self.predict_x0(
                self.real_score_adapter,
                real_score,
                values,
                active_level,
                score_prepared,
                "conditional",
            ),
            real_uncond_score=lambda values, active_level: self.predict_x0(
                self.real_score_adapter,
                real_score,
                values,
                active_level,
                score_prepared,
                "unconditional",
            ),
        )
        gradient = self.fake_score_adapter.generated_region(gradient, prepared)
        loss = self.generator_loss(generated, gradient, prepared)
        metrics = {
            "loss_scalar": loss.detach(),
            "gradient_norm": bundle_abs_mean(gradient),
        }
        if self.data.metrics is not None:
            metrics.update(self.data.metrics(level, gradient))
        return StepOutput(
            loss=loss,
            backward_losses=self.build_backward_losses(
                generator,
                prepared,
                generated,
                gradient,
                loss,
                trace,
            ),
            metrics=metrics,
            workload=self._workload(generated, generator),
        )

    def critic_step(
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
        self.retain_rollout(generated, prepared, "critic")
        score_generated, score_prepared = self.fake_score_adapter.score_window(generated, prepared)
        level, noisy, noise = self.data.sample_score_noise(
            score_generated,
            score_prepared,
            context.random_streams,
            CRITIC_RNG_SEED_OFFSET + 17,
        )
        fake_score = model_role(root, "fake_score")
        prediction = self.fake_score_adapter.predict_velocity(
            fake_score,
            noisy,
            level,
            score_prepared,
            "conditional",
        )
        project = self.fake_score_adapter.generated_region
        loss = self.critic_loss(
            project(prediction, prepared), generated, project(noise, prepared), prepared
        )
        metrics = {"loss_scalar": loss.detach()}
        if self.data.metrics is not None:
            metrics.update(self.data.metrics(level, None))
        return StepOutput(
            loss=loss,
            metrics=metrics,
            workload=self._workload(generated, generator),
        )

    def retain_rollout(self, generated, prepared, phase):
        """Variants may retain detached outputs until the complete update succeeds."""

    def generator_rollout(self, generator, prepared, random_streams):
        """Return generated latents and optional state for backward replay."""
        return self.data.rollout(
            generator, self.generator_adapter, prepared, random_streams, "generator", 0
        ), None

    def score_gradient(self, noisy, generated, level, prepared, **scores):
        return dmd_score_gradient_bundle(
            noisy,
            generated,
            level,
            **scores,
            real_guidance_scale=self.config.score.real_guidance_scale,
        )

    def generator_loss(self, generated, gradient, prepared):
        masks = self.data.valid_masks(prepared)
        loss = dmd_surrogate_loss_bundle(generated, gradient, valid_masks=masks)
        if self.config.score.mse_weight:
            loss = loss + self.config.score.mse_weight * data_regression_loss_bundle(
                generated,
                prepared.clean,
                valid_masks=masks,
            )
        return loss

    def critic_loss(self, prediction, generated, noise, prepared):
        return flow_critic_loss_bundle(
            prediction,
            generated,
            noise,
            valid_masks=self.data.valid_masks(prepared),
        )

    def build_backward_losses(self, generator, prepared, generated, gradient, loss, trace):
        """Default retained-graph backward is executed by the training engine."""
        return None

    @staticmethod
    def predict_x0(
        adapter: DmdCriticAdapter,
        tower: nn.Module,
        noisy: LatentBundle,
        level: DmdNoiseLevel,
        batch: PreparedDmdBatch,
        conditioning: Literal["conditional", "unconditional"],
    ) -> dict[str, torch.Tensor]:
        flow = adapter.predict_velocity(
            tower,
            noisy,
            level,
            batch,
            conditioning,
        )
        return {
            key: value - level.sigmas[key].to(value) * flow[key] for key, value in noisy.items()
        }

    def _workload(
        self,
        generated: LatentBundle,
        generator: nn.Module,
    ) -> Mapping[str, float | int] | None:
        if self.data.workload is None:
            return None
        return self.data.workload(generated, generator)


def model_role(root: nn.Module, role: str) -> nn.Module:
    if not isinstance(root, RoleModelBundle):
        raise TypeError("DMD phase root must be a RoleModelBundle")
    return root.model_for(role)


def bundle_abs_mean(bundle: LatentBundle) -> torch.Tensor:
    values = [value.abs().mean() for value in bundle.values()]
    return torch.stack(values).mean()


def _optimizer_factory(config: OptimizerSpec):
    if config.name != "adamw":
        raise ValueError(f"DMD optimizer must be adamw, got {config.name!r}")
    return build_optimizer_factory(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
    )


__all__ = ["DistributionMatchingAlgorithm", "DistributionMatchingSteps"]
