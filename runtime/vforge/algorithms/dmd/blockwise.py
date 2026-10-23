"""Block DMD differences over the shared assembly, steps and rollout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch

from vforge.algorithms.dmd.algorithm import DistributionMatchingAlgorithm, DistributionMatchingSteps
from vforge.algorithms.dmd.config import BlockDistributionMatchingConfig
from vforge.algorithms.dmd.model import LatentBundle
from vforge.algorithms.dmd.objectives import (
    BundleScoreCallable,
    bundle_score_predictions,
    component_masks,
    matching_keys,
    resolve_component_weights,
)
from vforge.algorithms.dmd.runtime import (
    DistributionMatchingRuntime,
    DmdRolloutSchedule,
    build_distribution_matching_data,
)
from vforge.training.diagnostics import torch_gradient_clipper


def build_block_distribution_matching_data(family, **kwargs):
    if family.loss_partitions is None or family.resolve_rollout_noise_level is None:
        raise ValueError("block DMD requires loss partitions and a rollout noise resolver")
    return build_distribution_matching_data(
        family, runtime_type=BlockDistributionMatchingRuntime, **kwargs
    )


class BlockDistributionMatchingRuntime(DistributionMatchingRuntime):
    """Specialize only the schedule, exit sampling and numerical boundaries."""

    noise_dtype = torch.float32
    clean_context = True

    def build_scheduler(self):
        # The model adapter resolves its own rollout and score timesteps.
        return None

    def build_rollout_schedule(self):
        return DmdRolloutSchedule(
            noise_levels=tuple(
                step / self.config.score.num_train_timesteps
                for step in self.config.rollout.denoising_steps
            )
        )

    def resolve_rollout_level(self, level, batch):
        return self.family.resolve_rollout_noise_level(None, level, batch)

    def select_exit_steps(self, block_count, forward_count, reference, random):
        count = 1 if self.config.rollout.same_step_across_blocks else block_count
        indices = torch.randint(forward_count, (count,), device=reference.device, generator=random)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(indices, src=0)
        return indices.tolist() * block_count if count == 1 else indices.tolist()

    def sample_score_timestep(self, prepared, reference, random):
        return torch.randint(
            self.config.score.sampled_timestep_min,
            self.config.score.sampled_timestep_max + 1,
            self.family.score_noise_shape(prepared),
            device=reference.device,
            generator=random,
        )


class BlockDistributionMatchingAlgorithm(DistributionMatchingAlgorithm):
    name = "block_distribution_matching"
    config_type = BlockDistributionMatchingConfig
    data_builder = staticmethod(build_block_distribution_matching_data)

    def build_components(self, request, *, integration):
        return replace(
            super().build_components(request, integration=integration),
            gradient_clipper_factory=torch_gradient_clipper,
        )

    @staticmethod
    def create_steps(**kwargs):
        return BlockDistributionMatchingSteps(**kwargs)


class BlockDistributionMatchingSteps(DistributionMatchingSteps):
    """Override only the numerical rules and optional replay of the shared steps."""

    def generator_rollout(self, generator, prepared, random_streams):
        if self.config.rollout.backward_mode == "block_replay":
            return self.data.rollout_trace(
                generator, self.generator_adapter, prepared, random_streams, "generator", 0
            )
        return super().generator_rollout(generator, prepared, random_streams)

    def score_gradient(self, noisy, generated, level, prepared, **scores):
        return self._dmd_score_gradient_bundle(
            noisy,
            generated,
            level,
            **scores,
            component_guidance_scales=self.config.score.component_guidance_scales,
            valid_masks=self.data.valid_masks(prepared),
        )

    def generator_loss(self, generated, gradient, prepared):
        return self._dmd_surrogate_loss_bundle(
            generated,
            gradient,
            valid_masks=self.data.valid_masks(prepared),
            component_weights=self.config.score.component_weights or None,
            partitions=self.data.loss_partitions(prepared) if self.data.loss_partitions else None,
        )

    def critic_loss(self, prediction, generated, noise, prepared):
        return self._flow_critic_loss_bundle(
            prediction,
            generated,
            noise,
            valid_masks=self.data.valid_masks(prepared),
            component_weights=self.config.score.component_weights or None,
        )

    @staticmethod
    def predict_x0(adapter, tower, noisy, level, batch, conditioning):
        return dict(adapter.predict_x0(tower, noisy, level, batch, conditioning))

    def build_backward_losses(self, generator, prepared, generated, gradient, loss, trace):
        if trace is None:
            return None
        targets = {
            key: (value - gradient[key].to(value)).detach() for key, value in generated.items()
        }
        partitions = self.data.loss_partitions(prepared)
        if len(trace) != len(partitions):
            raise ValueError("replay blocks and loss partitions must have the same length")

        def backward_losses():
            adapter = self.generator_adapter
            with adapter.open_rollout_context(generator, prepared) as cache:
                loss.zero_()
                for index, (state, region) in enumerate(zip(trace, partitions, strict=True)):
                    if state.is_anchor:
                        values = state.accepted
                    else:
                        values = adapter.denoise_block(
                            generator,
                            state.noisy,
                            state.selection,
                            state.level,
                            prepared,
                            cache,
                        )
                    target = {key: targets[key][:, region[key]] for key in values}
                    block_loss = self._dmd_loss_to_targets(
                        values,
                        target,
                        self.config.score.component_weights,
                        [dict.fromkeys(values, slice(None))],
                        divisor=len(trace),
                    )
                    loss.add_(block_loss.detach())
                    if not state.is_anchor:
                        yield block_loss
                    del block_loss, values, target
                    # Match reference: no final prefill after the last backward.
                    if index + 1 < len(trace):
                        with torch.no_grad():
                            adapter.commit_block(
                                generator,
                                state.selection,
                                state.accepted,
                                state.context_level,
                                prepared,
                                cache,
                            )

        return backward_losses

    @staticmethod
    def _dmd_surrogate_loss_bundle(
        generated: LatentBundle,
        score_gradient: LatentBundle,
        *,
        valid_masks: Mapping[str, torch.Tensor | None] | None = None,
        component_weights: Mapping[str, float] | None = None,
        partitions: Sequence[Mapping[str, slice]] | None = None,
    ) -> torch.Tensor:
        """Combine modalities; optional model partitions receive equal block weight.

        Partitioned surrogate keeps masked zero-gradient values in the denominator,
        matching a reference whose target equals the prediction on anchor tokens.
        """
        keys = matching_keys(generated, score_gradient)
        weights = resolve_component_weights(keys, component_weights)
        regions = partitions if partitions is not None else [dict.fromkeys(keys, slice(None))]
        if not regions:
            raise ValueError("surrogate partitions cannot be empty")
        values = generated
        targets = {
            key: (value - score_gradient[key].to(value)).detach() for key, value in values.items()
        }
        return BlockDistributionMatchingSteps._dmd_loss_to_targets(
            values, targets, weights, regions, valid_masks, partitions is None
        )

    @staticmethod
    def _dmd_loss_to_targets(
        values, targets, weights, regions, valid_masks=None, apply_mask=False, divisor=None
    ):
        """Apply the same block arithmetic to fixed, detached surrogate targets."""
        keys = matching_keys(values, targets)
        # Each BF16 block mean is rounded before the FP32 accumulation.
        total = next(iter(values.values())).new_zeros((), dtype=torch.float32)
        for region in regions:
            if set(region) != set(keys):
                raise ValueError("partition component keys must match the latent bundle")
            block = None
            for key in keys:
                value, target = (values[key][:, region[key]], targets[key][:, region[key]])
                if not value.numel():
                    raise ValueError("surrogate partition cannot select zero tokens")
                with torch.autocast(value.device.type, enabled=False):
                    squared = (value - target).pow(2)
                    if (
                        apply_mask
                        and valid_masks is not None
                        and (valid_masks.get(key) is not None)
                    ):
                        squared = squared[torch.broadcast_to(valid_masks[key], squared.shape)]
                    term = weights[key] * 0.5 * squared.mean()
                block = term if block is None else block + term
            term = block / (len(regions) if divisor is None else divisor)
            total = term if total is None else total + term
        return total

    @staticmethod
    def _flow_critic_loss_bundle(
        flow_prediction: LatentBundle,
        clean: LatentBundle,
        noise: LatentBundle,
        *,
        valid_masks: Mapping[str, torch.Tensor | None] | None = None,
        component_weights: Mapping[str, float] | None = None,
    ) -> torch.Tensor:
        """Combine per-component critic flow-regression losses."""
        keys = matching_keys(flow_prediction, clean, noise)
        weights = resolve_component_weights(keys, component_weights)
        masks = component_masks(keys, valid_masks)
        result: torch.Tensor | None = None
        for key in keys:
            with torch.autocast(flow_prediction[key].device.type, enabled=False):
                squared = (flow_prediction[key] - (noise[key] - clean[key])).pow(2)
                if masks[key] is not None:
                    squared = squared[torch.broadcast_to(masks[key], squared.shape)]
                if not squared.numel():
                    raise ValueError("critic mask must select at least one value")
                component = weights[key] * squared.mean()
            result = component if result is None else result + component
        assert result is not None
        return result

    @staticmethod
    def _dmd_score_gradient_bundle(
        noisy: LatentBundle,
        clean: LatentBundle,
        noise_level: Any,
        *,
        fake_score: BundleScoreCallable,
        real_cond_score: BundleScoreCallable,
        real_uncond_score: BundleScoreCallable,
        component_guidance_scales: Mapping[str, float],
        valid_masks: Mapping[str, torch.Tensor | None] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute detached per-component DMD gradients with three score calls."""
        with torch.no_grad():
            keys, fake, real_cond, real_uncond = bundle_score_predictions(
                noisy,
                clean,
                noise_level,
                fake_score=fake_score,
                real_cond_score=real_cond_score,
                real_uncond_score=real_uncond_score,
            )
            gradients: dict[str, torch.Tensor] = {}
            for key in keys:
                scale = component_guidance_scales[key] - 1.0
                guided_real = real_cond[key] + scale * (real_cond[key] - real_uncond[key])
                mask = None if valid_masks is None else valid_masks.get(key)
                gradient = fake[key] - guided_real
                normalizer = (
                    (clean[key] - guided_real)
                    .abs()
                    .mean(dim=tuple(range(1, clean[key].ndim)), keepdim=True)
                )
                if mask is not None:
                    expanded = torch.broadcast_to(mask, clean[key].shape)
                    dims = tuple(range(1, clean[key].ndim))
                    count = expanded.sum(dim=dims, keepdim=True)
                    if not bool((count > 0).all()):
                        raise ValueError("normalization mask selects no values")
                    normalizer = (
                        ((clean[key] - guided_real).abs().float() * expanded).sum(
                            dim=dims, keepdim=True
                        )
                        / count
                    ).to(clean[key].dtype)
                gradient = gradient / (normalizer + 1e-08)
                gradient = torch.nan_to_num(gradient)
                if mask is not None:
                    gradient = torch.where(mask, gradient, 0.0)
                gradients[key] = gradient.detach()
            return gradients
