"""Assembly and step ordering for causal consistency distillation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vforge.algorithms.base import Algorithm
from vforge.algorithms.cd.config import (
    CausalConsistencyConfig,
    CausalConsistencyMoeConfig,
)
from vforge.algorithms.cd.model import (
    CdFamilyComponents,
    CdModelFamily,
    CdSchedule,
    CdTeacherForcingAdapter,
    PreparedCdBatch,
    RoleModelBundle,
)
from vforge.models.loading import PretrainedWeightSource
from vforge.tasks.requests import TrainRequest
from vforge.training.diagnostics import LogicalReduction, clip_grad_norm_logical_
from vforge.training.ema import ShardedEmaState
from vforge.training.recipe import (
    GradientClipper,
    PhasedTrainComponents,
    PhaseSpec,
    StepContext,
    StepOutput,
)
from vforge.utils.training import build_optimizer_factory, build_scheduler_factory


class CausalConsistencyAlgorithm(Algorithm[CdModelFamily]):
    name = "causal_consistency_tf"
    required_roles = frozenset({"teacher", "student", "ema_model"})
    frozen_roles = frozenset({"teacher", "ema_model"})

    def validate_request(
        self,
        request: TrainRequest,
        *,
        engine: str,
        integration: CdModelFamily,
    ) -> None:
        roles = set(request.training_models)
        if roles != set(self.required_roles):
            raise ValueError(
                "causal consistency requires exactly training.models roles "
                f"{sorted(self.required_roles)}; got {sorted(roles)}"
            )
        if set(request.optimizers) != {"student"}:
            raise ValueError("causal consistency requires exactly optimizers.student")
        CausalConsistencyConfig.from_mapping(request.cd)
        integration.validate_request(request, engine=engine)

    def build_components(
        self,
        request: TrainRequest,
        *,
        integration: CdModelFamily,
    ) -> PhasedTrainComponents:
        config = CausalConsistencyConfig.from_mapping(request.cd)
        family = integration.build(request, required_roles=self.required_roles)
        if not isinstance(family, CdFamilyComponents):
            raise TypeError(
                f"CD model family must return CdFamilyComponents, got {type(family).__name__}"
            )
        if not isinstance(family.adapter, CdTeacherForcingAdapter):
            raise TypeError("CD model family returned an invalid adapter")
        if not all(
            isinstance(source, PretrainedWeightSource) for source in family.pretrained_weights
        ):
            raise TypeError("CD model family returned invalid pretrained weights")
        root = RoleModelBundle(family.models, frozen_roles=self.frozen_roles)
        dataloader, resume_contract = family.build_train_stream(0)
        steps = _CausalConsistencySteps(
            config=config,
            prepare_batch=family.prepare_batch,
            adapter=family.adapter,
            schedule=family.schedule,
            seed=request.seed,
        )
        return PhasedTrainComponents(
            model=root,
            dataloader=dataloader,
            phases={
                "student": PhaseSpec(
                    step_fn=steps.student_step,
                    target=lambda live_root: _model_role(live_root, "student"),
                    optimizer_factory=_optimizer_factory(request),
                    scheduler_factory=_scheduler_factory(request),
                    stream="student",
                    post_update_hook_factory=lambda live_root: _build_ema_hook(
                        live_root,
                        config=config,
                    ),
                )
            },
            phase_order=("student",),
            batch_selector=lambda batch, _stream: batch,
            pretrained_weights=family.pretrained_weights,
            dataloader_resume_contract=resume_contract,
            gradient_clipper_factory=_cd_gradient_clipper,
        )


class CausalConsistencyMoeAlgorithm(Algorithm[CdModelFamily]):
    """Alternating high/low expert consistency distillation for Wan A14B."""

    name = "causal_consistency_moe_tf"
    required_roles = CausalConsistencyAlgorithm.required_roles
    frozen_roles = CausalConsistencyAlgorithm.frozen_roles

    def validate_request(
        self,
        request: TrainRequest,
        *,
        engine: str,
        integration: CdModelFamily,
    ) -> None:
        roles = set(request.training_models)
        if roles != set(self.required_roles):
            raise ValueError(
                "MoE causal consistency requires exactly training.models roles "
                f"{sorted(self.required_roles)}; got {sorted(roles)}"
            )
        if set(request.optimizers) != {"student"}:
            raise ValueError("MoE causal consistency requires exactly optimizers.student")
        CausalConsistencyMoeConfig.from_mapping(request.cd)
        integration.validate_request(request, engine=engine)

    def build_components(
        self,
        request: TrainRequest,
        *,
        integration: CdModelFamily,
    ) -> PhasedTrainComponents:
        config = CausalConsistencyMoeConfig.from_mapping(request.cd)
        family = integration.build(request, required_roles=self.required_roles)
        if not isinstance(family, CdFamilyComponents):
            raise TypeError(
                f"CD model family must return CdFamilyComponents, got {type(family).__name__}"
            )
        if not isinstance(family.adapter, CdTeacherForcingAdapter):
            raise TypeError("CD model family returned an invalid adapter")
        if not all(
            isinstance(source, PretrainedWeightSource) for source in family.pretrained_weights
        ):
            raise TypeError("CD model family returned invalid pretrained weights")
        root = RoleModelBundle(family.models, frozen_roles=self.frozen_roles)
        dataloader, resume_contract = family.build_train_stream(0)
        steps = _CausalConsistencySteps(
            config=config,
            prepare_batch=family.prepare_batch,
            adapter=family.adapter,
            schedule=family.schedule,
            seed=request.seed,
        )
        return PhasedTrainComponents(
            model=root,
            dataloader=dataloader,
            phases={
                "student": PhaseSpec(
                    step_fn=steps.student_step,
                    target=lambda live_root: _model_role(live_root, "student"),
                    optimizer_factory=_optimizer_factory(request),
                    scheduler_factory=_scheduler_factory(request),
                    stream="student",
                    post_update_hook_factory=lambda live_root: _build_moe_ema_hook(
                        live_root,
                        config=config,
                    ),
                )
            },
            phase_order=("student",),
            batch_selector=lambda batch, _stream: batch,
            pretrained_weights=family.pretrained_weights,
            dataloader_resume_contract=resume_contract,
            gradient_clipper_factory=_cd_gradient_clipper,
        )


class _CausalConsistencySteps:
    def __init__(
        self,
        *,
        config: CausalConsistencyConfig | CausalConsistencyMoeConfig,
        prepare_batch,
        adapter,
        schedule: CdSchedule,
        seed: int,
    ) -> None:
        self.config = config
        self.prepare_batch = prepare_batch
        self.adapter = adapter
        self.schedule = schedule
        self.seed = int(seed)

    def student_step(
        self,
        root: nn.Module,
        batch: Any,
        step: int,
        context: StepContext,
    ) -> StepOutput:
        prepared = self.prepare_batch(batch, context.device, step, context.random_streams)
        if not isinstance(prepared, PreparedCdBatch):
            raise TypeError("CD data support must return PreparedCdBatch")
        generator = context.random_streams.generator(
            "cd.student", seed=self.seed, device=context.device
        )
        high_noise = step % 2 == 0 if isinstance(self.config, CausalConsistencyMoeConfig) else None
        pair = self.schedule.sample_pair(generator, high_noise)
        clean = prepared.clean_latent
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        noisy = self.schedule.add_noise(clean, noise, pair.sigma)
        anchor_first_frame = high_noise is None
        if anchor_first_frame:
            if prepared.first_frame_latent is None:
                raise ValueError("Wan 5B CD requires a first-frame anchor")
            noisy[:, :, 0:1] = prepared.first_frame_latent.to(noisy)
        clean_context = clean.clone()
        if anchor_first_frame:
            clean_context[:, :, 0:1] = prepared.first_frame_latent.to(clean_context)

        teacher = _model_role(root, "teacher")
        with torch.no_grad():
            teacher_cond = self.adapter.predict_velocity(
                teacher,
                noisy_latent=noisy,
                clean_context=clean_context,
                timestep=pair.timestep,
                batch=prepared,
            )
            if self.config.guidance_scale == 1.0:
                teacher_velocity = teacher_cond
            else:
                unconditional = replace(
                    prepared,
                    prompt_embeds=torch.zeros_like(prepared.prompt_embeds),
                )
                teacher_uncond = self.adapter.predict_velocity(
                    teacher,
                    noisy_latent=noisy,
                    clean_context=clean_context,
                    timestep=pair.timestep,
                    batch=unconditional,
                )
                teacher_velocity = teacher_uncond + self.config.guidance_scale * (
                    teacher_cond - teacher_uncond
                )
            next_noisy = self.schedule.teacher_step(
                noisy,
                teacher_velocity,
                pair.timestep,
                pair.next_timestep,
            )
            if anchor_first_frame:
                next_noisy[:, :, 0:1] = prepared.first_frame_latent.to(next_noisy)

        student = _model_role(root, "student")
        student_velocity = self.adapter.predict_velocity(
            student,
            noisy_latent=noisy,
            clean_context=clean_context,
            timestep=pair.timestep,
            batch=prepared,
        )
        student_x0 = self.schedule.velocity_to_x0(student_velocity, noisy, pair.sigma)

        ema_model = _model_role(root, "ema_model")
        with torch.no_grad():
            _sync_ema_model(root, ema_model, high_noise=high_noise)
            ema_velocity = self.adapter.predict_velocity(
                ema_model,
                noisy_latent=next_noisy,
                clean_context=clean_context,
                timestep=pair.next_timestep,
                batch=prepared,
            )
            ema_x0 = self.schedule.velocity_to_x0(ema_velocity, next_noisy, pair.next_sigma)

        loss = _x0_consistency_loss(
            student_x0,
            ema_x0,
            exclude_first_frame=anchor_first_frame,
        )
        metrics = {
            "loss_scalar": loss.detach(),
            "timestep_index": pair.index,
            "timestep": pair.timestep.detach(),
            "next_timestep": pair.next_timestep.detach(),
        }
        if high_noise is not None:
            metrics["high_noise_expert"] = torch.tensor(
                high_noise,
                device=loss.device,
            )
        return StepOutput(
            loss=loss,
            metrics=metrics,
        )


def _build_ema_hook(
    root: nn.Module,
    *,
    config: CausalConsistencyConfig,
):
    if not isinstance(root, RoleModelBundle):
        raise TypeError("CD EMA requires a RoleModelBundle")
    ema = ShardedEmaState(
        root.model_for("student"),
        decay=config.ema.decay,
        start_step=config.ema.start_step,
    )
    root.add_module("student_ema", ema)
    return lambda live_root, step: _ema_state(live_root).update(
        _model_role(live_root, "student"),
        step,
    )


def _build_moe_ema_hook(
    root: nn.Module,
    *,
    config: CausalConsistencyMoeConfig,
):
    if not isinstance(root, RoleModelBundle):
        raise TypeError("MoE CD EMA requires a RoleModelBundle")
    student = root.model_for("student")
    for high_noise, name in ((True, "wan"), (False, "wan_2")):
        expert = _expert_for_noise(student, high_noise=high_noise)
        root.add_module(
            f"student_ema_{name}",
            ShardedEmaState(
                expert,
                decay=config.ema.decay,
                start_step=config.ema.start_step,
            ),
        )

    def update_active(live_root: nn.Module, step: int) -> None:
        high_noise = int(step) % 2 == 0
        _moe_ema_state(live_root, high_noise=high_noise).update(
            _expert_for_noise(
                _model_role(live_root, "student"),
                high_noise=high_noise,
            ),
            step,
        )

    return update_active


def _sync_ema_model(
    root: nn.Module,
    ema_model: nn.Module,
    *,
    high_noise: bool | None = None,
) -> None:
    if high_noise is None:
        ema = _ema_state(root)
        target = ema_model
    else:
        ema = _moe_ema_state(root, high_noise=high_noise)
        target = _expert_for_noise(ema_model, high_noise=high_noise)
    if bool(ema.initialized.item()):
        ema.copy_to(target)


def _ema_state(root: nn.Module) -> ShardedEmaState:
    if not isinstance(root, RoleModelBundle):
        raise TypeError("CD phase root must be a RoleModelBundle")
    ema = getattr(root, "student_ema", None)
    if not isinstance(ema, ShardedEmaState):
        raise RuntimeError("CD student EMA state has not been configured")
    return ema


def _moe_ema_state(root: nn.Module, *, high_noise: bool) -> ShardedEmaState:
    if not isinstance(root, RoleModelBundle):
        raise TypeError("CD phase root must be a RoleModelBundle")
    name = "student_ema_wan" if high_noise else "student_ema_wan_2"
    ema = getattr(root, name, None)
    if not isinstance(ema, ShardedEmaState):
        raise RuntimeError(f"CD expert EMA state {name!r} has not been configured")
    return ema


def _expert_for_noise(module: nn.Module, *, high_noise: bool) -> nn.Module:
    name = "wan" if high_noise else "wan_2"
    expert = getattr(module, name, None)
    if not isinstance(expert, nn.Module):
        raise TypeError(f"Wan A14B CD model has no {name!r} expert")
    return expert


def _model_role(root: nn.Module, role: str) -> nn.Module:
    if not isinstance(root, RoleModelBundle):
        raise TypeError("CD phase root must be a RoleModelBundle")
    return root.model_for(role)


def _optimizer_factory(request: TrainRequest):
    config = request.optimizers["student"]
    return build_optimizer_factory(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
    )


def _scheduler_factory(request: TrainRequest):
    return build_scheduler_factory(
        name=request.scheduler_name,
        factor=request.scheduler_factor,
        warmup_steps=request.warmup_steps,
        start_factor=request.scheduler_start_factor,
        end_factor=request.scheduler_end_factor,
        total_iters=request.scheduler_total_iters,
    )


def _cd_gradient_clipper(norm_dtype: str) -> GradientClipper:
    """Preserve CD logical clipping, whose norm accumulation is FP32."""
    if norm_dtype != "float32":
        raise ValueError("CD logical gradient clipping requires gradient_norm_dtype=float32")
    return _clip_cd_grad_norm_


def _clip_cd_grad_norm_(
    model: nn.Module,
    optimizer: Any,
    max_norm: float,
    reduction: LogicalReduction | None,
) -> float:
    """Use the trainer's identity-based contribution filter for the student."""
    del optimizer
    return clip_grad_norm_logical_(model, max_norm, reduction)


def _x0_consistency_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    exclude_first_frame: bool,
) -> torch.Tensor:
    """Mean x0 consistency loss with an optional first-frame exclusion."""
    if prediction.shape != target.shape:
        raise ValueError("CD prediction and target must have identical shapes")
    if prediction.ndim != 5:
        raise ValueError("CD predictions must be shaped [B,C,T,H,W]")
    if exclude_first_frame and prediction.shape[2] < 2:
        raise ValueError("anchored CD loss requires at least two latent frames")
    prediction_value = prediction[:, :, 1:] if exclude_first_frame else prediction
    target_value = target[:, :, 1:] if exclude_first_frame else target
    # Taichu reduces in the prediction dtype. Keep the trainer's outer
    # autocast region from promoting BF16 MSE to FP32.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        loss = F.mse_loss(
            prediction_value,
            target_value.detach(),
            reduction="mean",
        )
    if not torch.isfinite(loss):
        raise FloatingPointError("CD consistency loss is not finite")
    return loss


__all__ = ["CausalConsistencyAlgorithm", "CausalConsistencyMoeAlgorithm"]
