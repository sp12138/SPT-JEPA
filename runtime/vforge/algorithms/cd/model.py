"""Model composition and family contracts for consistency distillation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from vforge.models.loading import PretrainedWeightSource
from vforge.tasks.requests import TrainRequest
from vforge.utils.rng import RandomStreams


@dataclass(frozen=True)
class CdNoisePair:
    index: int
    timestep: torch.Tensor
    next_timestep: torch.Tensor
    sigma: torch.Tensor
    next_sigma: torch.Tensor


@dataclass(frozen=True)
class CdSchedule:
    """Family-owned noise math; sampling consumes the caller's generator.

    ``sample_pair`` receives the generator and optional high/low expert band.
    ``teacher_step`` receives noisy latent, velocity, current and next timestep.
    The family owns time units, conversions and intermediate dtypes.
    """

    sample_pair: Callable[[torch.Generator, bool | None], CdNoisePair]
    add_noise: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    teacher_step: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    velocity_to_x0: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PreparedCdBatch:
    clean_latent: torch.Tensor
    condition_latents: Mapping[str, torch.Tensor]
    first_frame_latent: torch.Tensor | None
    prompt_embeds: torch.Tensor
    sample_ids: tuple[str, ...] = ()

    def condition(self, name: str) -> torch.Tensor:
        try:
            return self.condition_latents[name]
        except KeyError as exc:
            raise KeyError(
                f"CD batch has no condition stream {name!r}; "
                f"available={list(self.condition_latents)}"
            ) from exc


class CdTeacherForcingAdapter(ABC):
    @abstractmethod
    def predict_velocity(
        self,
        model: nn.Module,
        *,
        noisy_latent: torch.Tensor,
        clean_context: torch.Tensor,
        timestep: torch.Tensor,
        batch: PreparedCdBatch,
    ) -> torch.Tensor:
        """Run one doubled teacher-forcing forward and return velocity."""


class RoleModelBundle(nn.Module):
    """Register CD model roles using the DMD bundle lifecycle contract."""

    def __init__(
        self,
        models: Mapping[str, nn.Module],
        *,
        frozen_roles: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        super().__init__()
        resolved = dict(models)
        if not resolved:
            raise ValueError("role model bundle must contain at least one role")
        if any(not role.isidentifier() for role in resolved):
            raise ValueError("model role names must be valid Python identifiers")
        if any(not isinstance(module, nn.Module) for module in resolved.values()):
            raise TypeError("model role values must be nn.Module instances")
        unknown_frozen = set(frozen_roles) - set(resolved)
        if unknown_frozen:
            raise ValueError(f"unknown frozen model roles: {sorted(unknown_frozen)}")

        self._role_order = tuple(resolved)
        self._frozen_roles = frozenset(frozen_roles)
        for role, module in resolved.items():
            self.add_module(role, module)
        self._restore_frozen_roles()

    @property
    def roles(self) -> tuple[str, ...]:
        return self._role_order

    def model_for(self, role: str) -> nn.Module:
        if role not in self._role_order:
            raise KeyError(f"unknown model role {role!r}")
        module = getattr(self, role)
        if not isinstance(module, nn.Module):
            raise TypeError(f"model role {role!r} is not an nn.Module")
        return module

    def fsdp_execution_roots(self) -> tuple[nn.Module, ...]:
        roots: list[nn.Module] = []
        for role in self._role_order:
            module = self.model_for(role)
            resolver = getattr(module, "fsdp_execution_roots", None)
            resolved = tuple(resolver()) if callable(resolver) else (module,)
            if not resolved or any(not isinstance(root, nn.Module) for root in resolved):
                raise TypeError(f"model role {role!r} returned invalid FSDP execution roots")
            roots.extend(resolved)
        if len({id(root) for root in roots}) != len(roots):
            raise ValueError("CD model roles must expose disjoint FSDP execution roots")
        return tuple(roots)

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> None:
        kwargs = (
            {"use_reentrant": False}
            if gradient_checkpointing_kwargs is None
            else dict(gradient_checkpointing_kwargs)
        )
        checkpoint_func = partial(checkpoint, **kwargs)
        for role in self._role_order:
            if role in self._frozen_roles:
                continue
            module = self.model_for(role)
            enable = getattr(module, "enable_gradient_checkpointing", None)
            if callable(enable):
                enable(gradient_checkpointing_func=checkpoint_func)
                continue
            enable = getattr(module, "gradient_checkpointing_enable", None)
            if callable(enable):
                enable(gradient_checkpointing_kwargs=kwargs)

    def train(self, mode: bool = True) -> RoleModelBundle:
        super().train(mode)
        self._restore_frozen_roles()
        return self

    def _restore_frozen_roles(self) -> None:
        for role in self._frozen_roles:
            module = self.model_for(role)
            module.requires_grad_(False)
            module.eval()


BuildTrainStream = Callable[[int], tuple[Iterable[Any], Mapping[str, Any]]]
PrepareBatch = Callable[[Any, torch.device, int, RandomStreams], PreparedCdBatch]
ValidateFamily = Callable[[TrainRequest, str], None]
BuildRoleModel = Callable[[TrainRequest, str], nn.Module]
BuildDataSupport = Callable[[TrainRequest], "CdDataSupport"]
CreateAdapter = Callable[[TrainRequest], CdTeacherForcingAdapter]
CreateSchedule = Callable[[TrainRequest], CdSchedule]


@dataclass(frozen=True)
class CdDataSupport:
    build_train_stream: BuildTrainStream
    prepare_batch: PrepareBatch


@dataclass(frozen=True)
class CdFamilyComponents:
    models: Mapping[str, nn.Module]
    build_train_stream: BuildTrainStream
    prepare_batch: PrepareBatch
    adapter: CdTeacherForcingAdapter
    schedule: CdSchedule
    pretrained_weights: tuple[PretrainedWeightSource, ...]
    dataloader_resume_contract: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CdFamilySpec:
    model_name: str
    validate: ValidateFamily
    build_role_model: BuildRoleModel
    build_data_support: BuildDataSupport
    create_adapter: CreateAdapter
    create_schedule: CreateSchedule

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("CD family model_name must be non-empty")


class CdModelFamily:
    def __init__(self, spec: CdFamilySpec) -> None:
        self.spec = spec

    def validate_request(self, request: TrainRequest, *, engine: str) -> None:
        self._validate_model_names(request)
        self.spec.validate(request, engine)

    def build(
        self,
        request: TrainRequest,
        *,
        required_roles: frozenset[str],
    ) -> CdFamilyComponents:
        self._validate_model_names(request)
        missing = sorted(required_roles - set(request.training_models))
        if missing:
            raise ValueError(f"CD training.models is missing required roles: {missing}")
        schedule = self.spec.create_schedule(request)
        support = self.spec.build_data_support(request)
        roles = tuple(role for role in request.training_models if role in required_roles)
        models = {role: self.spec.build_role_model(request, role) for role in roles}
        return CdFamilyComponents(
            models=models,
            build_train_stream=support.build_train_stream,
            prepare_batch=support.prepare_batch,
            adapter=self.spec.create_adapter(request),
            schedule=schedule,
            pretrained_weights=_independent_pretrained_weights(request, roles),
        )

    def _validate_model_names(self, request: TrainRequest) -> None:
        mismatched = {
            role: model.model
            for role, model in request.training_models.items()
            if model.model != self.spec.model_name
        }
        if mismatched:
            raise ValueError(
                f"CD family {self.spec.model_name!r} received mismatched roles: {mismatched}"
            )


def _independent_pretrained_weights(
    request: TrainRequest,
    roles: tuple[str, ...],
) -> tuple[PretrainedWeightSource, ...]:
    sources = []
    for role in roles:
        model = request.training_models[role]
        if not model.init_weights:
            raise ValueError(f"CD training.models.{role}.init_weights is required")
        sources.append(
            PretrainedWeightSource(
                path=model.init_weights,
                model_path=role,
                loading_mode=model.weight_loading,
                source_prefix=str(model.config.get("init_source_prefix") or ""),
            )
        )
    return tuple(sources)


def family_from_spec(spec: Any) -> CdModelFamily:
    if isinstance(spec, CdModelFamily):
        return spec
    if not isinstance(spec, CdFamilySpec):
        raise TypeError(
            f"CD registry entry must be CdFamilySpec or CdModelFamily, got {type(spec).__name__}"
        )
    return CdModelFamily(spec)


__all__ = [
    "BuildTrainStream",
    "CdDataSupport",
    "CdFamilyComponents",
    "CdFamilySpec",
    "CdModelFamily",
    "CdNoisePair",
    "CdSchedule",
    "CdTeacherForcingAdapter",
    "PreparedCdBatch",
    "RoleModelBundle",
    "family_from_spec",
]
