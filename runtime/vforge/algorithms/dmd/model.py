"""Model composition, adapters, and runtime contracts for DMD."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Protocol

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from vforge.algorithms.base import Algorithm
from vforge.models.loading import PretrainedWeightSource
from vforge.tasks.requests import TrainRequest
from vforge.utils.rng import RandomStreams

LatentBundle = Mapping[str, torch.Tensor]
RoleCapability = Literal["generator", "critic"]


class FlowScheduler(Protocol):
    """Scheduler arrays required by DMD noise conversion."""

    timesteps: torch.Tensor
    sigmas: torch.Tensor


@dataclass(frozen=True)
class PreparedDmdBatch[ModelInputsT]:
    """Clean latent components plus model-specific conditioning."""

    clean: LatentBundle
    model_inputs: ModelInputsT


@dataclass(frozen=True)
class DmdNoiseLevel[ModelTimestepT]:
    """One normalized noise position resolved for a concrete model."""

    normalized: torch.Tensor
    sigmas: LatentBundle
    model_timestep: ModelTimestepT


@dataclass(frozen=True)
class DmdRolloutBlock[BlockSelectionT]:
    """One ordered rollout unit and its initial latent value."""

    selection: BlockSelectionT
    initial_value: LatentBundle
    is_anchor: bool = False


class DmdGeneratorAdapter(ABC):
    """Model operations required by the generic self-forcing rollout."""

    @abstractmethod
    def open_rollout_context(
        self,
        generator: nn.Module,
        batch: PreparedDmdBatch[Any],
    ) -> AbstractContextManager[Any]:
        """Open model-specific rollout state shared by all dependent blocks."""

    @abstractmethod
    def block_layout(
        self,
        batch: PreparedDmdBatch[Any],
        initial_noise: LatentBundle,
    ) -> Sequence[DmdRolloutBlock[Any]]:
        """Split one sample into ordered rollout blocks."""

    @abstractmethod
    def denoise_block(
        self,
        generator: nn.Module,
        noisy_block: LatentBundle,
        selection: Any,
        noise_level: DmdNoiseLevel[Any],
        batch: PreparedDmdBatch[Any],
        context: Any,
    ) -> LatentBundle:
        """Predict one block using the current rollout context."""

    @abstractmethod
    def commit_block(
        self,
        generator: nn.Module,
        selection: Any,
        context_value: LatentBundle,
        context_noise_level: DmdNoiseLevel[Any],
        batch: PreparedDmdBatch[Any],
        context: Any,
    ) -> None:
        """Make one completed block available to later dependent blocks."""

    @abstractmethod
    def concat_blocks(
        self,
        outputs: Sequence[tuple[Any, LatentBundle]],
        batch: PreparedDmdBatch[Any],
    ) -> LatentBundle:
        """Restore ordered block outputs to the canonical latent bundle."""


class DmdCriticAdapter(ABC):
    """Bidirectional teacher/critic prediction required by DMD."""

    @abstractmethod
    def predict_velocity(
        self,
        tower: nn.Module,
        noisy: LatentBundle,
        noise_level: DmdNoiseLevel[Any],
        batch: PreparedDmdBatch[Any],
        conditioning: Literal["conditional", "unconditional"],
    ) -> LatentBundle:
        """Return canonical velocity predictions for one score tower."""

    def score_window(self, generated, batch):
        """Default scores use the same window as the generator."""
        return generated, batch

    def generated_region(self, values, batch):
        """Project score-window values back to generated tokens."""
        return values

    def predict_x0(self, tower, noisy, noise_level, batch, conditioning) -> LatentBundle:
        """Convert velocity to x0; families may override the precision boundary."""
        velocity = self.predict_velocity(tower, noisy, noise_level, batch, conditioning)
        return {
            key: value - noise_level.sigmas[key].to(value) * velocity[key]
            for key, value in noisy.items()
        }


class RoleModelBundle(nn.Module):
    """Register arbitrary model roles as independent execution roots."""

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
        roots = []
        for role in self._role_order:
            model = self.model_for(role)
            resolver = getattr(model, "fsdp_execution_roots", None)
            roots.extend(resolver() if callable(resolver) else (model,))
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
            enable = getattr(module, "gradient_checkpointing_enable", None)
            if callable(enable):
                enable(gradient_checkpointing_kwargs=kwargs)
                continue
            enable = getattr(module, "enable_gradient_checkpointing", None)
            if callable(enable):
                enable(gradient_checkpointing_func=checkpoint_func)

    def train(self, mode: bool = True) -> RoleModelBundle:
        super().train(mode)
        self._restore_frozen_roles()
        return self

    def _restore_frozen_roles(self) -> None:
        for role in self._frozen_roles:
            module = self.model_for(role)
            module.requires_grad_(False)
            module.eval()


PrepareBatch = Callable[[Any, torch.device], PreparedDmdBatch]
Rollout = Callable[
    [
        nn.Module,
        DmdGeneratorAdapter,
        PreparedDmdBatch,
        RandomStreams,
        Literal["generator", "critic"],
        int,
    ],
    LatentBundle,
]
SampleScoreNoise = Callable[
    [LatentBundle, PreparedDmdBatch, RandomStreams, int],
    tuple[DmdNoiseLevel, LatentBundle, LatentBundle],
]
ValidMaskFactory = Callable[
    [PreparedDmdBatch],
    Mapping[str, torch.Tensor | None],
]
WorkloadFactory = Callable[
    [LatentBundle, nn.Module],
    Mapping[str, float | int] | None,
]
MetricsFactory = Callable[
    [DmdNoiseLevel, LatentBundle | None],
    Mapping[str, torch.Tensor],
]
BuildTrainStream = Callable[
    [int],
    tuple[Iterable[Any], Mapping[str, Any]],
]
ResolveNoiseLevel = Callable[
    [FlowScheduler, torch.Tensor, PreparedDmdBatch[Any]],
    DmdNoiseLevel[Any],
]
PostprocessScoreNoise = Callable[
    [LatentBundle, LatentBundle, DmdNoiseLevel[Any]],
    tuple[LatentBundle, DmdNoiseLevel[Any]],
]
ScoreNoiseShape = Callable[[PreparedDmdBatch[Any]], tuple[int, ...]]
LossPartitions = Callable[[PreparedDmdBatch[Any]], Sequence[Mapping[str, slice]]]


@dataclass(frozen=True)
class DmdFamilyComponents:
    """Model-owned values required by a generic DMD algorithm."""

    models: Mapping[str, nn.Module]
    build_train_stream: BuildTrainStream
    prepare_batch: PrepareBatch
    generator_adapter: DmdGeneratorAdapter
    critic_adapters: Mapping[str, DmdCriticAdapter]
    pretrained_weights: tuple[PretrainedWeightSource, ...]
    resolve_noise_level: ResolveNoiseLevel
    score_noise_shape: ScoreNoiseShape
    valid_masks: ValidMaskFactory
    postprocess_score_noise: PostprocessScoreNoise | None = None
    workload: WorkloadFactory | None = None
    metrics: MetricsFactory | None = None
    loss_partitions: LossPartitions | None = None
    resolve_rollout_noise_level: ResolveNoiseLevel | None = None


@dataclass(frozen=True)
class DistributionMatchingData:
    """Data stream and model-specific conversions used by one DMD run."""

    dataloader: Iterable[Any]
    prepare_batch: PrepareBatch
    rollout: Rollout
    sample_score_noise: SampleScoreNoise
    valid_masks: ValidMaskFactory
    batch_selector: Callable[[Any, str], Any]
    workload: WorkloadFactory | None = None
    metrics: MetricsFactory | None = None
    dataloader_resume_contract: Mapping[str, Any] | None = None
    loss_partitions: LossPartitions | None = None

    rollout_trace: Callable | None = None


ValidateFamily = Callable[[TrainRequest, str], None]
BuildRoleModel = Callable[[TrainRequest, str], nn.Module]
CreateGeneratorAdapter = Callable[[TrainRequest], DmdGeneratorAdapter]
CreateCriticAdapter = Callable[[TrainRequest, str], DmdCriticAdapter]


@dataclass(frozen=True)
class DmdDataSupport:
    """Model-specific data and noise hooks used by common family assembly."""

    build_train_stream: BuildTrainStream
    prepare_batch: PrepareBatch
    resolve_noise_level: ResolveNoiseLevel
    score_noise_shape: ScoreNoiseShape
    valid_masks: ValidMaskFactory
    postprocess_score_noise: PostprocessScoreNoise | None = None
    workload: WorkloadFactory | None = None
    metrics: MetricsFactory | None = None
    loss_partitions: LossPartitions | None = None
    resolve_rollout_noise_level: ResolveNoiseLevel | None = None


BuildDataSupport = Callable[[TrainRequest], DmdDataSupport]


@dataclass(frozen=True)
class DmdFamilySpec:
    """Declarative model hooks; lifecycle and role assembly stay common."""

    model_name: str
    validate: ValidateFamily
    build_role_model: BuildRoleModel
    build_data_support: BuildDataSupport
    create_generator_adapter: CreateGeneratorAdapter
    create_critic_adapter: CreateCriticAdapter
    build_weight_sources: (
        Callable[[TrainRequest, tuple[str, ...]], tuple[PretrainedWeightSource, ...]] | None
    ) = None

    algorithm_type: type[Algorithm] | None = None

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("DMD family model_name must be non-empty")


class DmdModelFamily:
    """Common validate/build lifecycle shared by every model family."""

    def __init__(self, spec: DmdFamilySpec) -> None:
        self.spec = spec

    def validate_request(
        self,
        request: TrainRequest,
        *,
        engine: str,
    ) -> None:
        self._validate_model_names(request)
        self.spec.validate(request, engine)

    def _validate_model_names(self, request: TrainRequest) -> None:
        mismatched = {
            role: model.model
            for role, model in request.training_models.items()
            if model.model != self.spec.model_name
        }
        if mismatched:
            raise ValueError(
                f"DMD family {self.spec.model_name!r} received mismatched roles: {mismatched}"
            )

    def build(
        self,
        request: TrainRequest,
        *,
        required_roles: frozenset[str],
        role_capabilities: Mapping[str, RoleCapability],
    ) -> DmdFamilyComponents:
        self._validate_model_names(request)
        missing = sorted(required_roles - set(request.training_models))
        if missing:
            raise ValueError(f"DMD training.models is missing required roles: {missing}")
        if set(role_capabilities) != set(required_roles):
            raise ValueError("DMD role capabilities must cover exactly required roles")
        support = self.spec.build_data_support(request)
        roles = tuple(role for role in request.training_models if role in required_roles)
        models = {role: self.spec.build_role_model(request, role) for role in roles}
        generator_roles = [
            role for role, capability in role_capabilities.items() if capability == "generator"
        ]
        if len(generator_roles) != 1:
            raise ValueError("DMD family assembly requires exactly one generator role")
        critic_roles = [
            role for role, capability in role_capabilities.items() if capability == "critic"
        ]
        return DmdFamilyComponents(
            models=models,
            build_train_stream=support.build_train_stream,
            prepare_batch=support.prepare_batch,
            generator_adapter=self.spec.create_generator_adapter(request),
            critic_adapters={
                role: self.spec.create_critic_adapter(request, role) for role in critic_roles
            },
            pretrained_weights=(self.spec.build_weight_sources or _independent_pretrained_weights)(
                request, roles
            ),
            resolve_noise_level=support.resolve_noise_level,
            score_noise_shape=support.score_noise_shape,
            valid_masks=support.valid_masks,
            postprocess_score_noise=support.postprocess_score_noise,
            workload=support.workload,
            metrics=support.metrics,
            loss_partitions=support.loss_partitions,
            resolve_rollout_noise_level=support.resolve_rollout_noise_level,
        )


def _independent_pretrained_weights(
    request: TrainRequest,
    roles: tuple[str, ...],
) -> tuple[PretrainedWeightSource, ...]:
    """Build one model-native initialization source for each role."""

    sources = []
    for role in roles:
        model = request.training_models[role]
        if not model.init_weights:
            raise ValueError(f"DMD training.models.{role}.init_weights is required")
        sources.append(
            PretrainedWeightSource(
                path=model.init_weights,
                model_path=role,
                loading_mode=model.weight_loading,
            )
        )
    return tuple(sources)


def family_from_spec(spec: Any) -> DmdModelFamily:
    """Resolve one registered definition into the common family runtime."""

    if isinstance(spec, DmdModelFamily):
        return spec
    if not isinstance(spec, DmdFamilySpec):
        raise TypeError(
            f"DMD registry entry must be DmdFamilySpec or DmdModelFamily, got {type(spec).__name__}"
        )
    return DmdModelFamily(spec)


__all__ = [
    "BuildTrainStream",
    "DmdCriticAdapter",
    "DmdDataSupport",
    "DmdFamilyComponents",
    "DmdFamilySpec",
    "DmdGeneratorAdapter",
    "DmdModelFamily",
    "DmdNoiseLevel",
    "DmdRolloutBlock",
    "DistributionMatchingData",
    "FlowScheduler",
    "LatentBundle",
    "MetricsFactory",
    "PostprocessScoreNoise",
    "PrepareBatch",
    "PreparedDmdBatch",
    "ResolveNoiseLevel",
    "RoleCapability",
    "RoleModelBundle",
    "Rollout",
    "SampleScoreNoise",
    "ScoreNoiseShape",
    "ValidMaskFactory",
    "WorkloadFactory",
    "family_from_spec",
]
