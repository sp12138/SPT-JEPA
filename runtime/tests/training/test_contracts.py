from dataclasses import FrozenInstanceError

import pytest
import torch

from vforge import (
    OptimizerFactory,
    OptimizerSpec,
    ParallelState,
    SchedulerFactory,
    SchedulerSpec,
    StepFunction,
    StepOutput,
    StepReporter,
    TrainComponents,
    TrainContext,
)
from vforge.models.factory import ModelSpec
from vforge.training import (
    StepOutput as TrainingStepOutput,
)
from vforge.training.recipe import session_from_components


def test_explicit_training_contracts_are_public():
    assert TrainingStepOutput is StepOutput
    assert OptimizerFactory is not None
    assert OptimizerSpec is not None
    assert SchedulerFactory is not None
    assert SchedulerSpec is not None
    assert StepFunction is not None
    assert StepReporter is not None


def test_train_components_describe_framework_independent_step():
    model = torch.nn.Linear(2, 1)
    dataloader = [torch.ones(1, 2)]

    def optimizer_factory(module):
        return torch.optim.SGD(module.parameters(), lr=0.1)

    def step_fn(module, batch, step, context):
        assert step == 3
        assert context.device.type == "cpu"
        loss = module(batch).square().mean()
        return StepOutput(
            loss=loss,
            metrics={"raw_loss": loss.detach()},
            workload={"local_batch": batch.shape[0]},
        )

    components = TrainComponents(
        model=model,
        dataloader=dataloader,
        optimizer_factory=optimizer_factory,
        step_fn=step_fn,
    )
    context = TrainContext(
        device=torch.device("cpu"),
        parallel_state=ParallelState(),
    )

    output = components.step_fn(
        components.model,
        next(iter(components.dataloader)),
        3,
        context,
    )
    optimizer = components.optimizer_factory(components.model)

    assert output.loss.ndim == 0
    assert set(output.metrics) == {"raw_loss"}
    assert output.workload == {"local_batch": 1}
    assert isinstance(optimizer, torch.optim.SGD)
    assert components.scheduler_factory is None


def test_native_session_exposes_dataloader_resume_contract_separately() -> None:
    components = TrainComponents(
        model=torch.nn.Linear(1, 1),
        dataloader=[torch.ones(1, 1)],
        optimizer_factory=lambda model: torch.optim.SGD(
            model.parameters(),
            lr=0.1,
        ),
        step_fn=lambda *_args: StepOutput(loss=torch.tensor(0.0)),
        dataloader_resume_contract={"source_kind": "unit", "digest": "stable"},
    )
    session = session_from_components(
        components,
        start_step=0,
        resume_path=None,
        reporters=(),
    )

    assert session.dataloader_resume_contract(components.dataloader) == {
        "source_kind": "unit",
        "digest": "stable",
    }


def test_native_session_exposes_engine_neutral_optimizer_specs() -> None:
    optimizer_spec = OptimizerSpec(learning_rate=2e-5)
    scheduler_spec = SchedulerSpec(name="constant", factor=0.5)
    components = TrainComponents(
        model=torch.nn.Linear(1, 1),
        dataloader=[],
        optimizer_factory=lambda _model: None,
        step_fn=lambda *_args: StepOutput(loss=torch.tensor(0.0)),
        optimizer_spec=optimizer_spec,
        scheduler_spec=scheduler_spec,
    )
    session = session_from_components(
        components,
        start_step=0,
        resume_path=None,
        reporters=(),
    )

    assert session.optimizer_spec() is optimizer_spec
    assert session.scheduler_spec() is scheduler_spec


def test_native_session_leaves_direct_meta_module_for_post_shard_materialization():
    class MetaModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(2, device="meta"))

    model = MetaModel()
    components = TrainComponents(
        model=model,
        dataloader=[],
        optimizer_factory=lambda _model: None,
        step_fn=lambda *_args, **_kwargs: StepOutput(loss=torch.tensor(0.0)),
    )
    session = session_from_components(
        components,
        start_step=0,
        resume_path=None,
        reporters=(),
    )

    session.setup(torch.device("cpu"), ParallelState())

    assert session.model() is model
    assert model.weight.is_meta


def test_legacy_model_spec_remains_a_compatible_input():
    model = torch.nn.Linear(1, 1)
    builds = []
    components = TrainComponents(
        model=ModelSpec(lambda: builds.append(True) or model),
        dataloader=[],
        optimizer_factory=lambda _model: None,
        step_fn=lambda *_args, **_kwargs: StepOutput(loss=torch.tensor(0.0)),
    )
    session = session_from_components(
        components,
        start_step=0,
        resume_path=None,
        reporters=(),
    )

    session.setup(torch.device("cpu"), ParallelState())

    assert builds == [True]
    assert session.model() is model


def test_contract_defaults_are_isolated_and_components_are_immutable():
    first = StepOutput(loss=torch.tensor(1.0))
    second = StepOutput(loss=torch.tensor(2.0))
    assert first.metrics == {}
    assert first.metrics is not second.metrics

    components = TrainComponents(
        model=torch.nn.Linear(1, 1),
        dataloader=[],
        optimizer_factory=lambda model: torch.optim.SGD(
            model.parameters(),
            lr=0.1,
        ),
        step_fn=lambda *_args: StepOutput(loss=torch.tensor(0.0)),
    )
    with pytest.raises(FrozenInstanceError):
        components.model = torch.nn.Identity()  # type: ignore[misc]
