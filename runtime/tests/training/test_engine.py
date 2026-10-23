import subprocess
import sys
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vforge import (
    FSDP2Config,
    MCoreDataParallelConfig,
    MixedPrecisionConfig,
    OptimizerSpec,
    ParallelConfig,
    ParallelizeConfig,
    SchedulerSpec,
    TrainerConfig,
)
from vforge.distributed.parallel_state import ParallelState
from vforge.training.checkpoint import CheckpointConfig
from vforge.training.engine import (
    ENGINE_NAMES,
    EnginePolicy,
    EngineRunContext,
    TrainingEngine,
    create_engine,
)
from vforge.training.engine.fsdp import FsdpEngine
from vforge.training.engine.megatron import MegatronEngine
from vforge.training.recipe import StepContext, StepOutput


def _parallelize() -> ParallelizeConfig:
    return ParallelizeConfig(
        parallel=ParallelConfig(),
        fsdp2=FSDP2Config(enable=False),
    )


def _policy() -> EnginePolicy:
    return EnginePolicy(parallelize=_parallelize())


def _megatron_policy() -> EnginePolicy:
    return EnginePolicy(
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(dp_mode="ddp"),
            fsdp2=FSDP2Config(enable=False),
        )
    )


def test_fsdp_stays_the_default_engine():
    config = TrainerConfig(max_steps=1, parallelize=_parallelize())

    assert config.engine == "fsdp2"
    engine = create_engine(config.engine, _policy())
    assert isinstance(engine, FsdpEngine)
    assert engine.name == "fsdp2"


def test_fsdp_update_changes_only_declared_update_target() -> None:
    class Bundle(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = nn.Linear(1, 1, bias=False)
            self.critic = nn.Linear(1, 1, bias=False)

    bundle = Bundle()
    generator_before = bundle.generator.weight.detach().clone()
    critic_before = bundle.critic.weight.detach().clone()
    optimizer = torch.optim.SGD(bundle.generator.parameters(), lr=0.1)
    engine = FsdpEngine(EnginePolicy(parallelize=_parallelize(), gradient_clip_val=0.5))
    context = StepContext(device=torch.device("cpu"), parallel_state=ParallelState())

    result = engine.run_update(
        model=bundle,
        microbatches=iter((torch.ones(1, 1),)),
        step=0,
        context=context,
        compute_step=lambda model, batch, _step, _context: StepOutput(
            loss=model.generator(batch).square().mean()
        ),
        optimizer=optimizer,
        scheduler=None,
        run=EngineRunContext(precision_context=nullcontext),
        update_target=bundle.generator,
    )

    assert result.output.loss.ndim == 0
    assert not torch.equal(bundle.generator.weight, generator_before)
    assert torch.equal(bundle.critic.weight, critic_before)
    assert bundle.generator.weight.grad is not None
    assert bundle.critic.weight.grad is None


def test_training_engine_is_a_runtime_enforced_contract() -> None:
    class IncompleteEngine(TrainingEngine):
        name = "incomplete"

    with pytest.raises(TypeError, match="abstract"):
        IncompleteEngine()

    assert issubclass(FsdpEngine, TrainingEngine)
    assert issubclass(MegatronEngine, TrainingEngine)


def test_engine_forward_enters_precision_context() -> None:
    engine = create_engine("fsdp2", _policy())
    events: list[str] = []

    @contextmanager
    def precision_context():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    result = engine.run_forward(
        lambda: events.append("forward") or "result",
        run=EngineRunContext(precision_context=precision_context),
    )

    assert result == "result"
    assert events == ["enter", "forward", "exit"]


def test_engine_names_are_the_single_source_of_valid_engines():
    assert ENGINE_NAMES == ("fsdp2", "megatron")

    with pytest.raises(ValueError, match="unsupported training engine"):
        create_engine("deepspeed", _policy())
    with pytest.raises(ValueError, match="unsupported engine"):
        TrainerConfig(
            max_steps=1,
            parallelize=_parallelize(),
            engine="deepspeed",
        )


def test_megatron_at_tp1_is_selectable_without_importing_mcore_eagerly():
    """Selecting it should not require a GPU or import MCore yet.

    MCore DDP and DistributedOptimizer are installed after distributed runtime
    initialization and model materialization, so engine policy validation must
    stay importable on a machine that cannot import triton.
    """
    before = {name for name in sys.modules if name.startswith("vforge.mcore")}

    engine = create_engine("megatron", _megatron_policy())

    assert engine.name == "megatron"
    assert isinstance(engine, MegatronEngine)
    # A delta rather than an absolute. sys.modules is process-wide, so whether
    # mcore is already in it says something about whichever test ran first, not
    # about this engine; what belongs to the constructor is that it imports nothing.
    assert not {name for name in sys.modules if name.startswith("vforge.mcore")} - before


def test_train_entry_runs_model_preflight_before_runtime(monkeypatch) -> None:
    import vforge.tasks.train as train_task

    runtime = SimpleNamespace(
        resolved_config_hash="test",
        config=SimpleNamespace(
            model=SimpleNamespace(name="wan22_a14b"),
            latent_cache=SimpleNamespace(mode="offline"),
            parallel=SimpleNamespace(engine="megatron"),
        ),
        trainer=TrainerConfig(
            max_steps=1,
            parallelize=_megatron_policy().parallelize,
            engine="megatron",
        ),
    )
    monkeypatch.setattr(train_task, "build_and_snapshot", lambda _cfg: runtime)
    request = object()
    monkeypatch.setattr(train_task, "train_request", lambda _cfg: request)

    def preflight(actual_request, engine):
        assert actual_request is request
        assert engine == "megatron"
        raise NotImplementedError("train_expert=high or low")

    monkeypatch.setattr(train_task, "resolve_train_preflight", lambda _model: preflight)

    def unexpected_runtime(*_args, **_kwargs):
        raise AssertionError("distributed runtime started before optimizer preflight")

    monkeypatch.setattr(train_task, "initialize_runtime", unexpected_runtime)

    with pytest.raises(NotImplementedError, match="train_expert=high or low"):
        train_task.run_train(object())


def test_megatron_rejects_ulysses_rather_than_reinterpreting_it():
    with pytest.raises(NotImplementedError, match="does not run Ulysses"):
        create_engine(
            "megatron",
            EnginePolicy(
                parallelize=ParallelizeConfig(
                    parallel=ParallelConfig(usp_size=2),
                    fsdp2=FSDP2Config(enable=False),
                )
            ),
        )
    with pytest.raises(NotImplementedError, match="pipeline parallel"):
        create_engine(
            "megatron",
            EnginePolicy(
                parallelize=ParallelizeConfig(
                    parallel=ParallelConfig(pp_size=2),
                    fsdp2=FSDP2Config(enable=False),
                )
            ),
        )
    with pytest.raises(NotImplementedError, match="single microbatch"):
        create_engine(
            "megatron",
            EnginePolicy(parallelize=_parallelize(), gradient_accumulation_steps=2),
        )


def _tp_trainer_config(checkpoint: CheckpointConfig) -> TrainerConfig:
    return TrainerConfig(
        max_steps=1,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(tp_size=2, dp_mode="ddp"),
            fsdp2=FSDP2Config(enable=False),
        ),
        engine="megatron",
        checkpoint=checkpoint,
    )


def test_mcore_distributed_optimizer_requires_native_checkpoint_format():
    with pytest.raises(NotImplementedError, match="format='mcore'"):
        _tp_trainer_config(
            CheckpointConfig(
                enabled=True,
                dirpath="/tmp/unused",
                save_final=True,
                state="full",
                format="dcp",
                save_optimizer=True,
            )
        )

    with pytest.raises(NotImplementedError, match="format='mcore'"):
        TrainerConfig(
            max_steps=1,
            parallelize=ParallelizeConfig(
                parallel=ParallelConfig(dp_mode="ddp"),
                fsdp2=FSDP2Config(enable=False),
            ),
            engine="megatron",
            checkpoint=CheckpointConfig(
                enabled=True,
                dirpath="/tmp/unused",
                save_final=True,
                state="full",
                format="dcp",
                save_optimizer=True,
            ),
        )

    # Weights are supported, by either route into them.
    _tp_trainer_config(
        CheckpointConfig(
            enabled=True,
            dirpath="/tmp/unused",
            save_final=True,
            state="full",
            format="dcp",
            save_optimizer=False,
        )
    )
    _tp_trainer_config(
        CheckpointConfig(enabled=True, dirpath="/tmp/unused", save_final=True, state="weights")
    )
    _tp_trainer_config(CheckpointConfig(enabled=False))
    _tp_trainer_config(
        CheckpointConfig(
            enabled=True,
            dirpath="/tmp/unused",
            save_final=True,
            state="full",
            format="mcore",
            save_optimizer=True,
        )
    )

    # The gap belongs to mcore, not to tp. On fsdp2 the parameters are DTensors
    # in their own right, so the optimizer state derived from them carries the
    # tp axis with nothing declared, and the same config is allowed.
    TrainerConfig(
        max_steps=1,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(tp_size=2),
            fsdp2=FSDP2Config(enable=True),
        ),
        engine="fsdp2",
        checkpoint=CheckpointConfig(
            enabled=True,
            dirpath="/tmp/unused",
            save_final=True,
            state="full",
            format="dcp",
            save_optimizer=True,
        ),
    )
    with pytest.raises(ValueError, match="requires engine='megatron'"):
        TrainerConfig(
            max_steps=1,
            parallelize=_parallelize(),
            engine="fsdp2",
            checkpoint=CheckpointConfig(
                enabled=True,
                dirpath="/tmp/unused",
                save_final=True,
                state="weights",
                format="mcore",
            ),
        )


def test_megatron_refuses_to_let_fsdp_reshard_what_mcore_already_split():
    """Two owners of one weight's layout, and the checkpoint would believe fsdp.

    ``fully_shard`` replaces every parameter with a DTensor on the data mesh --
    it does that even where the mesh holds a single rank and shards nothing --
    and what it wraps is already mcore's tp slice. A state dict then reports the
    data axis and no other, so a checkpoint records half a weight as if it were
    whole. Refused at construction, before a 14B model is built to be rejected.
    """
    with pytest.raises(NotImplementedError, match="cannot both shard"):
        create_engine(
            "megatron",
            EnginePolicy(
                parallelize=ParallelizeConfig(
                    parallel=ParallelConfig(tp_size=2),
                    fsdp2=FSDP2Config(enable=True),
                )
            ),
        )

    # MCore alone owns the layout and is supported.
    create_engine(
        "megatron",
        EnginePolicy(
            parallelize=ParallelizeConfig(
                parallel=ParallelConfig(tp_size=2, dp_mode="ddp"),
                fsdp2=FSDP2Config(enable=False),
            )
        ),
    )


def test_fsdp_accepts_tensor_parallelism_and_composes_it_with_sharding():
    """Both axes are DTensor here, so unlike megatron they can be stacked."""
    create_engine(
        "fsdp2",
        EnginePolicy(
            parallelize=ParallelizeConfig(
                parallel=ParallelConfig(tp_size=2, dp_size=2),
                fsdp2=FSDP2Config(enable=True),
            )
        ),
    )


def test_fsdp_rejects_a_policy_it_cannot_execute_at_construction():
    # Before collectives are up, so a rejected policy leaves nothing behind.
    with pytest.raises(NotImplementedError, match="pipeline parallel"):
        create_engine(
            "fsdp2",
            EnginePolicy(
                parallelize=ParallelizeConfig(
                    parallel=ParallelConfig(pp_size=2),
                    fsdp2=FSDP2Config(enable=False),
                )
            ),
        )
    assert (
        create_engine(
            "fsdp2", EnginePolicy(parallelize=_parallelize(), gradient_accumulation_steps=2)
        ).policy.gradient_accumulation_steps
        == 2
    )
    with pytest.raises(NotImplementedError, match="fsdp2 context-parallel backend"):
        create_engine(
            "fsdp2",
            EnginePolicy(
                parallelize=ParallelizeConfig(
                    parallel=ParallelConfig(cp_size=2),
                    fsdp2=FSDP2Config(enable=False),
                )
            ),
        )
    with pytest.raises(NotImplementedError, match="cannot combine.*DDP"):
        create_engine(
            "fsdp2",
            EnginePolicy(
                parallelize=ParallelizeConfig(
                    parallel=ParallelConfig(
                        dp_size=2,
                        tp_size=2,
                        dp_mode="ddp",
                    ),
                    fsdp2=FSDP2Config(enable=False),
                )
            ),
        )


def test_megatron_context_parallel_requires_replicated_ddp() -> None:
    with pytest.raises(NotImplementedError, match="set parallel.dp_mode=ddp"):
        create_engine(
            "megatron",
            EnginePolicy(
                parallelize=ParallelizeConfig(
                    parallel=ParallelConfig(cp_size=2, dp_mode="fsdp2"),
                    fsdp2=FSDP2Config(enable=True),
                )
            ),
        )

    with pytest.raises(NotImplementedError, match="set parallel.dp_mode=ddp"):
        create_engine(
            "megatron",
            EnginePolicy(
                parallelize=ParallelizeConfig(
                    parallel=ParallelConfig(cp_size=2, dp_mode="fsdp2"),
                    fsdp2=FSDP2Config(enable=False),
                )
            ),
        )

    create_engine(
        "megatron",
        EnginePolicy(
            parallelize=ParallelizeConfig(
                parallel=ParallelConfig(cp_size=2, dp_mode="ddp"),
                fsdp2=FSDP2Config(enable=False),
            )
        ),
    )


def test_megatron_installs_its_context_parallel_backend_before_model_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.nn as nn

    import vforge.training.engine.megatron as megatron_module
    from vforge.distributed.context_parallel import (
        get_context_parallel_backend,
        reset_context_parallel,
    )

    def assert_backend_is_ready(model: nn.Module, *_args, **_kwargs) -> nn.Module:
        assert get_context_parallel_backend().name == "mcore"
        return model

    monkeypatch.setattr(megatron_module, "apply_mcore_tensor_parallel", lambda model: None)
    monkeypatch.setattr(megatron_module, "install_tp_state_dict_hooks", lambda model: None)
    monkeypatch.setattr(megatron_module, "parallelize_model", assert_backend_is_ready)

    reset_context_parallel()
    engine = create_engine(
        "megatron",
        EnginePolicy(
            parallelize=ParallelizeConfig(
                parallel=ParallelConfig(cp_size=2, dp_mode="ddp"),
                fsdp2=FSDP2Config(enable=False),
            )
        ),
    )

    try:
        engine.prepare_model(nn.Linear(2, 2))
        assert get_context_parallel_backend().name == "mcore"
    finally:
        reset_context_parallel()


def test_megatron_delegates_replicated_data_parallel_to_mcore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.nn as nn

    import vforge.training.engine.megatron as megatron_module

    model = nn.Linear(2, 2)
    wrapped = nn.Sequential(model)
    communication = MCoreDataParallelConfig(
        overlap_grad_reduce=False,
        overlap_param_gather=True,
    )

    def apply(candidate, *, config):
        assert candidate is model
        assert config is communication
        return wrapped

    monkeypatch.setattr(
        megatron_module,
        "apply_mcore_data_parallel",
        apply,
    )

    engine = create_engine(
        "megatron",
        EnginePolicy(
            parallelize=ParallelizeConfig(
                parallel=ParallelConfig(dp_mode="ddp"),
                fsdp2=FSDP2Config(enable=False),
                mcore_ddp=communication,
            )
        ),
    )

    assert engine.prepare_data_parallel(model) is wrapped


def test_fsdp_engine_owns_model_materialization() -> None:
    with torch.device("meta"):
        model = torch.nn.Linear(2, 2)

    materialized = create_engine("fsdp2", _policy()).materialize_model(model, "cpu")

    assert materialized is model
    assert not model.weight.is_meta


def test_megatron_materializes_before_restoring_and_validating_tp_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.nn as nn

    import vforge.training.engine.megatron as megatron_module

    source = nn.Linear(2, 2)
    materialized = nn.Sequential(source)
    events: list[str] = []

    def materialize(candidate, device):
        assert candidate is source
        assert device == "cpu"
        events.append("materialize")
        return materialized

    def restore(candidate):
        assert candidate is materialized
        events.append("restore")

    def validate(candidate):
        assert candidate is materialized
        events.append("validate")

    monkeypatch.setattr(megatron_module, "materialize_meta_module", materialize)
    monkeypatch.setattr(
        megatron_module,
        "restore_mcore_tensor_parallel_parameter_attributes",
        restore,
    )
    monkeypatch.setattr(
        megatron_module,
        "validate_mcore_tensor_parallel_parameter_attributes",
        validate,
    )

    result = create_engine("megatron", _megatron_policy()).materialize_model(source, "cpu")

    assert result is materialized
    assert events == ["materialize", "restore", "validate"]


def test_megatron_materializes_model_parameters_in_compute_dtype() -> None:
    class ModelWithRuntimeBuffer(torch.nn.Linear):
        def reset_non_persistent_buffers(self) -> None:
            self.rope = torch.ones(2, dtype=torch.float64, device=self.weight.device)

    with torch.device("meta"):
        model = ModelWithRuntimeBuffer(2, 2)
        model.register_buffer(
            "rope",
            torch.ones(2, dtype=torch.float64),
            persistent=False,
        )
    engine = create_engine(
        "megatron",
        EnginePolicy(
            parallelize=ParallelizeConfig(
                parallel=ParallelConfig(dp_mode="ddp"),
                fsdp2=FSDP2Config(
                    enable=False,
                    mixed_precision=MixedPrecisionConfig(
                        enable=True,
                        param_dtype="bfloat16",
                    ),
                ),
            )
        ),
    )

    materialized = engine.materialize_model(model, "cpu")

    assert materialized.weight.dtype is torch.bfloat16
    assert materialized.bias.dtype is torch.bfloat16
    assert materialized.rope.dtype is torch.float64


def test_megatron_owns_distributed_optimizer_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.nn as nn

    import vforge.training.mcore_optimizer as optimizer_module

    model = nn.Linear(2, 2)
    optimizer = object()
    scheduler = object()
    optimizer_spec = OptimizerSpec(learning_rate=2e-5)
    scheduler_spec = SchedulerSpec(name="constant", factor=0.5)
    observed = {}

    def build(candidate, requested_optimizer, requested_scheduler, *, clip_grad):
        observed.update(
            model=candidate,
            optimizer_spec=requested_optimizer,
            scheduler_spec=requested_scheduler,
            clip_grad=clip_grad,
        )
        return optimizer, scheduler

    monkeypatch.setattr(optimizer_module, "build_mcore_distributed_optimizer", build)
    engine = create_engine(
        "megatron",
        EnginePolicy(
            parallelize=_megatron_policy().parallelize,
            gradient_clip_val=1.0,
        ),
    )

    assert engine.configure_optimizers(model, optimizer_spec, scheduler_spec) == (
        optimizer,
        scheduler,
    )
    assert observed == {
        "model": model,
        "optimizer_spec": optimizer_spec,
        "scheduler_spec": scheduler_spec,
        "clip_grad": 1.0,
    }


def test_megatron_reloads_fp32_pretrained_state_into_mcore_master(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.nn as nn

    from vforge.training.mcore_optimizer import MCoreDistributedOptimizer

    expected = {"weight": torch.tensor([1.001234], dtype=torch.float32)}
    observed = {}

    class Inner:
        def reload_model_params(self, *, state_dict):
            observed["state_dict"] = state_dict

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.module = nn.Linear(1, 1)

    wrapped = Wrapper()
    optimizer = MCoreDistributedOptimizer(wrapped, Inner(), clip_grad=0.0)
    sources = (object(),)
    monkeypatch.setattr(
        "vforge.models.loading.load_pretrained_optimizer_main_param_state",
        lambda model, requested: observed.update(model=model, sources=requested) or expected,
    )

    create_engine("megatron", _megatron_policy()).initialize_optimizer_from_pretrained(
        wrapped,
        optimizer,
        sources,
    )

    assert observed == {
        "model": wrapped.module,
        "sources": sources,
        "state_dict": expected,
    }


def test_megatron_requires_engine_neutral_optimizer_configuration() -> None:
    engine = create_engine("megatron", _megatron_policy())

    with pytest.raises(ValueError, match="OptimizerSpec"):
        engine.configure_optimizers(torch.nn.Linear(2, 2), None, None)


def test_megatron_rejects_fsdp_mode_before_model_construction() -> None:
    with pytest.raises(NotImplementedError, match="owns MCore DDP"):
        create_engine("megatron", _policy())


def test_importing_vforge_does_not_pull_in_megatron():
    probe = (
        "import sys, vforge; "
        "print(any(name == 'megatron' or name.startswith('megatron.') "
        "for name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"


@pytest.mark.parametrize("separate_target", [False, True])
def test_default_clipper_preserves_logical_contribution_filter(separate_target):
    from vforge.training.diagnostics import LogicalReduction

    model = nn.Linear(1, 1)
    root = nn.ModuleDict({"generator": model}) if separate_target else model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    engine = FsdpEngine(EnginePolicy(parallelize=_parallelize(), gradient_clip_val=1))
    result = engine.run_update(
        model=root,
        microbatches=iter((None,)),
        step=0,
        context=StepContext(device=torch.device("cpu"), parallel_state=ParallelState()),
        compute_step=lambda *_: StepOutput(loss=(3 * model.weight + 4 * model.bias).sum()),
        optimizer=optimizer,
        scheduler=None,
        run=EngineRunContext(
            precision_context=nullcontext,
            logical_reduction=LogicalReduction(frozenset({id(model.weight)})),
        ),
        update_target=model,
    )
    # Non-contributing replicas are omitted from the norm but must still be scaled.
    assert result.diagnostics["diag/grad_norm_post_reduce"] == 3.0
    torch.testing.assert_close(model.weight.grad, torch.full_like(model.weight, 3 / (3 + 1e-6)))
    torch.testing.assert_close(model.bias.grad, torch.full_like(model.bias, 4 / (3 + 1e-6)))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"gradient_norm_dtype": "float16"}, "gradient_norm_dtype"),
        ({"engine": "megatron", "gradient_norm_dtype": "float64"}, "FSDP2"),
    ],
)
def test_gradient_clipping_policy_rejects_invalid_combinations(overrides, message):
    with pytest.raises(ValueError, match=message):
        TrainerConfig(max_steps=1, parallelize=_parallelize(), **overrides)
    policy_overrides = {key: value for key, value in overrides.items() if key != "engine"}
    with pytest.raises(ValueError, match=message):
        create_engine(
            overrides.get("engine", "fsdp2"),
            EnginePolicy(parallelize=_parallelize(), **policy_overrides),
        )
