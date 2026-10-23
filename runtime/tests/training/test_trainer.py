import logging
import os
from dataclasses import replace

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

import vforge.training.engine.fsdp as fsdp_engine_module
import vforge.training.trainer as trainer_module
from vforge import (
    CheckpointConfig,
    FSDP2Config,
    ParallelConfig,
    ParallelizeConfig,
    Trainer,
    TrainerConfig,
)
from vforge.checkpointing.offload import (
    ActivationOffloadManager,
    AttentionOffloadPolicy,
)
from vforge.training.parity import ParityStepConfig
from vforge.training.recipe import StepOutput, TrainComponents


class _Reporter:
    def __init__(self):
        self.payloads = []
        self.closed = False

    def report_step(self, payload):
        self.payloads.append(payload)

    def close(self):
        self.closed = True


class _Recipe:
    def __init__(
        self,
        *,
        start: int = 0,
        batches=None,
        tokens_per_sample=0,
        resume_path=None,
    ):
        self.model = torch.nn.Linear(2, 1)
        self.reporter = _Reporter()
        self.start_step = start
        self.resume_path = resume_path
        self.tokens_per_sample = tokens_per_sample
        self.optimizer = None
        self.seen_models = []
        self.seen_steps = []
        # Keep pre-collated batches and a resumable cursor for full DCP tests.
        self.dataloader = StatefulDataLoader(
            list(batches) if batches is not None else [torch.ones(1, 2) for _ in range(8)],
            batch_size=None,
        )

    def build_components(self):
        return TrainComponents(
            model=self.model,
            dataloader=self.dataloader,
            optimizer_factory=self.build_optimizer,
            scheduler_factory=self.build_scheduler,
            step_fn=self.step,
        )

    def build_optimizer(self, model):
        self.seen_models.append(model)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        return self.optimizer

    def build_scheduler(self, optimizer):
        return None

    def loss(self, model, batch):
        return model(batch).square().mean()

    def step(self, model, batch, step, context):
        self.seen_steps.append((step, context.device.type, context.parallel_state.world_size))
        loss = self.loss(model, batch)
        workload = None
        if self.tokens_per_sample > 0:
            local_batch = int(batch.shape[0])
            workload = {
                "local_batch": local_batch,
                "local_tokens": local_batch * self.tokens_per_sample,
            }
        return StepOutput(loss=loss, metrics={"raw_loss": loss.detach()}, workload=workload)


def _fit_recipe(recipe, config) -> None:
    Trainer(config).fit(
        recipe.build_components(),
        start_step=recipe.start_step,
        resume_path=recipe.resume_path,
        reporters=(recipe.reporter,),
    )


def test_native_trainer_config_has_no_wan_or_parity_defaults(caplog):
    config = TrainerConfig(
        max_steps=1,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(),
            fsdp2=FSDP2Config(enable=False),
        ),
    )

    assert config.parity is None
    assert config.diagnostics.level == "basic"
    assert config.diagnostic_parameter_bucket is None
    assert not config.autocast
    assert "Global autocast is enabled" not in caplog.text


def test_trainer_config_warns_when_global_autocast_is_enabled(caplog):
    with caplog.at_level(logging.WARNING, logger="vforge.training.trainer"):
        config = TrainerConfig(
            max_steps=1,
            parallelize=ParallelizeConfig(
                parallel=ParallelConfig(),
                fsdp2=FSDP2Config(enable=False),
            ),
            autocast=True,
        )

    assert config.autocast
    assert "Global autocast is enabled (runtime.autocast=true)" in caplog.text
    assert "Set runtime.autocast=false" in caplog.text


def test_post_optimizer_callback_runs_after_weight_update_and_reports_metrics():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    observed: dict[str, float] = {}

    def step_fn(model, batch, step, context):
        del step, context
        return StepOutput(loss=model(batch).square().mean())

    def post_optimizer_step(model, step, context, output):
        del step, context, output
        observed["weight"] = float(model.weight.detach())
        return {"post_step_weight": model.weight.detach().square().mean()}

    reporter = _Reporter()
    components = TrainComponents(
        model=model,
        dataloader=StatefulDataLoader([torch.ones(1, 1)], batch_size=None),
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.1),
        step_fn=step_fn,
        post_optimizer_step=post_optimizer_step,
    )
    config = TrainerConfig(
        max_steps=1,
        precision="fp32",
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(),
            fsdp2=FSDP2Config(enable=False),
        ),
    )

    Trainer(config).fit(components, reporters=(reporter,))

    assert observed["weight"] == pytest.approx(0.8)
    assert reporter.payloads[-1]["metrics"]["post_step_weight"] == pytest.approx(0.64)


def test_offload_manager_uses_the_configured_policy():
    trainer = Trainer(
        TrainerConfig(
            max_steps=1,
            parallelize=ParallelizeConfig(
                parallel=ParallelConfig(),
                fsdp2=FSDP2Config(enable=False),
                gradient_checkpointing=True,
            ),
            attention_offload=AttentionOffloadPolicy(enable=True),
        )
    )
    manager = trainer._build_offload_manager()
    assert type(manager) is ActivationOffloadManager
    assert manager.config is trainer.config.attention_offload.config


@pytest.mark.parametrize("enabled", [False, True])
def test_trainer_passes_offload_context_during_model_preparation(monkeypatch, enabled):
    recipe = _Recipe()
    seen = []

    def parallelize(model, _config, *, gradient_checkpointing_kwargs=None, **_kwargs):
        assert model is recipe.model
        seen.append(gradient_checkpointing_kwargs)
        return model

    monkeypatch.setattr(fsdp_engine_module, "parallelize_model", parallelize)
    config = TrainerConfig(
        max_steps=1,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(),
            fsdp2=FSDP2Config(enable=False),
            gradient_checkpointing=True,
        ),
        attention_offload=AttentionOffloadPolicy(enable=enabled),
        precision="fp32",
    )
    _fit_recipe(recipe, config)

    assert len(seen) == 1
    if enabled:
        assert seen[0]["use_reentrant"] is False
        assert isinstance(seen[0]["context_fn"].__self__, ActivationOffloadManager)
    else:
        assert seen[0] is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"optimizer_step_mode": "invalid"}, "optimizer_step_mode"),
        (
            {"scale_grads_before_optimizer_step": 0},
            "scale_grads_before_optimizer_step",
        ),
        (
            {"quantize_after_optimizer_step": "fp16"},
            "quantize_after_optimizer_step",
        ),
        ({"emit_legacy_probe_aliases": 1}, "emit_legacy_probe_aliases"),
    ],
)
def test_parity_step_validation_is_centralized(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ParityStepConfig(**kwargs)


def test_trainer_owns_step_and_cleanup_lifecycle():
    recipe = _Recipe()
    before = recipe.model.weight.detach().clone()
    config = TrainerConfig(
        max_steps=2,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(),
            fsdp2=FSDP2Config(enable=False),
        ),
        precision="fp32",
    )

    _fit_recipe(recipe, config)

    assert recipe.reporter.closed
    assert [payload["step"] for payload in recipe.reporter.payloads] == [0, 1]
    assert not torch.equal(before, recipe.model.weight.detach().cpu())
    assert recipe.seen_models == [recipe.model]
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    assert recipe.seen_steps == [(0, device_type, 1), (1, device_type, 1)]


def test_trainer_closes_reporter_when_step_fails(monkeypatch):
    recipe = _Recipe()

    def fail_step(model, batch):
        raise RuntimeError("step failed")

    monkeypatch.setattr(recipe, "loss", fail_step)
    config = TrainerConfig(
        max_steps=1,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(),
            fsdp2=FSDP2Config(enable=False),
        ),
        precision="fp32",
    )

    with pytest.raises(RuntimeError, match="step failed"):
        _fit_recipe(recipe, config)

    assert recipe.reporter.closed


def test_trainer_saves_weights_only_pt_and_includes_checkpoint_in_step_time(
    tmp_path,
):
    recipe = _Recipe()
    config = TrainerConfig(
        max_steps=2,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(),
            fsdp2=FSDP2Config(enable=False),
        ),
        precision="fp32",
        checkpoint=CheckpointConfig(
            enabled=True,
            every_n_steps=2,
            dirpath=str(tmp_path / "checkpoints"),
        ),
    )

    _fit_recipe(recipe, config)

    assert os.path.isfile(tmp_path / "checkpoints" / "00002.pt")
    assert not os.path.exists(tmp_path / "checkpoints" / "00001.pt")
    final = recipe.reporter.payloads[-1]
    assert final["checkpoint_step"] == 2
    assert final["timing"]["checkpoint_s"] >= 0.0
    assert final["timing"]["step_s"] >= final["timing"]["step_compute_s"]
    assert (
        abs(
            final["timing"]["step_s"]
            - (final["timing"]["step_compute_s"] + final["timing"]["checkpoint_s"])
        )
        < 1e-3
    )


def test_trainer_respects_explicit_start_step():
    recipe = _Recipe(start=1)
    config = TrainerConfig(
        max_steps=3,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(),
            fsdp2=FSDP2Config(enable=False),
        ),
        precision="fp32",
    )

    _fit_recipe(recipe, config)

    assert [payload["step"] for payload in recipe.reporter.payloads] == [1, 2]


def test_pt_warmstart_ignores_full_resume_defaults_and_starts_from_zero(
    tmp_path,
):
    source = torch.nn.Linear(2, 1)
    path = tmp_path / "00001.pt"
    torch.save(source.state_dict(), path)
    recipe = _Recipe(start=1, resume_path=str(path))
    config = TrainerConfig(
        max_steps=2,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(),
            fsdp2=FSDP2Config(enable=False),
        ),
        precision="fp32",
    )

    _fit_recipe(recipe, config)

    assert [payload["step"] for payload in recipe.reporter.payloads] == [0, 1]


@pytest.mark.parametrize("accumulation", [1, 3])
def test_trainer_reports_throughput_from_step_output(accumulation):
    recipe = _Recipe(tokens_per_sample=1000)
    config = TrainerConfig(
        max_steps=1,
        gradient_accumulation_steps=accumulation,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(),
            fsdp2=FSDP2Config(enable=False),
        ),
        precision="fp32",
    )

    _fit_recipe(recipe, config)

    perf = recipe.reporter.payloads[-1]["perf"]
    assert perf["local_batch"] == accumulation
    assert perf["local_tokens"] == 1000 * accumulation
    assert perf["global_tokens"] == 1000 * accumulation
    assert perf["tokens_per_sec"] > 0.0
    assert perf["tokens_per_sec_per_gpu"] == perf["tokens_per_sec"]


class _RandomRecipe(_Recipe):
    def build_scheduler(self, optimizer):
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        return self.scheduler

    def step(self, model, batch, step, context):
        generator = context.random_streams.generator("input", seed=42, device=context.device)
        noise = torch.rand((), device=batch.device) + torch.rand(
            (), device=batch.device, generator=generator
        )
        loss = (model(batch) + noise).square().mean()
        return StepOutput(loss=loss, metrics={"raw_loss": loss.detach()})


def _full_dcp_config(tmp_path, *, max_steps, async_save=False, accumulation=1):
    return TrainerConfig(
        max_steps=max_steps,
        gradient_accumulation_steps=accumulation,
        seed=42,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(),
            fsdp2=FSDP2Config(enable=False),
        ),
        precision="fp32",
        checkpoint=CheckpointConfig(
            enabled=True,
            every_n_steps=2,
            dirpath=str(tmp_path / "checkpoints"),
            state="full",
            format="dcp",
            async_save=async_save,
        ),
    )


@pytest.mark.parametrize("async_save", [False, True])
@pytest.mark.parametrize("accumulation", [1, 2, 4])
def test_full_dcp_resume_matches_uninterrupted_training_bitwise(
    tmp_path,
    async_save,
    accumulation,
):
    batches = [torch.tensor([[float(index), float(index + 1)]]) for index in range(5)]

    torch.manual_seed(123)
    uninterrupted = _RandomRecipe(batches=batches)
    _fit_recipe(
        uninterrupted,
        _full_dcp_config(tmp_path / "continuous", max_steps=4, accumulation=accumulation),
    )

    torch.manual_seed(123)
    first = _RandomRecipe(batches=batches)
    split_root = tmp_path / "split"
    _fit_recipe(
        first,
        _full_dcp_config(
            split_root,
            max_steps=2,
            async_save=async_save,
            accumulation=accumulation,
        ),
    )
    resume_path = split_root / "checkpoints" / "00002.dcp"
    if async_save:
        assert resume_path.is_dir()
        assert first.reporter.payloads[-1]["checkpoint_pending_path"] == str(resume_path)
        assert (
            first.reporter.payloads[-1]["timing"]["checkpoint_s"]
            >= first.reporter.payloads[-1]["timing"]["checkpoint_foreground_s"]
        )

    torch.manual_seed(999)
    resumed = _RandomRecipe(
        batches=batches,
        resume_path=str(resume_path),
    )
    _fit_recipe(
        resumed,
        _full_dcp_config(split_root, max_steps=4, accumulation=accumulation),
    )

    assert [payload["step"] for payload in resumed.reporter.payloads] == [2, 3]
    assert [payload["loss"] for payload in resumed.reporter.payloads] == [
        payload["loss"] for payload in uninterrupted.reporter.payloads[2:]
    ]
    for actual, expected in zip(
        resumed.model.parameters(),
        uninterrupted.model.parameters(),
        strict=True,
    ):
        assert torch.equal(actual, expected)
    actual_optimizer = resumed.optimizer.state_dict()
    expected_optimizer = uninterrupted.optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    assert actual_optimizer["state"].keys() == expected_optimizer["state"].keys()
    for parameter_id, actual_state in actual_optimizer["state"].items():
        expected_state = expected_optimizer["state"][parameter_id]
        assert actual_state.keys() == expected_state.keys()
        for name, actual_value in actual_state.items():
            expected_value = expected_state[name]
            if isinstance(actual_value, torch.Tensor):
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()


def test_trainer_reuses_entrypoint_runtime_without_destroying_it():
    from vforge.distributed.runtime import (
        destroy_runtime,
        get_runtime,
        initialize_runtime,
    )
    from vforge.ops.config import OpsConfig

    config = TrainerConfig(
        max_steps=2,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(),
            fsdp2=FSDP2Config(enable=False),
        ),
        precision="fp32",
        log_timing=False,
    )
    context = initialize_runtime(
        config.parallelize.parallel,
        OpsConfig(backend="sdpa", allow_fallback=False),
        device=torch.device("cpu"),
    )
    components = TrainComponents(
        model=torch.nn.Linear(2, 1),
        dataloader=[torch.ones(1, 2)],
        optimizer_factory=lambda model: torch.optim.SGD(
            model.parameters(),
            lr=0.1,
        ),
        step_fn=lambda model, batch, step, _context: StepOutput(
            loss=model(batch).square().mean(),
        ),
    )
    try:
        Trainer(
            config=config,
            runtime_context=context,
        ).fit(components)
        assert get_runtime() is context
    finally:
        destroy_runtime()


def _failing_recipe(monkeypatch):
    recipe = _Recipe()

    def fail_step(model, batch):
        raise RuntimeError("step failed")

    monkeypatch.setattr(recipe, "loss", fail_step)
    config = TrainerConfig(
        max_steps=1,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(),
            fsdp2=FSDP2Config(enable=False),
        ),
        precision="fp32",
    )
    return recipe, config


def test_failing_step_logs_the_traceback_on_the_rank_that_failed(monkeypatch, caplog):
    recipe, config = _failing_recipe(monkeypatch)

    with caplog.at_level(logging.ERROR, logger="vforge.training.trainer"):
        with pytest.raises(RuntimeError, match="step failed"):
            _fit_recipe(recipe, config)

    assert "training failed on rank 0" in caplog.text
    assert "RuntimeError: step failed" in caplog.text


def test_failing_step_skips_the_teardown_that_blocks_on_other_ranks(monkeypatch):
    # Destroying a process group waits for every rank in it. A rank that is
    # already dying must not enter one, or the survivors hit the NCCL watchdog
    # instead of seeing the real error.
    recipe, config = _failing_recipe(monkeypatch)
    real_destroy = trainer_module.destroy_runtime
    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        return real_destroy(**kwargs)

    monkeypatch.setattr(trainer_module, "destroy_runtime", spy)

    with pytest.raises(RuntimeError, match="step failed"):
        _fit_recipe(recipe, config)

    assert calls == [{"destroy_process_group": False}]


def test_successful_run_destroys_the_runtime_it_owns(monkeypatch):
    recipe = _Recipe()
    config = TrainerConfig(
        max_steps=1,
        parallelize=ParallelizeConfig(
            parallel=ParallelConfig(),
            fsdp2=FSDP2Config(enable=False),
        ),
        precision="fp32",
    )
    real_destroy = trainer_module.destroy_runtime
    calls = []

    def spy(**kwargs):
        calls.append(kwargs)
        return real_destroy(**kwargs)

    monkeypatch.setattr(trainer_module, "destroy_runtime", spy)

    _fit_recipe(recipe, config)

    assert calls == [{}]


@pytest.mark.parametrize("norm_dtype", ["float32", "float64"])
def test_phased_clipper_factory_preserves_independent_tower_updates(norm_dtype):
    from copy import deepcopy

    from vforge.training.diagnostics import torch_gradient_clipper
    from vforge.training.phased_trainer import PhasedTrainer
    from vforge.training.recipe import PhasedTrainComponents, PhaseSpec

    model = torch.nn.ModuleDict({name: torch.nn.Linear(2, 1) for name in ("generator", "critic")})
    reference = deepcopy(model)
    batch = torch.tensor([[2.0, -3.0]])
    clip = torch_gradient_clipper(norm_dtype)
    for tower in reference.values():
        optimizer = torch.optim.SGD(tower.parameters(), lr=0.1)
        tower(batch).square().mean().backward()
        clip(tower, optimizer, 0.1)
        optimizer.step()
    phases = {
        name: PhaseSpec(
            step_fn=lambda root, batch, step, context, name=name: StepOutput(
                loss=root[name](batch).square().mean()
            ),
            target=lambda root, name=name: root[name],
            optimizer_factory=lambda tower: torch.optim.SGD(tower.parameters(), lr=0.1),
        )
        for name in model
    }
    config = TrainerConfig(
        max_steps=1,
        precision="fp32",
        gradient_clip_val=0.1,
        gradient_norm_dtype=norm_dtype,
        parallelize=ParallelizeConfig(parallel=ParallelConfig(), fsdp2=FSDP2Config(enable=False)),
        checkpoint=CheckpointConfig(enabled=False),
    )
    PhasedTrainer(config).fit(
        PhasedTrainComponents(
            model=model,
            dataloader=[batch],
            phases=phases,
            phase_order=tuple(phases),
            batch_selector=lambda batch, phase: batch,
            gradient_clipper_factory=torch_gradient_clipper,
        )
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)


def test_fp64_norm_without_recipe_support_fails_before_runtime(monkeypatch):
    def unexpected_runtime(*args, **kwargs):
        raise AssertionError("unsupported precision must fail before runtime creation")

    monkeypatch.setattr(trainer_module, "ensure_runtime", unexpected_runtime)
    config = TrainerConfig(
        max_steps=1,
        gradient_norm_dtype="float64",
        parallelize=ParallelizeConfig(parallel=ParallelConfig(), fsdp2=FSDP2Config(enable=False)),
    )
    with pytest.raises(ValueError, match="precision-aware clipper"):
        Trainer(config).fit(_Recipe().build_components())


@pytest.mark.parametrize("run_backward", [False, True])
def test_accumulation_streams_across_epochs_and_reports_whole_updates(run_backward):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    reporter = _Reporter()
    contexts, inputs, updates, callbacks, noises = [], [], [], [], []
    optimizer.register_step_post_hook(lambda *_: updates.append(True))
    reference = torch.Generator().manual_seed(42)

    def compute(model, batch, step, context):
        contexts.append((step, context))
        inputs.append(int(batch.item()))
        noise = torch.rand(
            8,
            device=context.device,
            generator=context.random_streams.generator("input", seed=42, device=context.device),
        )
        expected = torch.rand(8, generator=reference)
        torch.testing.assert_close(noise, expected, rtol=0, atol=0)
        noises.append(noise.cpu())
        loss = model(batch).square().mean()
        return StepOutput(
            loss,
            metrics={"loss_scalar": loss.detach()},
            workload={"local_batch": 1, "local_tokens": 10},
            metadata={"sample_id": int(batch.item())},
        )

    def post_update(model, step, context, output):
        callbacks.append(step)
        assert len(output.metadata["microbatches"]) == 2
        return {}

    config = TrainerConfig(
        max_steps=2,
        parallelize=ParallelizeConfig(parallel=ParallelConfig(), fsdp2=FSDP2Config(enable=False)),
        precision="fp32",
        checkpoint=CheckpointConfig(enabled=False),
        gradient_accumulation_steps=2,
        run_backward=run_backward,
    )
    components = TrainComponents(
        model=model,
        dataloader=StatefulDataLoader(
            [torch.tensor([[float(i)]]) for i in (1, 2, 3)], batch_size=None
        ),
        optimizer_factory=lambda _: optimizer,
        scheduler_factory=lambda _: scheduler,
        step_fn=compute,
        post_optimizer_step=post_update,
    )
    Trainer(config).fit(components, reporters=(reporter,))

    assert [step for step, _ in contexts] == [0, 0, 1, 1]
    assert all(context is contexts[0][1] for _, context in contexts)
    assert inputs == [1, 2, 3, 1]
    assert len(updates) == scheduler.last_epoch == (2 if run_backward else 0)
    assert callbacks == ([0, 1] if run_backward else [])
    assert all(not torch.equal(a, b) for a, b in zip(noises, noises[1:], strict=False))
    assert [row["loss"] for row in reporter.payloads] == [2.5, 5.0]
    for row in reporter.payloads:
        assert row["perf"]["local_batch"] == row["perf"]["global_batch"] == 2
        assert row["perf"]["local_tokens"] == 20
        assert all(value >= 0 for value in row["timing"].values())
        if not run_backward:
            assert row["timing"]["backward_s"] == row["timing"]["optimizer_s"] == 0


def test_accumulation_empty_dataloader_fails_and_closes_reporter():
    recipe = _Recipe(batches=[])
    config = TrainerConfig(
        max_steps=1,
        parallelize=ParallelizeConfig(parallel=ParallelConfig(), fsdp2=FSDP2Config(enable=False)),
        checkpoint=CheckpointConfig(enabled=False),
        gradient_accumulation_steps=2,
    )
    with pytest.raises(ValueError, match="dataloader must not be empty"):
        _fit_recipe(recipe, config)
    assert recipe.reporter.closed
    assert recipe.reporter.payloads == []


@pytest.mark.parametrize("strict", [False, True])
def test_full_resume_changed_accumulation_count_respects_strict(tmp_path, strict):
    recipe = _RandomRecipe()
    config = _full_dcp_config(tmp_path, max_steps=2, accumulation=2)
    _fit_recipe(recipe, config)
    resumed = _RandomRecipe(resume_path=str(tmp_path / "checkpoints" / "00002.dcp"))
    config = replace(config, max_steps=3, gradient_accumulation_steps=3)
    if strict:
        config = replace(config, resume=replace(config.resume, strict=True))
        with pytest.raises(RuntimeError, match="execution policy mismatch"):
            _fit_recipe(resumed, config)
    else:
        _fit_recipe(resumed, config)
        assert [row["step"] for row in resumed.reporter.payloads] == [2]


def test_failed_microbatch_keeps_only_completed_update_checkpoint(tmp_path):
    class FailingRecipe(_Recipe):
        def step(self, model, batch, step, context):
            if step == 1 and len(self.seen_steps) == 3:
                raise RuntimeError("failed second micro-batch")
            return super().step(model, batch, step, context)

    recipe = FailingRecipe()
    config = _full_dcp_config(tmp_path, max_steps=3, accumulation=2)
    config = replace(config, checkpoint=replace(config.checkpoint, every_n_steps=1))
    with pytest.raises(RuntimeError, match="failed second micro-batch"):
        _fit_recipe(recipe, config)
    assert [row["step"] for row in recipe.reporter.payloads] == [0]
    assert recipe.reporter.closed
    assert sorted(path.name for path in (tmp_path / "checkpoints").glob("*.dcp")) == ["00001.dcp"]
