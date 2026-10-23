"""Optimizer-window contracts; GPU topology gates live under distributed/."""

import copy
import weakref
from contextlib import contextmanager, nullcontext

import pytest
import torch

from vforge.data.loader import TrainingDataLoader
from vforge.distributed.parallel_state import ParallelConfig, ParallelState
from vforge.distributed.parallelize import FSDP2Config, ParallelizeConfig
from vforge.training.engine import EnginePolicy, EngineRunContext, create_engine
from vforge.training.engine.fsdp import FsdpEngine
from vforge.training.phased_trainer import PhasedTrainer
from vforge.training.recipe import StepContext, StepOutput
from vforge.training.trainer import TrainerConfig


def _parallelize():
    return ParallelizeConfig(parallel=ParallelConfig(), fsdp2=FSDP2Config(enable=False))


def _context():
    return StepContext(torch.device("cpu"), ParallelState())


def _run(
    engine,
    model,
    batches,
    optimizer,
    *,
    scheduler=None,
    compute=None,
    run=None,
    step=0,
    context=None,
):
    return engine.run_update(
        model=model,
        microbatches=iter(batches),
        step=step,
        context=context or _context(),
        compute_step=compute or (lambda model, batch, *_: StepOutput(model(batch).square().mean())),
        optimizer=optimizer,
        scheduler=scheduler,
        run=run or EngineRunContext(precision_context=nullcontext),
    )


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.AdamW])
def test_accumulation_matches_equal_weight_loss_and_updates_once(
    count, optimizer_type, monkeypatch
):
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 1).double()
    reference = copy.deepcopy(model)
    optimizer = optimizer_type(model.parameters(), lr=0.01)
    reference_optimizer = optimizer_type(reference.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    reference_scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, 1, gamma=0.8)
    # Unequal batch sizes deliberately distinguish equal-batch and sample-weighted means.
    batches = [torch.randn(index + 1, 2, dtype=torch.float64) for index in range(count)]
    engine = FsdpEngine(
        EnginePolicy(
            parallelize=_parallelize(),
            gradient_accumulation_steps=count,
            gradient_clip_val=0.25,
        )
    )
    updates = []
    clears = []
    zero_grad = optimizer.zero_grad

    def clear_gradients(*args, **kwargs):
        clears.append(True)
        zero_grad(*args, **kwargs)

    monkeypatch.setattr(optimizer, "zero_grad", clear_gradients)
    optimizer.register_step_post_hook(lambda *_: updates.append(True))

    for step in range(2):
        reference_optimizer.zero_grad(set_to_none=True)
        loss = sum(reference(batch).square().mean() for batch in batches) / count
        loss.backward()
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25)
        reference_optimizer.step()
        reference_scheduler.step()

        result = _run(
            engine,
            model,
            batches,
            optimizer,
            scheduler=scheduler,
            step=step,
        )

        assert len(updates) == len(clears) == step + 1
        assert scheduler.state_dict() == reference_scheduler.state_dict()
        assert result.output.loss.item() == pytest.approx(loss.item(), rel=1e-6)
        assert result.diagnostics["diag/grad_norm_post_reduce"] == pytest.approx(
            expected_norm.item(), rel=1e-6
        )
        for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-8)


def test_next_batch_is_requested_after_backward_and_graph_release():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = FsdpEngine(EnginePolicy(parallelize=_parallelize(), gradient_accumulation_steps=3))
    events, losses, inputs = [], [], []
    shared_context = _context()
    model.weight.register_hook(lambda grad: events.append("backward") or grad)
    optimizer.register_step_post_hook(lambda *_: events.append("update"))

    class Loader:
        def __iter__(self):
            return self

        def __next__(self):
            assert all(reference() is None for reference in losses + inputs)
            events.append("load")
            batch = torch.ones(1, 2)
            inputs.append(weakref.ref(batch))
            return batch

    def compute(model, batch, step, context):
        assert context is shared_context
        index = len(losses)
        events.append("forward")
        loss = model(batch).square().mean()
        losses.append(weakref.ref(loss))
        return StepOutput(
            loss,
            metrics={"loss_scalar": loss},
            metadata={"index": index},
        )

    batches = iter(TrainingDataLoader(Loader(), device=torch.device("cpu")))
    result = _run(engine, model, batches, optimizer, compute=compute, context=shared_context)

    assert events == ["load", "forward", "backward"] * 3 + ["update"]
    assert all(reference() is None for reference in losses + inputs)
    assert result.output.loss.grad_fn is None
    assert result.output.metadata == {"microbatches": [{"index": 0}, {"index": 1}, {"index": 2}]}


@pytest.mark.parametrize("count", [1, 2, 4])
def test_engine_consumes_one_window_and_leaves_later_batches_for_next_update(count):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = FsdpEngine(EnginePolicy(parallelize=_parallelize(), gradient_accumulation_steps=count))
    batches = iter(range(2 * count + 1))
    seen = []
    updates = []
    optimizer.register_step_post_hook(lambda *_: updates.append(True))

    def compute(model, batch, step, context):
        seen.append((step, batch))
        return StepOutput(model(torch.ones(1, 1)).sum())

    for step in range(2):
        _run(engine, model, batches, optimizer, compute=compute, step=step)
        assert len(seen) == (step + 1) * count
        assert len(updates) == step + 1

    assert seen == [(index // count, index) for index in range(2 * count)]
    assert next(batches) == 2 * count


@pytest.mark.parametrize("failure", ["forward", "backward", "short", "lazy"])
def test_incomplete_window_does_not_update(failure):
    model = torch.nn.Linear(1, 1)
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    engine = FsdpEngine(EnginePolicy(parallelize=_parallelize(), gradient_accumulation_steps=2))
    updates = []
    optimizer.register_step_post_hook(lambda *_: updates.append(True))

    def batches():
        count = 1 if failure == "short" else 2
        for index in range(count):
            yield index, torch.ones(1, 1)

    def compute(model, batch, step, context):
        index, inputs = batch
        if index == 1:
            if failure == "forward":
                raise RuntimeError("failed forward")
            if failure == "backward":
                return StepOutput(torch.tensor(1.0))
            if failure == "lazy":
                return StepOutput(model(inputs).sum(), backward_losses=lambda: iter(()))
        return StepOutput(model(inputs).sum())

    error = {"short": StopIteration, "lazy": ValueError}.get(failure, RuntimeError)
    with pytest.raises(error):
        _run(engine, model, batches(), optimizer, compute=compute, scheduler=scheduler)
    assert not updates
    assert scheduler.last_epoch == 0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_forward_only_aggregates_reports_and_keeps_model_unchanged():
    model = torch.nn.Linear(1, 1)
    before = copy.deepcopy(model.state_dict())
    engine = FsdpEngine(
        EnginePolicy(parallelize=_parallelize(), gradient_accumulation_steps=3, run_backward=False)
    )

    def compute(model, batch, step, context):
        index = batch
        return StepOutput(
            loss=torch.tensor(float(index + 1)),
            metrics={"common": index, **({"optional": 8.0} if index == 1 else {})},
            workload={"local_batch": index + 1, "local_tokens": (index + 1) * 10},
            metadata={"sample_id": index},
        )

    result = _run(engine, model, range(3), None, compute=compute)
    assert result.output.loss.item() == 2.0
    assert result.output.metrics == {"common": 1.0, "optional": 8.0}
    assert result.output.workload == {"local_batch": 6, "local_tokens": 60}
    assert result.diagnostics == {}
    assert result.backward_s == result.optimizer_s == 0.0
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.parametrize("run_backward", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_offload_lifecycle_is_per_microbatch(run_backward, fail):
    active = False
    events = []

    class Offload:
        def begin_step(self):
            nonlocal active
            assert not active
            active = True
            events.append("begin")

        def start_backward_prefetch(self):
            assert active
            events.append("prefetch")

        def finish_backward_prefetch(self):
            nonlocal active
            active = False
            events.append("finish")

        def abort_step(self):
            nonlocal active
            active = False
            events.append("abort")

    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine = FsdpEngine(
        EnginePolicy(
            parallelize=_parallelize(),
            gradient_accumulation_steps=2,
            run_backward=run_backward,
        )
    )

    def compute(model, batch, step, context):
        assert active
        index, inputs = batch
        if fail and index == 1:
            raise RuntimeError("failed micro-batch")
        return StepOutput(model(inputs).sum())

    expected_error = (
        pytest.raises(RuntimeError, match="failed micro-batch") if fail else nullcontext()
    )
    with expected_error:
        _run(
            engine,
            model,
            [(index, torch.ones(1, 1)) for index in range(2)],
            optimizer,
            compute=compute,
            run=EngineRunContext(precision_context=nullcontext, offload_manager=Offload()),
        )
    # Failed forwards propagate to the Trainer, which owns exit cleanup.
    assert active is fail
    completed = ["begin", "prefetch", "finish"] if run_backward else ["begin", "abort"]
    assert events == completed + (["begin"] if fail else completed)


def test_precision_context_excludes_backward_for_each_microbatch():
    active = False

    @contextmanager
    def precision():
        nonlocal active
        assert not active
        active = True
        try:
            yield
        finally:
            active = False

    model = torch.nn.Linear(1, 1)
    model.weight.register_hook(lambda gradient: gradient if not active else pytest.fail("autocast"))

    def compute(model, batch, *_):
        assert active
        return StepOutput(model(batch).sum())

    _run(
        FsdpEngine(EnginePolicy(parallelize=_parallelize(), gradient_accumulation_steps=2)),
        model,
        [torch.ones(1, 1) for _ in range(2)],
        torch.optim.SGD(model.parameters(), lr=0.1),
        compute=compute,
        run=EngineRunContext(precision_context=precision),
    )


@pytest.mark.parametrize("count", [1, 2, 4])
def test_random_sequence_does_not_depend_on_update_grouping(count):
    context = _context()
    reference = torch.Generator().manual_seed(42)
    values, steps = [], []
    model = torch.nn.Linear(1, 1)
    engine = FsdpEngine(
        EnginePolicy(
            parallelize=_parallelize(), gradient_accumulation_steps=count, run_backward=False
        )
    )

    def compute(model, batch, step, current):
        assert current is context
        steps.append(step)
        generator = current.random_streams.generator("input", seed=42, device=current.device)
        actual = torch.rand(16, generator=generator)
        assert torch.equal(actual, torch.rand(16, generator=reference))
        assert all(not torch.equal(actual, previous) for previous in values)
        values.append(actual)
        return StepOutput(actual.mean())

    for step in range(8 // count):
        _run(engine, model, range(count), None, compute=compute, step=step, context=context)
    assert steps == [step for step in range(8 // count) for _ in range(count)]


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "2"])
def test_runtime_contracts_reject_non_positive_or_non_integer_count(count):
    with pytest.raises(ValueError, match="positive integer"):
        EnginePolicy(parallelize=_parallelize(), gradient_accumulation_steps=count)
    with pytest.raises(ValueError, match="positive integer"):
        TrainerConfig(max_steps=1, parallelize=_parallelize(), gradient_accumulation_steps=count)


def test_unsupported_engines_and_phases_reject_before_setup():
    policy = EnginePolicy(parallelize=_parallelize(), gradient_accumulation_steps=2)
    with pytest.raises(NotImplementedError, match="single microbatch"):
        create_engine("megatron", policy)
    with pytest.raises(NotImplementedError, match="gradient_accumulation_steps=1"):
        PhasedTrainer(
            TrainerConfig(
                max_steps=1,
                parallelize=_parallelize(),
                gradient_accumulation_steps=2,
            )
        )


@pytest.mark.parametrize("run_backward", [False, True])
def test_single_batch_preserves_output_and_update_semantics(run_backward):
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    reference = copy.deepcopy(model)
    batch = torch.randn(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    expected_loss = reference(batch).square().mean()
    if run_backward:
        expected_loss.backward()
        reference_optimizer.step()
    outputs = []

    def compute(model, batch, *_):
        output = StepOutput(
            model(batch).square().mean(),
            metrics={"metric": 3},
            metadata={"sample_ids": [1, 2, 3]},
            workload={"local_batch": 3, "local_tokens": 6},
        )
        outputs.append(output)
        return output

    result = _run(
        FsdpEngine(EnginePolicy(parallelize=_parallelize(), run_backward=run_backward)),
        model,
        [batch],
        optimizer,
        compute=compute,
    )
    assert result.output is outputs[0]
    torch.testing.assert_close(result.output.loss, expected_loss, rtol=0, atol=0)
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_timing_sums_compute_regions_without_counting_interleaved_loading(monkeypatch):
    import vforge.training.engine.fsdp as fsdp_module

    clock = [0.0]
    monkeypatch.setattr(fsdp_module.time, "perf_counter", lambda: clock[0])
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.weight.register_hook(lambda grad: clock.__setitem__(0, clock[0] + 3) or grad)
    optimizer.register_step_post_hook(lambda *_: clock.__setitem__(0, clock[0] + 7))

    def batches():
        for _ in range(2):
            clock[0] += 5
            yield torch.ones(1, 1)

    def compute(model, batch, *_):
        clock[0] += 2
        return StepOutput(model(batch).sum())

    result = _run(
        FsdpEngine(EnginePolicy(parallelize=_parallelize(), gradient_accumulation_steps=2)),
        model,
        batches(),
        optimizer,
        compute=compute,
    )
    assert result.forward_s == 4
    assert result.dataloader_s == 10
    assert result.backward_s == 6
    assert result.optimizer_s == 7
    assert result.completed_at == 27
