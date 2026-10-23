from types import SimpleNamespace

import torch

from vforge.training.diagnostics import (
    DiagnosticsConfig,
    clip_and_step,
    global_grad_bucket_norms,
    global_grad_norm,
    global_grad_value_stats,
    global_optimizer_state_stats,
    global_param_norm,
    global_param_numel,
    global_param_update_stats,
    log_local_parameter_stats,
    run_backward,
    snapshot_trainable_params,
)
from vforge.training.parity import ParityStepConfig
from vforge.training.parity.probe_metrics import legacy_probe_aliases


def _parameter_bucket(name: str) -> str | None:
    return "patch_embedding" if name == "patch_embedding" else None


def test_local_parameter_stats_logs_each_parameter(capsys):
    model = torch.nn.Linear(2, 1)
    log_local_parameter_stats(model, phase="post_load", rank=3)

    output = capsys.readouterr().out
    assert output.count("VFORGE_LOCAL_PARAMETER_STATS") == 2
    assert "phase=post_load rank=3 name=weight" in output
    assert "name=bias" in output


def test_parameter_and_gradient_diagnostics_are_stable():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0]]))
    model.weight.grad = torch.tensor([[6.0, 8.0]])

    assert global_param_norm(model) == 5.0
    assert global_grad_norm(model) == 10.0
    assert global_param_numel(model) == 2.0
    assert global_grad_value_stats(model) == {
        "numel": 2.0,
        "sum": 14.0,
        "abs_sum": 14.0,
        "square_sum": 100.0,
        "max_abs": 8.0,
    }


def test_gradient_diagnostics_use_logical_parameter_buckets():
    model = torch.nn.Module()
    model.register_parameter("patch_embedding", torch.nn.Parameter(torch.tensor([0.0])))
    model.register_parameter("unclassified", torch.nn.Parameter(torch.tensor([0.0])))
    model.patch_embedding.grad = torch.tensor([3.0])
    model.unclassified.grad = torch.tensor([4.0])

    norms = global_grad_bucket_norms(model, _parameter_bucket)

    assert norms["probe/bucket_patch_embedding/grad_norm_post_reduce"] == 3.0
    assert norms["probe/bucket_other/grad_norm_post_reduce"] == 4.0


def test_logical_reduction_counts_only_one_hsdp_replica() -> None:
    from vforge.training.diagnostics import LogicalReduction

    model = torch.nn.Linear(2, 1)
    state = SimpleNamespace(
        initialized=True,
        config=SimpleNamespace(
            dp_mode="fsdp2",
            dp_size=4,
            dp_replicate_size=2,
        ),
        dp_rank=2,
        dp_cp_size=4,
        dp_cp_rank=2,
        dp_replicate_rank=1,
        tp_enabled=False,
    )

    reduction = LogicalReduction.for_model(model, state)

    assert reduction.contributing_parameters == frozenset()


def test_logical_reduction_survives_being_scoped_to_a_submodule() -> None:
    """Built from the bundle root, applied to one role: same parameters either way.

    A multi-model algorithm clips one role at a time, so the module it hands to
    the norm is a child of the module the reduction was built from and every
    name has lost its role prefix. If membership went by name the intersection
    would be empty, the norm would come back as zero, and clipping would quietly
    stop -- which reads as a gradient that never needed clipping.
    """
    from vforge.training.diagnostics import LogicalReduction

    root = torch.nn.Module()
    root.add_module("generator", torch.nn.Linear(2, 1))
    root.add_module("critic", torch.nn.Linear(2, 1))
    state = SimpleNamespace(
        initialized=True,
        config=SimpleNamespace(dp_mode="fsdp2", dp_size=8, dp_replicate_size=1),
        dp_rank=0,
        dp_cp_size=8,
        dp_cp_rank=0,
        dp_replicate_rank=0,
        tp_enabled=False,
    )

    reduction = LogicalReduction.for_model(root, state)

    generator = root.get_submodule("generator")
    assert [name for name, _ in generator.named_parameters()] == ["weight", "bias"]
    assert all(reduction.includes(parameter) for parameter in generator.parameters())

    for parameter in generator.parameters():
        parameter.grad = torch.ones_like(parameter)
    assert global_grad_norm(generator, reduction) > 0.0


def test_update_and_optimizer_diagnostics_preserve_metric_schema():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    snapshot = snapshot_trainable_params(model)
    loss = model(torch.ones(1, 2)).square().mean()
    loss.backward()
    optimizer.step()

    updates = global_param_update_stats(model, snapshot)
    optimizer_stats = global_optimizer_state_stats(optimizer, model)
    aliases = legacy_probe_aliases(
        {
            "probe/grad_norm_before_clip": 1.0,
            "probe/grad_norm_after_clip": 0.5,
            "probe/param_norm_before_update": 2.0,
            "probe/param_norm_after_update": 3.0,
        }
    )

    assert updates["probe/update_numel"] == 3.0
    assert updates["probe/update_nonzero_count"] > 0
    assert optimizer_stats["probe/adam_state_count"] == 2.0
    assert optimizer_stats["probe/adam_exp_avg_numel"] == 3.0
    assert optimizer_stats["probe/adam_exp_avg_sq_numel"] == 3.0
    assert optimizer_stats["probe/adam_step_min"] == 1.0
    assert aliases["diag/grad_norm_pre_clip"] == 1.0
    assert aliases["diag/grad_norm_post_clip"] == 0.5
    assert aliases["probe/param_norm_before_step"] == 2.0
    assert aliases["probe/param_norm_after_step"] == 3.0


def test_update_diagnostics_use_caller_parameter_buckets():
    model = torch.nn.Module()
    model.register_parameter(
        "patch_embedding",
        torch.nn.Parameter(torch.tensor([1.0])),
    )
    snapshot = snapshot_trainable_params(model)
    with torch.no_grad():
        model.patch_embedding.add_(0.25)

    updates = global_param_update_stats(
        model,
        snapshot,
        parameter_bucket=_parameter_bucket,
    )

    assert updates["probe/bucket_patch_embedding/update_numel"] == 1.0
    assert updates["probe/bucket_patch_embedding/update_max_abs"] == 0.25


def test_clip_and_step_owns_optimizer_diagnostic_sequence():
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.ones(1, 2)).square().mean()
    before = model.weight.detach().clone()

    run_backward(loss=loss)
    result = clip_and_step(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        usp_size=2,
        gradient_clip_val=0.5,
        parity=ParityStepConfig(emit_legacy_probe_aliases=True),
        diagnostics_config=DiagnosticsConfig(level="parity"),
    )

    assert not torch.equal(model.weight, before)
    assert result["probe/usp_size"] == 2.0
    assert result["probe/clip_active"] in {0.0, 1.0}
    assert "probe/update_norm" in result
    assert "diag/grad_norm_post_clip" in result


def test_basic_backward_update_only_emits_post_reduce_diagnostics():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.ones(1, 2)).square().mean()

    run_backward(loss=loss)
    result = clip_and_step(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        usp_size=1,
        gradient_clip_val=0.0,
        parity=None,
    )

    assert set(result) == {
        "diag/grad_norm_post_reduce",
        "diag/grad_norm_post_clip",
        "diag/clip_coefficient",
        "diag/clip_active",
    }
    assert result["diag/grad_norm_post_reduce"] == torch.sqrt(torch.tensor(32.0)).item()


def test_backward_scale_applies_before_clipping_and_optimizer_step() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = model(torch.ones(1, 1)).sum()

    run_backward(
        loss=loss,
        backward_scale=2.0,
    )
    result = clip_and_step(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        usp_size=1,
        gradient_clip_val=0.0,
        parity=None,
    )

    torch.testing.assert_close(model.weight, torch.tensor([[0.8]]))
    assert result["diag/grad_norm_post_reduce"] == 2.0


def test_diagnostics_level_does_not_change_clipping_update():
    initial_weight = torch.tensor([[1.0, -1.0]])

    def run(level):
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(initial_weight)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = model(torch.tensor([[2.0, -1.0]])).square().mean()
        run_backward(loss=loss)
        result = clip_and_step(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            usp_size=1,
            gradient_clip_val=0.5,
            parity=None,
            diagnostics_config=DiagnosticsConfig(level=level),
        )
        return model.weight.detach().clone(), result

    basic_weight, basic_metrics = run("basic")
    parity_weight, parity_metrics = run("parity")

    assert torch.equal(basic_weight, parity_weight)
    assert (
        basic_metrics["diag/grad_norm_post_reduce"] == parity_metrics["diag/grad_norm_post_reduce"]
    )
    assert basic_metrics["diag/grad_norm_post_clip"] == parity_metrics["diag/grad_norm_post_clip"]


def test_torch_clipper_honors_fp32_with_bf16_gradients():
    from vforge.training.diagnostics import clip_grad_norm_torch_

    model = torch.nn.Linear(3, 1, bias=False, dtype=torch.bfloat16)
    gradient = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16)
    model.weight.grad = gradient.clone()
    norm = clip_grad_norm_torch_(model, None, 1.0, norm_dtype="float32")
    expected_norm = gradient.float().norm()
    assert norm == expected_norm.item()
    torch.testing.assert_close(
        model.weight.grad, (gradient.float() / (expected_norm + 1e-6)).to(torch.bfloat16)
    )


def test_lazy_backward_interleaves_forwards_and_updates_once():
    from contextlib import contextmanager

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    events = []
    active = False

    @contextmanager
    def precision():
        nonlocal active
        active = True
        try:
            yield
        finally:
            active = False

    def on_backward(gradient):
        assert not active
        events.append("backward")
        return gradient

    model.weight.register_hook(on_backward)
    optimizer.register_step_post_hook(lambda *_: events.append("update"))

    def losses():
        try:
            for scale in [1, 3]:
                assert active
                events.append("forward")
                yield (model.weight * scale).sum()
        finally:
            events.append("close")

    run_backward(
        loss=torch.tensor(0.0),
        backward_losses=losses,
        backward_forward_context=precision,
    )
    clip_and_step(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        usp_size=1,
        gradient_clip_val=0,
        parity=None,
    )
    assert events == ["forward", "backward", "forward", "backward", "close", "update"]
    torch.testing.assert_close(model.weight, torch.tensor([[1.6]]))
    assert scheduler.last_epoch == 1
    assert optimizer.param_groups[0]["lr"] == 0.05


def test_lazy_backward_failure_closes_context_without_update():
    from contextlib import contextmanager

    import pytest

    model = torch.nn.Linear(1, 1, bias=False)
    before = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    events = []
    optimizer.register_step_post_hook(lambda *_: events.append("update"))

    @contextmanager
    def cache():
        try:
            yield
        finally:
            events.append("close")

    def losses():
        with cache():
            yield model.weight.sum()
            yield torch.tensor(1.0)  # No graph: fail inside backward, not in the iterator.

    with pytest.raises(RuntimeError, match="does not require grad"):
        run_backward(
            loss=torch.tensor(0.0),
            backward_losses=losses,
        )
        clip_and_step(
            model=model,
            optimizer=optimizer,
            scheduler=None,
            usp_size=1,
            gradient_clip_val=0,
            parity=None,
        )
    assert events == ["close"]
    torch.testing.assert_close(model.weight, before)
