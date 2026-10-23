"""Engine-neutral parts of the MCore optimizer adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vforge.training.mcore_optimizer import MCoreDistributedOptimizer, ParamGroupScheduler
from vforge.training.recipe import SchedulerSpec
from vforge.utils.gradients import set_main_gradient_active


def _fake_distributed_optimizer():
    model = torch.nn.Module()
    model.register_parameter("used", torch.nn.Parameter(torch.ones(2)))
    model.register_parameter("unused", torch.nn.Parameter(torch.ones(3)))
    used_shard = torch.nn.Parameter(torch.ones(2))
    unused_shard = torch.nn.Parameter(torch.ones(3))
    used_shard.grad = torch.tensor([3.0, 4.0])
    unused_shard.grad = torch.ones(3)
    for parameter, active in ((model.used, True), (model.unused, False)):
        parameter.main_grad = torch.ones_like(parameter)
        set_main_gradient_active(parameter, active)
    state = {
        used_shard: {
            "step": torch.tensor(1.0),
            "exp_avg": torch.ones(2),
            "exp_avg_sq": torch.ones(2),
        },
        unused_shard: {
            "step": torch.tensor(0.0),
            "exp_avg": torch.zeros(3),
            "exp_avg_sq": torch.zeros(3),
        },
    }
    inner = SimpleNamespace(
        config=SimpleNamespace(use_precision_aware_optimizer_no_fp8_or_ds_fp8=False),
        model_float16_groups=[],
        shard_fp32_from_float16_groups=[],
        model_fp32_groups=[[model.used, model.unused]],
        shard_fp32_groups=[[used_shard, unused_shard]],
        optimizer=SimpleNamespace(state=state),
        tp_group=None,
    )
    return model, MCoreDistributedOptimizer(model, inner, clip_grad=0.0)


def test_optimizer_state_stats_exclude_globally_unused_preallocated_state() -> None:
    _model, optimizer = _fake_distributed_optimizer()

    stats = optimizer.optimizer_state_stats()

    assert stats["probe/adam_exp_avg_numel"] == 2.0
    assert stats["probe/adam_exp_avg_sq_numel"] == 2.0
    assert stats["probe/adam_state_count"] == 1.0


def test_gradient_bucket_stats_include_the_unmapped_other_bucket() -> None:
    _model, optimizer = _fake_distributed_optimizer()

    optimizer._preserve_unused_parameter_semantics()
    stats = optimizer._logical_gradient_bucket_norms(lambda _name: None)

    assert stats == {"probe/bucket_other/grad_norm_post_reduce": 5.0}


@pytest.mark.parametrize(
    ("spec", "reference_factory"),
    [
        (
            SchedulerSpec(name="constant", factor=0.5),
            lambda optimizer: torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5),
        ),
        (
            SchedulerSpec(name="constant_with_warmup", warmup_steps=3),
            lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: min(float(step) / 3.0, 1.0),
            ),
        ),
    ],
    ids=["constant", "constant_with_warmup"],
)
def test_param_group_scheduler_matches_wan_torch_schedules(spec, reference_factory) -> None:
    reference_parameter = torch.nn.Parameter(torch.ones(()))
    candidate_parameter = torch.nn.Parameter(torch.ones(()))
    reference_optimizer = torch.optim.AdamW([reference_parameter], lr=1e-3)
    candidate_optimizer = torch.optim.AdamW([candidate_parameter], lr=1e-3)
    reference = reference_factory(reference_optimizer)
    candidate = ParamGroupScheduler(candidate_optimizer, spec)

    assert candidate_optimizer.param_groups[0]["lr"] == pytest.approx(
        reference_optimizer.param_groups[0]["lr"]
    )
    for _ in range(7):
        reference_optimizer.step()
        reference.step()
        candidate_optimizer.step()
        candidate.step()
        assert candidate_optimizer.param_groups[0]["lr"] == pytest.approx(
            reference_optimizer.param_groups[0]["lr"]
        )
