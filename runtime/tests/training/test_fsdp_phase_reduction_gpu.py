"""Phase clipping after a forward-only use of an independent FSDP root."""

from __future__ import annotations

import copy
import os
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
from torch import nn


@pytest.mark.gpu(gpus=2)
def test_phase_clipping_after_forward_only_rollout(torchrun):
    torchrun(__file__)


def _run() -> None:
    from torch.distributed.fsdp import fully_shard

    from vforge import FSDP2Config, ParallelConfig, ParallelizeConfig
    from vforge.distributed.parallel_state import destroy_parallel_state, init_parallel_state
    from vforge.training.diagnostics import LogicalReduction
    from vforge.training.engine import EnginePolicy, EngineRunContext
    from vforge.training.engine.fsdp import FsdpEngine
    from vforge.training.recipe import StepContext, StepOutput

    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    try:
        state = init_parallel_state(ParallelConfig(dp_size=2, device_type="cuda"))
        torch.manual_seed(42)
        generator = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4)).to(device)
        reference = copy.deepcopy(generator)
        fully_shard(generator[0], mesh=state.fsdp_mesh)
        fully_shard(generator, mesh=state.fsdp_mesh)
        bundle = nn.ModuleDict({"generator": generator})
        run = EngineRunContext(
            precision_context=nullcontext,
            logical_reduction=LogicalReduction.for_model(bundle, state),
        )
        engine = FsdpEngine(
            EnginePolicy(
                parallelize=ParallelizeConfig(parallel=state.config, fsdp2=FSDP2Config()),
                gradient_clip_val=0.1,
            )
        )
        optimizer = torch.optim.SGD(generator.parameters(), lr=0.1)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        inputs = torch.arange(8, device=device, dtype=torch.float32).reshape(2, 4)
        for step in range(2):
            # Critic rollout leaves the generator root unsharded. Backward later
            # restores the optimizer's sharded parameters before norm/clipping.
            with torch.no_grad():
                generator(inputs)
            reference_optimizer.zero_grad(set_to_none=True)
            reference(inputs).square().mean().backward()
            expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1)
            reference_optimizer.step()
            result = engine.run_update(
                model=bundle,
                microbatches=iter((inputs,)),
                step=step,
                context=StepContext(device=device, parallel_state=state),
                compute_step=lambda model, batch, *_: StepOutput(
                    loss=model["generator"](batch).square().mean()
                ),
                optimizer=optimizer,
                scheduler=None,
                run=run,
                update_target=generator,
            )
            assert result.diagnostics["diag/grad_norm_post_reduce"] == pytest.approx(
                expected_norm.item(),
                rel=1e-5,
            )
            for actual, expected in zip(
                generator.parameters(), reference.parameters(), strict=True
            ):
                torch.testing.assert_close(actual.grad.full_tensor(), expected.grad)
                torch.testing.assert_close(actual.full_tensor(), expected)
        destroy_parallel_state()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    _run()
