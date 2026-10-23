"""Replicated data parallelism over a tensor-parallel model.

Four ranks are arranged as DP=2 x TP=2.  Corresponding TP shards must average
their gradients over the DP axis, while logical diagnostics count one DP replica
and distinguish TP shards from TP-replicated parameters.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4


class _PartiallyUsedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, 1))
        self.unused = torch.nn.Parameter(torch.ones(1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value @ self.weight.T


def _init(rank: int, init_file: str) -> torch.device:
    use_cuda = os.environ.get("VFORGE_TEST_CUDA") == "1"
    if use_cuda:
        if torch.cuda.device_count() < WORLD:
            raise RuntimeError(f"VFORGE_TEST_CUDA=1 requires {WORLD} GPUs")
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    dist.init_process_group(
        backend="nccl" if use_cuda else "gloo",
        init_method=f"file://{init_file}",
        world_size=WORLD,
        rank=rank,
    )
    os.environ["LOCAL_RANK"] = str(rank)
    return device


def _ddp_worker(rank: int, init_file: str) -> None:
    device = _init(rank, init_file)
    try:
        from torch.nn.parallel import DistributedDataParallel

        from vforge.distributed.parallel_state import (
            ParallelConfig,
            destroy_parallel_state,
            init_parallel_state,
        )
        from vforge.distributed.parallelize import (
            apply_replicated_data_parallel,
            unwrap_replicated_data_parallel,
        )

        state = init_parallel_state(
            ParallelConfig(
                dp_size=2,
                tp_size=2,
                dp_mode="ddp",
                device_type=device.type,
            )
        )
        model = _PartiallyUsedModel().to(device)
        with torch.no_grad():
            model.weight.fill_(rank + 1)

        model = apply_replicated_data_parallel(model)
        assert isinstance(model, DistributedDataParallel)
        assert unwrap_replicated_data_parallel(model) is model.module

        # DCP's state-dict facade removes DDP's implementation-only ``module.``
        # prefix, so adding DP does not change checkpoint compatibility.
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        model_state = get_model_state_dict(model)
        assert set(model_state) == {"unused", "weight"}

        # DDP broadcasts within [rank 0, rank 2] and [rank 1, rank 3], not over
        # the world.  The two TP coordinates therefore keep distinct shards.
        expected_initial = float(state.tp_rank + 1)
        assert model.module.weight.item() == expected_initial

        # Different DP replicas process different values.  The gradient for a
        # fixed TP coordinate is their average: 1.5 for tp0 and 3.0 for tp1.
        value = float((state.dp_rank + 1) * (state.tp_rank + 1))
        model(torch.tensor([[value]], device=device)).sum().backward()
        expected_grad = 1.5 * float(state.tp_rank + 1)
        torch.testing.assert_close(
            model.module.weight.grad,
            torch.tensor([[expected_grad]], device=device),
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()
        expected_weight = expected_initial - 0.1 * expected_grad
        torch.testing.assert_close(
            model.module.weight,
            torch.tensor([[expected_weight]], device=device),
        )

        gathered = [torch.empty((), device=device) for _ in range(WORLD)]
        dist.all_gather(gathered, model.module.weight.detach().reshape(()))
        gathered_values = [float(item.item()) for item in gathered]
        assert gathered_values[0] == gathered_values[2]
        assert gathered_values[1] == gathered_values[3]
        assert gathered_values[0] != gathered_values[1]
        destroy_parallel_state()
    finally:
        dist.destroy_process_group()


class _LogicalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torch.distributed.tensor.parallel import ColwiseParallel

        from vforge.distributed.tensor_parallel import mark_tp_sharded

        self.sharded = torch.nn.Linear(1, 1, bias=False)
        mark_tp_sharded(self.sharded, ColwiseParallel())
        self.replicated = torch.nn.Parameter(torch.ones(1))


class _TorchTPModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

        from vforge.distributed.tensor_parallel import build_linear

        self.up = build_linear(4, 8, bias=True, style=ColwiseParallel())
        self.down = build_linear(8, 4, bias=True, style=RowwiseParallel())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.gelu(self.up(value)))


def _logical_norm_worker(rank: int, init_file: str) -> None:
    device = _init(rank, init_file)
    try:
        from vforge.distributed.parallel_state import (
            ParallelConfig,
            destroy_parallel_state,
            init_parallel_state,
        )
        from vforge.distributed.parallelize import apply_replicated_data_parallel
        from vforge.training.diagnostics import (
            LogicalReduction,
            clip_grad_norm_logical_,
        )

        state = init_parallel_state(
            ParallelConfig(
                dp_size=2,
                tp_size=2,
                dp_mode="ddp",
                device_type=device.type,
            )
        )
        model = apply_replicated_data_parallel(_LogicalModel().to(device))
        model.module.sharded.weight.grad = torch.tensor(
            [[3.0 if state.tp_rank == 0 else 4.0]],
            device=device,
        )
        model.module.replicated.grad = torch.tensor([2.0], device=device)

        reduction = LogicalReduction.for_model(model, state)
        norm = clip_grad_norm_logical_(model, max_norm=5.0, reduction=reduction)
        expected_norm = 29.0**0.5
        coefficient = 5.0 / (expected_norm + 1e-6)
        assert norm == pytest.approx(expected_norm)

        # Contribution filtering changes the norm only.  Every physical copy is
        # scaled by the common coefficient so replicas remain identical.
        local_shard = 3.0 if state.tp_rank == 0 else 4.0
        assert model.module.sharded.weight.grad.item() == pytest.approx(local_shard * coefficient)
        assert model.module.replicated.grad.item() == pytest.approx(2.0 * coefficient)

        gathered = [torch.empty((), device=device) for _ in range(WORLD)]
        dist.all_gather(gathered, torch.tensor(norm, device=device))
        assert all(item.item() == pytest.approx(expected_norm) for item in gathered)
        destroy_parallel_state()
    finally:
        dist.destroy_process_group()


def _torch_tp_ddp_rejection_worker(rank: int, init_file: str) -> None:
    device = _init(rank, init_file)
    try:
        from vforge.distributed.parallel_state import (
            ParallelConfig,
            destroy_parallel_state,
            init_parallel_state,
        )
        from vforge.distributed.parallelize import apply_replicated_data_parallel
        from vforge.distributed.tensor_parallel import apply_torch_tensor_parallel

        init_parallel_state(
            ParallelConfig(
                dp_size=2,
                tp_size=2,
                dp_mode="ddp",
                device_type=device.type,
            )
        )
        torch.manual_seed(17)
        model = _TorchTPModel().to(device)
        apply_torch_tensor_parallel(model)
        with pytest.raises(NotImplementedError, match="cannot wrap DTensor"):
            apply_replicated_data_parallel(model)
        destroy_parallel_state()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.cpu_count() is not None and os.cpu_count() < WORLD,
    reason="needs four processes for a DP=2 x TP=2 mesh",
)
@pytest.mark.parametrize(
    "worker",
    [_ddp_worker, _logical_norm_worker, _torch_tp_ddp_rejection_worker],
)
def test_replicated_dp_over_tensor_parallel(worker, tmp_path):
    mp.spawn(worker, args=(str(tmp_path / f"{worker.__name__}-pg"),), nprocs=WORLD, join=True)
