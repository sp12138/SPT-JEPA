"""Can mcore's tensor-parallel layers run on a process group we hand them?

The megatron engine's design rests on one assumption: vforge's DeviceMesh stays
the single owner of every process group, and mcore receives its tp group as an
argument rather than building one. If that does not hold we are pushed into two
parallel states over the same ranks -- duplicate NCCL communicators, two
independent lifecycles, and a membership agreement that nothing checks at
runtime. Far cheaper to find out here than after an engine is built on it.

mcore 0.18.2 says it holds: ``get_tensor_model_parallel_group_if_none`` passes a
supplied group straight through when its own state was never initialized, and
the layers derive shard widths from that group instead of a global. This asserts
that the claim survives contact with a running process group.

gloo on CPU, because the questions are which ranks talk to each other and how
wide each shard is; neither needs a GPU. It does need Linux: ``vforge.mcore``
imports triton at package level.
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _mcore_importable() -> bool:
    """Whether this machine can import mcore, asked without importing it.

    Importing any mcore submodule runs vforge/mcore/__init__.py, which reaches
    triton. There is no macOS wheel, so the package is unrunnable there rather
    than merely failing an assertion, and triton's presence stands in for it.

    Asking mcore directly is what must not happen. A failed
    ``import vforge.mcore...`` leaves some 120 half-initialized
    ``vforge.mcore.*`` entries in ``sys.modules`` -- including the module that
    raised -- so a second attempt in the same process succeeds against a partly
    built package. That makes ``pytest.importorskip`` order-dependent: the first
    test module to ask skips, every later one concludes mcore works, and their
    ``mp.spawn`` children then run against the wreckage.
    """
    return importlib.util.find_spec("triton") is not None


pytestmark = pytest.mark.skipif(
    not _mcore_importable(),
    reason="vforge.mcore imports triton at package level; absent on macOS and on CPU CI",
)

HIDDEN = 8
FFN = 16


def _mlp_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
    )
    os.environ["LOCAL_RANK"] = str(rank)
    try:
        from vforge.distributed.parallel_state import (
            ParallelConfig,
            destroy_parallel_state,
            init_parallel_state,
        )
        from vforge.mcore import parallel_state as mcore_state
        from vforge.mcore.model_parallel_config import ModelParallelConfig
        from vforge.mcore.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        state = init_parallel_state(ParallelConfig(dp_size=2, tp_size=2, device_type="cpu"))
        tp_group = state.tp_group
        tp_size = state.config.tp_size
        tp_rank = state.tp_rank
        shard = FFN // tp_size

        assert dist.get_process_group_ranks(tp_group) == [rank - rank % 2, rank - rank % 2 + 1]

        config = ModelParallelConfig(
            tensor_model_parallel_size=tp_size,
            # Without this the layers allocate on torch.cuda.current_device().
            use_cpu_initialization=True,
            # We install known weights below, so the random init would only cost
            # time and make the comparison depend on mcore's RNG handling.
            perform_initialization=False,
        )

        col = ColumnParallelLinear(
            HIDDEN,
            FFN,
            config=config,
            init_method=torch.nn.init.zeros_,
            bias=False,
            gather_output=False,
            tp_group=tp_group,
        )
        row = RowParallelLinear(
            FFN,
            HIDDEN,
            config=config,
            init_method=torch.nn.init.zeros_,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            tp_group=tp_group,
        )

        # First observable: had mcore ignored our group and consulted its own
        # uninitialized global, it could not have arrived at a width of 2.
        assert col.output_size_per_partition == shard
        assert col.weight.shape == (shard, HIDDEN)
        assert row.weight.shape == (HIDDEN, shard)
        assert not mcore_state.is_initialized(), "mcore's global state must stay untouched"

        # Same full weights everywhere, then each rank keeps its slice, so the
        # pair computes x @ w1.T @ w2.T split across the tp group.
        gen = torch.Generator().manual_seed(0)
        w1 = torch.randn(FFN, HIDDEN, generator=gen)
        w2 = torch.randn(HIDDEN, FFN, generator=gen)
        col.weight.data.copy_(w1[tp_rank * shard : (tp_rank + 1) * shard, :])
        row.weight.data.copy_(w2[:, tp_rank * shard : (tp_rank + 1) * shard])

        # Distinct data per dp replica, identical within the tp group: the real
        # layout, and what makes this discriminating. RowParallelLinear finishes
        # with an all-reduce; run over the dp group instead of the tp group it
        # would sum the two replicas and miss the reference below. Both groups
        # have size 2 here, so a size check alone would not catch the swap.
        x = torch.full((2, HIDDEN), float(state.dp_rank + 1), requires_grad=True)

        hidden, _ = col(x)
        assert hidden.shape == (2, shard)
        y, _ = row(hidden)

        torch.testing.assert_close(y, (x.detach() @ w1.T) @ w2.T)

        y.sum().backward()
        assert col.weight.grad is not None
        assert col.weight.grad.shape == (shard, HIDDEN)
        assert x.grad is not None

        assert not mcore_state.is_initialized()
        destroy_parallel_state()
    finally:
        dist.destroy_process_group()


def test_mcore_tp_layers_accept_an_injected_group(tmp_path):
    init_file = str(tmp_path / "mcore-tp-pg")
    mp.spawn(_mlp_worker, args=(4, init_file), nprocs=4, join=True)
