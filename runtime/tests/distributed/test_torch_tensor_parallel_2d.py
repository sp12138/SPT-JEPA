"""Tensor parallelism and fsdp2 sharding the same model at once.

This is the thing the torch-native backend is for and the megatron engine
refuses: both axes are DTensor here, so a weight split by tp and then by data
sharding carries both placements rather than the second overwriting the first.

Two ways it can go wrong without failing, and both are about the *order* the
engine applies them in:

* fsdp2 first would wrap ordinary tensors and describe the data mesh alone. The
  tp axis is then simply absent from the state dict, and every tp rank offers
  the same name at the same shape -- the case DCP settles by keeping one.
* a parameter the tp pass does not reach keeps its local width and gets that
  width recorded as though it were the whole parameter. Wan's q/k norm is such a
  parameter, since no stock ``ParallelStyle`` describes it.

So what is asserted is what those would break: that every sharded entry names
both meshes, and that the logical shape on disk is the unsharded one.

Four gloo ranks on CPU, arranged 2 x 2. No mcore, so this runs anywhere.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HIDDEN = 16
FFN = 32
TP = 2
DP = 2
WORLD = TP * DP


class Norm(torch.nn.Module):
    """Stands in for Wan's q/k norm: a weight the model itself sized locally.

    Marked rather than planned, because it is not a ``Linear`` and there is no
    stock style for "this parameter simply is one shard".
    """

    def __init__(self, local_features: int) -> None:
        super().__init__()
        from torch.distributed.tensor.parallel import ColwiseParallel

        from vforge.distributed.tensor_parallel import mark_tp_sharded

        self.weight = torch.nn.Parameter(torch.ones(local_features))
        mark_tp_sharded(self, ColwiseParallel())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        from vforge.distributed.tensor_parallel import _local_view

        return hidden * _local_view(self.weight)


def _build() -> torch.nn.Module:
    """Colwise projection, a hand-sized norm across it, then a rowwise one."""
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    from vforge.distributed.tensor_parallel import build_linear

    model = torch.nn.Sequential()
    model.add_module("up", build_linear(HIDDEN, FFN, bias=True, style=ColwiseParallel()))
    model.add_module("norm", Norm(FFN // TP))
    model.add_module("down", build_linear(FFN, HIDDEN, bias=True, style=RowwiseParallel()))
    return model


def _worker(rank: int, init_file: str, checkpoint: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=WORLD,
        rank=rank,
    )
    os.environ["LOCAL_RANK"] = str(rank)
    torch.manual_seed(0)
    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.checkpoint.state_dict import get_model_state_dict
        from torch.distributed.tensor import DTensor

        from vforge.distributed.parallel_state import (
            ParallelConfig,
            destroy_parallel_state,
            init_parallel_state,
        )
        from vforge.distributed.tensor_parallel import (
            apply_torch_tensor_parallel,
            tensor_parallel_sharded_parameter_names,
        )

        state = init_parallel_state(ParallelConfig(tp_size=TP, dp_size=DP, device_type="cpu"))
        assert state.tp_enabled

        with torch.device("meta"):
            model = _build()

        # The fsdp2 engine's order: tp first, so fully_shard wraps weights that
        # already carry a tp axis and composes with it rather than describing the
        # data mesh alone.
        apply_torch_tensor_parallel(model)
        assert tensor_parallel_sharded_parameter_names(model) == frozenset(
            {"up.weight", "up.bias", "norm.weight", "down.weight"}
        )
        fully_shard(model, mesh=state.fsdp_mesh)
        model.to_empty(device="cpu")
        with torch.no_grad():
            for index, parameter in enumerate(model.parameters()):
                parameter.fill_(0.25 + 0.5 * index)

        expected_shapes = {
            "up.weight": (FFN, HIDDEN),
            "up.bias": (FFN,),
            "norm.weight": (FFN,),
            "down.weight": (HIDDEN, FFN),
            "down.bias": (HIDDEN,),
        }
        reported = get_model_state_dict(model)
        assert set(reported) == set(expected_shapes), sorted(reported)
        for key, shape in expected_shapes.items():
            entry = reported[key]
            assert isinstance(entry, DTensor), f"{key} is {type(entry).__name__}"
            # The logical shape, which is the unsharded one: neither axis is
            # allowed to be mistaken for the size of the parameter.
            assert tuple(entry.shape) == shape, f"{key}: {tuple(entry.shape)} != {shape}"
            names = entry.device_mesh.mesh_dim_names
            assert names == ("dp_shard_usp", "tp"), f"{key}: {names}"

        # A forward, so the composition is exercised and not merely described.
        output = model(torch.randn(2, HIDDEN))
        assert tuple(output.shape) == (2, HIDDEN)

        dcp.save({"model": reported}, checkpoint_id=checkpoint)
        dist.barrier()

        whole = {"model": {key: torch.zeros(shape) for key, shape in expected_shapes.items()}}
        dcp.load(whole, checkpoint_id=checkpoint)
        for key, value in whole["model"].items():
            gathered = reported[key].full_tensor()
            assert torch.equal(value, gathered), key

        destroy_parallel_state()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.cpu_count() is not None and os.cpu_count() < WORLD,
    reason=f"needs {WORLD} processes to arrange a {DP}x{TP} mesh",
)
def test_tensor_and_data_parallelism_shard_the_same_weight(tmp_path):
    mp.spawn(
        _worker,
        args=(str(tmp_path / "pg"), str(tmp_path / "checkpoint")),
        nprocs=WORLD,
        join=True,
    )
