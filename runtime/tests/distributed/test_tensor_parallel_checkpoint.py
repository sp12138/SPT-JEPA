"""A tensor-parallel model must save and reload the weights it actually holds.

mcore gives each rank an ordinary tensor holding half a weight, and every path
vforge reads or writes weights with decides what it is looking at by asking
whether it is a ``DTensor``. Told nothing, each of them concludes this rank owns
the whole thing, and each then goes wrong in its own way without failing:

* DCP sees ``attn1.to_q.weight`` once per rank at the same shape, keeps one of
  them, and a checkpoint half the model is silently missing from reads back
  clean;
* the whole-tensor readers -- a PT export, ``broadcast_from_rank0`` -- gather
  nothing and take rank zero's fraction as the model;
* a load that ought to cut a pretrained tensor down to this rank hands it over
  whole instead.

:func:`install_tp_state_dict_hooks` is what closes all three, so these tests
check the properties the hooks are supposed to buy rather than the hooks
themselves: what a state dict says, that a round trip is bit-exact, that the
pieces join, and that a whole tensor arriving is narrowed.

Bit-exactness is the right bar here, unlike the arithmetic tests: saving and
loading moves bytes and must not change one.

gloo on CPU, and no forward pass: the question is which rank stores which bytes.
It does need Linux, since ``vforge.mcore`` imports triton.
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

    The presence of triton stands in for the whole package: mcore reaches it at
    package import, and no macOS wheel exists, which is the only reason these
    tests cannot run everywhere.

    Asking mcore directly is what must not happen. A failed
    ``import vforge.mcore...`` leaves some 120 half-initialized
    ``vforge.mcore.*`` entries in ``sys.modules`` -- including the module that
    raised -- so a second attempt in the same process succeeds against a partly
    built package. That makes ``pytest.importorskip`` order-dependent: the first
    test module to ask skips, every later one concludes mcore works, and their
    ``mp.spawn`` children then run against the wreckage and report a pass.
    """
    return importlib.util.find_spec("triton") is not None


pytestmark = pytest.mark.skipif(
    not _mcore_importable(),
    reason="vforge.mcore imports triton at package level; absent on macOS and on CPU CI",
)

DIM = 32
FFN_DIM = 64
HEADS = 4
WORLD = 2

#: Sharded the two ways that do not mirror each other, plus the norm that is
#: sharded by hand rather than by being an mcore type.
COLUMN_WEIGHT = "attn1.to_q.weight"
ROW_WEIGHT = "attn1.to_out.0.weight"
MARKED_NORM_WEIGHT = "attn1.norm_q.weight"


def _build_block():
    from vforge.models.wan.v21.transformer import WanTransformerBlock

    return WanTransformerBlock(
        dim=DIM,
        ffn_dim=FFN_DIM,
        num_heads=HEADS,
        cross_attn_norm=True,
    )


def _reference_weights() -> dict[str, torch.Tensor]:
    """Logical, unsharded weights, which is what any checkpoint would hold.

    Built before the parallel state exists so ``build_linear`` sees tp off and
    returns plain layers, then filled with values that differ everywhere: a
    shard taken along the wrong axis, or from the wrong rank, has to change
    something.
    """
    block = _build_block()
    generator = torch.Generator().manual_seed(23)
    return {
        key: torch.randn(value.shape, generator=generator, dtype=value.dtype)
        if value.is_floating_point()
        else value.clone()
        for key, value in block.state_dict().items()
    }


def _sharded_block(full: dict[str, torch.Tensor]):
    """This rank's half of ``full``, with the persistence hooks installed."""
    from vforge.distributed.tensor_parallel import (
        apply_mcore_tensor_parallel,
        install_tp_state_dict_hooks,
        shard_full_state_dict,
    )

    # What the megatron engine does: the model declares its splits, the engine
    # replaces each declared projection with one of mcore's parallel layers.
    block = _build_block()
    apply_mcore_tensor_parallel(block)
    missing, unexpected = block.load_state_dict(shard_full_state_dict(block, full), strict=False)
    assert not unexpected
    # mcore's layers carry a ``_extra_state`` entry for TE compatibility whose
    # value is None. Named rather than loaded non-strictly, which would also
    # swallow a genuinely missing weight.
    assert all(key.endswith("_extra_state") for key in missing), missing
    return install_tp_state_dict_hooks(block)


def _local_parameters(block) -> dict[str, torch.Tensor]:
    """Parameters as the module holds them, not as a state dict reports them.

    ``named_parameters`` bypasses the hooks, so this is the plain local shard --
    which is the thing that has to survive a round trip.
    """
    return {name: parameter.detach().clone() for name, parameter in block.named_parameters()}


def _phase_state_dict_reports_whole_tensors(state, full, _checkpoint) -> None:
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    from torch.distributed.tensor import DTensor

    block = _sharded_block(full)
    reported = get_model_state_dict(block)

    for key in (COLUMN_WEIGHT, ROW_WEIGHT, MARKED_NORM_WEIGHT):
        entry = reported[key]
        assert isinstance(entry, DTensor), f"{key} is {type(entry).__name__}"
        assert tuple(entry.shape) == tuple(full[key].shape), key
        assert tuple(entry.to_local().shape) == tuple(dict(block.named_parameters())[key].shape)

    # Row-parallel adds its bias after the all-reduce, so every rank holds the
    # whole of it. Declaring it sharded would divide it by tp_size on the way
    # back in, which no shape check would catch.
    row_bias = reported["attn1.to_out.0.bias"]
    assert not isinstance(row_bias, DTensor)
    assert tuple(row_bias.shape) == tuple(full["attn1.to_out.0.bias"].shape)

    # Nothing tp does not touch should have changed representation.
    untouched = reported["norm2.weight"]
    assert not isinstance(untouched, DTensor)


def _phase_round_trip_is_bit_exact(state, full, checkpoint) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        get_model_state_dict,
        set_model_state_dict,
    )

    saved = _sharded_block(full)
    expected = _local_parameters(saved)
    dcp.save({"model": get_model_state_dict(saved)}, checkpoint_id=checkpoint)
    dist.barrier()

    restored = _sharded_block(full)
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.zero_()
    assert all(value.abs().sum() == 0 for value in _local_parameters(restored).values())

    target = get_model_state_dict(restored)
    dcp.load({"model": target}, checkpoint_id=checkpoint)
    set_model_state_dict(restored, model_state_dict=target)

    actual = _local_parameters(restored)
    assert set(actual) == set(expected)
    for name, value in expected.items():
        assert torch.equal(actual[name], value), name


def _phase_the_pieces_join(state, full, checkpoint) -> None:
    """A reader asking for the whole tensor gets both shards, in rank order.

    This is what makes the checkpoint free of the layout that wrote it: the same
    directory is readable at another ``tp_size``, or by fsdp2, because what was
    recorded is the logical tensor.
    """
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    block = _sharded_block(full)
    dcp.save({"model": get_model_state_dict(block)}, checkpoint_id=checkpoint)
    dist.barrier()

    whole = {
        "model": {
            COLUMN_WEIGHT: torch.zeros_like(full[COLUMN_WEIGHT]),
            ROW_WEIGHT: torch.zeros_like(full[ROW_WEIGHT]),
        }
    }
    dcp.load(whole, checkpoint_id=checkpoint)
    for key in (COLUMN_WEIGHT, ROW_WEIGHT):
        torch.testing.assert_close(whole["model"][key], full[key], rtol=0, atol=0, msg=key)


def _phase_a_whole_tensor_load_is_narrowed(state, full, _checkpoint) -> None:
    """The warm-start shape: a PT export, or rank zero broadcasting its copy.

    Nothing sliced these, so the hook has to, and it has to pick this rank's
    slice rather than any slice -- hence comparing against what
    ``shard_full_state_dict`` independently computes.
    """
    from vforge.distributed.tensor_parallel import shard_full_state_dict

    block = _sharded_block(full)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()

    missing, unexpected = block.load_state_dict(full, strict=False)
    assert not unexpected
    assert all(key.endswith("_extra_state") for key in missing), missing

    expected = shard_full_state_dict(block, full)
    for name, value in _local_parameters(block).items():
        assert torch.equal(value, expected[name]), name


PHASES = {
    "reports_whole_tensors": _phase_state_dict_reports_whole_tensors,
    "round_trip": _phase_round_trip_is_bit_exact,
    "pieces_join": _phase_the_pieces_join,
    "whole_tensor_load": _phase_a_whole_tensor_load_is_narrowed,
}


def _worker(rank: int, phase: str, init_file: str, checkpoint: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=WORLD,
        rank=rank,
    )
    os.environ["LOCAL_RANK"] = str(rank)
    torch.manual_seed(0)
    try:
        from vforge.distributed.parallel_state import (
            ParallelConfig,
            destroy_parallel_state,
            get_parallel_state,
            init_parallel_state,
        )
        from vforge.ops import initialize_ops, reset_ops
        from vforge.ops.config import OpsConfig

        initialize_ops(
            OpsConfig(backend="sdpa", allow_fallback=False),
            device=torch.device("cpu"),
            parallel_state=get_parallel_state(),
        )
        full = _reference_weights()

        state = init_parallel_state(ParallelConfig(tp_size=WORLD, device_type="cpu"))
        assert state.tp_enabled

        PHASES[phase](state, full, checkpoint)

        destroy_parallel_state()
        reset_ops()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("phase", sorted(PHASES))
def test_tensor_parallel_weights_survive_the_checkpoint_boundary(phase, tmp_path):
    mp.spawn(
        _worker,
        args=(phase, str(tmp_path / "pg"), str(tmp_path / "checkpoint")),
        nprocs=WORLD,
        join=True,
    )
