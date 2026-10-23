"""The torch-native tensor-parallel backend, checked against the unsharded model.

This backend reaches the same arrangement as the mcore one by the opposite
route: the model is built with ordinary full-width layers that only carry a
mark, and ``parallelize_module`` replaces the tensors behind them once the tree
exists and its modules have names. The mistakes that route allows are its own:

* a mark never collected, leaving a projection unsharded on every rank -- which
  computes a plausible answer, just not a parallel one;
* a mark collected as the wrong style, so the shard is taken across the
  contraction axis instead of along the output;
* the q/k norm left out of the conversion, which is not an error and not visible
  in a forward pass: fsdp2 later wraps its local-width tensor as a DTensor over
  the data mesh, and the state dict then reports the local width as if it were
  the whole parameter, under one name on every tp rank.

So the properties checked here are the ones those would break: arithmetic
against a reference built before any group existed, and what a state dict says
about every sharded parameter afterwards.

The payoff over mcore is in ``optimizer_round_trip``. Parameters here are
DTensors in their own right, so the optimizer state derived from them carries
the tp axis without anything being declared -- the gap that is still fail-fast
on the megatron engine.

gloo on CPU, and no mcore, so unlike the mcore backend's tests this one runs
everywhere including macOS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DIM = 32
FFN_DIM = 64
HEADS = 4
SEQ = 6
BATCH = 2
TEXT_SEQ = 3
WORLD = 2

COLUMN_WEIGHT = "attn1.to_q.weight"
ROW_WEIGHT = "attn1.to_out.0.weight"
MARKED_NORM_WEIGHT = "attn1.norm_q.weight"
#: Row-parallel splits one contraction over its reduction axis into two partial
#: sums, so the additions happen in a different order than in a single matmul
#: and the last few bits differ. Measured worst case on this block is 1.6e-5
#: relative, on one element of 384; the mistakes this file exists to catch are
#: off by percent or more, so the gap between the two is four orders of
#: magnitude and the tolerance is not hiding anything.
TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}


def _inputs() -> dict[str, torch.Tensor]:
    """One fixed batch, identical on every rank: tp peers share a batch."""
    gen = torch.Generator().manual_seed(17)
    head_dim = DIM // HEADS
    return {
        "hidden_states": torch.randn(BATCH, SEQ, DIM, generator=gen),
        "encoder_hidden_states": torch.randn(BATCH, TEXT_SEQ, DIM, generator=gen),
        "temb": torch.randn(BATCH, 6, DIM, generator=gen),
        "rotary_emb": (
            torch.randn(1, SEQ, 1, head_dim, generator=gen),
            torch.randn(1, SEQ, 1, head_dim, generator=gen),
        ),
    }


def _build_block():
    from vforge.models.wan.v21.transformer import WanTransformerBlock

    return WanTransformerBlock(dim=DIM, ffn_dim=FFN_DIM, num_heads=HEADS, cross_attn_norm=True)


def _sharded_block(full: dict[str, torch.Tensor]):
    """A tp block holding ``full``, built and converted the torch-native way."""
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        set_model_state_dict,
    )

    from vforge.distributed.tensor_parallel import apply_torch_tensor_parallel

    # What the fsdp2 engine does, minus the data sharding: the model declares its
    # splits, the engine converts them.
    block = _build_block()
    apply_torch_tensor_parallel(block)
    # full_state_dict, because what we hold is the unsharded model: the options
    # are what tell DCP to cut each entry down to this rank rather than reject
    # it for having the wrong shape. Copied because it replaces entries in the
    # mapping it is given, and callers here compare against theirs afterwards.
    set_model_state_dict(
        block,
        model_state_dict=dict(full),
        options=StateDictOptions(full_state_dict=True, strict=False),
    )
    return block


@dataclass(frozen=True)
class _Reference:
    """The unsharded model, as weights and as the answer it produces.

    Built and run before ``init_parallel_state``, so ``build_linear`` sees tp off
    and returns exactly what upstream would: this is the model the sharded one
    has to reproduce, not another sharded model that happens to agree.
    """

    weights: dict[str, torch.Tensor]
    output: torch.Tensor
    grads: dict[str, torch.Tensor]


def _reference() -> _Reference:
    block = _build_block()
    generator = torch.Generator().manual_seed(23)
    weights = {
        key: (
            torch.randn(value.shape, generator=generator, dtype=value.dtype)
            if value.is_floating_point()
            else value.clone()
        )
        for key, value in block.state_dict().items()
    }
    block.load_state_dict(weights)
    output, grads = _run(block, _inputs())
    return _Reference(
        weights=weights,
        output=output,
        grads={name: gradient.detach().clone() for name, gradient in grads.items()},
    )


def _run(block, inputs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = block(**inputs)
    # A weighted sum rather than .sum(): a uniform loss lets a transposed or
    # misordered output still produce the right gradient.
    weights = torch.linspace(0.1, 1.0, output.numel()).reshape(output.shape)
    (output * weights).sum().backward()
    grads = {
        name: parameter.grad
        for name, parameter in block.named_parameters()
        if parameter.grad is not None
    }
    return output.detach(), grads


def _whole(tensor: torch.Tensor) -> torch.Tensor:
    """The logical tensor, gathering the shards when there are any."""
    full_tensor = getattr(tensor, "full_tensor", None)
    return full_tensor().detach() if callable(full_tensor) else tensor.detach()


def _phase_equivalence(reference: _Reference, _checkpoint) -> None:
    """Sharded arithmetic must reproduce the model the weights came from."""
    block = _sharded_block(reference.weights)
    assert block.attn1.heads == HEADS // WORLD
    actual_output, actual_grads = _run(block, _inputs())

    torch.testing.assert_close(actual_output, reference.output, **TOLERANCE)
    assert set(actual_grads) == set(reference.grads)
    for name, gradient in actual_grads.items():
        # Compared whole rather than sliced: a shard taken along the wrong axis
        # gathers back into a differently ordered tensor, which a per-rank
        # comparison against the matching slice could miss.
        torch.testing.assert_close(_whole(gradient), reference.grads[name], msg=name, **TOLERANCE)


def _phase_reports_whole_tensors(reference: _Reference, _checkpoint) -> None:
    full = reference.weights
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    from torch.distributed.tensor import DTensor

    block = _sharded_block(full)
    reported = get_model_state_dict(block)

    for key in (COLUMN_WEIGHT, ROW_WEIGHT, MARKED_NORM_WEIGHT):
        entry = reported[key]
        assert isinstance(entry, DTensor), f"{key} is {type(entry).__name__}"
        assert tuple(entry.shape) == tuple(full[key].shape), key

    # Row-parallel replicates its bias, because it is added after the reduction.
    # Declaring it sharded would divide it by tp_size on the way back in, which
    # no shape check would catch.
    row_bias = reported["attn1.to_out.0.bias"]
    assert tuple(row_bias.shape) == tuple(full["attn1.to_out.0.bias"].shape)

    for key, value in reported.items():
        gathered = _whole(value)
        assert not isinstance(gathered, DTensor), f"{key} did not gather: {type(value).__name__}"
        torch.testing.assert_close(gathered, full[key], rtol=0, atol=0, msg=key)


def _phase_optimizer_round_trip(reference: _Reference, checkpoint) -> None:
    """Model and optimizer state both survive, which mcore cannot yet manage."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    full = reference.weights
    saved = _sharded_block(full)
    saved_optimizer = torch.optim.AdamW(saved.parameters(), lr=1e-3)
    _run(saved, _inputs())
    saved_optimizer.step()

    model_state, optimizer_state = get_state_dict(saved, saved_optimizer)
    dcp.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=checkpoint)
    dist.barrier()

    restored = _sharded_block(full)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    # A step, so the optimizer has state to be overwritten rather than created.
    _run(restored, _inputs())
    restored_optimizer.step()

    target_model, target_optimizer = get_state_dict(restored, restored_optimizer)
    dcp.load({"model": target_model, "optimizer": target_optimizer}, checkpoint_id=checkpoint)
    set_state_dict(
        restored,
        restored_optimizer,
        model_state_dict=target_model,
        optim_state_dict=target_optimizer,
    )

    for (name, expected), (_, actual) in zip(
        saved.named_parameters(), restored.named_parameters(), strict=True
    ):
        assert torch.equal(_whole(actual), _whole(expected)), name

    expected_moments = _moments(saved, saved_optimizer)
    actual_moments = _moments(restored, restored_optimizer)
    assert expected_moments, "the optimizer produced no moments to compare"
    assert set(actual_moments) == set(expected_moments)
    for key, expected in expected_moments.items():
        assert torch.equal(actual_moments[key], expected), key


def _moments(block, optimizer) -> dict[str, torch.Tensor]:
    """Adam's per-parameter state, keyed by parameter name and gathered whole."""
    moments: dict[str, torch.Tensor] = {}
    for name, parameter in block.named_parameters():
        entry = optimizer.state.get(parameter)
        if not entry:
            continue
        for key, value in entry.items():
            if torch.is_tensor(value) and value.dim() > 0:
                moments[f"{name}.{key}"] = _whole(value)
    return moments


def _phase_pieces_join(reference: _Reference, checkpoint) -> None:
    """A reader wanting the whole tensor gets both shards, in rank order."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    full = reference.weights
    block = _sharded_block(full)
    dcp.save({"model": get_model_state_dict(block)}, checkpoint_id=checkpoint)
    dist.barrier()

    whole = {
        "model": {
            key: torch.zeros_like(full[key])
            for key in (COLUMN_WEIGHT, ROW_WEIGHT, MARKED_NORM_WEIGHT)
        }
    }
    dcp.load(whole, checkpoint_id=checkpoint)
    for key, value in whole["model"].items():
        torch.testing.assert_close(value, full[key], rtol=0, atol=0, msg=key)


PHASES = {
    "equivalence": _phase_equivalence,
    "optimizer_round_trip": _phase_optimizer_round_trip,
    "pieces_join": _phase_pieces_join,
    "reports_whole_tensors": _phase_reports_whole_tensors,
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
        reference = _reference()

        state = init_parallel_state(ParallelConfig(tp_size=WORLD, device_type="cpu"))
        assert state.tp_enabled

        PHASES[phase](reference, checkpoint)

        destroy_parallel_state()
        reset_ops()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("phase", sorted(PHASES))
def test_the_torch_backend_shards_and_persists_like_the_unsharded_model(phase, tmp_path):
    mp.spawn(
        _worker,
        args=(phase, str(tmp_path / "pg"), str(tmp_path / "checkpoint")),
        nprocs=WORLD,
        join=True,
    )


def test_the_plan_is_collected_from_the_tree_not_written_against_it():
    """A rename cannot desync a plan that was derived from the modules.

    Keying a plan by literal name is the usual way to write this, and its failure
    mode is a missing or stale entry: no error, just an unsharded projection or a
    shape mismatch several layers later. Collected off the modules, the keys are
    the names, so they cannot disagree.
    """
    block = _build_block()
    names = {name for name, module in block.named_modules() if _declared(module)}

    assert "attn1.to_q" in names
    assert "attn1.to_out.0" in names
    assert "ffn.net.0.proj" in names
    assert "ffn.net.2" in names
    # Every declared name is a real submodule, which is the property a
    # hand-written plan is the one that can lose.
    for name in names:
        assert block.get_submodule(name) is not None


def _declared(module) -> bool:
    from vforge.distributed.tensor_parallel import TP_STYLE_ATTR

    return getattr(module, TP_STYLE_ATTR, None) is not None
