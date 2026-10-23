"""A tensor-parallel Wan block must compute what the unsharded one computes.

Everything else about tp has been checked by reading: that the mesh hands out
the right groups, that mcore accepts an injected one, that a projection comes
back the expected shape. None of that says the arithmetic is right. Four
mistakes would pass every existing test and still train a wrong model:

* a projection sharded by row where it should be column, or the reverse;
* the all-reduce that ends ``RowParallelLinear`` never happening, leaving each
  rank holding a partial sum that looks entirely plausible;
* ``local_heads`` disagreeing with the reshape that follows q/k/v, so heads are
  read out of the wrong slice of the projection;
* the input-gradient all-reduce missing from the column-parallel backward,
  which no forward comparison can see.

So this builds the same block twice in one process -- once before the parallel
state exists, which is the unsharded reference, then once with tp on -- gives
both the same weights, and compares forward and gradients.

Not bit-exact, deliberately. Row-parallel splits one contraction over the
reduction axis into two partial sums, so the summation order differs from a
single matmul. In fp32 that costs a few ulps while any of the errors above is
off by percent or more, which is what makes a tight tolerance meaningful here;
in bf16 the two would be indistinguishable.

gloo on CPU: the questions are which ranks talk and what each holds, and
neither needs a GPU. It does need Linux, since ``vforge.mcore`` imports triton.
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
    ``mp.spawn`` children then run against the wreckage.
    """
    return importlib.util.find_spec("triton") is not None


pytestmark = pytest.mark.skipif(
    not _mcore_importable(),
    reason="vforge.mcore imports triton at package level; absent on macOS and on CPU CI",
)

DIM = 32
FFN_DIM = 64
HEADS = 4
SEQ = 6
BATCH = 2
TEXT_SEQ = 3
#: Row-parallel splits one contraction over its reduction axis into two partial
#: sums, so the additions happen in a different order than in a single matmul.
#: Same tolerance as the torch-native equivalence test, because both backends do
#: the same reassociation and there is no reason for one to be held tighter. The
#: mistakes this file exists to catch are off by percent or more across many
#: elements, four orders of magnitude away from this.
TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}

#: Gradients are compared by relative Frobenius norm rather than elementwise.
#:
#: Elementwise is the wrong instrument here: ``assert_close`` asks each element to
#: be within ``rtol`` of *its own* magnitude, so one entry that happens to sit
#: near zero can be thousands of eps away in relative terms while being
#: numerically irrelevant, and the test then flips on whichever weights it was
#: given. What matters is the deviation against the tensor's own scale.
#:
#: Measured worst case on this block is 6.9e-07, so the threshold leaves an order
#: of magnitude and the run prints where it actually sits. The mistakes this file
#: exists to catch -- a shard taken along the wrong axis, a missing reduction,
#: heads read from the wrong slice -- move a gradient by percent or more, four
#: orders above. A single badly wrong entry among 512 is not diluted away either;
#: it would land near its own relative size, far above the threshold.
GRADIENT_TOLERANCE = 1e-5


def _inputs(dim: int) -> dict[str, torch.Tensor]:
    """One fixed batch, identical on every rank: tp peers share a batch."""
    gen = torch.Generator().manual_seed(17)
    head_dim = dim // HEADS
    return {
        "hidden_states": torch.randn(BATCH, SEQ, dim, generator=gen),
        "encoder_hidden_states": torch.randn(BATCH, TEXT_SEQ, dim, generator=gen),
        "temb": torch.randn(BATCH, 6, dim, generator=gen),
        "rotary_emb": (
            torch.randn(1, SEQ, 1, head_dim, generator=gen),
            torch.randn(1, SEQ, 1, head_dim, generator=gen),
        ),
    }


def _build_block():
    from vforge.models.wan.v21.transformer import WanTransformerBlock

    return WanTransformerBlock(
        dim=DIM,
        ffn_dim=FFN_DIM,
        num_heads=HEADS,
        cross_attn_norm=True,
    )


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """How far ``actual`` is from ``expected``, against ``expected``'s own scale."""
    assert actual.shape == expected.shape, f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()


def _seeded_weights(block) -> dict[str, torch.Tensor]:
    """Every float entry of ``block``'s state dict, drawn from one fixed seed."""
    generator = torch.Generator().manual_seed(23)
    return {
        key: (
            torch.randn(value.shape, generator=generator, dtype=value.dtype)
            if value.is_floating_point()
            else value.clone()
        )
        for key, value in block.state_dict().items()
    }


def _run(block, inputs) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = block(**inputs)
    # A weighted sum rather than .sum(): a uniform loss lets a transposed or
    # misordered output still produce the right gradient.
    weights = torch.linspace(0.1, 1.0, output.numel()).reshape(output.shape)
    (output * weights).sum().backward()
    grads = {
        name: p.grad.detach().clone() for name, p in block.named_parameters() if p.grad is not None
    }
    return output.detach(), grads


def _worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=world_size,
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
        from vforge.distributed.tensor_parallel import (
            apply_mcore_tensor_parallel,
            shard_full_state_dict,
        )
        from vforge.ops import initialize_ops, reset_ops
        from vforge.ops.config import OpsConfig

        initialize_ops(
            OpsConfig(backend="sdpa", allow_fallback=False),
            device=torch.device("cpu"),
            parallel_state=get_parallel_state(),
        )
        inputs = _inputs(DIM)

        # Built before init_parallel_state, so nothing is sharded: this is the
        # model we must reproduce.
        reference = _build_block()
        # Weights from one seeded generator rather than from module init, so this
        # comparison does not depend on the order the modules happened to be
        # constructed in. It otherwise does: adding or moving a single module
        # shifts every later draw, the reference becomes a different model, and a
        # test sitting near its tolerance flips for a reason unrelated to tp.
        full_state = _seeded_weights(reference)
        reference.load_state_dict(full_state)
        expected_output, expected_grads = _run(reference, inputs)

        state = init_parallel_state(ParallelConfig(tp_size=world_size, device_type="cpu"))
        assert state.tp_enabled

        # What the megatron engine does: the declared projections are replaced
        # with mcore's parallel layers, whose weights are already narrow.
        sharded = _build_block()
        apply_mcore_tensor_parallel(sharded)
        local_state = shard_full_state_dict(sharded, full_state)
        # mcore's layers carry a ``_extra_state`` entry for TE compatibility
        # whose value is None. Named here rather than loading non-strictly,
        # which would also swallow a genuinely missing weight.
        missing, unexpected = sharded.load_state_dict(local_state, strict=False)
        assert not unexpected
        assert all(key.endswith("_extra_state") for key in missing), missing

        # Same q/k/v projection, a quarter of it per rank at tp=2 with 4 heads.
        assert sharded.attn1.heads == HEADS // world_size
        assert sharded.attn1.to_q.weight.shape == (DIM // world_size, DIM)
        assert sharded.attn1.to_out[0].weight.shape == (DIM, DIM // world_size)

        actual_output, actual_grads = _run(sharded, inputs)
        torch.testing.assert_close(actual_output, expected_output, **TOLERANCE)

        # Gradients per rank are the slice of the reference gradient that this
        # rank's shard is responsible for, so compare against that slice rather
        # than gathering: a wrong shard axis shows up as a shape mismatch and a
        # missing reduction as a value mismatch. The worst error is reported
        # whether or not it fails, so a drift towards the threshold is visible in
        # a passing log rather than only in the run that finally trips it.
        sliced = shard_full_state_dict(sharded, expected_grads)
        assert set(sliced) == set(actual_grads)
        errors = {
            name: _relative_error(actual_grads[name], expected) for name, expected in sliced.items()
        }
        worst = max(errors, key=errors.__getitem__)
        if rank == 0:
            ranked = sorted(errors.items(), key=lambda item: -item[1])[:5]
            print("worst gradient errors: " + ", ".join(f"{k}={v:.2e}" for k, v in ranked))
        assert errors[worst] <= GRADIENT_TOLERANCE, f"{worst} off by {errors[worst]:.3e}"

        destroy_parallel_state()
        reset_ops()
    finally:
        dist.destroy_process_group()


def test_a_sharded_block_matches_the_unsharded_one(tmp_path):
    init_file = str(tmp_path / "tp-equivalence-pg")
    mp.spawn(_worker, args=(2, init_file), nprocs=2, join=True)
