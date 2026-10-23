from contextlib import nullcontext
from functools import partial

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from vforge.checkpointing.offload import (
    ActivationOffloadConfig,
    ActivationOffloadManager,
)
from vforge.distributed.parallel_state import ParallelConfig, ParallelState
from vforge.distributed.parallelize import FSDP2Config, ParallelizeConfig
from vforge.ops._attention import dispatch as attention
from vforge.training.engine import EnginePolicy, EngineRunContext
from vforge.training.engine.fsdp import FsdpEngine
from vforge.training.recipe import StepContext, StepOutput

pytestmark = [pytest.mark.gpu(gpus=1), pytest.mark.usefixtures("fa4_backend")]


def _manager() -> ActivationOffloadManager:
    return ActivationOffloadManager(
        ActivationOffloadConfig(
            min_tensor_bytes=1,
            pin_memory=True,
            async_transfer=True,
        )
    )


def _inputs(shape, *, seed=42):
    torch.manual_seed(seed)
    return tuple(
        torch.randn(
            *shape,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(3)
    )


def _run_offloaded(attention_fn, inputs, manager):
    manager.begin_step()
    output = checkpoint(
        attention_fn,
        *inputs,
        use_reentrant=False,
        context_fn=manager.checkpoint_contexts,
    )
    manager.start_backward_prefetch()
    try:
        output.float().sum().backward()
    finally:
        manager.finish_backward_prefetch()
    return output


def _assert_parity(attention_fn, shape):
    direct_inputs = _inputs(shape)
    offload_inputs = tuple(value.detach().clone().requires_grad_(True) for value in direct_inputs)
    direct_output = attention_fn(*direct_inputs)
    direct_output.float().sum().backward()

    manager = _manager()
    offload_output = _run_offloaded(
        attention_fn,
        offload_inputs,
        manager,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(offload_output, direct_output, rtol=0, atol=0)
    for offload_input, direct_input in zip(
        offload_inputs,
        direct_inputs,
        strict=True,
    ):
        assert torch.isfinite(offload_input.grad).all()
        torch.testing.assert_close(
            offload_input.grad,
            direct_input.grad,
            rtol=5e-3,
            atol=5e-3,
        )
    stats = manager.pop_stats()
    assert stats["packed_tensors"] == 2
    assert stats["restored_tensors"] == 2
    assert stats["current_live_bytes"] == 0
    return stats


@pytest.mark.parametrize("mode", ["dense", "mask", "varlen"])
def test_real_fa4_checkpoint_offload_matches_direct_gradients(mode):
    shape = (2, 132, 4, 64)
    if mode == "mask":
        lengths = torch.tensor([127, 126], device="cuda")
        mask = torch.arange(132, device="cuda").view(1, 1, 1, 132)
        attention_fn = partial(
            attention.unified_attention,
            attn_mask=mask < lengths.view(2, 1, 1, 1),
        )
    elif mode == "varlen":
        attention_fn = partial(
            attention.unified_attention_varlen,
            real_len=127,
        )
    else:
        attention_fn = attention.unified_attention
    _assert_parity(attention_fn, shape)


def test_engine_accumulates_real_fa4_offloaded_microbatches():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0, device="cuda"))

        def forward(self, q, k, v):
            return attention.unified_attention(q * self.scale.to(q.dtype), k, v)

    model, reference = Model(), Model()
    batches = [
        tuple(value.detach() for value in _inputs((1, length, 4, 64), seed=42 + index))
        for index, length in enumerate((96, 160, 128, 192))
    ]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    oracle_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    for batch in batches:
        (reference(*batch).float().square().mean() / len(batches)).backward()
    oracle_optimizer.step()

    manager = _manager()
    engine = FsdpEngine(
        EnginePolicy(
            parallelize=ParallelizeConfig(
                parallel=ParallelConfig(), fsdp2=FSDP2Config(enable=False)
            ),
            gradient_accumulation_steps=len(batches),
        )
    )

    def compute(root, batch, *_):
        output = checkpoint(
            root, *batch, use_reentrant=False, context_fn=manager.checkpoint_contexts
        )
        return StepOutput(output.float().square().mean())

    engine.run_update(
        model=model,
        microbatches=iter(batches),
        context=StepContext(torch.device("cuda"), ParallelState()),
        step=0,
        compute_step=compute,
        optimizer=optimizer,
        scheduler=None,
        run=EngineRunContext(precision_context=nullcontext, offload_manager=manager),
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(model.scale.grad, reference.scale.grad, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(model.scale, reference.scale, rtol=5e-3, atol=5e-3)
    stats = manager.pop_stats()
    assert stats["packed_tensors"] == stats["restored_tensors"] == 2 * len(batches)
    assert stats["current_live_bytes"] == 0


def test_async_restore_handles_changing_shapes_across_steps():
    manager = _manager()
    correctness_records = []
    for sequence_length in (160, 96, 224):
        inputs = _inputs((1, sequence_length, 4, 64), seed=sequence_length)
        reference_inputs = tuple(value.detach().clone().requires_grad_(True) for value in inputs)
        manager.begin_step()
        output = inputs[0]
        for _ in range(2):
            output = checkpoint(
                attention.unified_attention,
                output,
                inputs[1],
                inputs[2],
                use_reentrant=False,
                context_fn=manager.checkpoint_contexts,
            )
        manager.start_backward_prefetch()
        try:
            output.float().sum().backward()
        finally:
            manager.finish_backward_prefetch()
        step_stats = manager.pop_stats()
        assert step_stats["current_live_bytes"] == 0
        correctness_records.append(
            (
                output.detach().clone(),
                tuple(value.grad.detach().clone() for value in inputs),
                reference_inputs,
            )
        )

    # Dynamic restores rely on allocator stream tracking, not host synchronization.
    torch.cuda.synchronize()
    for offload_output, offload_grads, reference_inputs in correctness_records:
        reference_output = reference_inputs[0]
        for _ in range(2):
            reference_output = attention.unified_attention(
                reference_output,
                reference_inputs[1],
                reference_inputs[2],
            )
        reference_output.float().sum().backward()
        torch.testing.assert_close(
            offload_output,
            reference_output,
            rtol=0,
            atol=0,
        )
        for offload_grad, reference_input in zip(
            offload_grads,
            reference_inputs,
            strict=True,
        ):
            torch.testing.assert_close(
                offload_grad,
                reference_input.grad,
                rtol=5e-3,
                atol=5e-3,
            )
