from functools import partial

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from vforge.checkpointing.offload import ActivationOffloadConfig, ActivationOffloadManager
from vforge.ops._attention import dispatch as attention
from vforge.ops._attention import flash
from vforge.ops._attention.flash import FA4Backend
from vforge.ops._attention.teacher_forcing import TeacherForcingSparsity


def _run_two_attention_calls(monkeypatch, preserve_indices):
    calls = {"public": 0, "forward": 0, "backward": 0}

    def public(q, k, v, *, causal=False):
        assert not causal
        calls["public"] += 1
        return q + k + v

    def private_forward(q, k, v, **_kwargs):
        calls["forward"] += 1
        return q + k + v, torch.zeros(q.shape[:-1], dtype=torch.float32)

    def private_backward(_q, _k, _v, _out, dout, _lse, **_kwargs):
        calls["backward"] += 1
        return dout, dout, dout

    monkeypatch.setattr(
        attention,
        "_active_flash_backend",
        FA4Backend(public, public, private_forward, private_backward),
    )
    manager = ActivationOffloadManager(preserve_call_indices=preserve_indices)
    manager.begin_step()

    def block(q, k, v):
        first = attention.unified_attention(q, k, v)
        return attention.unified_attention(first, k, v)

    inputs = tuple(torch.randn(2, 3, requires_grad=True) for _ in range(3))
    output = checkpoint(
        block,
        *inputs,
        use_reentrant=False,
        context_fn=manager.checkpoint_contexts,
    )
    manager.start_backward_prefetch()
    try:
        output.sum().backward()
    finally:
        manager.finish_backward_prefetch()
    return calls, inputs, manager.pop_stats()


@pytest.mark.parametrize(
    ("preserve_indices", "expected_private_calls"),
    [
        ((0,), 1),
        ((1,), 1),
        ((0, 1), 2),
        (None, 2),
    ],
)
def test_fa_operator_selects_calls_and_skips_private_recompute(
    monkeypatch,
    preserve_indices,
    expected_private_calls,
):
    calls, inputs, stats = _run_two_attention_calls(monkeypatch, preserve_indices)

    assert calls["forward"] == expected_private_calls
    assert calls["backward"] == expected_private_calls
    assert calls["public"] >= 2 - expected_private_calls
    assert all(torch.isfinite(value.grad).all() for value in inputs)
    assert stats["kept_tensors"] == 2 * expected_private_calls
    assert stats["current_live_bytes"] == 0


def test_manager_checkpoint_context_allows_no_matching_fa_call(monkeypatch):
    calls, inputs, stats = _run_two_attention_calls(monkeypatch, (3,))

    assert calls["forward"] == 0
    assert calls["backward"] == 0
    assert calls["public"] == 2
    assert all(torch.isfinite(value.grad).all() for value in inputs)
    assert stats["kept_tensors"] == 0


def test_sparse_fa4_keeps_original_metadata_during_recompute(monkeypatch):
    generations = []
    mask_mod = object()
    seen = {}
    calls = {"forward": 0, "backward": 0}

    def build_metadata(*_args, **_kwargs):
        metadata = (object(), object())
        generations.append(metadata)
        return metadata

    def public(*_args, **_kwargs):
        raise AssertionError("selected sparse attention must use private ops")

    def private_forward(q, k, v, **kwargs):
        calls["forward"] += 1
        seen["forward"] = kwargs
        return q + k + v, torch.zeros(q.shape[:-1], dtype=torch.float32)

    def private_backward(_q, _k, _v, _out, dout, _lse, **kwargs):
        calls["backward"] += 1
        seen["backward"] = kwargs
        return dout, dout, dout

    monkeypatch.setattr(
        attention,
        "_active_flash_backend",
        FA4Backend(public, public, private_forward, private_backward),
    )
    monkeypatch.setattr(flash, "build_metadata", build_metadata)
    monkeypatch.setattr(flash, "mask_mod_cute", lambda _sparsity: mask_mod)
    manager = ActivationOffloadManager()
    manager.begin_step()
    sparsity = TeacherForcingSparsity(
        num_latent_frames=1,
        tokens_per_frame=2,
        frame_per_block=1,
    )
    inputs = tuple(torch.randn(1, sparsity.seqlen, 1, 4, requires_grad=True) for _ in range(3))

    output = checkpoint(
        partial(attention.unified_attention, sparsity=sparsity),
        *inputs,
        use_reentrant=False,
        context_fn=manager.checkpoint_contexts,
    )
    output.sum().backward()
    manager.finish_backward_prefetch()

    assert len(generations) == 2
    assert calls == {"forward": 1, "backward": 1}
    forward_sparse, backward_sparse = generations[0]
    assert seen["forward"]["mask_mod"] is mask_mod
    assert seen["forward"]["block_sparse_tensors"] is forward_sparse
    assert seen["backward"]["mask_mod"] is mask_mod
    assert seen["backward"]["block_sparse_tensors"] is backward_sparse
    assert all(torch.isfinite(value.grad).all() for value in inputs)


def test_checkpointed_fa_prefetches_previous_block_and_releases_tensors(monkeypatch):
    events = []
    forward_calls = 0

    class RecordingManager(ActivationOffloadManager):
        def _schedule_checkpoint_block(self, block):
            if not block.prefetched:
                events.append(("prefetch", self._checkpoint_blocks.index(block)))
            super()._schedule_checkpoint_block(block)

        def release(self, *packed_tensors):
            if packed_tensors:
                events.append(("release", len(packed_tensors)))
            super().release(*packed_tensors)

    def public(*_args, **_kwargs):
        raise AssertionError("selected attention must use private ops")

    def private_forward(q, k, v, **_kwargs):
        nonlocal forward_calls
        forward_calls += 1
        return q + k + v, torch.zeros(q.shape[:-1], dtype=torch.float32)

    def private_backward(_q, _k, _v, _out, dout, _lse, **_kwargs):
        events.append(("backward", None))
        return dout, dout, dout

    monkeypatch.setattr(
        attention,
        "_active_flash_backend",
        FA4Backend(public, public, private_forward, private_backward),
    )
    manager = RecordingManager(ActivationOffloadConfig(async_transfer=True))
    manager.begin_step()
    inputs = tuple(torch.randn(2, 3, requires_grad=True) for _ in range(3))
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
        output.sum().backward()
    finally:
        manager.finish_backward_prefetch()

    assert forward_calls == 2
    assert events == [
        ("prefetch", 1),
        ("prefetch", 0),
        ("backward", None),
        ("release", 2),
        ("backward", None),
        ("release", 2),
    ]
    for value, expected_grad in zip(inputs, (1, 2, 2), strict=True):
        torch.testing.assert_close(value.grad, torch.full_like(value, expected_grad))
    assert manager.pop_stats()["current_live_bytes"] == 0
