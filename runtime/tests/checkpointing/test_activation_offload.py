import pytest
import torch
from torch.utils.checkpoint import checkpoint

from vforge.checkpointing.offload import (
    ActivationOffloadConfig,
    ActivationOffloadManager,
    AttentionOffloadPolicy,
    _CheckpointBlockState,
)


def test_public_offload_exports_share_the_native_policy():
    from vforge import AttentionOffloadPolicy as PublicPolicy
    from vforge.training import AttentionOffloadPolicy as TrainingPolicy

    assert PublicPolicy is TrainingPolicy is AttentionOffloadPolicy


def test_cpu_tensor_is_kept_and_accounted():
    manager = ActivationOffloadManager(
        ActivationOffloadConfig(min_tensor_bytes=1, pin_memory=False)
    )
    value = torch.randn(8)

    packed = manager.pack(value)

    stats = manager.pop_stats()
    assert isinstance(packed, torch.Tensor)
    assert stats["packed_tensors"] == 0
    assert stats["kept_tensors"] == 1
    assert stats["kept_bytes"] == value.numel() * value.element_size()


def test_pop_stats_resets_counters():
    manager = ActivationOffloadManager()

    first = manager.pop_stats()
    second = manager.pop_stats()

    assert first == second
    assert second["packed_bytes"] == 0


def test_async_transfer_requires_pinned_memory():
    with pytest.raises(ValueError, match="requires pinned"):
        ActivationOffloadManager(
            ActivationOffloadConfig(
                pin_memory=False,
                async_transfer=True,
            )
        )


def test_backward_prefetch_primes_last_group_then_advances(monkeypatch):
    manager = ActivationOffloadManager()
    first = _CheckpointBlockState(manager, (0,))
    second = _CheckpointBlockState(manager, (0,))
    second.previous = first
    manager._checkpoint_blocks = [first, second]
    events = []

    monkeypatch.setattr(
        manager,
        "_schedule_checkpoint_block",
        lambda group: events.append(group),
    )

    manager.start_backward_prefetch()
    second.advance_backward_prefetch()

    assert events == [second, first]


def test_step_lifecycle_rejects_overlap_and_abort_resets_state():
    manager = ActivationOffloadManager()

    manager.begin_step()
    with pytest.raises(RuntimeError, match="not finalized"):
        manager.begin_step()

    manager.abort_step()
    manager.begin_step()
    manager.finish_backward_prefetch()


def test_manager_aware_checkpoint_rejects_nested_invocations():
    manager = ActivationOffloadManager()
    value = torch.randn(4, requires_grad=True)

    def managed_checkpoint(function, *args):
        return checkpoint(
            function,
            *args,
            use_reentrant=False,
            context_fn=manager.checkpoint_contexts,
        )

    def outer(tensor):
        return managed_checkpoint(lambda inner: inner.square(), tensor)

    with pytest.raises(RuntimeError, match="nested manager-aware checkpoint"):
        managed_checkpoint(outer, value)

    assert torch.equal(
        managed_checkpoint(lambda tensor: tensor + 1, value),
        value + 1,
    )
