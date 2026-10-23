from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def test_mcore_attention_adapter_observes_torch_autocast_dtype() -> None:
    from vforge.distributed.context_parallel_mcore import _autocast_attention_inputs

    query = torch.ones(1, 4, 2, 8)
    outside = _autocast_attention_inputs(query, query, query)
    assert all(tensor is query for tensor in outside)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        inside = _autocast_attention_inputs(query, query, query)

    assert all(tensor.dtype == torch.bfloat16 for tensor in inside)


def test_context_parallel_backend_lifecycle_is_explicit_and_idempotent() -> None:
    from vforge.distributed.context_parallel import (
        get_context_parallel_backend,
        initialize_context_parallel,
        reset_context_parallel,
    )

    reset_context_parallel()
    with pytest.raises(RuntimeError, match="no backend is initialized"):
        get_context_parallel_backend()

    backend = initialize_context_parallel("mcore")
    assert backend.name == "mcore"
    assert initialize_context_parallel("mcore") is backend

    reset_context_parallel()
    with pytest.raises(RuntimeError, match="no backend is initialized"):
        get_context_parallel_backend()


def _sequence_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        world_size=world_size,
        rank=rank,
    )
    os.environ["LOCAL_RANK"] = str(rank)
    try:
        from vforge.distributed.context_parallel import (
            gather_sequence,
            initialize_context_parallel,
            reset_context_parallel,
            shard_sequence,
        )
        from vforge.distributed.parallel_state import (
            ParallelConfig,
            destroy_parallel_state,
            init_parallel_state,
        )

        state = init_parallel_state(
            ParallelConfig(cp_size=world_size, dp_mode="ddp", device_type="cpu")
        )
        initialize_context_parallel("mcore")
        full = torch.arange(16, dtype=torch.float32).view(1, 16, 1).requires_grad_(True)
        local = shard_sequence(full, dim=1, group=state.cp_group)
        expected = [0, 1, 2, 3, 12, 13, 14, 15] if rank == 0 else [4, 5, 6, 7, 8, 9, 10, 11]
        assert local.flatten().tolist() == expected

        restored = gather_sequence(local, dim=1, group=state.cp_group)
        torch.testing.assert_close(restored, full)
        restored.square().sum().backward()

        expected_grad = torch.zeros_like(full)
        indices = torch.tensor(expected)
        expected_grad[:, indices] = 2 * full.detach()[:, indices]
        torch.testing.assert_close(full.grad, expected_grad)
        assert dist.get_world_size(state.attention_tp_group) == 1
        assert dist.get_process_group_ranks(state.dp_cp_group) == list(range(world_size))
        reset_context_parallel()
        destroy_parallel_state()
    finally:
        dist.destroy_process_group()


def test_context_parallel_zigzag_roundtrip_and_backward(tmp_path) -> None:
    mp.spawn(
        _sequence_worker,
        args=(2, str(tmp_path / "cp-sequence-pg")),
        nprocs=2,
        join=True,
    )
