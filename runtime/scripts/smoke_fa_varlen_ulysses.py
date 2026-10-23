#!/usr/bin/env python3
"""Validate masked FA4 recompute offload through eight-rank Ulysses."""

from __future__ import annotations

import json
import os
from functools import partial

import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from vforge.checkpointing.offload import (
    ActivationOffloadConfig,
    ActivationOffloadManager,
)
from vforge.distributed.ulysses import ulysses_attention
from vforge.ops._attention import unified_attention, unified_attention_varlen


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if world_size != 8:
        raise RuntimeError(f"expected world_size=8, got {world_size}")

    batch = 2
    padded_length = 1024
    local_length = padded_length // world_size
    heads = 16
    head_dim = 64
    valid_lengths = torch.tensor(
        [1019, 1007],
        device="cuda",
        dtype=torch.int64,
    )
    mask = torch.arange(padded_length, device="cuda").view(
        1, 1, 1, padded_length
    ) < valid_lengths.view(batch, 1, 1, 1)

    torch.manual_seed(42 + rank)
    direct_inputs = tuple(
        torch.randn(
            batch,
            local_length,
            heads,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(3)
    )
    offload_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in direct_inputs)
    attention_fn = partial(
        ulysses_attention,
        group=dist.group.WORLD,
        world_size=world_size,
        attention_fn=unified_attention,
        varlen_attention_fn=unified_attention_varlen,
        attn_mask=mask,
    )

    direct_output = attention_fn(*direct_inputs)
    direct_output.float().sum().backward()

    controller = ActivationOffloadManager(
        ActivationOffloadConfig(
            min_tensor_bytes=1,
            pin_memory=True,
            async_transfer=True,
        )
    )
    controller.begin_step()
    offload_output = checkpoint(
        attention_fn,
        *offload_inputs,
        use_reentrant=False,
        context_fn=controller.checkpoint_contexts,
    )
    controller.start_backward_prefetch()
    try:
        offload_output.float().sum().backward()
    finally:
        controller.finish_backward_prefetch()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        offload_output,
        direct_output,
        rtol=0,
        atol=0,
    )
    for offload_input, direct_input in zip(
        offload_inputs,
        direct_inputs,
        strict=True,
    ):
        torch.testing.assert_close(
            offload_input.grad,
            direct_input.grad,
            rtol=5e-3,
            atol=5e-3,
        )

    global_start = rank * local_length
    global_positions = torch.arange(local_length, device="cuda") + global_start
    local_keep = global_positions.unsqueeze(0) < valid_lengths.unsqueeze(1)
    padded_output = offload_output.masked_select(~local_keep[:, :, None, None])
    if torch.count_nonzero(padded_output):
        raise AssertionError("Ulysses padding output was not zero")
    for tensor in offload_inputs:
        padded_grad = tensor.grad.masked_select(~local_keep[:, :, None, None])
        if torch.count_nonzero(padded_grad):
            raise AssertionError("Ulysses padding gradient was not zero")

    stats = controller.pop_stats()
    if stats["current_live_bytes"] != 0:
        raise AssertionError(f"offload tensors leaked: {stats}")
    success = torch.ones((), device="cuda", dtype=torch.int32)
    dist.all_reduce(success, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(
            "FA4_ULYSSES_VARLEN "
            + json.dumps(
                {
                    "world_size": world_size,
                    "padded_length": padded_length,
                    "valid_lengths": valid_lengths.cpu().tolist(),
                    "stats": stats,
                    "success": bool(success.item()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
