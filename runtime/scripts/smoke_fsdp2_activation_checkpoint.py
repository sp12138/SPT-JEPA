"""Distributed smoke for FSDP2 mixed precision plus activation checkpointing."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from vforge import (
    FSDP2Config,
    MixedPrecisionConfig,
    ParallelConfig,
    ParallelizeConfig,
    parallelize_model,
)


class Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.input_dtypes: list[torch.dtype] = []

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.input_dtypes.append(value.dtype)
        return torch.nn.functional.silu(self.linear(value))


class CheckpointedModel(nn.Module):
    _no_split_modules = ["Block"]

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block(width), Block(width)])
        self.gradient_checkpointing = True

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            # Reproduce an fp32 checkpoint input that FSDP2 casts for forward.
            value = checkpoint(block, value.float(), use_reentrant=False)
        return value


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        world_size = dist.get_world_size()
        model = CheckpointedModel().cuda()
        model = parallelize_model(
            model,
            ParallelizeConfig(
                parallel=ParallelConfig(
                    dp_size=world_size,
                    dp_mode="fsdp2",
                    device_type="cuda",
                ),
                fsdp2=FSDP2Config(
                    mixed_precision=MixedPrecisionConfig(
                        enable=True,
                        param_dtype="bfloat16",
                        reduce_dtype="float32",
                    )
                ),
            ),
        )
        loss = model(torch.randn(4, 32, device="cuda")).float().square().mean()
        loss.backward()
        for block in model.blocks:
            if block.input_dtypes != [torch.bfloat16, torch.bfloat16]:
                raise AssertionError(f"forward/recompute dtype mismatch: {block.input_dtypes}")
        if dist.get_rank() == 0:
            print(
                "vforge FSDP2 activation-checkpoint smoke ok "
                f"torch={torch.__version__} world_size={world_size}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
