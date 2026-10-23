"""8-GPU distributed smoke for vforge's first milestone.

Run inside the debug container:

    PYTHONPATH=/data/work torchrun --standalone --nproc-per-node=8 \
        scripts/smoke_distributed.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn

from vforge import (
    FSDP2Config,
    MixedPrecisionConfig,
    ParallelConfig,
    ParallelizeConfig,
    init_parallel_state,
    parallelize_model,
)
from vforge.distributed.ulysses import gather_sequence, slice_sequence
from vforge.utils.comparison import step_snapshot_json
from vforge.utils.rng import seed_everything


class TinyBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x))


class TinyModel(nn.Module):
    _no_split_modules = ["TinyBlock"]

    def __init__(self, dim: int = 16) -> None:
        super().__init__()
        self.layers = nn.Sequential(TinyBlock(dim), TinyBlock(dim))
        self.out = nn.Linear(dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.layers(x))


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    usp_size = int(os.environ.get("VFORGE_ULYSSES_SIZE", "1"))
    if world_size % usp_size != 0:
        raise RuntimeError(f"world_size={world_size} must divide usp_size={usp_size}")
    dp_size = world_size // usp_size

    seed_everything(20260713 + rank, deterministic=False)
    state = init_parallel_state(
        ParallelConfig(
            dp_size=dp_size,
            usp_size=usp_size,
            dp_mode="fsdp2",
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )
    )

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if usp_size > 1:
        sequence = torch.arange(
            usp_size * 8,
            device=device,
            dtype=torch.float32,
        ).reshape(usp_size * 2, 4)
        gathered = gather_sequence(slice_sequence(sequence, dim=0), dim=0)
        if not torch.equal(sequence, gathered):
            raise RuntimeError(f"sequence parallel gather mismatch on rank {rank}")

    model = TinyModel().to(device)
    model = parallelize_model(
        model,
        ParallelizeConfig(
            parallel=state.config,
            fsdp2=FSDP2Config(
                enable=True,
                mixed_precision=MixedPrecisionConfig(enable=False),
            ),
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.arange(32 * 16, device=device, dtype=torch.float32).reshape(32, 16) / 1000
    y = model(x).float().square().mean()
    y.backward()
    optimizer.step()

    loss = y.detach().float()
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss = loss / world_size

    if rank == 0:
        snapshot = step_snapshot_json(
            model,
            loss,
            extra={
                "world_size": world_size,
                "dp_size": dp_size,
                "usp_size": usp_size,
            },
        )

    optimizer.zero_grad(set_to_none=True)

    if rank == 0:
        print(
            "vforge distributed smoke ok "
            f"world_size={world_size} dp_size={dp_size} usp_size={usp_size} "
            f"avg_loss={loss.item():.8f}"
        )
        print(snapshot)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
