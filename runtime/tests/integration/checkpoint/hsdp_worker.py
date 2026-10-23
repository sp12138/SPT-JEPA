"""Distributed worker for the pytest HSDP smoke."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from vforge.distributed.parallel_state import ParallelConfig
from vforge.distributed.parallelize import FSDP2Config, ParallelizeConfig, parallelize_model


class TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj_in = nn.Linear(width, width * 2)
        self.proj_out = nn.Linear(width * 2, width)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.proj_out(torch.nn.functional.silu(self.proj_in(hidden_states)))


class TinyModel(nn.Module):
    _no_split_modules = ["TinyBlock"]

    def __init__(self, width: int = 64, depth: int = 4) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(TinyBlock(width) for _ in range(depth))
        self.output = nn.Linear(width, width)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.output(hidden_states)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-replicate-size", type=int, required=True)
    parser.add_argument("--dp-shard-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument(
        "--checkpoint-mode",
        choices=("none", "save", "load", "roundtrip"),
        default="none",
    )
    return parser.parse_args()


def _checkpoint_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    model_state, optimizer_state = get_state_dict(model, optimizer)
    return {"model": model_state, "optimizer": optimizer_state}


def _load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
) -> None:
    state = _checkpoint_state(model, optimizer)
    dcp.load(state_dict=state, checkpoint_id=str(checkpoint_dir))
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
    )


def _local_parameter_checksum(model: nn.Module) -> torch.Tensor:
    first_parameter = next(model.parameters())
    checksum = torch.zeros((), device=first_parameter.device, dtype=torch.float64)
    for parameter in model.parameters():
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        checksum += local.detach().double().sum()
    return checksum


def run() -> None:
    args = _parse_args()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())

    config = ParallelConfig(
        dp_size=world_size,
        dp_replicate_size=args.dp_replicate_size,
        dp_shard_size=args.dp_shard_size,
        device_type="cuda",
    ).normalized(world_size)
    torch.manual_seed(1234)
    model = TinyModel().to(device)
    model = parallelize_model(
        model,
        ParallelizeConfig(
            parallel=config,
            fsdp2=FSDP2Config(enable=True),
        ),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    if args.checkpoint_mode == "load":
        if args.checkpoint_dir is None:
            raise ValueError("--checkpoint-dir is required for checkpoint load")
        _load_checkpoint(model, optimizer, args.checkpoint_dir)

    losses: list[float] = []
    for step in range(args.steps):
        generator = torch.Generator(device=device).manual_seed(1000 + rank * 100 + step)
        inputs = torch.randn(2, 8, 64, device=device, generator=generator)
        targets = torch.randn(2, 8, 64, device=device, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss on rank {rank}: {loss}")
        losses.append(float(loss.detach()))

    if args.checkpoint_mode in {"save", "roundtrip"}:
        if args.checkpoint_dir is None:
            raise ValueError("--checkpoint-dir is required for checkpoint save")
        checkpoint_dir = args.checkpoint_dir.resolve()
        if rank == 0:
            checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        state = _checkpoint_state(model, optimizer)
        dcp.save(state_dict=state, checkpoint_id=str(checkpoint_dir))
        if args.checkpoint_mode == "roundtrip":
            expected_checksum = _local_parameter_checksum(model)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            _load_checkpoint(model, optimizer, checkpoint_dir)
            torch.testing.assert_close(
                _local_parameter_checksum(model),
                expected_checksum,
                rtol=0,
                atol=0,
            )

    dist.barrier()
    if rank == 0:
        print(
            "VFORGE_HSDP_SMOKE: PASS "
            f"world_size={world_size} replicate={config.dp_replicate_size} "
            f"shard={config.dp_shard_size} losses={losses}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
