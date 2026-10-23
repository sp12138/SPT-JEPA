"""B300 numerical, resume, recompute, and representative dual-bank EP gate."""

import argparse
import faulthandler
import json
import os
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from expert_parallel_models import MoE
from test_expert_checkpoint import run_checkpoint_case
from test_expert_parallel import run_numerical_cases

from vforge.distributed.expert_parallel import (
    ExpertBankSpec,
    ExpertParallelPlan,
)
from vforge.distributed.parallel_state import ParallelConfig, destroy_parallel_state
from vforge.distributed.parallelize import (
    FSDP2Config,
    MixedPrecisionConfig,
    ParallelizeConfig,
    parallelize_model,
)


class TargetMoE(torch.nn.Module):
    _no_split_modules = ["MoE"]

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [MoE(main=400, block=16, hidden=16, intermediate=32, top_k=20) for _ in range(39)]
        )

    def get_parallel_plan(self):
        return ExpertParallelPlan((ExpertBankSpec("layers.*.main", ("gate_up", "down")),))

    def forward(self, hidden):
        for layer in self.layers:
            hidden = hidden + layer(hidden) / len(self.layers)
        return hidden


def measure_target(rank, world, device):
    destroy_parallel_state()
    torch.manual_seed(42)
    model = parallelize_model(
        TargetMoE().to(device),
        ParallelizeConfig(
            parallel=ParallelConfig(dp_size=world, extra_parallel_sizes=(world // 2,)),
            fsdp2=FSDP2Config(mixed_precision=MixedPrecisionConfig(enable=True)),
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5, weight_decay=0.0, foreach=False, fused=False
    )
    counts = torch.zeros(39, 400, device=device, dtype=torch.long)

    def count_assignments(layer_index, bank, args):
        ids = args[1] + bank._vforge_expert_layout.start
        counts[layer_index].add_(torch.bincount(ids, minlength=400))

    handles = [
        layer.main.register_forward_pre_hook(
            lambda bank, args, index=index: count_assignments(index, bank, args)
        )
        for index, layer in enumerate(model.layers)
    ]
    torch.manual_seed(42 + rank)
    inputs = torch.randn(8, 16, device=device)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        optimizer.zero_grad()
        torch.cuda.synchronize()
        started = time.perf_counter()
        model(inputs).float().square().mean().backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    for handle in handles:
        handle.remove()
    gathered = [None] * world
    dist.all_gather_object(
        gathered,
        {
            "rank": rank,
            "step_seconds": times,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "received_expert_tokens_per_layer": counts.cpu().tolist(),
            # Forward dispatch + return, and their backward adjoints. Includes
            # self-destined payloads; excludes metadata and FSDP communication.
            "ep_activation_payload_bytes_per_step": 39 * 8 * 20 * 16 * 2 * 4,
        },
    )
    if rank == 0:
        print("VFORGE_EP_TARGET " + json.dumps(gathered), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--usp", type=int, default=1)
    args = parser.parse_args()
    assert torch.__version__.split("+")[0].startswith("2.13."), torch.__version__
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180), device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    log_path = args.log_dir
    log_path.mkdir(parents=True, exist_ok=True)
    trace = (log_path / f"rank-{rank}.log").open("w", buffering=1)
    faulthandler.dump_traceback_later(90, repeat=True, file=trace)

    def progress(message):
        print(f"{time.monotonic():.3f} rank={rank} {message}", file=trace, flush=True)

    try:
        for amp in (False, True):
            for recompute in (False, True):
                label = f"usp{args.usp}-{'bf16' if amp else 'fp32'}-recompute{int(recompute)}"
                if rank == 0:
                    print(f"VFORGE_EP_GATE world={world} ep={args.ep} {label}", flush=True)
                progress(label)
                run_numerical_cases(
                    rank,
                    world,
                    device=device,
                    ep=args.ep,
                    usp=args.usp,
                    dtype=torch.float32,
                    amp=amp,
                    recompute=recompute,
                    progress=progress,
                )
                path = log_path / label
                path.mkdir(parents=True, exist_ok=True)
                run_checkpoint_case(
                    rank,
                    world,
                    str(path),
                    device=device,
                    ep=args.ep,
                    usp=args.usp,
                    dtype=torch.float32,
                    amp=amp,
                    recompute=recompute,
                    progress=progress,
                )
        if args.usp == 1:
            measure_target(rank, world, device)
    except BaseException:
        # A failing rank must report before NCCL teardown waits for peers that
        # may already be entering the next collective. Torchrun then stops them.
        traceback.print_exc(file=trace)
        traceback.print_exc()
        trace.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        faulthandler.cancel_dump_traceback_later()
        trace.close()
        destroy_parallel_state()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
