"""Multi-node RepWorld gate; Native owns model execution and checkpointing.

Run once per node under a KAP task. Assets, output and rendezvous are supplied
externally; no cluster, user path or credentials belong in this worker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def wait_for(path, timeout=600):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Waiting for {path}")
        time.sleep(2)


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f"{prefix}/{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from flatten(item, f"{prefix}/{index}")
    else:
        yield prefix, value


def equal_state(left, right):
    a, b = dict(flatten(left)), dict(flatten(right))
    if a.keys() != b.keys():
        raise AssertionError("State keys differ")
    tensors, elements = 0, 0
    for name, value in a.items():
        other = b[name]
        if isinstance(value, torch.Tensor):
            tensors += 1
            elements += value.numel()
            same = isinstance(other, torch.Tensor) and torch.equal(value, other)
        elif isinstance(value, np.ndarray):
            same = np.array_equal(value, other)
        else:
            same = value == other
        if not same:
            raise AssertionError(f"Checkpoint state differs: {name}")
    return dict(tensors=tensors, elements=elements, bitwise_equal=True)


def verify_resume(output, world):
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    before = output / "hsdp_latent_continuous/checkpoints/00004.dcp"
    after = output / "hsdp_latent_resumed/checkpoints/00004.dcp"
    temporary = output / "comparison"
    temporary.mkdir(exist_ok=True)
    results = {}
    for part in ("weights", "optimizer"):
        paths = []
        for label, checkpoint in (("continuous", before), ("resumed", after)):
            path = temporary / f"{label}-{part}.pt"
            dcp_to_torch_save(checkpoint / part, path)
            paths.append(path)
        states = [
            torch.load(path, map_location="cpu", weights_only=False, mmap=True) for path in paths
        ]
        results[part] = equal_state(*states)
        del states
        for path in paths:
            path.unlink()
    results["trainer"] = []
    for rank in range(world):
        relative = f"trainer/rank-{rank:05}.pt"
        a, b = [
            torch.load(path / relative, map_location="cpu", weights_only=False)
            for path in (before, after)
        ]
        results["trainer"].append(dict(rank=rank, **equal_state(a, b)))
    losses = []
    for name in ("hsdp_latent_continuous", "hsdp_latent_resumed"):
        record = json.loads((output / f"{name}-node0.json").read_text())
        losses.append([row["loss"] for row in record["metrics"] if row["step"] == 3])
    assert losses[0] == losses[1] and len(losses[0]) == 1, losses
    results.update(losses=losses, passed=True)
    write(output / "resume_comparison.json", results)
    return results


def verify_eval(output, samples):
    import av

    folder = output / "eval_hsdp_checkpoint/samples"
    videos = sorted(folder.glob("*.mp4"))
    assert len(videos) == samples, len(videos)
    rows = []
    for path in videos:
        with av.open(str(path)) as container:
            frames = list(container.decode(video=0))
            assert len(frames) == 48
            assert all((frame.width, frame.height) == (256, 256) for frame in frames)
        metadata = json.loads(path.with_suffix(".mp4.json").read_text())["metadata"]
        assert metadata["cfg_scale"] == 2 and metadata["total_states"] == 3
        rows.append(dict(case=path.stem, frames=48, metadata=metadata))
    assert len({row["case"] for row in rows}) == samples
    write(output / "eval_validation.json", dict(passed=True, samples=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    torch.set_num_threads(4)
    assert args.nodes >= 2 and args.gpus % 2 == 0
    assert torch.cuda.device_count() == args.gpus
    world = args.nodes * args.gpus
    dp = world // 2
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    configs = {
        "video": (
            "rep_world",
            "VideoRAE/vjepa21_3d.pth",
            "VideoRAE/solarwm_latent_stats_32clips.pt",
        ),
        "oned": (
            "rep_world_1d",
            "VideoRAE/vjepa_rae.pth",
            "VideoRAE/solarwm_1d_latent_stats_32clips.pt",
        ),
        "eupe": ("rep_world_eupe", "V-RAE/vrae/vrae_eupe.pt", None),
        "vjepa": ("rep_world_vjepa", "V-RAE/vrae/vrae_vjepa2.1.pt", None),
    }
    for _, checkpoint, statistics in configs.values():
        assert (args.assets / checkpoint).is_file(), checkpoint
        if statistics:
            assert (args.assets / statistics).is_file(), statistics
    examples = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    assert len(examples) >= world
    for row in examples:
        path = Path(row.get("rgb_path", row.get("video", "")))
        if not path.is_absolute():
            path = args.manifest.parent / path
        assert path.is_file(), str(path)
    environment = dict(
        node_rank=args.node_rank,
        hostname=socket.gethostname(),
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
        visible_gpus=args.gpus,
        nodes=args.nodes,
        world=world,
        source=json.loads((ROOT / ".ci/metadata.json").read_text()),
    )
    write(output / f"environment-node{args.node_rank}.json", environment)
    for rank in range(args.nodes):
        wait_for(output / f"environment-node{rank}.json", timeout=180)
    hosts = [
        json.loads((output / f"environment-node{rank}.json").read_text())["hostname"]
        for rank in range(args.nodes)
    ]
    assert len(set(hosts)) == args.nodes, hosts
    print("MULTINODE_ENV " + json.dumps(environment), flush=True)
    stage = 0
    results = []

    def launch(name, entry, extra, *, training=False):
        nonlocal stage
        port = args.master_port + stage
        stage += 1
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nnodes={args.nodes}",
            f"--node_rank={args.node_rank}",
            f"--nproc_per_node={args.gpus}",
            f"--master_addr={args.master_addr}",
            f"--master_port={port}",
            "--max_restarts=0",
            "--monitor_interval=1",
            *entry,
            *extra,
        ]
        log = output / "logs" / f"{name}-node{args.node_rank}.log"
        started = time.monotonic()
        print(f"MULTINODE_STAGE_START {name}", flush=True)
        with log.open("x") as handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT),
                    "OMP_NUM_THREADS": "2",
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=900,
            )
        metrics = []
        for line in log.read_text().splitlines():
            if line.startswith("VFORGE_TRAIN "):
                metrics.append(json.loads(line.removeprefix("VFORGE_TRAIN ")))
        record = dict(
            name=name,
            node_rank=args.node_rank,
            argv=command,
            exit_code=completed.returncode,
            seconds=time.monotonic() - started,
            log=str(log),
            metrics=metrics,
        )
        write(output / f"{name}-node{args.node_rank}.json", record)
        results.append(record)
        print(f"MULTINODE_STAGE_END {name} exit={completed.returncode}", flush=True)
        if completed.returncode:
            print(log.read_text()[-16000:], flush=True)
        assert completed.returncode == 0, name
        if training and args.node_rank == 0:
            assert metrics and all(math.isfinite(row["loss"]) for row in metrics), name
        for rank in range(args.nodes):
            marker = output / f"{name}-node{rank}.json"
            wait_for(marker, timeout=180)
            assert json.loads(marker.read_text())["exit_code"] == 0, str(marker)

    for name, usp, replicas in [
        ("oracle_cross_node_usp", world, 1),
        ("oracle_fsdp_usp2", 2, 1),
        ("oracle_hsdp_usp2", 2, args.nodes),
    ]:
        launch(
            name,
            [str(ROOT / "tests/models/rep_world/sequence_parallel_worker.py")],
            [
                "--usp",
                str(usp),
                "--replicate",
                str(replicas),
                "--min-hosts",
                str(args.nodes),
                "--output",
                str(output / f"{name}.json"),
            ],
        )
        wait_for(output / f"{name}.json", timeout=30)
        oracle = json.loads((output / f"{name}.json").read_text())
        assert oracle["passed"] and len(oracle["cases"]) == 20

    def common(codec):
        experiment, checkpoint, statistics = configs[codec]
        overrides = [
            f"experiment={experiment}",
            "attention.backend=sdpa",
            f"model.extras.codec.checkpoint={args.assets / checkpoint}",
        ]
        if statistics:
            overrides.append(f"model.extras.codec.statistics={args.assets / statistics}")
        return overrides

    def train(name, overrides, codec="video"):
        launch(
            name,
            ["-m", "vforge.tasks.train"],
            common(codec)
            + [
                f"data.train_manifest={args.manifest}",
                "data.batch_size=1",
                "data.shuffle=false",
                "model.extras.random_clip=false",
                "parallel.dp_mode=fsdp2",
                "parallel.mixed_precision.enabled=false",
                "checkpoint.enabled=false",
                "runtime.max_steps=3",
                "logging.wandb_enabled=false",
                f"runtime.output_dir={output / name}",
            ]
            + overrides,
            training=True,
        )

    hsdp = [
        "parallel.usp_size=2",
        f"parallel.dp_replicate_size={args.nodes}",
        f"parallel.dp_shard_size={args.gpus // 2}",
        "parallel.gradient_checkpointing=true",
    ]
    train("fsdp_rgb", [])
    train("cross_node_usp_rgb", [f"parallel.usp_size={world}"])
    train("fsdp_usp2_rgb", ["parallel.usp_size=2"])
    train("hsdp_usp2_rgb", hsdp)
    for codec in ("oned", "eupe", "vjepa"):
        train(f"fsdp_usp2_{codec}", ["parallel.usp_size=2"], codec=codec)
    resume_overrides = hsdp + [
        "model.extras.conditioner.input=latent",
        "runtime.gradient_accumulation_steps=2",
        "runtime.max_steps=4",
        "runtime.deterministic=true",
        "checkpoint.enabled=true",
        "checkpoint.async_save=false",
        "checkpoint.every_n_steps=3",
    ]
    train("hsdp_latent_continuous", resume_overrides)
    checkpoint = output / "hsdp_latent_continuous/checkpoints/00003.dcp"
    train(
        "hsdp_latent_resumed",
        resume_overrides
        + [
            f"resume.path={checkpoint}",
            "resume.strict=true",
        ],
    )
    if args.node_rank == 0:
        verify_resume(output, world)
    wait_for(output / "resume_comparison.json", timeout=300)
    launch(
        "eval_hsdp_checkpoint",
        ["-m", "vforge.tasks.eval"],
        common("video")
        + [
            "model.extras.conditioner.input=latent",
            "parallel.dp_mode=fsdp2",
            "parallel.usp_size=2",
            f"evaluation.checkpoint_path={checkpoint}",
            "evaluation.data_source=inference",
            f"evaluation.manifest={args.manifest}",
            f"evaluation.num_samples={dp}",
            "evaluation.num_inference_steps=5",
            "evaluation.guidance_scale=2",
            "model.extras.rollout.total_states=3",
            f"runtime.output_dir={output / 'eval_hsdp_checkpoint'}",
            "logging.wandb_enabled=false",
        ],
    )
    if args.node_rank == 0:
        verify_eval(output, dp)
        write(
            output / "result.json",
            dict(
                passed=True,
                nodes=args.nodes,
                gpus_per_node=args.gpus,
                world=world,
                hosts=hosts,
                stages=[{k: v for k, v in row.items() if k != "metrics"} for row in results],
            ),
        )
    wait_for(output / "result.json", timeout=120)
    print("REPWORLD_MULTINODE_GATE_PASS", flush=True)


if __name__ == "__main__":
    main()
