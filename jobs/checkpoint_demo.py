"""Generate a finite SPT EMA demo after its full checkpoint is committed."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("run_root", type=Path)
parser.add_argument("step", type=int)
parser.add_argument("output", type=Path)
parser.add_argument("--timeout", type=int, default=1800)
args = parser.parse_args()
plan = json.loads((args.run_root / "plan.json").read_text())
cache = Path(plan.get("cache_root", args.run_root.parents[1] / "cache"))
runtime = args.run_root / "source/runtime"
jobs = args.run_root / "source/jobs"
checkpoint = args.run_root / "output/checkpoints" / f"{args.step:05d}.dcp"
args.output.mkdir(parents=True, exist_ok=True)
lock = (args.output / "demo.lock").open("a+")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
sys.path.insert(0, str(runtime))
from vforge.training.checkpoint import read_manifest


def status(state, **values):
    record = dict(status=state, host=socket.gethostname(), pid=os.getpid(),
                  time=time.time(), checkpoint=str(checkpoint), step=args.step, **values)
    temp = args.output / "demo_status.json.tmp"
    temp.write_text(json.dumps(record, indent=2) + "\n")
    temp.replace(args.output / "demo_status.json")
    print(json.dumps(record), flush=True)


try:
    deadline = time.monotonic() + args.timeout
    while not (checkpoint / "manifest.json").is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("No committed checkpoint before timeout")
        status("waiting_for_checkpoint")
        time.sleep(15)
    manifest = read_manifest(str(checkpoint))
    assert manifest.completed_steps == args.step and manifest.state == "full"
    assert all(manifest.components_present[k] for k in ["weights", "optimizer", "trainer"])
    expected = json.loads((jobs / "runtime_hashes.json").read_text())
    for name, digest in expected.items():
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == digest, name
    busy = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                    "--format=csv,noheader"], text=True).strip()
    if busy:
        raise RuntimeError("Demo node GPUs are occupied: " + busy)
    env = dict(os.environ, PYTHONPATH=str(runtime), PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               HYDRA_FULL_ERROR="1", FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED="1",
               FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=str(cache / "scaling_fa4_h200/fa4"),
               CUTE_DSL_CACHE_DIR=str(cache / "scaling_fa4_h200/cute"),
               TRITON_CACHE_DIR=str(cache / "triton"),
               TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
               CUDA_CACHE_PATH=str(cache / "cuda"),
               XDG_CACHE_HOME=str(cache))
    command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
               "--nproc-per-node=8", "--max-restarts=0", str(jobs / "sample_entry.py"),
               str(jobs / "configs/pretrain_50k.yaml"), str(checkpoint), str(args.output)]
    status("sampling", command=command)
    with (args.output / "sample.log").open("a") as handle:
        subprocess.run(command, cwd=runtime, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True, timeout=900)
    videos = sorted(args.output.glob("*.mp4"))
    assert len(videos) == 8, len(videos)
    for path in videos:
        sidecar = json.loads(Path(str(path) + ".json").read_text())
        assert sidecar["metadata"]["evaluation_weights"] == "ema"
        assert sidecar["metadata"]["ema_updates"] == args.step
    status("completed", samples=len(videos))
except BaseException as exc:
    status("failed", error=repr(exc))
    raise
