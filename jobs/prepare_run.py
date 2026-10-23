"""Freeze a committed SPT revision and prepare an independent EFS run."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("run_name")
parser.add_argument("--steps", type=int, default=50000)
parser.add_argument("--resume", default="none")
parser.add_argument("--checkpoint-every", type=int, default=2500)
parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("SPT_DATA_ROOT", "/efs/prescottsun/SPT")))
parser.add_argument("--train-manifest", type=Path)
parser.add_argument("--val-manifest", type=Path)
args = parser.parse_args()
if Path(args.run_name).name != args.run_name or args.steps < 1:
    raise SystemExit("run_name must be one path component and steps must be positive")
repo = Path(__file__).resolve().parents[1]
if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
    raise SystemExit("Commit SPT source before freezing a run")
data_root = args.data_root.resolve()
train_manifest = (args.train_manifest or data_root / "data/train_windows_820343.jsonl").resolve()
val_manifest = (args.val_manifest or data_root / "data/test_windows_2048.jsonl").resolve()
for manifest in [train_manifest, val_manifest]:
    if not manifest.is_file():
        raise SystemExit("Missing data manifest: " + str(manifest))
root = data_root / "runs" / args.run_name
root.mkdir(parents=True, exist_ok=False)
archive = root / "source.tar"
with archive.open("wb") as handle:
    subprocess.run(["git", "-C", str(repo), "archive", "HEAD"], stdout=handle, check=True)
source = root / "source"
source.mkdir()
with tarfile.open(archive) as handle:
    handle.extractall(source, filter="data")
archive.unlink()
jobs = source / "jobs"
runtime = source / "runtime"
cfg = OmegaConf.load(jobs / "configs/pretrain_50k.yaml")
cfg.runtime.max_steps = args.steps
cfg.runtime.output_dir = str(root / "output")
cfg.checkpoint.dirpath = str(root / "output/checkpoints")
cfg.checkpoint.every_n_steps = args.checkpoint_every
cfg.resume.path = args.resume
cfg.data.train_manifest = str(train_manifest)
cfg.data.val_manifest = str(val_manifest)
OmegaConf.save(cfg, jobs / "configs/pretrain_50k.yaml")
commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
plan = dict(root=str(root), runtime=str(runtime), jobs=str(jobs), max_steps=args.steps,
            pretrain_config=str(jobs / "configs/pretrain_50k.yaml"),
            pretrain_output=str(root / "output"), source_commit=commit,
            python=sys.executable, data_root=str(data_root), cache_root=str(data_root / "cache"),
            initialized_from="rgb16_scale_budget_20261010")
(root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
(jobs / "runtime_hashes.json").write_bytes(
    (source / "provenance/upstream_runtime_hashes.json").read_bytes())
names = ["plan.json", "source/jobs/worker.py", "source/jobs/train_entry.py",
         "source/jobs/sample_entry.py", "source/jobs/configs/pretrain_50k.yaml"]
hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
(jobs / "job_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
packages = sorted((dist.metadata.get("Name", ""), dist.version)
                  for dist in importlib.metadata.distributions())
(root / "environment.json").write_text(json.dumps(
    dict(python=sys.version, executable=sys.executable, packages=packages), indent=2) + "\n")
(root / "workers/pretrain").mkdir(parents=True)
print(json.dumps(plan, indent=2))
