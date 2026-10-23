"""Freeze repository-owned RGB16 FVD code and configure external data assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import numpy as np
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("evaluation_name")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True, help="Fixed 2048-case JSONL manifest")
parser.add_argument("--bank", type=Path, required=True, help="Frozen RGB/metadata/real-feature bank")
parser.add_argument("--weights", type=Path, required=True, help="Pinned I3D TorchScript weights")
parser.add_argument("--config", type=Path, help="Training config; defaults to repository RGB16 baseline")
parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("SPT_DATA_ROOT", "/efs/prescottsun/SPT")))
parser.add_argument("--allowed-host", default="dev8")
args = parser.parse_args()
if Path(args.evaluation_name).name != args.evaluation_name:
    raise SystemExit("evaluation_name must be one path component")
repo = Path(__file__).resolve().parents[1]
if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
    raise SystemExit("Commit SPT source before freezing an evaluation")
checkpoint = args.checkpoint.resolve()
manifest = args.manifest.resolve()
bank = args.bank.resolve()
weights = args.weights.resolve()
for path in [manifest, weights, checkpoint / "manifest.json"]:
    if not path.is_file():
        raise SystemExit("Missing asset: " + str(path))
rows = [json.loads(line) for line in manifest.read_text().splitlines()]
if len(rows) != 2048 or len({row["sample_id"] for row in rows}) != 2048:
    raise SystemExit("The reference protocol requires exactly 2048 unique cases")
sys.path.insert(0, str(repo / "runtime"))
from vforge.training.checkpoint import read_manifest

saved = read_manifest(str(checkpoint))
assert saved.state == "full" and all(saved.components_present[k] for k in ["weights", "optimizer", "trainer"])
assert int(checkpoint.name.split(".")[0]) == saved.completed_steps
detector_source = repo / "jobs/fvd_reference/detector/fvd.py"
sys.path.insert(0, str(detector_source.parent))
from fvd import WEIGHTS_SHA256, digest_file

assert digest_file(weights) == WEIGHTS_SHA256
for row in rows:
    sid = row["sample_id"]
    if not sid.startswith("fvd_") or Path(sid).name != sid:
        raise ValueError("Invalid bank case identity: " + sid)
    meta = json.loads((bank / (sid + ".json")).read_text())
    assert meta["detector"]["implementation_sha256"] == digest_file(detector_source)
    assert meta["detector"]["weights_sha256"] == WEIGHTS_SHA256
    feature = np.load(bank / (sid + ".feature.npy"), allow_pickle=False)
    assert feature.shape == (400,) and np.isfinite(feature).all()
    assert (bank / (sid + ".npy")).is_file()
data_root = args.data_root.resolve()
root = data_root / "evaluations" / args.evaluation_name
root.mkdir(parents=True, exist_ok=False)
archive = root / "source.tar"
with archive.open("wb") as handle:
    subprocess.run(["git", "-C", str(repo), "archive", "HEAD"], stdout=handle, check=True)
source = root / "source"
source.mkdir()
with tarfile.open(archive) as handle:
    handle.extractall(source, filter="data")
archive.unlink()
jobs = root / "jobs"
shutil.copytree(source / "jobs/fvd_reference", jobs)
shutil.copyfile(source / "jobs/fvd_worker.py", jobs / "fvd_worker.py")
(root / "data").mkdir()
(root / "fvd").mkdir()
shutil.copyfile(manifest, root / "data/eval_2048.jsonl")
shutil.copyfile(source / "provenance/fvd_reference/protocol.json", root / "fvd/protocol.json")
(root / "fvd/bank").symlink_to(bank, target_is_directory=True)
runtime = source / "runtime"
runtime_hashes = json.loads((source / "provenance/upstream_runtime_hashes.json").read_text())
for name, expected in runtime_hashes.items():
    assert digest_file(runtime / name) == expected, name
(jobs / "source_hashes.json").write_text(json.dumps(runtime_hashes, indent=2) + "\n")
name = args.evaluation_name + "__ema"
output = root / "fvd/results" / name / f"{saved.completed_steps:05d}"
state = dict(jobs=str(jobs), report=str(root), runtime_snapshot=str(runtime),
             source_digest=hashlib.sha256(json.dumps(runtime_hashes, sort_keys=True).encode()).hexdigest(),
             sampling_batch_sizes={name: 8}, eval_entrypoint="eval_entry_batch_frozen.py",
             profile_name="rgb16-128frames", gpu="H200", detector_weights=str(weights),
             cache_root=str(data_root / "cache"))
(jobs / "state.json").write_text(json.dumps(state, indent=2) + "\n")
cfg = OmegaConf.load(args.config or source / "jobs/configs/pretrain_50k.yaml")
cfg.runtime.output_dir = str(output)
cfg.runtime.weights_strict = True
cfg.checkpoint.enabled = False
cfg.checkpoint.dirpath = str(checkpoint.parent)
cfg.resume.path = "none"
cfg.model.extras.conditioner.context_mask_ratio = 0.0
cfg.model.extras.flow.distant_target_prob = 0.0
cfg.model.extras.ema.evaluation_weights = "ema"
cfg.data.val_manifest = str(root / "data/eval_2048.jsonl")
cfg.evaluation.output_dir = str(output)
config = jobs / "evaluation_configs" / (name + ".yaml")
config.parent.mkdir()
OmegaConf.save(cfg, config)
identity = dict(name=name, config=str(config), output=str(checkpoint.parent.parent),
                training_name=args.evaluation_name, weights="ema", sampling_batch_size=8)
(jobs / "eval_runs.json").write_text(json.dumps([identity], indent=2) + "\n")
sys.path.insert(0, str(jobs))
from fvd_coordination import protocol_hash

plan = dict(name=name, step=saved.completed_steps, checkpoint=str(checkpoint), output=str(output),
            runtime=str(runtime), cache_root=state["cache_root"], cases=2048, phase="checkpoint_evaluation",
            sampling_batch_size=8, allowed_host=args.allowed_host, protocol_sha256=protocol_hash(),
            historical_reference_protocol_sha256="649a3e06e6028bae012db80c53f27594e5c64f3ee6966c336f9362e2aff471dd",
            source_commit=subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
            source_hashes={str(path.relative_to(root)): digest_file(path) for path in jobs.rglob("*") if path.is_file() and path.suffix != ".pyc"},
            manifest_sha256=digest_file(root / "data/eval_2048.jsonl"), detector_weights_sha256=WEIGHTS_SHA256,
            portability_note="Source paths changed; actual evaluator/runtime hashes define a new protocol identity. Sampling and detector math are preserved.")
(root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
(root / "bank_audit.json").write_text(json.dumps(dict(cases=2048, complete=True, bank=str(bank), real_features_valid=True), indent=2) + "\n")
print(json.dumps(plan, indent=2))
