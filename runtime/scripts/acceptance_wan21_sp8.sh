#!/usr/bin/env bash
# Single-allocation Wan21 SP8 acceptance (Tier C). Fail fast on first error.
# Requires: 8 GPUs, project weights/cache paths via env, no local laptop use.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python3}"
NPROC="${NPROC_PER_NODE:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/acceptance_wan21_sp8}"
CACHE_DIR="${CACHE_DIR:?set CACHE_DIR to a prepared latent cache}"
PRETRAINED="${PRETRAINED_PATH:?set PRETRAINED_PATH}"
MANIFEST="${TRAIN_MANIFEST:?set TRAIN_MANIFEST matching the indexed cache}"

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[acceptance] 1/4 continuous offline train (3 steps, ckpt at 2)"
torchrun --standalone --nproc_per_node="${NPROC}" -m vforge.tasks.train \
  experiment=wan21_i2v_14b \
  latent_cache.mode=offline \
  data.cache_dir="${CACHE_DIR}" \
  data.train_manifest="${MANIFEST}" \
  model.pretrained_path="${PRETRAINED}" \
  runtime.max_steps=3 \
  runtime.seed=42 \
  runtime.output_dir="${OUTPUT_DIR}/cont" \
  parallel.usp_size=8 \
  checkpoint.enabled=true \
  checkpoint.every_n_steps=2 \
  checkpoint.state=full \
  checkpoint.format=dcp \
  checkpoint.dirpath="${OUTPUT_DIR}/cont/checkpoints" \
  resume.path=none

CKPT="${OUTPUT_DIR}/cont/checkpoints/00002.dcp"
test -d "${CKPT}"
EVAL_CHECKPOINT_PATH="${EVAL_CHECKPOINT_PATH:-${CKPT}}"
EVAL_CHECKPOINT_FORMAT="${EVAL_CHECKPOINT_FORMAT:-dcp}"

echo "[acceptance] 2/4 resume from step 2 to step 3"
torchrun --standalone --nproc_per_node="${NPROC}" -m vforge.tasks.train \
  experiment=wan21_i2v_14b \
  latent_cache.mode=offline \
  data.cache_dir="${CACHE_DIR}" \
  data.train_manifest="${MANIFEST}" \
  model.pretrained_path="${PRETRAINED}" \
  runtime.max_steps=3 \
  runtime.seed=42 \
  runtime.output_dir="${OUTPUT_DIR}/resume" \
  parallel.usp_size=8 \
  checkpoint.enabled=false \
  resume.path="${CKPT}" \
  resume.strict=true

echo "[acceptance] comparing continuous and resumed step-3 traces"
"${PY}" - <<PY
import json
from pathlib import Path

continuous = [
    json.loads(line)
    for line in Path("${OUTPUT_DIR}/cont/train_steps.jsonl").read_text().splitlines()
]
resumed = [
    json.loads(line)
    for line in Path("${OUTPUT_DIR}/resume/train_steps.jsonl").read_text().splitlines()
]
expected = next(item for item in continuous if item["step"] == 2)
actual = next(item for item in resumed if item["step"] == 2)
for key in (
    "loss",
    "loss_backward",
    "learning_rate_before_step",
    "learning_rate_after_step",
):
    assert actual[key] == expected[key], (key, actual[key], expected[key])
assert actual.get("step_metadata", {}) == expected.get("step_metadata", {})
print("resume_trace_parity_ok")
PY

echo "[acceptance] 3/4 DCP -> PT convert"
"${PY}" -m vforge.tasks.convert_checkpoint \
  experiment=wan21_i2v_14b \
  convert_checkpoint.input_path="${CKPT}" \
  convert_checkpoint.output_path="${OUTPUT_DIR}/00002.pt" \
  runtime.output_dir="${OUTPUT_DIR}/convert"
test -f "${OUTPUT_DIR}/00002.pt"
"${PY}" - <<PY
import hashlib
import json
from pathlib import Path
import torch

path = Path("${OUTPUT_DIR}/00002.pt")
payload = torch.load(path, map_location="cpu", weights_only=True)
assert isinstance(payload, dict) and payload, "converted PT has no weights"
assert not any(key.startswith("_inner.") for key in payload), "wrapper keys leaked"
hasher = hashlib.sha256()
with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        hasher.update(chunk)
digest = hasher.hexdigest()
sidecar = json.loads(Path(str(path) + ".json").read_text())
assert sidecar["sha256"] == digest
print("converted_pt_load_hash_ok")
PY

echo "[acceptance] 4/4 SP8 eval checkpoint=${EVAL_CHECKPOINT_PATH}"
EVAL_OVERRIDES=(
  experiment=wan21_i2v_14b
  evaluation.data_source=cache
  evaluation.manifest="${CACHE_DIR}/manifest.json"
  model.pretrained_path="${PRETRAINED}"
  evaluation.checkpoint_path="${EVAL_CHECKPOINT_PATH}"
  evaluation.checkpoint_format="${EVAL_CHECKPOINT_FORMAT}"
  evaluation.output_dir="${OUTPUT_DIR}/eval"
  evaluation.num_samples=1
  evaluation.skip_existing=false
  evaluation.num_inference_steps=20
  parallel.usp_size=8
  runtime.output_dir="${OUTPUT_DIR}/eval_run"
)
torchrun --standalone --nproc_per_node="${NPROC}" -m vforge.tasks.eval \
  "${EVAL_OVERRIDES[@]}"

echo "[acceptance] checking MP4 and metadata outputs"
"${PY}" - <<PY
import json
from pathlib import Path
cur = Path("${OUTPUT_DIR}/eval")
videos = sorted(cur.glob("*.mp4"))
assert len(videos) == 1, videos
assert videos[0].stat().st_size > 0, videos[0]
sidecar = Path(str(videos[0]) + ".json")
metadata = json.loads(sidecar.read_text())
assert metadata["sample_id"]
assert metadata["metadata"]["num_inference_steps"] > 0
print("eval_mp4_ok")
PY

echo "[acceptance] done"
