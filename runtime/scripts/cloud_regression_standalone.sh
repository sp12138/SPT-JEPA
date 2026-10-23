#!/usr/bin/env bash
# Consolidated cloud regression for standalone Hydra/tasks Phase 0+.
# Run on a pod with the project Python env; do not use as a local torch gate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="${PYTHON:-python3}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[standalone-regression] import smoke (no taichu)"
"${PY}" - <<'PY'
import sys
import importlib

assert "taichu" not in sys.modules
for name in (
    "vforge.tasks.train",
    "vforge.tasks.prepare",
    "vforge.tasks.eval",
    "vforge.tasks.convert_checkpoint",
    "vforge.tasks.wan.v21.train",
    "vforge.tasks.wan.v21.eval",
    "vforge.tasks.wan.v21.encode",
    "vforge.distributed.runtime",
    "vforge.ops",
):
    importlib.import_module(name)
assert "taichu" not in sys.modules
print("import_ok")
PY

echo "[standalone-regression] pytest tasks/hydra + data protocol"
"${PY}" -m pytest -q \
  tests/tasks/test_hydra_entries.py \
  tests/data/test_loader_protocol.py \
  tests/data/test_stateful_loader.py

echo "[standalone-regression] pytest checkpoint v2 / convert / eval / wan21"
"${PY}" -m pytest -q \
  tests/checkpoint/test_v1_v2_resume.py \
  tests/checkpoint/test_convert.py \
  tests/checkpoint/test_selective_resume.py \
  tests/evaluation/test_runner.py \
  tests/tasks/wan/v21/test_task_contracts.py \
  tests/data/test_cache.py \
  tests/data/test_video_io.py \
  tests/models/test_loading.py \
  tests/distributed/test_ulysses_ops.py \
  tests/distributed/test_wan_ulysses.py

echo "[standalone-regression] torchrun Tier-B subgroup/eval smoke"
torchrun --standalone --nproc_per_node=4 tests/distributed/ulysses_runtime_smoke.py

echo "[standalone-regression] torchrun native MMDiT FSDP2 + USP smoke"
torchrun --standalone --nproc_per_node=4 \
  tests/models/gen_engine/native_mmdit_distributed_smoke.py

echo "[standalone-regression] existing CPU-safe unit suite"
"${PY}" -m pytest -q \
  tests/training/test_contracts.py \
  tests/training/test_rng.py \
  tests/checkpoint/test_resume.py \
  tests/checkpoint/test_distributed.py \
  tests/ops/attention/test_dispatch_contract.py \
  tests/ops/attention/test_lazy_loading.py \
  tests/ops/attention/test_backends.py \
  tests/ops/attention/test_debug.py \
  tests/distributed/test_precision_modes.py \
  tests/distributed/test_fsdp2_patch.py \
  tests/training/test_reporting.py \
  tests/training/test_parity_helpers.py \
  tests/training/test_throughput.py \
  tests/training/test_diagnostics.py \
  tests/distributed/test_wan_ulysses.py \
  tests/training/test_dual_entry.py \
  tests/training/test_trainer.py \
  tests/checkpoint/test_checkpoint.py \
  tests/checkpointing/test_wrapper.py \
  tests/checkpointing/test_preserved_kernel.py

echo "[standalone-regression] done"
