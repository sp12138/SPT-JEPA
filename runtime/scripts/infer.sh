#!/usr/bin/env bash
# Single-sample inference smoke built on the standalone eval contract.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29503}"

exec torchrun \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  -m vforge.tasks.eval \
  evaluation.num_samples=1 \
  evaluation.sample_index="${SAMPLE_INDEX:-0}" \
  "$@"
