#!/usr/bin/env bash
# Grid/normal ablation evaluation built on the standalone eval contract.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GRIDNORMAL_VARIANTS="${GRIDNORMAL_VARIANTS:-}"
has_experiment=0
for arg in "$@"; do
  if [[ "${arg}" == experiment=* ]]; then
    has_experiment=1
    break
  fi
done
if [[ "${has_experiment}" -eq 0 ]]; then
  echo "usage: $0 experiment=<experiment-config> [hydra overrides ...]" >&2
  exit 2
fi

args=(
  evaluation.gridnormal.enabled=true
)

if [[ -n "${GRIDNORMAL_VARIANTS}" ]]; then
  args+=("evaluation.gridnormal.variants=${GRIDNORMAL_VARIANTS}")
fi

exec "${ROOT}/scripts/eval.sh" "${args[@]}" "$@"
