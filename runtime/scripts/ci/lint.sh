#!/usr/bin/env bash
# Cheap CPU checks, runnable locally or in a KAP CPU worker.
#
#   ruff check / format --check   style and import order
#   compileall                    every module parses
#   vendored mcore                vforge/mcore is unmodified upstream source
#
# The import smoke needs torch. KAP workers have it; lightweight local lint
# environments may omit it and run only the static checks.
set -euo pipefail

VFORGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${VFORGE_ROOT}"

PYTHON="${PYTHON:-python3}"
status=0

# A lint run that skips every real check still exits 0, which would let a CI
# image with a missing toolchain look green. The pipeline sets CI_STRICT=1 so a
# skip becomes a failure; local runs leave it unset.
STRICT="${CI_STRICT:-0}"

step() {
  echo ""
  echo "=== $* ==="
}

skip() {
  if [ "${STRICT}" = "1" ]; then
    echo "FAIL (CI_STRICT=1): $*"
    status=1
  else
    echo "SKIP: $*"
  fi
}

# For absences that are by design rather than a broken image. CI_STRICT must not
# turn these red, or the gate fails on the very setup it was built for.
note_skip() {
  echo "SKIP (expected): $*"
}

# The repo is not ruff-clean yet, so a whole-tree gate would fail every MR on
# debt its author did not create. LINT_SCOPE=changed narrows ruff to the files
# this branch actually touches: pre-existing debt blocks nobody, but anything
# you edit you are expected to leave clean. LINT_SCOPE=all lints everything and
# is what a cleanup MR would use.
LINT_SCOPE="${LINT_SCOPE:-all}"
BASE_REF="${BASE_REF:-origin/master}"

changed_python_files() {
  local base
  # Compare against the merge base so commits landing on the target branch
  # after this one forked are not attributed to it.
  base="$(git merge-base "${BASE_REF}" HEAD 2>/dev/null)" || return 1
  git diff --name-only --diff-filter=ACMR "${base}" HEAD -- '*.py'
  # Uncommitted work, so a local run sees what CI will see once it is pushed.
  git diff --name-only --diff-filter=ACMR -- '*.py'
  git diff --name-only --diff-filter=ACMR --cached -- '*.py'
  git ls-files --others --exclude-standard -- '*.py'
}

# bash 3.2 (macOS) has no mapfile, so collect into an array the portable way.
RUFF_TARGETS=()
ruff_scope_note=""
if [ "${LINT_SCOPE}" = "changed" ]; then
  if [ -n "${LINT_FILES_FILE:-}" ]; then
    if [ ! -f "${LINT_FILES_FILE}" ]; then
      echo "FAIL: missing ${LINT_FILES_FILE}; prepare the bundle with LINT_BASE_REF" >&2
      exit 1
    fi
    while IFS= read -r -d '' f; do
      if [ ! -f "${f}" ]; then
        echo "FAIL: lint manifest names a missing file: ${f}" >&2
        exit 1
      fi
      RUFF_TARGETS+=("${f}")
    done < "${LINT_FILES_FILE}"
    ruff_scope_note="bundle manifest (${#RUFF_TARGETS[@]} file(s))"
  else
    if ! git merge-base "${BASE_REF}" HEAD >/dev/null 2>&1; then
      echo "FAIL: cannot resolve merge base against ${BASE_REF}" >&2
      echo "      fetch the target branch first, or set BASE_REF" >&2
      status=1
    fi
    # A delete or rename still shows up in the diff, so drop paths that are gone.
    while IFS= read -r f; do
      if [ -n "${f}" ] && [ -f "${f}" ]; then
        RUFF_TARGETS+=("${f}")
      fi
    done < <(changed_python_files | sort -u)
    ruff_scope_note="changed vs ${BASE_REF} (${#RUFF_TARGETS[@]} file(s))"
  fi
else
  RUFF_TARGETS=(".")
  ruff_scope_note="whole tree"
fi

run_ruff() {
  local label="$1"
  shift
  step "${label} [${ruff_scope_note}]"
  if ! command -v ruff >/dev/null 2>&1; then
    skip "ruff not installed (uv sync --only-group lint)"
    return
  fi
  if [ "${#RUFF_TARGETS[@]}" -eq 0 ]; then
    echo "no Python files changed; nothing to check"
    return
  fi
  ruff "$@" -- "${RUFF_TARGETS[@]}" || status=1
}

run_ruff "ruff check" check
run_ruff "ruff format --check" format --check

step "compileall"
# Compiling with an interpreter older than the project targets is worse than
# not compiling at all: it cannot accept syntax the project is allowed to use,
# so it would reject valid code. ruff parses with target-version = py312
# regardless of the interpreter running it, so syntax stays covered.
if "${PYTHON}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
  # Keep .pyc out of the source tree, and away from the macOS system cache that
  # a sandboxed dev run cannot write to.
  PYCACHE_DIR="$(mktemp -d)"
  trap 'rm -rf "${PYCACHE_DIR}"' EXIT
  # vforge/mcore is vendored upstream source (see vforge/mcore/VENDOR.md): 437
  # files we neither wrote nor may fix, so compiling them only adds ~150k lines
  # of parse time and reports we would have to ignore.
  PYTHONPYCACHEPREFIX="${PYCACHE_DIR}" "${PYTHON}" -m compileall -q \
    -x '(^|/)vforge/mcore/' vforge scripts || status=1
else
  note_skip "$("${PYTHON}" -V 2>&1) predates the 3.12 this project targets; ruff covers syntax"
fi

step "vendored mcore"
# vforge/mcore may be referenced but never edited: a local change is a conflict
# at every future upgrade, and this is the only cheap way to notice one. The
# check is offline (hashes against a committed manifest), so it costs about a
# second and does not drag an external clone into the lint image.
if [ -x scripts/vendor_mcore.sh ]; then
  scripts/vendor_mcore.sh --verify || status=1
else
  skip "scripts/vendor_mcore.sh missing or not executable"
fi

step "import smoke"
# Reuse the worker runtime for the import check; local lint-only environments
# can omit torch. The CPU suite separately requires the complete runtime.
if "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
  "${PYTHON}" scripts/smoke_import.py || status=1
else
  note_skip "torch absent in this environment; CPU suite requires it"
fi

echo ""
if [ "${status}" -eq 0 ]; then
  echo "[lint] PASS"
else
  echo "[lint] FAIL"
fi
exit "${status}"
