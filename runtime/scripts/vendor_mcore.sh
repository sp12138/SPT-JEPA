#!/usr/bin/env bash
# Sync the vendored Megatron-Core tree from upstream Megatron-LM into vforge/mcore.
#
#   scripts/vendor_mcore.sh --verify         offline: tree matches its checksums
#   scripts/vendor_mcore.sh --check          online: tree matches upstream at tag
#   scripts/vendor_mcore.sh                  re-sync at the currently pinned tag
#   scripts/vendor_mcore.sh core_v0.19.0     upgrade to a new tag
#
# Upstream's `megatron.core` import path is rewritten to `vforge.mcore` so the
# tree lives inside our package and does not claim the global `megatron` name.
# The rewrite is mechanical and reproducible, which is what keeps upgrades cheap:
# --check reapplies it to a fresh upstream checkout and diffs, so any hand edit
# to the vendored tree shows up. See vforge/mcore/VENDOR.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM="https://github.com/NVIDIA/Megatron-LM.git"
DEST="${REPO_ROOT}/vforge/mcore"

# Subtrees dropped from the vendored copy. When updating this list, check that
# retained modules do not import the removed subtrees, including optional
# GPU-only branches. Several packages look equally unused and are not
# here because they are reachable in ways a quick grep misses -- `post_training`
# is imported from transformer/ under `if HAVE_TE:`, which does not execute on a
# laptop but does on a GPU box, and `models`/`inference` hold generic pieces
# (rope helpers, inference contexts) that transformer/ imports at module level.
#
# Applied to the staging tree, so --check compares against upstream-minus-this
# rather than reporting the absence as drift.
PRUNE=(
  export      # TensorRT-LLM export path
  resharding  # RLHF weight refit between train and inference layouts

  # Nothing references it once the T5 inference wrapper above is gone.
  datasets

  # The model zoo. What stays is the part that is not a model at all:
  # models/common holds the rope helpers transformer/attention.py imports at
  # module level, and models/backends.py the spec-provider protocol. hybrid
  # stays too -- attention.py -> inference.contexts -> dynamic_context imports
  # models.hybrid.hybrid_layer_allocation eagerly, a cycle upstream documents
  # and plans to break in 0.14.
  models/T5
  models/bert
  models/huggingface
  models/mamba
  models/mimo
  models/multimodal
  models/vision

  # Tokenizer implementations. inference_request.py needs the MegatronTokenizer
  # name for two deprecated type hints, which lives in the package root, so the
  # root plus base_tokenizer and megatron_tokenizer stay and the libraries go.
  tokenizers/text
  tokenizers/utils
  tokenizers/vision

  # Serving: engines, the HTTP server, samplers, request scheduling. The parts
  # of inference/ that the trainer does reach -- contexts, moe, quantization,
  # symmetric_memory, utils, communication -- are not here.
  inference/apis
  inference/async_stream.py
  inference/common_inference_params.py
  inference/communication_utils.py
  inference/data_parallel_inference_coordinator.py
  inference/engines
  inference/headers.py
  inference/inference_client.py
  inference/model_inference_wrappers
  inference/sampling
  inference/scheduler.py
  inference/text_generation_controllers
  inference/text_generation_server

  # Only modelopt/layers.py is reached, from transformer/ under `if HAVE_TE:`.
  # The model specs beside it are not, and the package __init__ files are empty
  # so nothing drags them in.
  post_training/modelopt/gpt
  post_training/modelopt/hybrid
  post_training/modelopt/mamba
)
VENDOR_DOC="${DEST}/VENDOR.md"
MANIFEST="${DEST}/CHECKSUMS.sha256"
WORKTREE="${REPO_ROOT}/.mcore-src"
PYTHON="${PYTHON:-python3}"

die() { echo "error: $*" >&2; exit 1; }

# Hashing lives in a standalone script so it can be tested against a throwaway
# tree, and in Python rather than sha256sum/shasum, whose names and output
# formats differ across the macOS and Linux boxes this has to run on.
manifest_tool() {
  "${PYTHON}" "${REPO_ROOT}/scripts/vendor_manifest.py" "$@"
}

# The pinned tag lives in VENDOR.md so there is exactly one place to read it
# from and no chance of the doc disagreeing with the script. The doc is written
# in Chinese but this row's key is deliberately kept as ASCII "Tag" -- translating
# it would silently break this lookup.
pinned_tag() {
  grep -E '^\| Tag \|' "${VENDOR_DOC}" | sed -E 's/.*`([^`]+)`.*/\1/'
}

# Rewrite `megatron.core` -> `vforge.mcore` in place, across imports and the
# string module paths that mcore resolves at runtime (ModuleSpec, the pickle
# allowlist in safe_globals). Only the dotted token is touched: bare `megatron`
# appears solely in docstrings, tokenizer library names and HF model ids, and
# `megatron/core` / `megatron-core` are URLs. Verified against core_v0.18.2.
rewrite_namespace() {
  local tree="$1"
  find "${tree}" -type f \( -name '*.py' -o -name '*.cpp' -o -name '*.cu' \
    -o -name '*.h' -o -name '*.txt' -o -name '*.md' \) -print0 \
    | xargs -0 -n 64 perl -pi -e 's/\bmegatron\.core\b/vforge.mcore/g'
}

CHECK_ONLY=0
TAG=""
case "${1:-}" in
  --verify)
    # Offline and fast, so CI can afford it on every MR. This is the gate that
    # enforces "reference this tree, never edit it": a hand edit changes a hash.
    # Regenerating the manifest to bless an edit is possible, but it shows up as
    # a CHECKSUMS.sha256 diff in review, which is exactly the signal we want.
    echo "==> verifying vforge/mcore against ${MANIFEST#"${REPO_ROOT}/"}"
    if manifest_tool verify "${DEST}" "${MANIFEST}"; then
      echo "==> clean: vendored tree is unmodified"
      exit 0
    fi
    echo >&2
    echo "vforge/mcore is vendored upstream source and must not be edited." >&2
    echo "Wrap or subclass it from vforge/ instead. If a patch is truly" >&2
    echo "unavoidable, add it to scripts/vendor_mcore.sh so every re-sync" >&2
    echo "reapplies it, then run the script to regenerate the manifest." >&2
    echo "See ${VENDOR_DOC#"${REPO_ROOT}/"}." >&2
    exit 1
    ;;
  --check) CHECK_ONLY=1 ;;
  "")      ;;
  -*)      die "unknown flag: $1" ;;
  *)       TAG="$1" ;;
esac
[[ -n "${TAG}" ]] || TAG="$(pinned_tag)"
[[ -n "${TAG}" ]] || die "could not read the pinned tag from ${VENDOR_DOC}"

echo "==> upstream ${UPSTREAM} @ ${TAG}"

# Reuse the cache only if it is genuinely a clone of upstream. A half-deleted
# .mcore-src still has a .git directory, and git would then walk up and resolve
# this repository as the remote -- a confusing failure a long way from here.
cached_upstream=""
if [[ -d "${WORKTREE}/.git" ]]; then
  cached_upstream="$(git -C "${WORKTREE}" config --local remote.origin.url 2>/dev/null || true)"
fi

if [[ "${cached_upstream}" == "${UPSTREAM}" ]]; then
  git -C "${WORKTREE}" fetch --tags --quiet origin
else
  rm -rf "${WORKTREE}"
  git clone --filter=blob:none --no-checkout --quiet "${UPSTREAM}" "${WORKTREE}"
fi

git -C "${WORKTREE}" rev-parse -q --verify "refs/tags/${TAG}" >/dev/null \
  || die "tag ${TAG} does not exist upstream"

# Materialise only the subtree we vendor. Cone mode also keeps root-level files,
# which is how we still get LICENSE without checking out the whole repository.
git -C "${WORKTREE}" sparse-checkout init --cone
git -C "${WORKTREE}" sparse-checkout set megatron/core
git -C "${WORKTREE}" checkout --quiet --force "${TAG}"

SHA="$(git -C "${WORKTREE}" rev-list -n1 "${TAG}")"
SRC="${WORKTREE}/megatron/core"
[[ -d "${SRC}" ]] || die "${TAG} has no megatron/core directory"

# Stage upstream in a scratch tree and rewrite there, so both --check and a real
# sync compare exactly the same artefact.
STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
rsync -a --exclude='__pycache__' --exclude='*.pyc' "${SRC}/" "${STAGE}/"
rewrite_namespace "${STAGE}"

# -e, not -d: entries may be single modules as well as packages. The check is
# what turns a stale list into an error at the next upgrade instead of a silent
# no-op that quietly re-vendors something we meant to drop.
for subtree in "${PRUNE[@]}"; do
  [[ -e "${STAGE}/${subtree}" ]] || die "PRUNE lists ${subtree}, which ${TAG} does not have"
  rm -rf "${STAGE:?}/${subtree}"
done

DIFF_EXCLUDES=(--exclude='__pycache__' --exclude='VENDOR.md' --exclude='LICENSE'
               --exclude='CHECKSUMS.sha256')

if (( CHECK_ONLY )); then
  # VENDOR.md, LICENSE and the manifest are ours / not part of the upstream subtree.
  if diff -r "${DIFF_EXCLUDES[@]}" "${STAGE}" "${DEST}" >/dev/null 2>&1; then
    echo "==> clean: vforge/mcore matches ${TAG} (${SHA:0:12}) after namespace rewrite"
    exit 0
  fi
  echo "==> DRIFT: vforge/mcore differs from ${TAG} (${SHA:0:12})" >&2
  diff -r "${DIFF_EXCLUDES[@]}" "${STAGE}" "${DEST}" | head -50 >&2
  echo >&2
  echo "Either the tree was edited locally (record it in ${VENDOR_DOC#"${REPO_ROOT}/"})" >&2
  echo "or re-sync with: scripts/vendor_mcore.sh ${TAG}" >&2
  exit 1
fi

echo "==> syncing into ${DEST#"${REPO_ROOT}/"}"
KEEP_DOC=""
[[ -f "${VENDOR_DOC}" ]] && KEEP_DOC="$(cat "${VENDOR_DOC}")"
rm -rf "${DEST}"
mkdir -p "${DEST}"
rsync -a "${STAGE}/" "${DEST}/"
cp "${WORKTREE}/LICENSE" "${DEST}/LICENSE"
[[ -n "${KEEP_DOC}" ]] && printf '%s\n' "${KEEP_DOC}" > "${VENDOR_DOC}"

echo -n "==> writing manifest: "
manifest_tool write "${DEST}" "${MANIFEST}"

FILES="$(find "${DEST}" -name '*.py' | wc -l | tr -d ' ')"
echo "==> done: ${FILES} python files at ${TAG} (${SHA:0:12})"
echo
echo "Now update the Tag / Commit / Vendor 日期 rows in ${VENDOR_DOC#"${REPO_ROOT}/"},"
echo "and re-verify that bare \`megatron\` is still never a module path at this tag."
