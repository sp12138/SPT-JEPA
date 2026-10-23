#!/usr/bin/env bash
# Install a checkout in the dependency image without fetching optional submodules.
# Dependency changes require rebuilding the image, not resolving them at startup.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
: "${UV_PROJECT_ENVIRONMENT:?Use a dependency image built from Dockerfile}"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
uv lock --check --offline
cmp uv.lock /opt/vforge-image/uv.lock
uv pip install --offline --no-deps --no-build-isolation -e .
uv pip check
