# syntax=docker/dockerfile:1
# B300 / Linux amd64 dependencies only. Users install their own vforge checkout.
ARG CUDA_IMAGE=nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04@sha256:c2621d98e7de80c2aec5eb8403b19c67454c8f5b0c929e8588fd3563c9b6558d
FROM ghcr.io/astral-sh/uv:0.12.10@sha256:2bb3ebca0a796a155094a27773d290c4b074572e6107f171d88d086682fd2500 AS uv
FROM ${CUDA_IMAGE} AS aws-b300
COPY --from=uv /uv /uvx /usr/local/bin/

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    VFORGE_IMAGE_KIND=dependencies \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential ca-certificates cmake curl ffmpeg git git-lfs \
        libglib2.0-0 libsm6 libxext6 libxrender1 ninja-build pkg-config \
        python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*
# Native dependencies do not depend on the Python lockfile.
ARG BUILD_EFA=true
ARG EFA_VERSION=1.48.0
ARG EFA_SHA256=f3a8aadc2b94288694a9b9375bcad84a82f0f843e0eefafda7b158b2200549cc
# aws-ofi-nccl v1.19.1, peeled commit (not a moving branch).
ARG OFI_NCCL_COMMIT=206c02c478c6d724af09c3cbca59c06863a0b9c0
COPY docker/install_efa.sh /tmp/install_efa.sh
RUN case "${BUILD_EFA}" in \
        true) bash /tmp/install_efa.sh ;; \
        false) ;; \
        *) echo "BUILD_EFA must be true or false" >&2; exit 1 ;; \
    esac \
    && rm /tmp/install_efa.sh
ENV PATH=/opt/amazon/efa/bin:$PATH \
    LD_LIBRARY_PATH=/opt/amazon/ofi-nccl/lib:/opt/amazon/efa/lib:$LD_LIBRARY_PATH
# KAP bootstraps peers over SSH; bake this in instead of letting pod startup apt install it.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends openssh-server openssh-client rsync \
    && rm -rf /var/lib/apt/lists/*

# No application source or submodule is part of the build context.
ARG UV_RUNTIME_GROUP=cu130
ARG PYTHON_VERSION=3.12.3
ENV VFORGE_UV_RUNTIME_GROUP=${UV_RUNTIME_GROUP}
WORKDIR /opt/vforge-image
COPY pyproject.toml uv.lock .python-version ./
RUN case "${UV_RUNTIME_GROUP}" in cu130|cu130-torch29) ;; \
        *) echo "Unsupported UV_RUNTIME_GROUP=${UV_RUNTIME_GROUP}" >&2; exit 1 ;; esac \
    && python3 -c 'import platform, sys; assert platform.python_version() == sys.argv[1], platform.python_version()' "${PYTHON_VERSION}" \
    && uv sync --locked --no-install-project \
        --group "${UV_RUNTIME_GROUP}" --group build --group docs \
    && uv cache clean
COPY docker/repair_decord.py /tmp/repair_decord.py
RUN python /tmp/repair_decord.py /opt/vforge-image/decord-wheel.json \
    && uv pip check && rm /tmp/repair_decord.py

ARG VFORGE_REVISION=unknown
# This identifies the dependency configuration, not installed application code.
ENV VFORGE_IMAGE_REVISION=${VFORGE_REVISION}
LABEL org.opencontainers.image.source="https://git.woa.com/ieg-infra/vforge" \
      org.opencontainers.image.revision="${VFORGE_REVISION}" \
      io.vforge.platform="aws-b300-amd64" \
      io.vforge.image-kind="dependencies" \
      io.vforge.uv-runtime-group="${UV_RUNTIME_GROUP}"
# Preserve the lock and installed versions with the image for audit and comparison.
COPY docker/verify_runtime.py /opt/vforge-image/verify_runtime.py
RUN uv --version > /opt/vforge-image/uv-version.txt \
    && uv pip freeze > /opt/vforge-image/requirements.txt \
    && uv pip list --format=json > /opt/vforge-image/packages.json \
    && dpkg-query -W > /opt/vforge-image/dpkg.txt \
    && printf '%s\n' "${VFORGE_REVISION}" > /opt/vforge-image/revision.txt \
    && python /opt/vforge-image/verify_runtime.py \
        --group "${UV_RUNTIME_GROUP}" --python-version "${PYTHON_VERSION}"
WORKDIR /workspace
