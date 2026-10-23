#!/usr/bin/env bash
# Container userspace only: the host owns the EFA kernel driver and limits.
set -euo pipefail
[ "$(dpkg --print-architecture)" = amd64 ]
apt-get update
apt-get install --yes --no-install-recommends autoconf automake libtool libhwloc-dev
curl --fail --location --retry 3 \
  "https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_VERSION}.tar.gz" \
  --output /tmp/efa.tar.gz
printf '%s  %s\n' "${EFA_SHA256}" /tmp/efa.tar.gz | sha256sum --check -
tar -xzf /tmp/efa.tar.gz -C /tmp
# Plain CUDA images do not contain NGC's RDMA packages. Disable both NGC
# heuristics so a GPU-less Docker builder installs rdma-core before libfabric.
# Install one plugin below; don't also install the bundle's older plugin.
(
  cd /tmp/aws-efa-installer
  ./efa_installer.sh --yes --skip-kmod --skip-limit-conf \
    --skip-mpi --skip-plugin --no-verify --disable-ngc --disable-build-ngc
)
# Keep PyTorch's NCCL. aws-ofi-nccl is a network plugin, not a replacement NCCL.
git init /tmp/aws-ofi-nccl
git -C /tmp/aws-ofi-nccl remote add origin https://github.com/aws/aws-ofi-nccl.git
git -C /tmp/aws-ofi-nccl fetch --depth 1 origin "${OFI_NCCL_COMMIT}"
git -C /tmp/aws-ofi-nccl checkout --detach FETCH_HEAD
[ "$(git -C /tmp/aws-ofi-nccl rev-parse HEAD)" = "${OFI_NCCL_COMMIT}" ]
cd /tmp/aws-ofi-nccl
./autogen.sh
./configure --prefix=/opt/amazon/ofi-nccl --with-libfabric=/opt/amazon/efa \
  --with-cuda=/usr/local/cuda --enable-platform-aws --disable-tests
make -j8
make install
ldd /opt/amazon/ofi-nccl/lib/libnccl-net.so > /tmp/ofi-ldd.txt
cat /tmp/ofi-ldd.txt
! grep -q 'not found' /tmp/ofi-ldd.txt
mkdir -p /opt/vforge-image
printf 'efa=%s\naws-ofi-nccl=%s\n' "${EFA_VERSION}" "${OFI_NCCL_COMMIT}" \
  > /opt/vforge-image/fabric.txt
cd /
rm -rf /tmp/efa.tar.gz /tmp/aws-efa-installer /tmp/aws-ofi-nccl /tmp/ofi-ldd.txt \
  /var/lib/apt/lists/*
