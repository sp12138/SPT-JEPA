#!/usr/bin/env bash
# Deploy an already-built site from the CI host. The static server is managed separately.
set -euo pipefail
SITE="${1:?site directory required}"
test -f "${SITE}/index.html"
: "${DEVCLOUD_SSH_KEY:?deployment key required}"
: "${DEVCLOUD_KNOWN_HOSTS:?trusted SSH host keys required}"
HOST="${DEVCLOUD_HOST:-ieg-infra-vforge.devcloud.woa.com}"
PORT="${DEVCLOUD_PORT:-36000}"
REMOTE_USER="${DEVCLOUD_USER:-root}"
REMOTE_DIR="${DEVCLOUD_SITE_DIR:-/data/workspace/vforge/site}"
RSYNC_ARGS=(-a --delete-delay --delay-updates)
case "${2:-publish}" in
  publish) BASE_PATH=/; RSYNC_ARGS+=(--exclude=/debug/) ;;
  debug) BASE_PATH=/debug/ ;;
  *) echo 'deployment mode must be publish or debug' >&2; exit 1 ;;
esac
TARGET_DIR="${REMOTE_DIR}${BASE_PATH}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
printf '%s\n' "${DEVCLOUD_SSH_KEY}" > "${TMP}/key"
printf '%s\n' "${DEVCLOUD_KNOWN_HOSTS}" > "${TMP}/known_hosts"
chmod 600 "${TMP}/key" "${TMP}/known_hosts"
ssh-keygen -y -f "${TMP}/key" >/dev/null
SSH=(ssh -i "${TMP}/key" -p "${PORT}" -o BatchMode=yes -o IdentitiesOnly=yes
     -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=${TMP}/known_hosts")
printf -v RSYNC_RSH '%q ' "${SSH[@]}"
export RSYNC_RSH
"${SSH[@]}" "${REMOTE_USER}@${HOST}" "mkdir -p '${TARGET_DIR}'"
# The remote lock also serializes deployments started outside this pipeline.
rsync "${RSYNC_ARGS[@]}" \
  --rsync-path="flock -x -w 900 '${REMOTE_DIR}.deploy.lock' rsync" \
  "${SITE}/" "${REMOTE_USER}@${HOST}:${TARGET_DIR}"
"${SSH[@]}" "${REMOTE_USER}@${HOST}" "curl -fsS -o /dev/null 'http://127.0.0.1${BASE_PATH}'"
echo "[publish] deployed and HTTP check passed: https://${HOST}${BASE_PATH}"
