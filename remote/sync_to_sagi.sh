#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/runtime.env"

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_PROJECT_DIR}'"
rsync -az --delete \
  --exclude .git \
  --exclude data \
  --exclude outputs \
  --exclude checkpoints \
  --exclude remote/runtime.env \
  "${REPO_DIR}/" "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/"

echo "Synced ${REPO_DIR} -> ${REMOTE_HOST}:${REMOTE_PROJECT_DIR}"

