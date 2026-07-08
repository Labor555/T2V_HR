#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_PROJECT_DIR}' && \
  source '${REMOTE_CONDA_SH}' && \
  (conda env list | awk '{print \$1}' | grep -qx '${REMOTE_CONDA_ENV}' || conda create -y -n '${REMOTE_CONDA_ENV}' python=3.10) && \
  conda activate '${REMOTE_CONDA_ENV}' && \
  cd '${REMOTE_PROJECT_DIR}' && \
  pip install -e ."

echo "Bootstrapped ${REMOTE_CONDA_ENV} on ${REMOTE_HOST}:${REMOTE_PROJECT_DIR}"

