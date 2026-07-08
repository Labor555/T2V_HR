#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"
source "${SCRIPT_DIR}/remote_common.sh"

REMOTE_BOOTSTRAP='
set -e
if [ -n "${REMOTE_PYTHON}" ] && [ -x "${REMOTE_PYTHON}" ]; then
  mkdir -p "${REMOTE_PROJECT_DIR}"
  cd "${REMOTE_PROJECT_DIR}"
  "${REMOTE_PYTHON}" -m pip install --no-deps -e .
  exit 0
fi
if [ -n "${REMOTE_CONDA_EXE}" ] && [ -x "${REMOTE_CONDA_EXE}" ]; then
  mkdir -p "${REMOTE_PROJECT_DIR}"
  ("${REMOTE_CONDA_EXE}" env list | awk "{print \$1}" | grep -qx "${REMOTE_CONDA_ENV}" || "${REMOTE_CONDA_EXE}" create -y -n "${REMOTE_CONDA_ENV}" python=3.10)
  cd "${REMOTE_PROJECT_DIR}"
  "${REMOTE_CONDA_EXE}" run -n "${REMOTE_CONDA_ENV}" pip install --no-deps -e .
  exit 0
fi
if [ -n "${REMOTE_CONDA_SH}" ] && [ -f "${REMOTE_CONDA_SH}" ]; then
  CONDA_SH="${REMOTE_CONDA_SH}"
else
  CONDA_SH=""
  for candidate in \
    "$HOME/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/anaconda3/etc/profile.d/conda.sh" \
    "/mnt/nfs/users/lbzhu/conda/etc/profile.d/conda.sh" \
    "/data/lbzhu/miniconda3/etc/profile.d/conda.sh" \
    "/data/lbzhu/anaconda3/etc/profile.d/conda.sh" \
    "$HOME/nfs/miniconda3/etc/profile.d/conda.sh" \
    "$HOME/nfs/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$candidate" ]; then
      CONDA_SH="$candidate"
      break
    fi
  done
fi
if [ -z "$CONDA_SH" ]; then
  echo "Could not find conda.sh; set REMOTE_CONDA_SH in remote/runtime.env" >&2
  exit 10
fi
mkdir -p "${REMOTE_PROJECT_DIR}"
source "$CONDA_SH"
(conda env list | awk "{print \$1}" | grep -qx "${REMOTE_CONDA_ENV}" || conda create -y -n "${REMOTE_CONDA_ENV}" python=3.10)
conda activate "${REMOTE_CONDA_ENV}"
cd "${REMOTE_PROJECT_DIR}"
pip install --no-deps -e .
'

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_PROJECT_DIR}' && \
  REMOTE_PROJECT_DIR='${REMOTE_PROJECT_DIR}' REMOTE_PYTHON='${REMOTE_PYTHON:-}' REMOTE_CONDA_SH='${REMOTE_CONDA_SH:-}' REMOTE_CONDA_EXE='${REMOTE_CONDA_EXE:-}' REMOTE_CONDA_ENV='${REMOTE_CONDA_ENV}' bash -c '${REMOTE_BOOTSTRAP}'"

echo "Bootstrapped ${REMOTE_CONDA_ENV} on ${REMOTE_HOST}:${REMOTE_PROJECT_DIR}"
exit 0
