#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config> [extra train_lsr.py args...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"

CONFIG="$1"
shift

ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT_DIR}' && \
  source '${REMOTE_CONDA_SH}' && conda activate '${REMOTE_CONDA_ENV}' && \
  pip install -e . && \
  python scripts/train_lsr.py --config '${CONFIG}' $*"

