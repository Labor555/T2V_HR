#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <remote command...>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"
source "${SCRIPT_DIR}/remote_common.sh"

REMOTE_COMMAND="$*"
ENV_PREFIX="$(remote_env_prefix)"
REMOTE_COMMAND_B64="$(printf '%s' "${REMOTE_COMMAND}" | base64 | tr -d '\n')"

ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT_DIR}' && \
  printf '%s' '${REMOTE_COMMAND_B64}' | base64 -d | ${ENV_PREFIX} bash"
