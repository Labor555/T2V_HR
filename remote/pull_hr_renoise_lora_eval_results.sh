#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/runtime.env"

REMOTE_HOST="${HR_RENOISE_REMOTE_HOST:-${REMOTE_HOST}}"
REMOTE_PROJECT_DIR="${HR_RENOISE_REMOTE_PROJECT_DIR:-${REMOTE_PROJECT_DIR}}"
REMOTE_EVAL_ROOT="${HR_RENOISE_EVAL_ROOT:-outputs/hr_renoise_lora_checkpoint_evals}"
LOCAL_EVAL_ROOT="${HR_RENOISE_LOCAL_EVAL_ROOT:-outputs/hr_renoise_lora_checkpoint_evals}"

mkdir -p "${REPO_DIR}/${LOCAL_EVAL_ROOT}"
if ssh "${REMOTE_HOST}" "test -d '${REMOTE_PROJECT_DIR}/${REMOTE_EVAL_ROOT}'"; then
  rsync -az --prune-empty-dirs \
    --include '*/' \
    --include 'metrics.json' \
    --include 'compare_lsrna_hrdenoise_target_540p.mp4' \
    --include 'hr_renoise_lora.mp4' \
    --include 'lsrna_coarse.mp4' \
    --include 'wan_vae_target.mp4' \
    --exclude '*' \
    "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/${REMOTE_EVAL_ROOT}/" \
    "${REPO_DIR}/${LOCAL_EVAL_ROOT}/" || true
fi

latest_metrics="$(find "${REPO_DIR}/${LOCAL_EVAL_ROOT}" -name metrics.json -type f | sort | tail -1 || true)"
latest_compare="$(find "${REPO_DIR}/${LOCAL_EVAL_ROOT}" -name compare_lsrna_hrdenoise_target_540p.mp4 -type f | sort | tail -1 || true)"
echo "local_eval_root=${REPO_DIR}/${LOCAL_EVAL_ROOT}"
echo "latest_metrics=${latest_metrics}"
echo "latest_compare=${latest_compare}"
if [[ -n "${latest_metrics}" ]]; then
  cat "${latest_metrics}"
fi
