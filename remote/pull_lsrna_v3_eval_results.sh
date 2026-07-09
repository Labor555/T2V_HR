#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"

SAMPLE="${LSRNA_V3_EVAL_SAMPLE:-00000004}"
REMOTE_EVAL_ROOT="${LSRNA_EVAL_ROOT:-outputs/wan13b_lsrna_v3_checkpoint_evals}"
LOCAL_EVAL_ROOT="${LSRNA_LOCAL_EVAL_ROOT:-${LSRNA_V3_LOCAL_EVAL_ROOT:-outputs/wan13b_lsrna_v3_checkpoint_evals}}"

mkdir -p "${LOCAL_EVAL_ROOT}"

remote_files="$(
  ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT_DIR}' && find '${REMOTE_EVAL_ROOT}' -maxdepth 3 -type f \\( -name metrics.json -o -name compare_lsrna_vs_wan_target_540p.mp4 \\) -printf '%P\\n' | sort" || true
)"

if [[ -z "${remote_files}" ]]; then
  echo "pull_status=no_remote_eval_results"
  exit 0
fi

copied=0
while IFS= read -r rel; do
  [[ -z "${rel}" ]] && continue
  local_path="${LOCAL_EVAL_ROOT}/${rel}"
  mkdir -p "$(dirname "${local_path}")"
  scp -p -q "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/${REMOTE_EVAL_ROOT}/${rel}" "${local_path}"
  copied=$((copied + 1))
done <<< "${remote_files}"

echo "pull_status=ok"
echo "copied_files=${copied}"

latest_video="$(find "${LOCAL_EVAL_ROOT}" -path "*/${SAMPLE}/compare_lsrna_vs_wan_target_540p.mp4" -type f | sort | tail -1 || true)"
latest_metrics="$(find "${LOCAL_EVAL_ROOT}" -path "*/${SAMPLE}/metrics.json" -type f | sort | tail -1 || true)"

if [[ -n "${latest_video}" ]]; then
  latest_video_abs="$(cd "$(dirname "${latest_video}")" && pwd)/$(basename "${latest_video}")"
  echo "latest_local_video=${latest_video_abs}"
fi
if [[ -n "${latest_metrics}" ]]; then
  latest_metrics_abs="$(cd "$(dirname "${latest_metrics}")" && pwd)/$(basename "${latest_metrics}")"
  echo "latest_local_metrics=${latest_metrics_abs}"
fi
