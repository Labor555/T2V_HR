#!/usr/bin/env bash
set -euo pipefail

REPAIR=0
if [[ "${1:-}" == "--repair" ]]; then
  REPAIR=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"

STATUS="$(ssh "${REMOTE_HOST}" "PROJECT='${REMOTE_PROJECT_DIR}' REMOTE_PYTHON='${REMOTE_PYTHON:-python}' bash -s" <<'EOF'
set -euo pipefail
cd "${PROJECT}"
PYTHON_BIN="${REMOTE_PYTHON:-python}"
latest_pid_file="$(ls -t logs/720_4k_full_*.pid 2>/dev/null | head -1 || true)"
latest_log=""
pid=""
parent_alive=0
if [[ -n "${latest_pid_file}" ]]; then
  pid="$(cat "${latest_pid_file}" 2>/dev/null || true)"
  latest_log="${latest_pid_file%.pid}.log"
  if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
    parent_alive=1
  fi
fi

hr_count="$(find cache/wan13b_latents_720_4k_1000/hr -type f -name '*.pt' 2>/dev/null | wc -l)"
lr_count="$(find cache/wan13b_latents_720_4k_1000/lr -type f -name '*.pt' 2>/dev/null | wc -l)"
cache_workers="$(ps -eo args= | awk '$0 ~ /^python scripts\/cache_wan_latents.py .*cache_wan_latents_720_4k_1000.yaml/ {count++} END {print count+0}')"
train_workers="$(ps -eo args= | awk '$0 ~ /^python scripts\/train_lsr.py --config configs\/train_lsr_720_4k_1000.yaml/ {count++} END {print count+0}')"
torchrun_workers="$(ps -eo args= | awk '$0 ~ /torch\.distributed\.run .*train_lsr_720_4k_1000.yaml/ {count++} END {print count+0}')"
expected_cache_workers="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path

limit = 1000
num_shards = 10
cache_dir = Path("cache/wan13b_latents_720_4k_1000")
expected = 0
for shard in range(num_shards):
    target = (limit + num_shards - 1 - shard) // num_shards
    path = cache_dir / f"index_shard_{shard}.jsonl"
    count = 0
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
    if count < target:
        expected += 1
print(expected)
PY
)"
step="0"
latest_ckpt="outputs/wan13b_lsr_720_4k_1000/checkpoints/latest.pt"
if [[ -f "${latest_ckpt}" ]]; then
  step="$("${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import torch
path = Path("outputs/wan13b_lsr_720_4k_1000/checkpoints/latest.pt")
print(int(torch.load(path, map_location="cpu").get("step", 0)))
PY
)"
fi

echo "latest_pid_file=${latest_pid_file}"
echo "latest_log=${latest_log}"
echo "pid=${pid}"
echo "parent_alive=${parent_alive}"
echo "hr_count=${hr_count}"
echo "lr_count=${lr_count}"
echo "cache_workers=${cache_workers}"
echo "expected_cache_workers=${expected_cache_workers}"
echo "train_workers=${train_workers}"
echo "torchrun_workers=${torchrun_workers}"
echo "train_step=${step}"
echo "gpu_summary_begin"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "gpu_summary_end"
if [[ -n "${latest_log}" && -f "${latest_log}" ]]; then
  echo "log_tail_begin"
  tail -20 "${latest_log}"
  echo "log_tail_end"
fi
EOF
)"

printf '%s\n' "${STATUS}"

if [[ "${REPAIR}" != "1" ]]; then
  exit 0
fi

parent_alive="$(awk -F= '$1=="parent_alive"{print $2}' <<<"${STATUS}")"
hr_count="$(awk -F= '$1=="hr_count"{print $2}' <<<"${STATUS}")"
cache_workers="$(awk -F= '$1=="cache_workers"{print $2}' <<<"${STATUS}")"
expected_cache_workers="$(awk -F= '$1=="expected_cache_workers"{print $2}' <<<"${STATUS}")"
train_workers="$(awk -F= '$1=="train_workers"{print $2}' <<<"${STATUS}")"
torchrun_workers="$(awk -F= '$1=="torchrun_workers"{print $2}' <<<"${STATUS}")"
pid="$(awk -F= '$1=="pid"{print $2}' <<<"${STATUS}")"

repair_reason=""
if [[ "${parent_alive}" != "1" ]]; then
  repair_reason="parent job is not alive"
elif (( hr_count < 1000 && expected_cache_workers > 0 && cache_workers == 0 )); then
  repair_reason="cache is incomplete and no cache workers are alive"
elif (( hr_count >= 1000 && train_workers == 0 && torchrun_workers == 0 )); then
  repair_reason="cache appears complete but training is not alive"
fi

state_file="${SCRIPT_DIR}/../logs/monitor_mismatch_count.local"
if [[ -z "${repair_reason}" && "${parent_alive}" == "1" && "${hr_count}" -lt 1000 && "${cache_workers}" -lt "${expected_cache_workers}" ]]; then
  mkdir -p "$(dirname "${state_file}")"
  count=0
  [[ -f "${state_file}" ]] && count="$(cat "${state_file}")"
  count=$((count + 1))
  echo "${count}" > "${state_file}"
  if (( count >= 2 )); then
    repair_reason="cache worker count stayed below expected ${expected_cache_workers} for ${count} checks"
  else
    echo "repair_deferred=cache worker count ${cache_workers}/${expected_cache_workers}; will repair if repeated"
  fi
else
  rm -f "${state_file}"
fi

if [[ -n "${repair_reason}" ]]; then
  echo "repair_reason=${repair_reason}"
  if [[ -n "${pid}" ]]; then
    ssh "${REMOTE_HOST}" "PID='${pid}' bash -s" <<'EOF'
if [[ -n "${PID}" ]] && ps -p "${PID}" >/dev/null 2>&1; then
  pkill -TERM -P "${PID}" 2>/dev/null || true
  kill -TERM "${PID}" 2>/dev/null || true
  sleep 3
  pkill -KILL -P "${PID}" 2>/dev/null || true
  kill -KILL "${PID}" 2>/dev/null || true
fi
EOF
  fi
  bash "${SCRIPT_DIR}/launch_720_4k_full.sh"
fi
