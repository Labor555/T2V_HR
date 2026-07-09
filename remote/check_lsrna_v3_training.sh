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
PY="${REMOTE_PYTHON:-python}"
CONFIG="configs/train_lsrna_720_4k_1000_v3_full.yaml"
OUT_DIR="$("${PY}" - <<'PY'
import yaml
cfg = yaml.safe_load(open("configs/train_lsrna_720_4k_1000_v3_full.yaml", "r", encoding="utf-8"))
print(cfg["output_dir"])
PY
)"
latest_pid_file="$(ls -t logs/lsrna_v2_train_*.pid 2>/dev/null | head -1 || true)"
latest_log=""
pid=""
supervisor_alive=0
if [[ -n "${latest_pid_file}" ]]; then
  pid="$(cat "${latest_pid_file}" 2>/dev/null || true)"
  latest_log="${latest_pid_file%.pid}.log"
  if [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1; then
    supervisor_alive=1
  fi
fi

torchrun_workers="$(ps -eo args= | awk '$0 ~ /torch\.distributed\.run .*train_lsr\.py --config configs\/train_lsrna_720_4k_1000_v3_full\.yaml/ {count++} END {print count+0}')"
train_workers="$(ps -eo args= | awk '$0 ~ /python .*scripts\/train_lsr\.py --config configs\/train_lsrna_720_4k_1000_v3_full\.yaml/ {count++} END {print count+0}')"
step="0"
max_steps="$("${PY}" - <<'PY'
import yaml
cfg = yaml.safe_load(open("configs/train_lsrna_720_4k_1000_v3_full.yaml", "r", encoding="utf-8"))
print(int(cfg["train"]["max_steps"]))
PY
)"
latest_ckpt="${OUT_DIR}/checkpoints/latest.pt"
if [[ -f "${latest_ckpt}" ]]; then
  step="$("${PY}" - <<PY
import torch
print(int(torch.load("${latest_ckpt}", map_location="cpu").get("step", 0)))
PY
)"
fi
latest_progress=""
if [[ -n "${latest_log}" && -f "${latest_log}" ]]; then
  latest_progress="$(tr '\r' '\n' < "${latest_log}" | grep -E 'train_lsr:|starting v2 LSRNA|Traceback|OutOfMemory|ERROR|Error' | tail -5 | sed 's/[[:cntrl:]]//g' || true)"
fi

echo "latest_pid_file=${latest_pid_file}"
echo "latest_log=${latest_log}"
echo "pid=${pid}"
echo "supervisor_alive=${supervisor_alive}"
echo "torchrun_workers=${torchrun_workers}"
echo "train_workers=${train_workers}"
echo "train_step=${step}"
echo "train_max_steps=${max_steps}"
echo "output_dir=${OUT_DIR}"
echo "gpu_summary_begin"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "gpu_summary_end"
echo "latest_progress_begin"
printf '%s\n' "${latest_progress}"
echo "latest_progress_end"
EOF
)"

printf '%s\n' "${STATUS}"

if [[ "${REPAIR}" != "1" ]]; then
  exit 0
fi

supervisor_alive="$(awk -F= '$1=="supervisor_alive"{print $2}' <<<"${STATUS}")"
torchrun_workers="$(awk -F= '$1=="torchrun_workers"{print $2}' <<<"${STATUS}")"
train_workers="$(awk -F= '$1=="train_workers"{print $2}' <<<"${STATUS}")"
step="$(awk -F= '$1=="train_step"{print $2}' <<<"${STATUS}")"
max_steps="$(awk -F= '$1=="train_max_steps"{print $2}' <<<"${STATUS}")"
pid="$(awk -F= '$1=="pid"{print $2}' <<<"${STATUS}")"

repair_reason=""
if (( step >= max_steps )); then
  repair_reason=""
elif (( torchrun_workers == 0 || train_workers == 0 )); then
  repair_reason="v3 full-latent training workers are not alive"
elif [[ "${supervisor_alive}" != "1" ]]; then
  repair_reason="v3 training supervisor is not alive"
fi

if [[ -n "${repair_reason}" ]]; then
  echo "repair_reason=${repair_reason}"
  if [[ -n "${pid}" ]]; then
    ssh "${REMOTE_HOST}" "PID='${pid}' bash -s" <<'EOF'
if [[ -n "${PID}" ]] && ps -p "${PID}" >/dev/null 2>&1; then
  pkill -TERM -P "${PID}" 2>/dev/null || true
  kill -TERM "${PID}" 2>/dev/null || true
  sleep 5
  pkill -KILL -P "${PID}" 2>/dev/null || true
  kill -KILL "${PID}" 2>/dev/null || true
fi
EOF
  fi
  ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT_DIR}' && pkill -TERM -f 'train_lsr.py --config configs/train_lsrna_720_4k_1000_v3_full.yaml' 2>/dev/null || true; pkill -TERM -f 'torch.distributed.run.*train_lsrna_720_4k_1000_v3_full.yaml' 2>/dev/null || true"
  LSRNA_V2_CONFIG="configs/train_lsrna_720_4k_1000_v3_full.yaml" bash "${SCRIPT_DIR}/launch_lsrna_v2_train.sh"
fi
