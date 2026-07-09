#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"
source "${SCRIPT_DIR}/remote_common.sh"

ENV_PREFIX="$(remote_env_prefix)"
TRAIN_GPUS="${LSRNA_V2_GPUS:-${REMOTE_CUDA_VISIBLE_DEVICES:-0,4,7}}"
MEM_THRESHOLD_MB="${LSRNA_V2_WAIT_MEM_MB:-3000}"
CONFIG="${LSRNA_V2_CONFIG:-configs/train_lsrna_720_4k_1000_v2.yaml}"
RUN_NAME="lsrna_v2_train_$(date +%Y%m%d_%H%M%S)"
NUM_TRAIN_GPUS="$(awk -F',' '{print NF}' <<<"${TRAIN_GPUS}")"

ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT_DIR}' && ${ENV_PREFIX} bash -s" <<EOF
set -euo pipefail
mkdir -p logs
LOG="logs/${RUN_NAME}.log"
PID_FILE="logs/${RUN_NAME}.pid"

nohup bash -c '
set -euo pipefail
cd "${REMOTE_PROJECT_DIR}"
TRAIN_GPUS="${TRAIN_GPUS}"
CONFIG="${CONFIG}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS}"
MEM_THRESHOLD_MB="${MEM_THRESHOLD_MB}"

gpu_mem_used() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
}

gpus_ready() {
  IFS="," read -ra GPU_LIST <<< "\${TRAIN_GPUS}"
  local blocked=0
  local summary
  summary="\$(gpu_mem_used)"
  for gpu in "\${GPU_LIST[@]}"; do
    used="\$(awk -F, -v g="\${gpu}" '"'"'\$1+0 == g+0 {gsub(/ /, "", \$2); print \$2}'"'"' <<<"\${summary}")"
    used="\${used:-999999}"
    if (( used > MEM_THRESHOLD_MB )); then
      blocked=1
    fi
  done
  return "\${blocked}"
}

echo "[\$(date)] waiting for train_gpus=\${TRAIN_GPUS} mem_threshold_mb=\${MEM_THRESHOLD_MB}"
while ! gpus_ready; do
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  sleep 300
done

echo "[\$(date)] starting v2 LSRNA training config=\${CONFIG} gpus=\${TRAIN_GPUS}"
CUDA_VISIBLE_DEVICES="\${TRAIN_GPUS}" python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="\${NUM_TRAIN_GPUS}" \
  scripts/train_lsr.py --config "\${CONFIG}" --no-resume

echo "[\$(date)] v2 training exited; holding process until manually stopped"
while true; do
  sleep 3600
  python - <<PY
from pathlib import Path
import torch, yaml
cfg = yaml.safe_load(open("${CONFIG}", "r", encoding="utf-8"))
latest = Path(cfg["output_dir"]) / "checkpoints" / "latest.pt"
step = int(torch.load(latest, map_location="cpu").get("step", 0)) if latest.exists() else 0
max_steps = int(cfg["train"]["max_steps"])
print("[\$(date)] hold: v2 step=%d/%d" % (step, max_steps))
PY
done
' > "\${LOG}" 2>&1 &

echo \$! > "\${PID_FILE}"
echo "started ${RUN_NAME}"
echo "pid_file=${REMOTE_PROJECT_DIR}/\${PID_FILE}"
echo "log=${REMOTE_PROJECT_DIR}/\${LOG}"
EOF
