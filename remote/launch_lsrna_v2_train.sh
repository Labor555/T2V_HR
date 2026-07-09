#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"
source "${SCRIPT_DIR}/remote_common.sh"

ENV_PREFIX="$(remote_env_prefix)"
TRAIN_GPUS="${LSRNA_V2_GPUS:-auto}"
MIN_GPUS="${LSRNA_V2_MIN_GPUS:-1}"
MEM_THRESHOLD_MB="${LSRNA_V2_WAIT_MEM_MB:-3000}"
UTIL_THRESHOLD="${LSRNA_V2_WAIT_UTIL:-20}"
CONFIG="${LSRNA_V2_CONFIG:-configs/train_lsrna_720_4k_1000_v3_full.yaml}"
RUN_NAME="lsrna_v2_train_$(date +%Y%m%d_%H%M%S)"

ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT_DIR}' && ${ENV_PREFIX} bash -s" <<EOF
set -euo pipefail
mkdir -p logs
LOG="logs/${RUN_NAME}.log"
PID_FILE="logs/${RUN_NAME}.pid"
RUNNER="logs/${RUN_NAME}.runner.sh"

cat > "\${RUNNER}" <<'RUNNER_EOF'
#!/usr/bin/env bash
set -euo pipefail

available_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v mem_limit="\${MEM_THRESHOLD_MB}" -v util_limit="\${UTIL_THRESHOLD}" '
      {
        gsub(/ /, "", \$1); gsub(/ /, "", \$2); gsub(/ /, "", \$3);
        if (\$2 + 0 <= mem_limit + 0 && \$3 + 0 <= util_limit + 0) {
          if (out == "") out = \$1; else out = out "," \$1;
        }
      }
      END { print out }
    '
}

count_gpus() {
  local value="\${1:-}"
  if [[ -z "\${value}" ]]; then
    echo 0
  else
    awk -F, "{print NF}" <<<"\${value}"
  fi
}

cd "\${PROJECT_DIR}"
echo "[\$(date)] auto training request=\${REQUESTED_GPUS} min_gpus=\${MIN_GPUS} mem_threshold_mb=\${MEM_THRESHOLD_MB} util_threshold=\${UTIL_THRESHOLD}"
while true; do
  if [[ "\${REQUESTED_GPUS}" == "auto" ]]; then
    TRAIN_GPUS="\$(available_gpus)"
  else
    TRAIN_GPUS="\${REQUESTED_GPUS}"
  fi
  NUM_TRAIN_GPUS="\$(count_gpus "\${TRAIN_GPUS}")"
  if (( NUM_TRAIN_GPUS >= MIN_GPUS )); then
    break
  fi
  echo "[\$(date)] waiting: selected_gpus=\${TRAIN_GPUS:-none} count=\${NUM_TRAIN_GPUS}/\${MIN_GPUS}"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  sleep 300
done

echo "[\$(date)] starting v2 LSRNA training config=\${CONFIG} gpus=\${TRAIN_GPUS} n=\${NUM_TRAIN_GPUS}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="\${TRAIN_GPUS}" python -m torch.distributed.run \
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
RUNNER_EOF
chmod +x "\${RUNNER}"

PROJECT_DIR="${REMOTE_PROJECT_DIR}" \
REQUESTED_GPUS="${TRAIN_GPUS}" \
MIN_GPUS="${MIN_GPUS}" \
CONFIG="${CONFIG}" \
MEM_THRESHOLD_MB="${MEM_THRESHOLD_MB}" \
UTIL_THRESHOLD="${UTIL_THRESHOLD}" \
nohup bash "\${RUNNER}" > "\${LOG}" 2>&1 &

echo \$! > "\${PID_FILE}"
echo "started ${RUN_NAME}"
echo "pid_file=${REMOTE_PROJECT_DIR}/\${PID_FILE}"
echo "log=${REMOTE_PROJECT_DIR}/\${LOG}"
EOF
