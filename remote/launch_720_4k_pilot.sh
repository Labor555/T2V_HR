#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"
source "${SCRIPT_DIR}/remote_common.sh"

ENV_PREFIX="$(remote_env_prefix)"
JOB_NAME="720_4k_pilot_$(date +%Y%m%d_%H%M%S)"
REMOTE_CUDA_VISIBLE_DEVICES="${REMOTE_CUDA_VISIBLE_DEVICES:-4}"

ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT_DIR}' && ${ENV_PREFIX} CUDA_VISIBLE_DEVICES='${REMOTE_CUDA_VISIBLE_DEVICES}' bash -s" <<EOF
set -euo pipefail
mkdir -p logs
LOG="logs/${JOB_NAME}.log"
PID_FILE="logs/${JOB_NAME}.pid"
nohup bash -c '
set -euo pipefail
echo "[\$(date)] CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-}"
echo "[\$(date)] cache 720p/4K latent pairs"
python scripts/cache_wan_latents.py --config configs/cache_wan_latents_720_4k_pilot.yaml --limit 4
echo "[\$(date)] train 3x Video-LSR"
python scripts/train_lsr.py --config configs/train_lsr_720_4k_pilot.yaml
echo "[\$(date)] build Video-LSRNA output"
python scripts/infer_lsrna.py --config configs/infer_lsrna_720_4k_pilot.yaml --lr-latent /data/user/lbzhu/T2V_HR/cache/wan13b_latents_720_4k_pilot/lr/00000000.pt
echo "[\$(date)] summarize tensors"
python - <<PY
from pathlib import Path
import torch
path = Path("/data/user/lbzhu/T2V_HR/outputs/infer_lsrna_720_4k_pilot/lsrna_latents.pt")
obj = torch.load(path, map_location="cpu")
print(path)
for key, value in obj.items():
    if hasattr(value, "shape"):
        print(key, tuple(value.shape), value.dtype, float(value.float().mean()))
PY
echo "[\$(date)] done"
' > "\${LOG}" 2>&1 &
echo \$! > "\${PID_FILE}"
echo "started ${JOB_NAME}"
echo "pid_file=${REMOTE_PROJECT_DIR}/\${PID_FILE}"
echo "log=${REMOTE_PROJECT_DIR}/\${LOG}"
EOF
