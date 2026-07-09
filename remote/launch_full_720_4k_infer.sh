#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"

REMOTE_GPU="${REMOTE_INFER_CUDA_VISIBLE_DEVICES:-7}"
SAMPLE_ID="${1:-00000000}"
STEPS="${FULL_INFER_STEPS:-50}"
REFINE_STRENGTH="${FULL_INFER_REFINE_STRENGTH:-0.35}"
GUIDANCE_SCALE="${FULL_INFER_GUIDANCE_SCALE:-5.0}"
TILE_HW="${FULL_INFER_TILE_LATENT_HW:-64,64}"
TILE_OVERLAP="${FULL_INFER_TILE_OVERLAP:-16}"
RUN_NAME="full_720_4k_infer_${SAMPLE_ID}_$(date +%Y%m%d_%H%M%S)"

ssh "${REMOTE_HOST}" "PROJECT='${REMOTE_PROJECT_DIR}' PY='${REMOTE_PYTHON:-python}' GPU='${REMOTE_GPU}' SAMPLE='${SAMPLE_ID}' STEPS='${STEPS}' REFINE='${REFINE_STRENGTH}' GUIDANCE='${GUIDANCE_SCALE}' TILE_HW='${TILE_HW}' TILE_OVERLAP='${TILE_OVERLAP}' RUN_NAME='${RUN_NAME}' bash -s" <<'EOF'
set -euo pipefail
cd "${PROJECT}"
mkdir -p logs
LR="cache/wan13b_latents_720_4k_1000/lr/${SAMPLE}.pt"
HR="cache/wan13b_latents_720_4k_1000/hr/${SAMPLE}.pt"
OUT="outputs/${RUN_NAME}"
LOG="logs/${RUN_NAME}.log"
PID_FILE="logs/${RUN_NAME}.pid"

if [[ ! -f "${LR}" ]]; then
  echo "Missing LR latent: ${LR}" >&2
  exit 1
fi
if [[ ! -f "${HR}" ]]; then
  echo "Missing HR latent: ${HR}" >&2
  exit 1
fi

echo "Launching full 720p->4K inference"
echo "sample=${SAMPLE}"
echo "gpu=${GPU}"
echo "steps=${STEPS} refine=${REFINE} guidance=${GUIDANCE}"
echo "tile_hw=${TILE_HW} tile_overlap=${TILE_OVERLAP}"
echo "out=${OUT}"
echo "log=${LOG}"

nohup bash -c "
set -euo pipefail
cd '${PROJECT}'
CUDA_VISIBLE_DEVICES='${GPU}' '${PY}' scripts/infer_lsrna.py \
  --config configs/infer_lsrna_720_4k_1000.yaml \
  --lr-latent '${LR}' \
  --hr-latent '${HR}' \
  --hr-denoise-mode tiled \
  --tile-latent-hw '${TILE_HW}' \
  --tile-overlap '${TILE_OVERLAP}' \
  --num-inference-steps '${STEPS}' \
  --refine-strength '${REFINE}' \
  --guidance-scale '${GUIDANCE}' \
  --fps 8 \
  --save-intermediates \
  --output-dir '${OUT}'
" > "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "pid=$(cat "${PID_FILE}")"
echo "pid_file=${PID_FILE}"
echo "log=${LOG}"
EOF
