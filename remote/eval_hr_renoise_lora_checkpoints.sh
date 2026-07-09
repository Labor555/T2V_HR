#!/usr/bin/env bash
set -euo pipefail

REPAIR=0
if [[ "${1:-}" == "--repair" ]]; then
  REPAIR=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"

REMOTE_HOST="${HR_RENOISE_REMOTE_HOST:-${REMOTE_HOST}}"
REMOTE_PROJECT_DIR="${HR_RENOISE_REMOTE_PROJECT_DIR:-${REMOTE_PROJECT_DIR}}"
REMOTE_PYTHON="${HR_RENOISE_REMOTE_PYTHON:-${REMOTE_PYTHON:-python}}"
CONFIG="${HR_RENOISE_EVAL_CONFIG:-configs/infer_hr_renoise_lora_v1.yaml}"
EVAL_ROOT="${HR_RENOISE_EVAL_ROOT:-outputs/hr_renoise_lora_checkpoint_evals}"
TRAIN_DIR="${HR_RENOISE_TRAIN_DIR:-outputs/wan13b_hr_renoise_lora_v1}"
SAMPLE="${HR_RENOISE_EVAL_SAMPLE:-00000004}"
LR_LATENT="${HR_RENOISE_EVAL_LR:-cache/wan13b_latents_720_4k_1000/lr/${SAMPLE}.pt}"
HR_LATENT="${HR_RENOISE_EVAL_HR:-cache/wan13b_latents_720_4k_1000/hr/${SAMPLE}.pt}"
GPU_MEM_THRESHOLD="${HR_RENOISE_EVAL_WAIT_MEM_MB:-12000}"
GPU_UTIL_THRESHOLD="${HR_RENOISE_EVAL_WAIT_UTIL:-40}"

STATUS="$(ssh "${REMOTE_HOST}" "PROJECT='${REMOTE_PROJECT_DIR}' PY='${REMOTE_PYTHON}' CONFIG='${CONFIG}' TRAIN_DIR='${TRAIN_DIR}' EVAL_ROOT='${EVAL_ROOT}' SAMPLE='${SAMPLE}' LR_LATENT='${LR_LATENT}' HR_LATENT='${HR_LATENT}' MEM_LIMIT='${GPU_MEM_THRESHOLD}' UTIL_LIMIT='${GPU_UTIL_THRESHOLD}' bash -s" <<'EOF'
set -euo pipefail
cd "${PROJECT}"
latest_ckpt=""
latest_step="base"
if [[ -f "${TRAIN_DIR}/checkpoints/latest.pt" ]]; then
  latest_ckpt="${TRAIN_DIR}/checkpoints/latest.pt"
  latest_step="$("${PY}" - <<PY
import torch
print(int(torch.load("${latest_ckpt}", map_location="cpu").get("step", 0)))
PY
)"
fi
eval_dir="${EVAL_ROOT}/step_${latest_step}/${SAMPLE}"
eval_alive="$(ps -eo args= | awk -v cfg="${CONFIG}" '$0 ~ /scripts\/infer_hr_renoise_lora\.py/ && index($0, cfg) {count++} END {print count+0}')"
selected_gpu="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F, -v mem_limit="${MEM_LIMIT}" -v util_limit="${UTIL_LIMIT}" '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if ($2+0 <= mem_limit+0 && $3+0 <= util_limit+0) {print $1; exit}}')"
metrics="${eval_dir}/metrics.json"
compare="${eval_dir}/compare_lsrna_hrdenoise_target_540p.mp4"
echo "latest_step=${latest_step}"
echo "latest_ckpt=${latest_ckpt}"
echo "eval_dir=${eval_dir}"
echo "eval_alive=${eval_alive}"
echo "selected_gpu=${selected_gpu}"
echo "metrics=${metrics}"
echo "compare=${compare}"
echo "metrics_exists=$([[ -f "${metrics}" ]] && echo 1 || echo 0)"
echo "compare_exists=$([[ -f "${compare}" ]] && echo 1 || echo 0)"
if [[ -f "${metrics}" ]]; then
  echo "metrics_json_begin"
  cat "${metrics}"
  echo "metrics_json_end"
fi
echo "gpu_summary_begin"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "gpu_summary_end"
EOF
)"

printf '%s\n' "${STATUS}"

if [[ "${REPAIR}" != "1" ]]; then
  exit 0
fi

eval_alive="$(awk -F= '$1=="eval_alive"{print $2}' <<<"${STATUS}")"
metrics_exists="$(awk -F= '$1=="metrics_exists"{print $2}' <<<"${STATUS}")"
selected_gpu="$(awk -F= '$1=="selected_gpu"{print $2}' <<<"${STATUS}")"
latest_step="$(awk -F= '$1=="latest_step"{print $2}' <<<"${STATUS}")"
eval_dir="$(awk -F= '$1=="eval_dir"{print $2}' <<<"${STATUS}")"
latest_ckpt="$(awk -F= '$1=="latest_ckpt"{print $2}' <<<"${STATUS}")"

if [[ "${metrics_exists}" == "1" || "${eval_alive}" != "0" ]]; then
  exit 0
fi
if [[ -z "${selected_gpu}" ]]; then
  echo "eval_deferred=no_free_gpu_under_${GPU_MEM_THRESHOLD}MiB_util_${GPU_UTIL_THRESHOLD}"
  exit 0
fi

ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT_DIR}' && PY='${REMOTE_PYTHON}' GPU='${selected_gpu}' CONFIG='${CONFIG}' EVAL_DIR='${eval_dir}' LR='${LR_LATENT}' HR='${HR_LATENT}' LORA='${latest_ckpt}' bash -s" <<'EOF'
set -euo pipefail
mkdir -p logs "${EVAL_DIR}"
run_name="eval_hr_renoise_lora_$(date +%Y%m%d_%H%M%S)"
log="logs/${run_name}.log"
cmd=("${PY}" scripts/infer_hr_renoise_lora.py
  --config "${CONFIG}"
  --lr-latent "${LR}"
  --hr-latent "${HR}"
  --output-dir "${EVAL_DIR}"
  --latent-crop-lr 5,24,40
  --num-inference-steps 20
  --refine-strength 0.35
  --guidance-scale 1.0
  --hr-denoise-mode tiled
  --tile-latent-hw 48,80
  --tile-overlap 12)
if [[ -n "${LORA}" ]]; then
  cmd+=(--lora-checkpoint "${LORA}")
else
  cmd+=(--no-lora)
fi
CUDA_VISIBLE_DEVICES="${GPU}" nohup "${cmd[@]}" > "${log}" 2>&1 &
echo $! > "logs/${run_name}.pid"
echo "started_eval_step=${EVAL_DIR}"
echo "eval_gpu=${GPU}"
echo "eval_log=${PWD}/${log}"
EOF
