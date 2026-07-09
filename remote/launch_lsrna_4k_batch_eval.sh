#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"

GPUS="${LSRNA_EVAL_GPUS:-4,7}"
SAMPLES="${*:-00000000 00000001 00000002 00000003}"
RUN_NAME="lsrna_4k_batch_eval_$(date +%Y%m%d_%H%M%S)"

ssh "${REMOTE_HOST}" "PROJECT='${REMOTE_PROJECT_DIR}' PY='${REMOTE_PYTHON:-python}' GPUS='${GPUS}' SAMPLES='${SAMPLES}' RUN_NAME='${RUN_NAME}' bash -s" <<'EOF'
set -euo pipefail
cd "${PROJECT}"
mkdir -p logs
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
read -r -a SAMPLE_LIST <<< "${SAMPLES}"
LOG="logs/${RUN_NAME}.log"
PID_FILE="logs/${RUN_NAME}.pid"

if [[ "${#GPU_LIST[@]}" -eq 0 ]]; then
  echo "No GPUs specified." >&2
  exit 1
fi

echo "Launching no-tile LSRNA 4K batch eval"
echo "samples=${SAMPLES}"
echo "gpus=${GPUS}"
echo "run=${RUN_NAME}"
echo "log=${LOG}"

nohup bash -c '
set -euo pipefail
cd "'"${PROJECT}"'"
IFS="," read -r -a GPU_LIST <<< "'"${GPUS}"'"
read -r -a SAMPLE_LIST <<< "'"${SAMPLES}"'"

for worker_idx in "${!GPU_LIST[@]}"; do
  GPU="${GPU_LIST[$worker_idx]}"
  (
    set -euo pipefail
    for i in "${!SAMPLE_LIST[@]}"; do
      if (( i % ${#GPU_LIST[@]} != worker_idx )); then
        continue
      fi
      SAMPLE="${SAMPLE_LIST[$i]}"
      LR="cache/wan13b_latents_720_4k_1000/lr/${SAMPLE}.pt"
      HR="cache/wan13b_latents_720_4k_1000/hr/${SAMPLE}.pt"
      OUT="outputs/'"${RUN_NAME}"'/${SAMPLE}"
      mkdir -p "${OUT}"
      if [[ ! -f "${LR}" ]]; then
        echo "[sample=${SAMPLE}] missing LR latent: ${LR}" >&2
        exit 1
      fi
      if [[ ! -f "${HR}" ]]; then
        echo "[sample=${SAMPLE}] missing HR latent: ${HR}" >&2
        exit 1
      fi
      echo "[sample=${SAMPLE}] gpu=${GPU} start_lsrna out=${OUT}"
      CUDA_VISIBLE_DEVICES="${GPU}" "'"${PY}"'" scripts/infer_lsrna.py \
        --config configs/infer_lsrna_720_4k_1000.yaml \
        --lr-latent "${LR}" \
        --hr-latent "${HR}" \
        --mode lsrna \
        --output-dir "${OUT}"
      echo "[sample=${SAMPLE}] gpu=${GPU} start_decode"
      CUDA_VISIBLE_DEVICES="${GPU}" "'"${PY}"'" scripts/decode_lsrna_latents.py \
        --config configs/infer_lsrna_720_4k_1000.yaml \
        --latents "${OUT}/lsrna_latents.pt" \
        --output-dir "${OUT}" \
        --keys z_hr_ref,z_noisy,z_hr_target \
        --fps 8
      echo "[sample=${SAMPLE}] done"
    done
  ) &
done
wait
echo "all_samples_done"
' > "${LOG}" 2>&1 &

echo $! > "${PID_FILE}"
echo "pid=$(cat "${PID_FILE}")"
echo "pid_file=${PID_FILE}"
echo "log=${LOG}"
echo "outputs=outputs/${RUN_NAME}"
EOF
