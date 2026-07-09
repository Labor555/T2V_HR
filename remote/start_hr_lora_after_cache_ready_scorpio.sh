#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/mnt/nfs/users/lbzhu/T2V_HR}"
PY="${PY:-/mnt/nfs/users/lbzhu/conda/envs/video-huawei-diffsynth/bin/python}"
CONFIG="${HR_RENOISE_CONFIG:-configs/train_hr_renoise_lora_v1_nfs_nocoarse.yaml}"
GPUS="${HR_RENOISE_GPUS:-auto}"
MIN_GPUS="${HR_RENOISE_MIN_GPUS:-1}"
MEM_THRESHOLD_MB="${HR_RENOISE_WAIT_MEM_MB:-12000}"
UTIL_THRESHOLD="${HR_RENOISE_WAIT_UTIL:-35}"
RUN_NAME="hr_renoise_lora_after_cache_$(date +%Y%m%d_%H%M%S)"

available_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v mem_limit="${MEM_THRESHOLD_MB}" -v util_limit="${UTIL_THRESHOLD}" '
      {
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
        if ($2 + 0 <= mem_limit + 0 && $3 + 0 <= util_limit + 0) {
          if (out == "") out = $1; else out = out "," $1;
        }
      }
      END { print out }
    '
}

count_gpus() {
  local value="${1:-}"
  if [[ -z "${value}" ]]; then
    echo 0
  else
    awk -F, "{print NF}" <<<"${value}"
  fi
}

cd "${PROJECT_DIR}"
mkdir -p logs
echo "[$(date)] watcher start config=${CONFIG} requested_gpus=${GPUS}"
while true; do
  base="${PROJECT_DIR}/cache/wan13b_latents_720_4k_1000"
  lr=$(find "${base}/lr" -type f -name '*.pt' 2>/dev/null | wc -l || true)
  hr=$(find "${base}/hr" -type f -name '*.pt' 2>/dev/null | wc -l || true)
  idx=0
  test -f "${base}/index_nfs.jsonl" && idx=$(wc -l < "${base}/index_nfs.jsonl")
  echo "[$(date)] waiting_cache lr=${lr} hr=${hr} idx=${idx}"
  if [[ "${lr}" -ge 1000 && "${hr}" -ge 1000 && "${idx}" -ge 1000 ]]; then
    break
  fi
  sleep 300
done

while true; do
  if [[ "${GPUS}" == "auto" ]]; then
    selected_gpus="$(available_gpus)"
  else
    selected_gpus="${GPUS}"
  fi
  num_gpus="$(count_gpus "${selected_gpus}")"
  if (( num_gpus >= MIN_GPUS )); then
    break
  fi
  echo "[$(date)] waiting_gpus selected=${selected_gpus:-none} count=${num_gpus}/${MIN_GPUS}"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  sleep 300
done

echo "[$(date)] starting HR renoise LoRA config=${CONFIG} gpus=${selected_gpus} n=${num_gpus}"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="${selected_gpus}" "${PY}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${num_gpus}" \
  scripts/train_hr_renoise_lora.py --config "${CONFIG}"

echo "[$(date)] HR renoise LoRA exited; holding watcher"
while true; do
  sleep 3600
  "${PY}" - <<PY
from pathlib import Path
import torch, yaml
cfg = yaml.safe_load(open("${CONFIG}", "r", encoding="utf-8"))
latest = Path(cfg["output_dir"]) / "checkpoints" / "latest.pt"
step = int(torch.load(latest, map_location="cpu").get("step", 0)) if latest.exists() else 0
print("[hold] hr_renoise_lora step=%d" % step)
PY
done
