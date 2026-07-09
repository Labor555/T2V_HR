#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DST_HOST="${HR_RENOISE_REMOTE_HOST:-scorpio}"
DST_PROJECT_DIR="${HR_RENOISE_REMOTE_PROJECT_DIR:-/mnt/nfs/users/lbzhu/T2V_HR}"
PY="${HR_RENOISE_REMOTE_PYTHON:-/mnt/nfs/users/lbzhu/conda/envs/video-huawei-diffsynth/bin/python}"
CONFIG="${HR_RENOISE_CONFIG:-configs/train_hr_renoise_lora_v1_nfs.yaml}"

cd "${REPO_DIR}"
while true; do
  status="$(
    ssh "${DST_HOST}" "base='${DST_PROJECT_DIR}/cache/wan13b_latents_720_4k_1000'; ckpt='${DST_PROJECT_DIR}/outputs/wan13b_lsrna_720_4k_1000_v5_recurrent_detail/checkpoints/latest.pt'; lr=\$(find \"\$base/lr\" -type f -name '*.pt' 2>/dev/null | wc -l || true); hr=\$(find \"\$base/hr\" -type f -name '*.pt' 2>/dev/null | wc -l || true); idx=0; test -f \"\$base/index_nfs.jsonl\" && idx=\$(wc -l < \"\$base/index_nfs.jsonl\"); ckpt_ready=0; test -f \"\$ckpt\" && ckpt_ready=1; echo lr=\$lr hr=\$hr idx=\$idx ckpt=\$ckpt_ready"
  )"
  echo "[$(date)] ${status}"
  lr="$(awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="lr") print $(i+1)}' <<<"${status}")"
  hr="$(awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="hr") print $(i+1)}' <<<"${status}")"
  idx="$(awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="idx") print $(i+1)}' <<<"${status}")"
  ckpt="$(awk -F'[ =]' '{for(i=1;i<=NF;i++) if($i=="ckpt") print $(i+1)}' <<<"${status}")"
  if [[ "${lr:-0}" -ge 1000 && "${hr:-0}" -ge 1000 && "${idx:-0}" -ge 1000 && "${ckpt:-0}" == "1" ]]; then
    echo "[$(date)] cache ready; launching HR renoise LoRA"
    HR_RENOISE_REMOTE_HOST="${DST_HOST}" \
    HR_RENOISE_REMOTE_PROJECT_DIR="${DST_PROJECT_DIR}" \
    HR_RENOISE_REMOTE_PYTHON="${PY}" \
    HR_RENOISE_CONFIG="${CONFIG}" \
    HR_RENOISE_GPUS="${HR_RENOISE_GPUS:-auto}" \
    HR_RENOISE_MIN_GPUS="${HR_RENOISE_MIN_GPUS:-1}" \
    HR_RENOISE_WAIT_MEM_MB="${HR_RENOISE_WAIT_MEM_MB:-12000}" \
    HR_RENOISE_WAIT_UTIL="${HR_RENOISE_WAIT_UTIL:-35}" \
      bash "${SCRIPT_DIR}/launch_hr_renoise_lora_train.sh"
    break
  fi
  sleep "${HR_RENOISE_WATCH_INTERVAL:-300}"
done
