#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"
source "${SCRIPT_DIR}/remote_common.sh"

ENV_PREFIX="$(remote_env_prefix)"
REMOTE_CUDA_VISIBLE_DEVICES="${REMOTE_CUDA_VISIBLE_DEVICES:-4,5,6,7}"
REMOTE_CACHE_CUDA_VISIBLE_DEVICES="${REMOTE_CACHE_CUDA_VISIBLE_DEVICES:-${REMOTE_CUDA_VISIBLE_DEVICES}}"
LIMIT="${REMOTE_CACHE_LIMIT:-1000}"
NUM_TRAIN_GPUS="$(awk -F',' '{print NF}' <<<"${REMOTE_CUDA_VISIBLE_DEVICES}")"
NUM_CACHE_WORKERS="$(awk -F',' '{print NF}' <<<"${REMOTE_CACHE_CUDA_VISIBLE_DEVICES}")"
JOB_NAME="720_4k_full_$(date +%Y%m%d_%H%M%S)"

ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT_DIR}' && ${ENV_PREFIX} bash -s" <<EOF
set -euo pipefail
mkdir -p logs
LOG="logs/${JOB_NAME}.log"
PID_FILE="logs/${JOB_NAME}.pid"
nohup bash -c '
set -euo pipefail
CACHE_GPUS="${REMOTE_CACHE_CUDA_VISIBLE_DEVICES}"
TRAIN_GPUS="${REMOTE_CUDA_VISIBLE_DEVICES}"
LIMIT="${LIMIT}"
NUM_CACHE_WORKERS="${NUM_CACHE_WORKERS}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS}"
CACHE_CONFIG="configs/cache_wan_latents_720_4k_1000.yaml"
TRAIN_CONFIG="configs/train_lsr_720_4k_1000.yaml"
CACHE_DIR="/data/user/lbzhu/T2V_HR/cache/wan13b_latents_720_4k_1000"
TRAIN_DIR="/data/user/lbzhu/T2V_HR/outputs/wan13b_lsr_720_4k_1000"
INTERVAL=300
echo "[\$(date)] cache_gpus=\${CACHE_GPUS} train_gpus=\${TRAIN_GPUS} limit=\${LIMIT} cache_workers=\${NUM_CACHE_WORKERS} train_gpus_n=\${NUM_TRAIN_GPUS}"

cache_count() {
  if [ -f "\${CACHE_DIR}/index.jsonl" ]; then
    wc -l < "\${CACHE_DIR}/index.jsonl"
  else
    echo 0
  fi
}

merge_cache() {
  python scripts/merge_cache_indices.py --cache-dir "\${CACHE_DIR}" --output index.jsonl
}

while [ "\$(cache_count)" -lt "\${LIMIT}" ]; do
  attempt=\$((\${attempt:-0} + 1))
  echo "[\$(date)] cache attempt \${attempt}; current=\$(cache_count)/\${LIMIT}"
  pids=""
  shard=0
  IFS="," read -ra GPU_LIST <<< "\${CACHE_GPUS}"
  for gpu in "\${GPU_LIST[@]}"; do
    log_path="logs/${JOB_NAME}_cache_shard_\${shard}.log"
    echo "[\$(date)] launch cache shard \${shard}/\${NUM_CACHE_WORKERS} on gpu \${gpu} -> \${log_path}"
    CUDA_VISIBLE_DEVICES="\${gpu}" python scripts/cache_wan_latents.py \
      --config "\${CACHE_CONFIG}" \
      --limit "\${LIMIT}" \
      --num-shards "\${NUM_CACHE_WORKERS}" \
      --shard-id "\${shard}" \
      --index-name "index_shard_\${shard}.jsonl" \
      > "\${log_path}" 2>&1 &
    pids="\${pids} \$!"
    shard=\$((shard + 1))
  done

  failed=0
  for pid in \${pids}; do
    if ! wait "\${pid}"; then
      failed=1
    fi
  done
  merge_cache || failed=1
  echo "[\$(date)] cache merged count=\$(cache_count)/\${LIMIT} failed=\${failed}"
  if [ "\$(cache_count)" -ge "\${LIMIT}" ]; then
    break
  fi
  echo "[\$(date)] cache is not complete; retrying in 60s and reusing valid latent files"
  sleep 60
done

read_step() {
  python - <<PY
from pathlib import Path
import torch, yaml
cfg = yaml.safe_load(open("\${TRAIN_CONFIG}", "r", encoding="utf-8"))
latest = Path(cfg["output_dir"]) / "checkpoints" / "latest.pt"
if latest.exists():
    print(int(torch.load(latest, map_location="cpu").get("step", 0)))
else:
    print(0)
PY
}

read_max_steps() {
  python - <<PY
import yaml
cfg = yaml.safe_load(open("\${TRAIN_CONFIG}", "r", encoding="utf-8"))
print(int(cfg["train"]["max_steps"]))
PY
}

MAX_STEPS="\$(read_max_steps)"
mkdir -p "\${TRAIN_DIR}"
while true; do
  STEP="\$(read_step)"
  echo "[\$(date)] train watchdog step=\${STEP}/\${MAX_STEPS}"
  if [ "\${STEP}" -ge "\${MAX_STEPS}" ]; then
    echo "[\$(date)] training complete"
    break
  fi
  set +e
  CUDA_VISIBLE_DEVICES="\${TRAIN_GPUS}" python -m torch.distributed.run \
    --standalone \
    --nproc_per_node="\${NUM_TRAIN_GPUS}" \
    scripts/train_lsr.py --config "\${TRAIN_CONFIG}" &
  train_pid=\$!
  while kill -0 "\${train_pid}" 2>/dev/null; do
    sleep "\${INTERVAL}"
    STEP="\$(read_step)"
    echo "[\$(date)] train watchdog alive pid=\${train_pid}; step=\${STEP}/\${MAX_STEPS}"
    if [ "\${STEP}" -ge "\${MAX_STEPS}" ]; then
      break
    fi
  done
  wait "\${train_pid}"
  status=\$?
  set -e
  STEP="\$(read_step)"
  echo "[\$(date)] train process exited status=\${status}; step=\${STEP}/\${MAX_STEPS}"
  if [ "\${STEP}" -ge "\${MAX_STEPS}" ]; then
    echo "[\$(date)] training complete"
    break
  fi
  echo "[\$(date)] restarting training in \${INTERVAL}s"
  sleep "\${INTERVAL}"
done
echo "[\$(date)] full 720/4K job reached max steps; holding process until manually stopped"
while true; do
  sleep 3600
  echo "[\$(date)] hold: training already reached \$(read_step)/\${MAX_STEPS}"
done
' > "\${LOG}" 2>&1 &
echo \$! > "\${PID_FILE}"
echo "started ${JOB_NAME}"
echo "pid_file=${REMOTE_PROJECT_DIR}/\${PID_FILE}"
echo "log=${REMOTE_PROJECT_DIR}/\${LOG}"
EOF
