#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"

REMOTE_HOST="${CACHE_REMOTE_HOST:-scorpio}"
REMOTE_PROJECT_DIR="${CACHE_REMOTE_PROJECT_DIR:-/mnt/nfs/users/lbzhu/T2V_HR}"
REMOTE_PYTHON="${CACHE_REMOTE_PYTHON:-/mnt/nfs/users/lbzhu/conda/envs/video-huawei-diffsynth/bin/python}"
CONFIG="${CACHE_CONFIG:-configs/cache_wan_latents_720_4k_1000_nfs.yaml}"
LIMIT="${CACHE_LIMIT:-1000}"
GPUS="${CACHE_GPUS:-0,1}"
JOB_NAME="cache_720_4k_1000_nfs_$(date +%Y%m%d_%H%M%S)"

ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT_DIR}' && REMOTE_PROJECT_DIR='${REMOTE_PROJECT_DIR}' PY='${REMOTE_PYTHON}' CONFIG='${CONFIG}' LIMIT='${LIMIT}' GPUS='${GPUS}' JOB_NAME='${JOB_NAME}' bash -s" <<'EOF'
set -euo pipefail
mkdir -p logs
RUNNER="logs/${JOB_NAME}.runner.sh"
LOG="logs/${JOB_NAME}.log"
PID_FILE="logs/${JOB_NAME}.pid"
cat > "${RUNNER}" <<'RUNNER_EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT_DIR}"
IFS=',' read -ra GPU_LIST <<< "${GPUS}"
num_shards="${#GPU_LIST[@]}"
echo "[$(date)] cache start config=${CONFIG} limit=${LIMIT} gpus=${GPUS} num_shards=${num_shards}"
for shard_id in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$shard_id]}"
  shard_log="logs/${JOB_NAME}_shard${shard_id}_gpu${gpu}.log"
  echo "[$(date)] start shard=${shard_id}/${num_shards} gpu=${gpu} log=${shard_log}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/cache_wan_latents.py \
    --config "${CONFIG}" \
    --limit "${LIMIT}" \
    --num-shards "${num_shards}" \
    --shard-id "${shard_id}" \
    --index-name "index_shard_${shard_id}.jsonl" > "${shard_log}" 2>&1 &
  echo $! > "logs/${JOB_NAME}_shard${shard_id}.pid"
done

failed=0
for shard_id in "${!GPU_LIST[@]}"; do
  pid="$(cat "logs/${JOB_NAME}_shard${shard_id}.pid")"
  if ! wait "${pid}"; then
    failed=1
  fi
done
if (( failed != 0 )); then
  echo "[$(date)] cache failed; see shard logs"
  exit 1
fi
"${PY}" scripts/merge_cache_indices.py \
  --cache-dir /mnt/nfs/users/lbzhu/T2V_HR/cache/wan13b_latents_720_4k_1000 \
  --pattern 'index_shard_*.jsonl' \
  --output index_nfs.jsonl
echo "[$(date)] cache complete; holding supervisor"
while true; do
  sleep 3600
  base=/mnt/nfs/users/lbzhu/T2V_HR/cache/wan13b_latents_720_4k_1000
  lr=$(find "$base/lr" -type f -name '*.pt' 2>/dev/null | wc -l || true)
  hr=$(find "$base/hr" -type f -name '*.pt' 2>/dev/null | wc -l || true)
  idx=0
  test -f "$base/index_nfs.jsonl" && idx=$(wc -l < "$base/index_nfs.jsonl")
  echo "[$(date)] hold: lr=${lr} hr=${hr} idx=${idx}"
done
RUNNER_EOF
chmod +x "${RUNNER}"
PROJECT_DIR="$(pwd)" PY="${PY}" CONFIG="${CONFIG}" LIMIT="${LIMIT}" GPUS="${GPUS}" JOB_NAME="${JOB_NAME}" \
  nohup bash "${RUNNER}" > "${LOG}" 2>&1 &
echo $! > "${PID_FILE}"
echo "started ${JOB_NAME}"
echo "pid_file=${REMOTE_PROJECT_DIR}/${PID_FILE}"
echo "log=${REMOTE_PROJECT_DIR}/${LOG}"
EOF
