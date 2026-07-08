#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"
source "${SCRIPT_DIR}/remote_common.sh"

ENV_PREFIX="$(remote_env_prefix)"
JOB_NAME="cache_tail_accel_$(date +%Y%m%d_%H%M%S)"
CONFIG="${REMOTE_TAIL_CACHE_CONFIG:-configs/cache_wan_latents_720_4k_1000.yaml}"
LIMIT="${REMOTE_CACHE_LIMIT:-1000}"
GPU_GROUPS="${REMOTE_TAIL_CACHE_GPU_GROUPS:-4:0,1,2;5:3,4,5;7:6,7,8}"
START_POS="${REMOTE_TAIL_CACHE_START_POS:-85}"
END_POS="${REMOTE_TAIL_CACHE_END_POS:-99}"

ssh "${REMOTE_HOST}" "cd '${REMOTE_PROJECT_DIR}' && ${ENV_PREFIX} bash -s" <<EOF
set -euo pipefail
mkdir -p logs
echo "[\$(date)] launch ${JOB_NAME}; groups=${GPU_GROUPS}; positions=${START_POS}-${END_POS}"

group_id=0
IFS=";" read -ra CACHE_GROUP_LIST <<< "${GPU_GROUPS}"
for group in "\${CACHE_GROUP_LIST[@]}"; do
  gpu="\${group%%:*}"
  shards="\${group#*:}"
  indices=""
  IFS="," read -ra SHARD_LIST <<< "\${shards}"
  for shard in "\${SHARD_LIST[@]}"; do
    for pos in \$(seq "${START_POS}" "${END_POS}"); do
      idx=\$((shard + 10 * pos))
      if [ "\${idx}" -lt "${LIMIT}" ]; then
        indices="\${indices},\${idx}"
      fi
    done
  done
  indices="\${indices#,}"
  if [ -z "\${indices}" ]; then
    continue
  fi
  log_path="logs/${JOB_NAME}_gpu\${gpu}_group\${group_id}.log"
  index_name="index_shard_${JOB_NAME}_group\${group_id}.jsonl"
  echo "[\$(date)] gpu=\${gpu} shards=\${shards} indices=\$(tr ',' ' ' <<< "\${indices}") -> \${log_path}"
  nohup bash -c "CUDA_VISIBLE_DEVICES='\${gpu}' python scripts/cache_wan_latents_indices.py --config '${CONFIG}' --limit '${LIMIT}' --indices '\${indices}' --index-name '\${index_name}'" > "\${log_path}" 2>&1 &
  echo \$! > "logs/${JOB_NAME}_group\${group_id}.pid"
  group_id=\$((group_id + 1))
done
EOF
