#!/usr/bin/env bash
set -euo pipefail

SRC_HOST="${SRC_HOST:-sagittarius}"
DST_HOST="${DST_HOST:-scorpio}"
SRC_PATH="${SRC_PATH:-/data/user/lbzhu/T2V_HR/outputs/wan13b_lsrna_720_4k_1000_v5_recurrent_detail/checkpoints/latest.pt}"
DST_PATH="${DST_PATH:-/mnt/nfs/users/lbzhu/T2V_HR/outputs/wan13b_lsrna_720_4k_1000_v5_recurrent_detail/checkpoints/latest.pt}"

dst_dir="$(dirname "${DST_PATH}")"
ssh "${DST_HOST}" "mkdir -p '${dst_dir}' && rm -f '${DST_PATH}.tmp'"
ssh "${SRC_HOST}" "cat '${SRC_PATH}'" | ssh "${DST_HOST}" "cat > '${DST_PATH}.tmp' && mv '${DST_PATH}.tmp' '${DST_PATH}' && ls -lh '${DST_PATH}'"
echo "copy_v5_checkpoint_done"
