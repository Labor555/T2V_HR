#!/usr/bin/env bash
set -euo pipefail

SRC="${SRC:-/data/user/lbzhu/T2V_HR/cache/wan13b_latents_720_4k_1000}"
DST="${DST:-/mnt/nfs/users/lbzhu/T2V_HR/cache/wan13b_latents_720_4k_1000}"
SAGI_HOST="${SAGI_HOST:-sagittarius}"
NFS_HOST="${NFS_HOST:-scorpio}"

echo "[$(date)] stream_start src=${SAGI_HOST}:${SRC} dst=${NFS_HOST}:${DST}"
ssh "${SAGI_HOST}" "cd '${SRC}' && tar -cf - lr hr index.jsonl" |
  ssh "${NFS_HOST}" "set -euo pipefail
    mkdir -p '${DST}'
    tar -C '${DST}' -xf -
    sed 's#/data/user/lbzhu/T2V_HR/cache/wan13b_latents_720_4k_1000#/mnt/nfs/users/lbzhu/T2V_HR/cache/wan13b_latents_720_4k_1000#g' '${DST}/index.jsonl' > '${DST}/index_nfs.jsonl.tmp'
    mv '${DST}/index_nfs.jsonl.tmp' '${DST}/index_nfs.jsonl'
    printf '[remote_done] lr='
    find '${DST}/lr' -type f -name '*.pt' | wc -l
    printf '[remote_done] hr='
    find '${DST}/hr' -type f -name '*.pt' | wc -l
    printf '[remote_done] idx='
    wc -l < '${DST}/index_nfs.jsonl'
  "
echo "[$(date)] stream_done"
