#!/usr/bin/env bash
set -euo pipefail

SRC="${SRC:-/data/user/lbzhu/T2V_HR/cache/wan13b_latents_720_4k_1000}"
DST="${DST:-/mnt/nfs/users/lbzhu/T2V_HR/cache/wan13b_latents_720_4k_1000}"
SAGI_HOST="${SAGI_HOST:-sagittarius}"
NFS_OWNER_HOST="${NFS_OWNER_HOST:-scorpio}"

cleanup_perms() {
  ssh -n "${NFS_OWNER_HOST}" "chmod o-w '${DST}' '${DST}/lr' '${DST}/hr' 2>/dev/null || true"
}
trap cleanup_perms EXIT

echo "[$(date)] direct_sync_prepare dst=${NFS_OWNER_HOST}:${DST}"
ssh -n "${NFS_OWNER_HOST}" "mkdir -p '${DST}/lr' '${DST}/hr' && chmod o+w '${DST}' '${DST}/lr' '${DST}/hr'"

echo "[$(date)] direct_sync_start src=${SAGI_HOST}:${SRC} dst=${DST}"
ssh -n "${SAGI_HOST}" "set -euo pipefail
  rsync -rlt --omit-dir-times --no-owner --no-group --ignore-existing --info=progress2 '${SRC}/lr' '${SRC}/hr' '${DST}/'
  sed 's#/data/user/lbzhu/T2V_HR/cache/wan13b_latents_720_4k_1000#/mnt/nfs/users/lbzhu/T2V_HR/cache/wan13b_latents_720_4k_1000#g' '${SRC}/index.jsonl' > '${DST}/index_nfs.jsonl.tmp'
  mv '${DST}/index_nfs.jsonl.tmp' '${DST}/index_nfs.jsonl'
  printf '[sagi_done] lr='
  find '${DST}/lr' -type f -name '*.pt' | wc -l
  printf '[sagi_done] hr='
  find '${DST}/hr' -type f -name '*.pt' | wc -l
  printf '[sagi_done] idx='
  wc -l < '${DST}/index_nfs.jsonl'
"

cleanup_perms
echo "[$(date)] direct_sync_done"
