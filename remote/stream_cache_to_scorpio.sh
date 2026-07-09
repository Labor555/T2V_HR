#!/usr/bin/env bash
set -euo pipefail

SRC_HOST="${SRC_HOST:-sagittarius}"
DST_HOST="${DST_HOST:-scorpio}"
DST_PROJECT_DIR="${DST_PROJECT_DIR:-/mnt/nfs/users/lbzhu/T2V_HR}"
SRC_PROJECT_DIR="${SRC_PROJECT_DIR:-/data/user/lbzhu/T2V_HR}"
PY="${REMOTE_PYTHON:-/mnt/nfs/users/lbzhu/conda/envs/video-huawei-diffsynth/bin/python}"

ssh "${DST_HOST}" "mkdir -p '${DST_PROJECT_DIR}/cache' '${DST_PROJECT_DIR}/outputs/wan13b_lsrna_720_4k_1000_v5_recurrent_detail/checkpoints'"
ssh "${SRC_HOST}" "cd '${SRC_PROJECT_DIR}' && tar -cf - cache/wan13b_latents_720_4k_1000 outputs/wan13b_lsrna_720_4k_1000_v5_recurrent_detail/checkpoints/latest.pt" \
  | ssh "${DST_HOST}" "cd '${DST_PROJECT_DIR}' && tar -xf -"
ssh "${DST_HOST}" "PROJECT='${DST_PROJECT_DIR}' '${PY}' - <<'PY'
from pathlib import Path
import json
import sys

project = Path('/mnt/nfs/users/lbzhu/T2V_HR')
src = project / 'cache' / 'wan13b_latents_720_4k_1000' / 'index.jsonl'
dst = project / 'cache' / 'wan13b_latents_720_4k_1000' / 'index_nfs.jsonl'
rows = 0
with src.open('r', encoding='utf-8') as fin, dst.open('w', encoding='utf-8') as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        for key in ('lr_latent', 'hr_latent'):
            row[key] = row[key].replace('/data/user/lbzhu/T2V_HR', str(project))
        fout.write(json.dumps(row, ensure_ascii=False) + '\\n')
        rows += 1
print(f'wrote {dst} rows={rows}', flush=True)
PY"
echo "stream_cache_to_scorpio_done"
