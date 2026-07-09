#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LSRNA_EVAL_TRAIN_DIR="outputs/wan13b_lsrna_720_4k_1000_v5_recurrent_detail" \
LSRNA_EVAL_ROOT="outputs/wan13b_lsrna_v5_recurrent_detail_checkpoint_evals" \
LSRNA_EVAL_INFER_CONFIG="configs/infer_lsrna_720_4k_1000_v5_recurrent_detail.yaml" \
LSRNA_EVAL_LOG_PREFIX="eval_lsrna_v5_recurrent_detail" \
  bash "${SCRIPT_DIR}/eval_lsrna_v3_checkpoints.sh" "$@"
