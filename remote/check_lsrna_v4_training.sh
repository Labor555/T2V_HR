#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LSRNA_TRAIN_CONFIG="configs/train_lsrna_720_4k_1000_v4_detail_temporal.yaml" \
  bash "${SCRIPT_DIR}/check_lsrna_v3_training.sh" "$@"
