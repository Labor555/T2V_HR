#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LSRNA_EVAL_ROOT="outputs/wan13b_lsrna_v4_detail_temporal_checkpoint_evals" \
LSRNA_LOCAL_EVAL_ROOT="outputs/wan13b_lsrna_v4_detail_temporal_checkpoint_evals" \
  bash "${SCRIPT_DIR}/pull_lsrna_v3_eval_results.sh" "$@"
