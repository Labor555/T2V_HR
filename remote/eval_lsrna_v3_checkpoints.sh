#!/usr/bin/env bash
set -euo pipefail

REPAIR=0
if [[ "${1:-}" == "--repair" ]]; then
  REPAIR=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"

SAMPLE="${LSRNA_V3_EVAL_SAMPLE:-00000004}"
MEM_THRESHOLD_MB="${LSRNA_V3_EVAL_MEM_MB:-25000}"
UTIL_THRESHOLD="${LSRNA_V3_EVAL_UTIL:-100}"
FORCE_GPU="${LSRNA_V3_EVAL_GPU:-}"
TRAIN_DIR="${LSRNA_EVAL_TRAIN_DIR:-outputs/wan13b_lsrna_720_4k_1000_v3_full}"
EVAL_ROOT="${LSRNA_EVAL_ROOT:-outputs/wan13b_lsrna_v3_checkpoint_evals}"
CONFIG="${LSRNA_EVAL_INFER_CONFIG:-configs/infer_lsrna_720_4k_1000_v3_full.yaml}"
LOG_PREFIX="${LSRNA_EVAL_LOG_PREFIX:-eval_lsrna_v3}"

ssh "${REMOTE_HOST}" "PROJECT='${REMOTE_PROJECT_DIR}' PY='${REMOTE_PYTHON:-python}' SAMPLE='${SAMPLE}' MEM_THRESHOLD_MB='${MEM_THRESHOLD_MB}' UTIL_THRESHOLD='${UTIL_THRESHOLD}' FORCE_GPU='${FORCE_GPU}' REPAIR='${REPAIR}' TRAIN_DIR='${TRAIN_DIR}' EVAL_ROOT='${EVAL_ROOT}' CONFIG='${CONFIG}' LOG_PREFIX='${LOG_PREFIX}' bash -s" <<'EOF'
set -euo pipefail
cd "${PROJECT}"
mkdir -p logs "${EVAL_ROOT}"

if [[ "${REPAIR}" == "1" ]]; then
  while IFS= read -r running_marker; do
    step_dir="$(dirname "${running_marker}")"
    pid_file="${step_dir}/eval.pid"
    if [[ ! -s "${pid_file}" ]] || ! kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
      rm -f "${running_marker}"
      echo "cleared_stale_eval=$(basename "${step_dir}")"
    fi
  done < <(find "${EVAL_ROOT}" -mindepth 2 -maxdepth 2 -name RUNNING 2>/dev/null | sort)
fi

running_eval="$(ps -eo pid=,args= | awk -v root="${EVAL_ROOT}" -v train="${TRAIN_DIR}" '
  $0 ~ /awk -v root=/ {
    next
  }
  ($0 ~ /python .*infer_lsrna.py/ && (index($0, train) || index($0, root))) ||
  ($0 ~ /python .*decode_lsrna_latents.py/ && index($0, root)) ||
  ($0 ~ /bash .*eval_runner.sh/ && index($0, root)) ||
  ($0 ~ /ffmpeg/ && index($0, root)) {
    print
  }
' || true)"
if [[ -n "${running_eval}" ]]; then
  echo "eval_status=already_running"
  printf '%s\n' "${running_eval}" | sed 's/^/running_eval_process=/'
  exit 0
fi

available_gpus() {
  if [[ -n "${FORCE_GPU}" ]]; then
    printf '%s\n' "${FORCE_GPU}"
    return
  fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v mem_limit="${MEM_THRESHOLD_MB}" -v util_limit="${UTIL_THRESHOLD}" '
      {
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
        mem = $2 + 0;
        util = $3 + 0;
        if (mem <= mem_limit + 0 && util <= util_limit + 0) {
          if (best == "" || mem < best_mem) {
            best = $1;
            best_mem = mem;
          }
        }
      }
      END { print best }
    '
}

next_ckpt="$("${PY}" - "${TRAIN_DIR}" "${EVAL_ROOT}" <<'PY'
from pathlib import Path
import sys

train_dir = Path(sys.argv[1]) / "checkpoints"
eval_root = Path(sys.argv[2])
ckpts = sorted(train_dir.glob("step_*.pt"))
for ckpt in ckpts:
    step = ckpt.stem.split("_")[-1]
    done = eval_root / f"step_{step}" / "DONE"
    running = eval_root / f"step_{step}" / "RUNNING"
    if not done.exists() and not running.exists():
        print(ckpt)
        break
PY
)"

if [[ -z "${next_ckpt}" ]]; then
  echo "eval_status=no_new_checkpoint"
  latest_done="$(find "${EVAL_ROOT}" -maxdepth 2 -name DONE 2>/dev/null | sort | tail -1 || true)"
  [[ -n "${latest_done}" ]] && echo "latest_done=${latest_done}"
  exit 0
fi

step="$(basename "${next_ckpt}" .pt | sed 's/^step_//')"
out="${EVAL_ROOT}/step_${step}/${SAMPLE}"
step_dir="${EVAL_ROOT}/step_${step}"
log="logs/${LOG_PREFIX}_step_${step}_${SAMPLE}.log"
gpu="$(available_gpus)"

echo "eval_status=found_checkpoint"
echo "checkpoint=${next_ckpt}"
echo "step=${step}"
echo "sample=${SAMPLE}"
echo "out=${out}"
echo "log=${log}"
echo "selected_gpu=${gpu:-none}"
[[ -n "${FORCE_GPU}" ]] && echo "forced_gpu=${FORCE_GPU}"

if [[ "${REPAIR}" != "1" ]]; then
  exit 0
fi
if [[ -z "${gpu}" ]]; then
  echo "eval_deferred=no_free_gpu_under_${MEM_THRESHOLD_MB}MiB_util_${UTIL_THRESHOLD}"
  exit 0
fi

mkdir -p "${out}"
touch "${step_dir}/RUNNING"
rm -f "${step_dir}/DONE"

cat > "${step_dir}/eval_runner.sh" <<RUNNER
#!/usr/bin/env bash
set -euo pipefail
cd "${PROJECT}"
CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/infer_lsrna.py \\
  --config "${CONFIG}" \\
  --lsr-checkpoint "${next_ckpt}" \\
  --lr-latent "cache/wan13b_latents_720_4k_1000/lr/${SAMPLE}.pt" \\
  --hr-latent "cache/wan13b_latents_720_4k_1000/hr/${SAMPLE}.pt" \\
  --mode lsrna \\
  --output-dir "${out}"
CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/latent_lsrna_metrics.py \\
  --latents "${out}/lsrna_latents.pt" \\
  --output "${out}/metrics.json"
CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/decode_lsrna_latents.py \\
  --config "${CONFIG}" \\
  --latents "${out}/lsrna_latents.pt" \\
  --output-dir "${out}" \\
  --keys z_noisy,z_hr_target \\
  --fps 8
ffmpeg -y \\
  -i "${out}/lsrna_noisy.mp4" \\
  -i "${out}/target_reconstruction.mp4" \\
  -filter_complex "[0:v]scale=960:540,setsar=1,drawtext=text='Video-LSRNA':x=24:y=24:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.55[v0];[1:v]scale=960:540,setsar=1,drawtext=text='Baseline Wan target':x=24:y=24:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.55[v1];[v0][v1]hstack=inputs=2[v]" \\
  -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 18 \\
  "${out}/compare_lsrna_vs_wan_target_540p.mp4"
rm -f "${step_dir}/RUNNING"
touch "${step_dir}/DONE"
RUNNER
chmod +x "${step_dir}/eval_runner.sh"
nohup bash "${step_dir}/eval_runner.sh" > "${log}" 2>&1 &
echo $! > "${step_dir}/eval.pid"
echo "eval_launched_pid=$(cat "${step_dir}/eval.pid")"
EOF
