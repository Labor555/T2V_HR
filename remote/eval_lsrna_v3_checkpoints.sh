#!/usr/bin/env bash
set -euo pipefail

REPAIR=0
if [[ "${1:-}" == "--repair" ]]; then
  REPAIR=1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime.env"

SAMPLE="${LSRNA_V3_EVAL_SAMPLE:-00000004}"
MEM_THRESHOLD_MB="${LSRNA_V3_EVAL_MEM_MB:-3000}"
UTIL_THRESHOLD="${LSRNA_V3_EVAL_UTIL:-25}"

ssh "${REMOTE_HOST}" "PROJECT='${REMOTE_PROJECT_DIR}' PY='${REMOTE_PYTHON:-python}' SAMPLE='${SAMPLE}' MEM_THRESHOLD_MB='${MEM_THRESHOLD_MB}' UTIL_THRESHOLD='${UTIL_THRESHOLD}' REPAIR='${REPAIR}' bash -s" <<'EOF'
set -euo pipefail
cd "${PROJECT}"
TRAIN_DIR="outputs/wan13b_lsrna_720_4k_1000_v3_full"
EVAL_ROOT="outputs/wan13b_lsrna_v3_checkpoint_evals"
CONFIG="configs/infer_lsrna_720_4k_1000_v3_full.yaml"
mkdir -p logs "${EVAL_ROOT}"

running_eval="$(pgrep -af 'eval_lsrna_v3_checkpoint|infer_lsrna.py .*train_lsrna_720_4k_1000_v3_full|decode_lsrna_latents.py .*wan13b_lsrna_v3_checkpoint_evals' || true)"
if [[ -n "${running_eval}" ]]; then
  echo "eval_status=already_running"
  printf '%s\n' "${running_eval}" | sed 's/^/running_eval_process=/'
  exit 0
fi

available_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
    awk -F, -v mem_limit="${MEM_THRESHOLD_MB}" -v util_limit="${UTIL_THRESHOLD}" '
      {
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
        if (first == "" && $2 + 0 <= mem_limit + 0 && $3 + 0 <= util_limit + 0) {
          first = $1;
        }
      }
      END { print first }
    '
}

next_ckpt="$("${PY}" - <<'PY'
from pathlib import Path

train_dir = Path("outputs/wan13b_lsrna_720_4k_1000_v3_full/checkpoints")
eval_root = Path("outputs/wan13b_lsrna_v3_checkpoint_evals")
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
log="logs/eval_lsrna_v3_step_${step}_${SAMPLE}.log"
gpu="$(available_gpus)"

echo "eval_status=found_checkpoint"
echo "checkpoint=${next_ckpt}"
echo "step=${step}"
echo "sample=${SAMPLE}"
echo "out=${out}"
echo "log=${log}"
echo "selected_gpu=${gpu:-none}"

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
  -filter_complex "[0:v]scale=960:540,setsar=1[v0];[1:v]scale=960:540,setsar=1[v1];[v0][v1]hstack=inputs=2[v]" \\
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
