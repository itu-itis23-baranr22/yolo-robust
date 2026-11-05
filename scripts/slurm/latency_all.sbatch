#!/bin/bash
#SBATCH -A gh0002
#SBATCH -p gpu2dq
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -J yr_latall
#SBATCH -o /ari/users/rbaran/yolo-robust/logs/latency_all_%j.out
#SBATCH -e /ari/users/rbaran/yolo-robust/logs/latency_all_%j.err

# Comparable batch-1 latency: every frozen checkpoint measured in ONE job on ONE GPU (per-job latency files
# differ by ~1 ms between nodes). Two passes in opposite order show whether order/thermal drift matters.
# Afterwards (CPU only, GPU idle): reproducibility manifest of the on-the-fly corrupted test-dev images
# (sha256 of the corrupted arrays; no labels, no model).
#   sbatch scripts/slurm/latency_all.sbatch
set -uo pipefail

export ROOT=/ari/users/rbaran/yolo-robust
source "$ROOT/scripts/env.sh"
export YOLO_OFFLINE=true
export YOLO_AUTOINSTALL=false
cd "$ROOT"

echo "== host $(hostname) job $SLURM_JOB_ID start $(date -Is)"
echo "== git $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm \
    --format=csv -l 5 > "$ROOT/logs/latency_all_gpu_$SLURM_JOB_ID.csv" &
MON=$!

for pass in fwd rev; do
    flag=""; [ "$pass" = rev ] && flag="--reverse"
    args=()
    while read -r spec; do args+=(--model "$spec"); done < <(python scripts/freeze_models.py --specs $flag)
    echo "== latency pass $pass: ${#args[@]} args $(date -Is)"
    python scripts/latency.py --tag "all_${SLURM_JOB_ID}_${pass}" "${args[@]}"
    echo "== latency $pass rc=$? $(date -Is)"
done
kill "$MON"

python scripts/corruption_manifest.py --split test-dev --procs 32
echo "== manifest rc=$? $(date -Is)"
echo "== end $(date -Is)"
