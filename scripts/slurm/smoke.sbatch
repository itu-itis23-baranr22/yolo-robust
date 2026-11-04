#!/bin/bash
#SBATCH -J yr_smoke
#SBATCH -A gh0002
#SBATCH -p gpu2dq
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 01:30:00
#SBATCH -o /ari/users/rbaran/yolo-robust/logs/smoke_%j.out
#SBATCH -e /ari/users/rbaran/yolo-robust/logs/smoke_%j.err

# Stage-2 smoke test: offline/GPU checks, short trainings, resume, val, batch-1 latency.
# No -e: a failing step is reported and the remaining steps still run.
set -uo pipefail

export ROOT=/ari/users/rbaran/yolo-robust
source "$ROOT/scripts/env.sh"
export YOLO_OFFLINE=true
export YOLO_AUTOINSTALL=false
cd "$ROOT"

TAG="$SLURM_JOB_ID"
OUT="$ROOT/results/smoke/$TAG"
mkdir -p "$OUT"

echo "== host $(hostname) job $SLURM_JOB_ID partition $SLURM_JOB_PARTITION start $(date -Is)"
echo "== git $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l)"
nvidia-smi | tee "$OUT/nvidia-smi.txt"

echo "== internet check (expected to FAIL on compute nodes)"
curl -sS -m 10 -o /dev/null -w "pypi http %{http_code}\n" https://pypi.org/simple/ || echo "curl failed -> no internet"

echo "== cuda module check (subshell only; jobs use the CUDA libs bundled with the torch wheel)"
( module load cuda/cuda-12.5-a100q && which nvcc && nvcc --version | tail -2 ) || echo "cuda module load FAILED"

nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
    --format=csv -l 5 > "$OUT/gpu_monitor.csv" &
MON=$!

run_step() {
    echo "== STEP $1 start $(date -Is)"
    local t=$SECONDS
    python scripts/smoke_test.py "$1" --tag "$TAG"
    echo "== STEP $1 rc=$? elapsed=$((SECONDS - t))s"
}
for s in env throughput resume val latency; do
    run_step "$s"
done

kill "$MON"
echo "== end $(date -Is)"
