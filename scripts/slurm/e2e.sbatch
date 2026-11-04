#!/bin/bash
#SBATCH -J yr_e2e
#SBATCH -A gh0002
#SBATCH -p gpu2dq
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 01:00:00
#SBATCH -o /ari/users/rbaran/yolo-robust/logs/e2e_%j.out
#SBATCH -e /ari/users/rbaran/yolo-robust/logs/e2e_%j.err

# Stage-3 end-to-end check on small subsets / short runs (smoke-level, no model selection):
#   tests, corruption manifest (+ determinism re-check), 2-epoch E3 vs E4 (corruption-augment overhead),
#   evaluation of the Stage-2 smoke checkpoints (all 16 sets; backends compared), batch-1 latency.
# No -e: a failing step is reported and the remaining steps still run.
set -uo pipefail

export ROOT=/ari/users/rbaran/yolo-robust
source "$ROOT/scripts/env.sh"
export YOLO_OFFLINE=true
export YOLO_AUTOINSTALL=false
cd "$ROOT"

TAG="e2e_$SLURM_JOB_ID"
OUT="$ROOT/results/e2e/$SLURM_JOB_ID"
EVAL="$OUT/eval"
SMOKE="$ROOT/runs/smoke/555069"
mkdir -p "$OUT"

echo "== host $(hostname) job $SLURM_JOB_ID partition $SLURM_JOB_PARTITION start $(date -Is)"
echo "== git $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
nvidia-smi | tee "$OUT/nvidia-smi.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
    --format=csv -l 5 > "$OUT/gpu_monitor.csv" &
MON=$!

step() {
    local name=$1
    shift
    echo "== STEP $name start $(date -Is)"
    local t=$SECONDS
    "$@"
    local rc=$?
    echo "== STEP $name rc=$rc elapsed=$((SECONDS - t))s"
    echo "$name,$rc,$((SECONDS - t))" >> "$OUT/steps.csv"
}
echo "step,rc,elapsed_s" > "$OUT/steps.csv"

step pytest python -m pytest -q -p no:cacheprovider tests
step manifest python scripts/corruption_manifest.py --split val --procs 32 --samples 2
step manifest_check python scripts/corruption_manifest.py --split val --procs 32 --check

# corruption-augment overhead: same short schedule as the Stage-2 throughput runs
step train_E3 python scripts/train.py --exp E3 --short "$TAG" --epochs 2 --fraction 0.25
step train_E4 env ROBUST_DATA_STATS_DIR="$OUT/train_augment_stats" \
    python scripts/train.py --exp E4 --short "$TAG" --epochs 2 --fraction 0.25

# evaluation (2-epoch smoke checkpoints: mAP values are meaningless, the pipeline is tested)
step eval_y26s960_full python scripts/evaluate.py --weights "$SMOKE/tp_y26s_960/weights/best.pt" --imgsz 960 \
    --tag y26s_960 --results-root "$EVAL"
step eval_y26s960_backends python scripts/evaluate.py --weights "$SMOKE/tp_y26s_960/weights/best.pt" --imgsz 960 \
    --tag y26s_960 --results-root "$EVAL" --subset 100 --sets clean,noise_s3,blur_s3 --backend both
step eval_y11s640_sub python scripts/evaluate.py --weights "$SMOKE/tp_y11s_640/weights/best.pt" --imgsz 640 \
    --tag y11s_640 --results-root "$EVAL" --subset 100
step eval_E4short_sub python scripts/evaluate.py --weights "$ROOT/runs/e2e/$TAG/E4_s0/weights/best.pt" --imgsz 960 \
    --tag E4_short --results-root "$EVAL" --subset 100 --sets clean,noise_s3

step latency python scripts/latency.py --tag "$TAG" --out "$OUT/latency.json" \
    --model "y11s_640=$SMOKE/tp_y11s_640/weights/best.pt:640" \
    --model "y26s_640=$SMOKE/tp_y26s_640/weights/best.pt:640" \
    --model "y26s_960=$SMOKE/tp_y26s_960/weights/best.pt:960"

kill "$MON"
du -sh "$ROOT/runs/e2e/$TAG" "$ROOT/runs/eval" 2>/dev/null
echo "== end $(date -Is)"
