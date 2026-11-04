#!/bin/bash
#SBATCH -A gh0002
#SBATCH -p gpu2dq
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -o /ari/users/rbaran/yolo-robust/logs/train_eval_%x_%j.out
#SBATCH -e /ari/users/rbaran/yolo-robust/logs/train_eval_%x_%j.err

# One experiment end to end: train (or resume) -> clean + 15 corrupted val sets -> batch-1 latency.
#   sbatch -J yr_E1 -t 02:00:00 --export=ALL,EXP=E1,SEED=0 scripts/slurm/train_eval.sbatch
# Re-submitting the same EXP/SEED after an interruption resumes training from last.pt, or skips training
# when it already completed. -t is set per experiment at submission time.
set -uo pipefail

export ROOT=/ari/users/rbaran/yolo-robust
source "$ROOT/scripts/env.sh"
export YOLO_OFFLINE=true
export YOLO_AUTOINSTALL=false
cd "$ROOT"

: "${EXP:?EXP not set}"
SEED="${SEED:-0}"
RUN="${EXP}_s${SEED}"
RUN_DIR="$ROOT/runs/train/$RUN"
META="$ROOT/results/train/$RUN/meta.json"
IMGSZ=$(python -c "from ultralytics.utils import YAML; print(YAML.load('configs/experiments.yaml')['experiments']['$EXP']['imgsz'])")

echo "== host $(hostname) job $SLURM_JOB_ID $RUN imgsz=$IMGSZ start $(date -Is)"
echo "== git $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mkdir -p "$ROOT/results/train/$RUN"
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
    --format=csv -l 30 > "$ROOT/results/train/$RUN/gpu_monitor_$SLURM_JOB_ID.csv" &
MON=$!

# Concurrent jobs must never rebuild the shared labels.cache (unlink race in ultralytics 8.4.202).
if ! python scripts/check_label_caches.py; then
    echo "== stale label cache -> abort (rebuild once on the login node)"
    kill "$MON"
    exit 1
fi

status=$(python -c "import json,sys; print(json.load(open('$META'))['extra']['status'])" 2>/dev/null || echo none)
if [ "$status" = "completed" ] && [ -f "$RUN_DIR/weights/best.pt" ]; then
    echo "== training already completed -> skip"
    rc=0
elif [ -f "$RUN_DIR/weights/last.pt" ]; then
    echo "== resume from $RUN_DIR/weights/last.pt"
    python scripts/train.py --exp "$EXP" --seed "$SEED" --resume
    rc=$?
else
    python scripts/train.py --exp "$EXP" --seed "$SEED"
    rc=$?
fi
echo "== train rc=$rc $(date -Is)"

if [ "$rc" -eq 0 ] && [ -f "$RUN_DIR/weights/best.pt" ]; then
    python scripts/evaluate.py --weights "$RUN_DIR/weights/best.pt" --imgsz "$IMGSZ" --tag "$RUN"
    echo "== eval rc=$? $(date -Is)"
    python scripts/latency.py --tag "$RUN" --model "$RUN=$RUN_DIR/weights/best.pt:$IMGSZ"
    echo "== latency rc=$? $(date -Is)"
else
    echo "== skipping evaluation (train rc=$rc)"
fi

kill "$MON"
du -sh "$RUN_DIR" "$ROOT/runs/eval/$RUN" 2>/dev/null
echo "== end $(date -Is)"
