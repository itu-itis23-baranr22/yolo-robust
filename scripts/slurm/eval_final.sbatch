#!/bin/bash
#SBATCH -A gh0002
#SBATCH -p gpu2dq
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -o /ari/users/rbaran/yolo-robust/logs/eval_final_%x_%j.out
#SBATCH -e /ari/users/rbaran/yolo-robust/logs/eval_final_%x_%j.err

# Stage 6: the single test-dev evaluation (clean + 15 corrupted sets) of one frozen checkpoint.
#   sbatch -J yr_T_E4s0 -t 01:30:00 --export=ALL,EXP=E4,SEED=0 scripts/slurm/eval_final.sbatch
# Refuses to run when the checkpoint or a protocol file differs from results/manifests/final_models.json, or when
# a complete test-dev result already exists (re-running is only for a job that died before finishing).
set -uo pipefail

export ROOT=/ari/users/rbaran/yolo-robust
source "$ROOT/scripts/env.sh"
export YOLO_OFFLINE=true
export YOLO_AUTOINSTALL=false
cd "$ROOT"

: "${EXP:?EXP not set}"
SEED="${SEED:-0}"
RUN="${EXP}_s${SEED}"
IMGSZ=$(python -c "from ultralytics.utils import YAML; print(YAML.load('configs/experiments.yaml')['experiments']['$EXP']['imgsz'])")

echo "== host $(hostname) job $SLURM_JOB_ID $RUN imgsz=$IMGSZ start $(date -Is)"
echo "== git $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

if [ -f "$ROOT/results/eval/$RUN/test-dev/summary.json" ]; then
    echo "== test-dev result for $RUN already complete -> refuse to re-run"
    exit 1
fi
python scripts/freeze_models.py --verify "$RUN" || exit 1
# Concurrent jobs must never rebuild a shared labels.cache (unlink race in ultralytics 8.4.202).
python scripts/check_label_caches.py --test-dev || { echo "== stale label cache -> abort"; exit 1; }

python scripts/evaluate.py --weights "$ROOT/runs/train/$RUN/weights/best.pt" --imgsz "$IMGSZ" --tag "$RUN" \
    --split test-dev --final
echo "== eval rc=$? $(date -Is)"
du -sh "$ROOT/runs/eval/$RUN/test-dev" 2>/dev/null
echo "== end $(date -Is)"
