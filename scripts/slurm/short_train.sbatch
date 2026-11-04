#!/bin/bash
#SBATCH -A gh0002
#SBATCH -p gpu2dq
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 00:45:00
#SBATCH -o /ari/users/rbaran/yolo-robust/logs/short_%x_%j.out
#SBATCH -e /ari/users/rbaran/yolo-robust/logs/short_%x_%j.err

# Short smoke-level training of one experiment on the FULL train split (no `fraction`, so the shared
# labels.cache stays valid for concurrently starting jobs) + batch-1 latency of the resulting checkpoint.
#   sbatch -J yr_E5short --export=ALL,EXP=E5,EPOCHS=2,ALLOW_OPTIONAL_EXPERIMENTS=1 scripts/slurm/short_train.sbatch
set -uo pipefail

export ROOT=/ari/users/rbaran/yolo-robust
source "$ROOT/scripts/env.sh"
export YOLO_OFFLINE=true
export YOLO_AUTOINSTALL=false
cd "$ROOT"

: "${EXP:?EXP not set}"
EPOCHS="${EPOCHS:-2}"
TAG="short_$SLURM_JOB_ID"
IMGSZ=$(python -c "from ultralytics.utils import YAML; print(YAML.load('configs/experiments.yaml')['experiments']['$EXP']['imgsz'])")
echo "== host $(hostname) job $SLURM_JOB_ID $EXP epochs=$EPOCHS imgsz=$IMGSZ start $(date -Is)"
echo "== git $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"

if ! python scripts/check_label_caches.py; then
    echo "== stale label cache -> abort"
    exit 1
fi
python scripts/train.py --exp "$EXP" --short "$TAG" --epochs "$EPOCHS"
echo "== train rc=$? $(date -Is)"
W="$ROOT/runs/e2e/$TAG/${EXP}_s0/weights/best.pt"
if [ -f "$W" ]; then
    python scripts/latency.py --tag "$TAG" --out "$ROOT/results/e2e/$TAG/latency.json" --model "${EXP}short=$W:$IMGSZ"
    echo "== latency rc=$? $(date -Is)"
fi
echo "== end $(date -Is)"
