#!/usr/bin/env bash
# Create the isolated project venv. Run on the LOGIN node (needs internet).
#   bash scripts/setup_env.sh
set -euo pipefail
ROOT="${ROOT:-$HOME/yolo-robust}"
source "$ROOT/scripts/env.sh"

VENV="$ROOT/.venv"
if [ ! -x "$VENV/bin/python" ]; then
    python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -V

# 1) torch/torchvision from the cu124 index (driver 555.42.02 supports up to CUDA 12.5).
"$PY" -m pip install --no-cache-dir \
    "torch==2.6.0" "torchvision==0.21.0" \
    --index-url https://download.pytorch.org/whl/cu124

# 2) Everything else from PyPI; constraints keep torch/torchvision pinned.
"$PY" -m pip install --no-cache-dir -r "$ROOT/requirements.txt" -c "$ROOT/constraints.txt"

"$PY" -m pip check
"$PY" -m pip freeze --all > "$ROOT/requirements.lock"
du -sh "$VENV"
echo "setup_env: OK"
