#!/usr/bin/env bash
# Download + unzip VisDrone2019-DET (train, val, test-dev). LOGIN node only (needs internet).
# Source URLs: ASSETS_URL in ultralytics/cfg/datasets/VisDrone.yaml (ultralytics 8.4.202).
# Original zips are kept in datasets/_downloads until deletion is explicitly approved.
set -euo pipefail
ROOT="${ROOT:-$HOME/yolo-robust}"
source "$ROOT/scripts/env.sh"

BASE="https://github.com/ultralytics/assets/releases/download/v0.0.0"
DL="$ROOT/datasets/_downloads"
RAW="$ROOT/datasets/VisDrone/raw"
MAN="$ROOT/results/manifests/visdrone_zips.txt"
mkdir -p "$DL" "$RAW" "$(dirname "$MAN")"

# Expected sizes from HTTP Content-Length (checked 2026-11-04).
declare -A EXPECTED=(
    [train]=1549875511
    [val]=81638851
    [test-dev]=311251787
)

: > "$MAN"
for split in train val test-dev; do
    f="$DL/VisDrone2019-DET-$split.zip"
    if [ ! -f "$f" ] || [ "$(stat -c %s "$f")" != "${EXPECTED[$split]}" ]; then
        curl -L --fail --retry 3 -C - -o "$f" "$BASE/VisDrone2019-DET-$split.zip"
    fi
    size=$(stat -c %s "$f")
    if [ "$size" != "${EXPECTED[$split]}" ]; then
        echo "size mismatch for $f: $size != ${EXPECTED[$split]}" >&2
        exit 1
    fi
    sum=$(sha256sum "$f" | cut -d' ' -f1)
    echo "$split $size $sum" | tee -a "$MAN"
    if [ ! -d "$RAW/VisDrone2019-DET-$split" ]; then
        unzip -q -n "$f" -d "$RAW"
    fi
done
ls -la "$RAW"
du -sh "$DL" "$RAW"
echo "download_visdrone: OK"
