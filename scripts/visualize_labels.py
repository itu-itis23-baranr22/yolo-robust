"""Draw YOLO-format labels on a few random images for a visual sanity check.

Example:
    python scripts/visualize_labels.py --images datasets/VisDrone/images/val \
        --labels datasets/VisDrone/labels/val --names configs/visdrone_names.txt \
        --n 6 --seed 0 --out results/label_checks/val
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def read_yolo_labels(path: Path) -> np.ndarray:
    """Return an (N, 5) float array [cls, xc, yc, w, h] (normalized); empty if no file."""
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float64)
    arr = np.loadtxt(path, ndmin=2, dtype=np.float64)
    assert arr.shape[1] == 5, f"{path}: expected 5 columns, got {arr.shape[1]}"
    return arr


def draw(img: np.ndarray, labels: np.ndarray, names: list[str]) -> np.ndarray:
    h, w = img.shape[:2]
    rng = np.random.default_rng(12345)
    colors = [tuple(int(c) for c in rng.integers(64, 256, 3)) for _ in range(max(len(names), 1))]
    out = img.copy()
    for cls, xc, yc, bw, bh in labels:
        c = int(cls)
        x1, y1 = int(round((xc - bw / 2) * w)), int(round((yc - bh / 2) * h))
        x2, y2 = int(round((xc + bw / 2) * w)), int(round((yc + bh / 2) * h))
        color = colors[c % len(colors)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        name = names[c] if c < len(names) else f"cls{c}"
        cv2.putText(out, name, (x1, max(y1 - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--names", type=Path, required=True, help="text file, one class name per line")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1600, help="downscale saved previews")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names = [s.strip() for s in args.names.read_text().splitlines() if s.strip()]
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMG_EXTS)
    random.Random(args.seed).shuffle(images)
    args.out.mkdir(parents=True, exist_ok=True)

    for p in images[: args.n]:
        img = cv2.imread(str(p))
        assert img is not None, f"cannot read {p}"
        labels = read_yolo_labels(args.labels / f"{p.stem}.txt")
        vis = draw(img, labels, names)
        s = args.max_side / max(vis.shape[:2])
        if s < 1:
            vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        dst = args.out / f"{p.stem}.jpg"
        cv2.imwrite(str(dst), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"{dst}  boxes={len(labels)}  size={img.shape[1]}x{img.shape[0]}")


if __name__ == "__main__":
    main()
