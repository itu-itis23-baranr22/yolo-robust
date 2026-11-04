"""Estimate disk needed to store corrupted copies of a split (JPEG and PNG), from a random image sample.

    python scripts/estimate_corruption_disk.py --split val --n 24
Encodes in memory only; writes nothing except results/manifests/corruption_disk_estimate_<split>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2

ROOT = Path(os.environ["ROOT"]).resolve()
sys.path.insert(0, str(ROOT / "src"))
import corruptions as C  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = C.load_config()
    q = int(cfg["storage"]["jpeg_quality"])
    imgs = sorted((ROOT / "datasets" / "VisDrone" / a.split / "images").glob("*.jpg"))
    orig_total = sum(p.stat().st_size for p in imgs)
    sample = random.Random(a.seed).sample(imgs, min(a.n, len(imgs)))
    s_orig = sum(p.stat().st_size for p in sample)

    sets = [("clean_reencoded", 0)] + [(t, s) for t in C.TYPES for s in (1, 2, 3)]
    jpg = {f"{t}_s{s}": 0 for t, s in sets}
    png = dict.fromkeys(jpg, 0)
    for p in sample:
        im = cv2.imread(str(p))
        for t, s in sets:
            out = im if t == "clean_reencoded" else C.corrupt(im, t, s, p.stem, cfg)
            k = f"{t}_s{s}"
            jpg[k] += len(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])[1])
            png[k] += len(cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 3])[1])

    est = lambda d: {k: v / s_orig * orig_total for k, v in d.items()}
    ej, ep = est(jpg), est(png)
    corrupted_keys = [k for k in jpg if not k.startswith("clean")]
    out = {
        "split": a.split,
        "n_images_split": len(imgs),
        "n_sample": len(sample),
        "orig_split_bytes": orig_total,
        "jpeg_quality": q,
        "per_set_gb_jpg": {k: round(v / 1e9, 3) for k, v in ej.items()},
        "per_set_gb_png": {k: round(v / 1e9, 3) for k, v in ep.items()},
        "total_15_sets_gb_jpg": round(sum(ej[k] for k in corrupted_keys) / 1e9, 2),
        "total_15_sets_gb_png": round(sum(ep[k] for k in corrupted_keys) / 1e9, 2),
        "clean_reencoded_gb_jpg": round(ej["clean_reencoded_s0"] / 1e9, 3),
    }
    dst = ROOT / "results" / "manifests" / f"corruption_disk_estimate_{a.split}.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
