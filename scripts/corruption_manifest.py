"""Reproducibility manifest for the on-the-fly corrupted sets (no corrupted image is stored).

    python scripts/corruption_manifest.py --split val --procs 32 [--samples 2]

For every image x 15 corrupted sets: sha256 of the corrupted uint8 array (shape + raw bytes) as fed to the
pipeline (corruptions.corrupt on the decoded original JPEG). Re-running later must give identical hashes;
--check compares against the committed manifest instead of writing it.
Outputs: results/manifests/corruptions_<split>_sha256.csv (stem,set,sha256)
         results/manifests/corruptions_<split>_digest.json (one digest per set + config sha256)
         results/figures/corruption_grid_<stem>.jpg with --samples N (downscaled, for the README)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ["ROOT"]).resolve()
sys.path.insert(0, str(ROOT / "src"))
import corruptions as C  # noqa: E402
import robustness as RB  # noqa: E402

_CFG = None


def array_sha256(a: np.ndarray) -> str:
    h = hashlib.sha256(f"{a.shape}|{a.dtype}|".encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _init():
    global _CFG
    cv2.setNumThreads(1)
    _CFG = C.load_config()


def _hash_one(path: str) -> list[tuple[str, str, str]]:
    p = Path(path)
    im = cv2.imread(str(p))
    out = [(p.stem, RB.CLEAN, array_sha256(im))]
    for s in RB.CORRUPTED_SETS:
        t, sev = RB.parse_set(s)
        out.append((p.stem, s, array_sha256(C.corrupt(im, t, sev, p.stem, _CFG))))
    return out


def grid(path: Path, cfg: dict, tile_w: int = 340) -> np.ndarray:
    im = cv2.imread(str(path))
    tiles = []
    for s in RB.ALL_SETS:
        t, sev = RB.parse_set(s)
        x = im if s == RB.CLEAN else C.corrupt(im, t, sev, path.stem, cfg)
        x = cv2.resize(x, (tile_w, round(im.shape[0] * tile_w / im.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.putText(x, s, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(x, s, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(x)
    rows = [np.hstack(tiles[i : i + 4]) for i in range(0, 16, 4)]
    return np.vstack(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test-dev"])
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    imgs = sorted((ROOT / "datasets" / "VisDrone" / a.split / "images").glob("*.jpg"))
    t0 = time.perf_counter()
    with Pool(a.procs, initializer=_init) as pool:
        rows = [r for chunk in pool.imap(_hash_one, [str(p) for p in imgs], chunksize=4) for r in chunk]
    elapsed = time.perf_counter() - t0
    rows.sort()

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["stem", "set", "sha256"])
    w.writerows(rows)
    digests = {}
    for s in RB.ALL_SETS:
        h = hashlib.sha256()
        for stem, set_, d in rows:
            if set_ == s:
                h.update(f"{stem}:{d}\n".encode())
        digests[s] = h.hexdigest()
    cfg_sha = hashlib.sha256(C.CFG_PATH.read_bytes()).hexdigest()
    man = ROOT / "results" / "manifests"
    csv_path, dig_path = man / f"corruptions_{a.split}_sha256.csv", man / f"corruptions_{a.split}_digest.json"
    summary = {"split": a.split, "images": len(imgs), "sets": len(RB.ALL_SETS), "rows": len(rows),
               "corruptions_yaml_sha256": cfg_sha, "set_digests": digests}  # fmt: skip

    if a.check:
        ref = json.loads(dig_path.read_text())
        bad = [s for s in RB.ALL_SETS if ref["set_digests"][s] != digests[s]]
        print(json.dumps({"check": "PASS" if not bad else "FAIL", "mismatched_sets": bad, "elapsed_s": elapsed}))
        sys.exit(1 if bad else 0)

    man.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(buf.getvalue())
    dig_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({**{k: v for k, v in summary.items() if k != "set_digests"}, "elapsed_s": round(elapsed, 1)}))

    if a.samples:
        import random

        cfg = C.load_config()
        fig = ROOT / "results" / "figures"
        fig.mkdir(parents=True, exist_ok=True)
        for p in random.Random(0).sample(imgs, a.samples):
            dst = fig / f"corruption_grid_{p.stem}.jpg"
            cv2.imwrite(str(dst), grid(p, cfg), [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"figure: {dst} ({dst.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
