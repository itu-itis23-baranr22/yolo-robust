"""Evaluate one checkpoint on clean + corrupted copies of a split (corruptions applied on the fly, nothing stored).

    python scripts/evaluate.py --weights runs/train/E3_s0/weights/best.pt --imgsz 960 --tag E3_s0
    python scripts/evaluate.py ... --subset 100 --sets clean,noise_s3 --backend both     # quick check
    python scripts/evaluate.py ... --split test-dev --final                                # Stage 6 only

Per set: Ultralytics val (conf=0.001, iou=0.7, max_det=1000, nms=False, FP16, rect batches, save_json) ->
predictions.json (original-pixel boxes) -> src/coco_eval.py with VisDrone ignored regions.
Outputs (small, committed): results/eval/<tag>/<split>[_subN]/{<set>.json, breakdown.csv, summary.json, meta.json}
Raw predictions stay in runs/eval/<tag>/... (ignored by git).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ["ROOT"]).resolve()
sys.path.insert(0, str(ROOT / "src"))
import coco_eval  # noqa: E402
import robust_data  # noqa: E402  (installs the imread dispatcher before any dataset is built)
import robustness as RB  # noqa: E402
import runmeta  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402

VD = ROOT / "datasets" / "VisDrone"
VAL_ARGS = dict(conf=0.001, iou=0.7, max_det=1000, nms=False, quantize=16, rect=True, save_json=True, plots=False)


def clean_json(o):
    """NaN -> None so the files are strict JSON."""
    if isinstance(o, float):
        return None if math.isnan(o) else o
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    return o


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(obj), indent=2, default=str) + "\n")


def select_stems(split: str, subset: int | None) -> list[str]:
    stems = sorted(p.stem for p in (VD / split / "images").glob("*.jpg"))
    if subset:
        stems = sorted(random.Random(0).sample(stems, subset))
    return stems


def write_data_yaml(run_root: Path, split: str, stems: list[str], subset: int | None) -> Path:
    """Data YAML whose `val` entry is the evaluated split (a list file for subsets)."""
    names = YAML.load(ROOT / "configs" / "visdrone.yaml")["names"]
    if subset:
        lst = run_root / "images.txt"
        lst.parent.mkdir(parents=True, exist_ok=True)
        lst.write_text("".join(f"{VD / split / 'images' / s}.jpg\n" for s in stems))
        val = str(lst)
    else:
        val = str(VD / split / "images")
    y = run_root / "data.yaml"
    y.parent.mkdir(parents=True, exist_ok=True)
    YAML.save(y, {"path": str(VD), "train": "train/images", "val": val, "names": names})
    return y


def git_clean() -> bool:
    out = subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True)
    return out.strip() == ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--tag", required=True, help="results/eval/<tag>/...")
    ap.add_argument("--split", default="val", choices=["val", "test-dev"])
    ap.add_argument("--sets", default="all", help="'all' or comma list, e.g. clean,noise_s3")
    ap.add_argument("--subset", type=int, help="evaluate a fixed random subset of N images (seed 0)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--backend", default="faster", choices=["faster", "pycocotools", "both"])
    ap.add_argument("--final", action="store_true", help="required for test-dev (Stage 6, run once)")
    ap.add_argument("--results-root", default=str(ROOT / "results" / "eval"), help="default results/eval")
    ap.add_argument("--device", default="0", help="'0' (GPU) or 'cpu' (dry runs only)")
    a = ap.parse_args()

    if a.split == "test-dev":
        if not a.final or a.subset:
            raise SystemExit("test-dev is evaluated only once, in full, with --final (Stage 6)")
        if not git_clean():
            raise SystemExit("test-dev requires a clean, committed tree (frozen selections)")
    sets = RB.ALL_SETS if a.sets == "all" else a.sets.split(",")
    for s in sets:
        RB.parse_set(s)

    from ultralytics import YOLO

    split_key = f"{a.split}_sub{a.subset}" if a.subset else a.split
    res_root = Path(a.results_root).resolve() / a.tag / split_key
    run_root = ROOT / "runs" / "eval" / a.tag / split_key
    stems = select_stems(a.split, a.subset)
    data_yaml = write_data_yaml(run_root, a.split, stems, a.subset)
    gt = coco_eval.build_gt(VD / a.split, stems)
    cfg = robust_data.C.load_config()

    weights = Path(a.weights).resolve()
    meta = runmeta.collect(ROOT, config={"weights": str(weights), "imgsz": a.imgsz, "split": a.split, "subset": a.subset,
                                         "sets": sets, "val_args": VAL_ARGS, "batch": a.batch, "backend": a.backend,
                                         "corruptions_version": cfg["version"]},
                           extra={"gt_counts": coco_eval.gt_counts(gt), "final": a.final})  # fmt: skip
    dump(res_root / "meta.json", meta)

    model = YOLO(str(weights))
    per_set: dict[str, dict] = {}
    for s in sets:
        ctype, sev = RB.parse_set(s)
        validator = robust_data.make_validator(None if ctype == RB.CLEAN else ctype, sev, cfg)
        t0 = time.perf_counter()
        r = model.val(validator=validator, data=str(data_yaml), split="val", imgsz=a.imgsz, batch=a.batch,
                      workers=a.workers, device=a.device, project=str(run_root), name=s, exist_ok=True, verbose=False,
                      **VAL_ARGS)  # fmt: skip
        t_val = time.perf_counter() - t0
        pred_json = Path(r.save_dir) / "predictions.json"
        dets = coco_eval.load_predictions(pred_json, gt) if pred_json.exists() else []
        backends = ["faster", "pycocotools"] if a.backend == "both" else [a.backend]
        out = {"set": s, "type": ctype, "severity": sev, "val_wall_s": t_val, "speed_ms_per_img": r.speed,
               "ultralytics_metrics": {k: float(v) for k, v in r.results_dict.items()},
               "predictions_json": str(pred_json)}  # fmt: skip
        for b in backends:
            t1 = time.perf_counter()
            m = coco_eval.evaluate(gt, dets, backend=b)
            m["eval_wall_s"] = time.perf_counter() - t1
            out[b] = m
        primary = out[backends[0]]
        if a.backend == "both":
            keys = [k for k, v in out["faster"].items() if isinstance(v, float) and k != "eval_wall_s"]
            out["backend_max_abs_diff"] = max(abs(out["faster"][k] - out["pycocotools"][k]) for k in keys)
        out.update({k: v for k, v in primary.items() if k not in ("backend",)})
        per_set[s] = out
        dump(res_root / f"{s}.json", out)
        print(f"[{s}] mAP50-95={out['mAP50_95']:.4f} mAP50={out['mAP50']:.4f} AP_small={out['AP_small']:.4f} "
              f"(ultralytics mAP50-95={r.box.map:.4f}) val {t_val:.1f}s", flush=True)  # fmt: skip

    RB.write_csv(res_root / "breakdown.csv", RB.breakdown_rows(per_set))
    if all(s in per_set for s in RB.ALL_SETS):
        summ = RB.summarize(per_set)
        dump(res_root / "summary.json", summ)
        print(f"mPC={summ['mAP50_95']['mPC']:.4f} rPC={summ['mAP50_95']['rPC']:.4f} clean={summ['mAP50_95']['clean']:.4f}")
    print(f"results: {res_root}")


if __name__ == "__main__":
    main()
