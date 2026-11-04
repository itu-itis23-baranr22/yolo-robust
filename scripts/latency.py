"""Batch-1 latency on the GPU, after warm-up, with torch.cuda.synchronize(). Same precision for every model.

    python scripts/latency.py --model E1=runs/train/E1_s0/weights/best.pt:640 --model E3=...:960 --tag stage4

Per image (val images, preloaded as BGR arrays, so disk I/O is excluded):
  * Ultralytics' own per-stage timings (ops.Profile syncs the device): preprocess, inference, postprocess
    (postprocess contains NMS for YOLO11; YOLO26 with nms=False uses its NMS-free head + top-k)
  * wall: whole predict() call bracketed by torch.cuda.synchronize()
  * gpu_forward_event: CUDA-event time between forward pre-/post-hooks of the network (GPU timeline;
    includes GPU idle gaps while the CPU launches kernels, so it equals launch-bound time when overhead-bound)
Also: parameter count, peak allocated VRAM, mean detections. Settings: deploy (conf 0.25, max_det 300) and
eval (conf 0.001, max_det 1000).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ["ROOT"]).resolve()
sys.path.insert(0, str(ROOT / "src"))
import runmeta  # noqa: E402

SETTINGS = {"deploy": {"conf": 0.25, "max_det": 300}, "eval": {"conf": 0.001, "max_det": 1000}}


def stats(a) -> dict:
    a = np.asarray(a, dtype=float)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "n": int(a.size)}


def measure(weights: str, imgsz: int, imgs: list, warmup: int, quantize: int, setting: dict) -> dict:
    from ultralytics import YOLO

    m = YOLO(weights)
    pk = dict(imgsz=imgsz, quantize=quantize, iou=0.7, nms=False, device=0, verbose=False, **setting)
    for im in imgs[:warmup]:
        m.predict(im, **pk)
    net = m.predictor.model  # AutoBackend
    ev: dict = {}

    def pre(mod, inp):
        ev["s"] = torch.cuda.Event(enable_timing=True)
        ev["s"].record()

    def post(mod, inp, out):
        ev["e"] = torch.cuda.Event(enable_timing=True)
        ev["e"].record()

    h1, h2 = net.register_forward_pre_hook(pre), net.register_forward_hook(post)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    try:
        for im in imgs[warmup:]:
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = m.predict(im, **pk)[0]
            torch.cuda.synchronize()
            wall = (time.perf_counter() - t) * 1e3
            rows.append({**r.speed, "wall": wall, "gpu_forward_event": ev["s"].elapsed_time(ev["e"]), "n_det": len(r.boxes)})
    finally:
        h1.remove()
        h2.remove()
    inner = m.model
    out = {
        "weights": weights,
        "imgsz": imgsz,
        "params": sum(p.numel() for p in inner.parameters()),
        "end2end": bool(getattr(inner, "end2end", False)),
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
        "predict_args": pk,
        **{k: stats([r[k] for r in rows]) for k in ("preprocess", "inference", "postprocess", "wall", "gpu_forward_event", "n_det")},
    }
    out["sum_of_stages_mean"] = out["preprocess"]["mean"] + out["inference"]["mean"] + out["postprocess"]["mean"]
    del m
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="NAME=weights:imgsz (repeatable)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--quantize", type=int, default=16, choices=[16, 32])
    ap.add_argument("--settings", default="deploy,eval")
    ap.add_argument("--out", help="default results/latency/<tag>.json")
    a = ap.parse_args()

    import cv2

    assert torch.cuda.is_available(), "latency needs a GPU"
    files = sorted((ROOT / "datasets" / "VisDrone" / "val" / "images").glob("*.jpg"))[: a.n + a.warmup]
    imgs = [cv2.imread(str(p)) for p in files]
    res: dict = {}
    for spec in a.model:
        name, rest = spec.split("=", 1)
        w, imgsz = rest.rsplit(":", 1)
        for sname in a.settings.split(","):
            key = f"{name}_{sname}"
            res[key] = measure(str(Path(w).resolve()), int(imgsz), imgs, a.warmup, a.quantize, SETTINGS[sname])
            r = res[key]
            print(f"[{key}] wall p50 {r['wall']['p50']:.2f} ms (pre {r['preprocess']['mean']:.2f} / inf {r['inference']['mean']:.2f} "
                  f"/ post {r['postprocess']['mean']:.2f}; gpu-fwd {r['gpu_forward_event']['p50']:.2f}) params {r['params']}", flush=True)  # fmt: skip
    out = Path(a.out) if a.out else ROOT / "results" / "latency" / f"{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = runmeta.collect(ROOT, config=vars(a), extra={"settings": SETTINGS, "n_images": len(imgs) - a.warmup, "unit": "ms"})
    out.write_text(json.dumps({"meta": meta, "results": res}, indent=2, default=str) + "\n")
    print(f"results: {out}")


if __name__ == "__main__":
    main()
