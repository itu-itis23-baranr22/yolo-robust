"""Stage-2 smoke test. Run inside a SLURM GPU job after `source scripts/env.sh` with YOLO_OFFLINE=true.

    python scripts/smoke_test.py <step> --tag <jobid>

steps:
  env         GPU / driver / offline checks + run metadata
  throughput  short trainings (fraction of train, full val) -> epoch time, it/s, peak VRAM
  resume      start 3-epoch run, SIGKILL it inside epoch 2, resume from last.pt, check epochs/rows
  val         standalone val with save_json -> predictions.json sanity (boxes in original pixels)
  latency     batch=1 predict, warm-up, CUDA-synchronized timing, FP32 and FP16

Outputs: results/smoke/<tag>/<step>.json (small, committed) and runs/smoke/<tag>/ (ignored).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ["ROOT"]).resolve()
sys.path.insert(0, str(ROOT / "src"))
import runmeta  # noqa: E402

DATA = str(ROOT / "configs" / "visdrone.yaml")
W = {"yolo11s": str(ROOT / "weights" / "yolo11s.pt"), "yolo26s": str(ROOT / "weights" / "yolo26s.pt")}

# Hyper-parameters shared by all compared runs (smoke uses them too, except epochs/fraction).
COMMON = dict(
    data=DATA,
    batch=16,
    workers=16,
    seed=0,
    deterministic=True,
    optimizer="SGD",
    lr0=0.01,
    momentum=0.937,
    amp=True,
    cache=False,
    plots=True,
)


def dump(out: Path, name: str, obj: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(obj, indent=2, default=str) + "\n")
    print(json.dumps(obj, indent=2, default=str), flush=True)


def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=float), q))


def summarize(a) -> dict:
    a = np.asarray(a, dtype=float)
    return {"mean": float(a.mean()), "p50": pct(a, 50), "p95": pct(a, 95), "n": int(a.size)}


# --------------------------------------------------------------------------------------------- env
def step_env(out: Path, tag: str) -> None:
    from ultralytics.utils import ONLINE, SETTINGS

    meta = runmeta.collect(ROOT, extra={"step": "env"})
    meta["ultralytics_ONLINE"] = ONLINE
    meta["ultralytics_settings"] = {k: SETTINGS[k] for k in ("datasets_dir", "weights_dir", "runs_dir", "sync")}
    import socket

    try:
        socket.create_connection(("pypi.org", 443), timeout=5).close()
        meta["pypi_tcp_443"] = "reachable"
    except OSError as e:
        meta["pypi_tcp_443"] = f"unreachable: {e.__class__.__name__}: {e}"
    if torch.cuda.is_available():
        x = torch.randn(4096, 4096, device="cuda")
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            y = x @ x
        torch.cuda.synchronize()
        meta["matmul_4096_fp32_ms"] = (time.perf_counter() - t) / 10 * 1e3
        del x, y
    dump(out, "env", meta)


# -------------------------------------------------------------------------------------- throughput
def timed_train(weights: str, project: Path, name: str, **kw) -> dict:
    from ultralytics import YOLO

    m = YOLO(weights)
    ev: dict[str, list] = {
        k: [] for k in ("ep_start", "tr_end", "val_start", "val_end", "peak_gb", "iters", "imgs", "batch")
    }

    def sync():
        torch.cuda.synchronize()
        return time.perf_counter()

    def on_train_epoch_start(tr):
        torch.cuda.reset_peak_memory_stats()
        ev["ep_start"].append(sync())

    def on_train_epoch_end(tr):
        ev["tr_end"].append(sync())
        ev["peak_gb"].append(torch.cuda.max_memory_reserved() / 1024**3)
        ev["iters"].append(len(tr.train_loader))
        ev["imgs"].append(len(tr.train_loader.dataset))
        ev["batch"].append(tr.batch_size)  # actual batch (Ultralytics may shrink it after an OOM)

    m.add_callback("on_train_epoch_start", on_train_epoch_start)
    m.add_callback("on_train_epoch_end", on_train_epoch_end)
    m.add_callback("on_val_start", lambda v: ev["val_start"].append(sync()))
    m.add_callback("on_val_end", lambda v: ev["val_end"].append(sync()))

    t0 = time.perf_counter()
    m.train(project=str(project), name=name, exist_ok=False, **kw)
    total = time.perf_counter() - t0

    epochs = []
    for i in range(len(ev["tr_end"])):
        tr_s = ev["tr_end"][i] - ev["ep_start"][i]
        va_s = ev["val_end"][i] - ev["val_start"][i] if i < len(ev["val_end"]) else None
        epochs.append(
            {
                "train_s": tr_s,
                "val_s": va_s,
                "iters": ev["iters"][i],
                "imgs": ev["imgs"][i],
                "batch": ev["batch"][i],
                "it_per_s": ev["iters"][i] / tr_s,
                "img_per_s": ev["imgs"][i] / tr_s,
                "peak_reserved_gb": ev["peak_gb"][i],
            }
        )
    last = epochs[-1]
    full_train_s = last["train_s"] * 6471 / last["imgs"]
    return {
        "weights": weights,
        "args": {k: v for k, v in kw.items() if k != "data"},
        "total_wall_s": total,
        "epochs": epochs,
        "val_calls_s": [e - s for s, e in zip(ev["val_start"], ev["val_end"])],
        "est_full_epoch_train_s": full_train_s,
        "est_full_epoch_train_plus_val_s": full_train_s + (last["val_s"] or 0),
        "save_dir": str(project / name),
    }


def step_throughput(out: Path, tag: str) -> None:
    project = ROOT / "runs" / "smoke" / tag
    cfgs = [
        ("tp_y11s_640", "yolo11s", 640, {}),
        ("tp_y26s_640", "yolo26s", 640, {}),
        ("tp_y26s_960", "yolo26s", 960, {}),
        ("tp_y26s_960_musgd", "yolo26s", 960, {"optimizer": "MuSGD", "momentum": 0.9}),
        ("tp_y26s_960_b32", "yolo26s", 960, {"batch": 32}),
    ]
    res = {}
    for name, model, imgsz, over in cfgs:
        kw = {**COMMON, "imgsz": imgsz, "epochs": 2, "fraction": 0.25, **over}
        try:
            res[name] = timed_train(W[model], project, name, **kw)
        except Exception as e:  # keep going; report the failure
            res[name] = {"error": f"{e.__class__.__name__}: {e}"}
        torch.cuda.empty_cache()
        dump(out, "throughput", res)


# ------------------------------------------------------------------------------------------ resume
def read_csv_rows(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


def step_resume(out: Path, tag: str) -> None:
    from ultralytics import YOLO

    project = ROOT / "runs" / "smoke" / tag
    run = project / "resume"
    kw = {**COMMON, "imgsz": 640, "epochs": 3, "fraction": 0.15, "save_period": 1, "plots": False}
    code = (
        "from ultralytics import YOLO; "
        f"YOLO({W['yolo26s']!r}).train(project={str(project)!r}, name='resume', exist_ok=False, **{kw!r})"
    )
    log = open(out / "resume_child.log", "w")
    p = subprocess.Popen([sys.executable, "-c", code], start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
    res_csv, last = run / "results.csv", run / "weights" / "last.pt"
    t0, killed = time.time(), False
    while p.poll() is None and time.time() - t0 < 1800:
        rows = read_csv_rows(res_csv)
        if rows and last.exists() and last.stat().st_mtime >= res_csv.stat().st_mtime:
            time.sleep(3)  # last.pt written after epoch 1 -> now inside epoch 2
            os.killpg(p.pid, signal.SIGKILL)
            killed = True
            break
        time.sleep(1)
    p.wait()
    log.close()
    before = {
        "killed": killed,
        "child_returncode": p.returncode,
        "csv_rows": len(read_csv_rows(res_csv)),
        "weights": sorted(x.name for x in (run / "weights").glob("*.pt")),
        "last_pt_epoch": torch.load(last, map_location="cpu", weights_only=False).get("epoch") if last.exists() else None,
    }

    t1 = time.perf_counter()
    YOLO(str(last)).train(resume=True)
    rows = read_csv_rows(res_csv)
    after = {
        "resume_wall_s": time.perf_counter() - t1,
        "csv_epochs": [r.get("epoch") for r in rows],
        "weights": sorted(x.name for x in (run / "weights").glob("*.pt")),
        "final_last_pt_epoch": torch.load(last, map_location="cpu", weights_only=False).get("epoch"),
    }
    ok = killed and before["csv_rows"] == 1 and after["csv_epochs"] == ["1", "2", "3"]
    dump(out, "resume", {"before_kill": before, "after_resume": after, "pass": ok})


# --------------------------------------------------------------------------------------------- val
def step_val(out: Path, tag: str) -> None:
    from PIL import Image
    from ultralytics import YOLO

    project = ROOT / "runs" / "smoke" / tag
    w = project / "tp_y26s_960" / "weights" / "best.pt"
    m = YOLO(str(w))
    t = time.perf_counter()
    r = m.val(data=DATA, imgsz=960, batch=16, conf=0.001, iou=0.7, half=True, save_json=True, plots=True,
              project=str(project), name="val_y26s_960", exist_ok=False)
    wall = time.perf_counter() - t
    pj = Path(r.save_dir) / "predictions.json"
    preds = json.loads(pj.read_text()) if pj.exists() else []
    # are boxes in original-image pixels? compare max x2/y2 against the native image size
    val_dir = ROOT / "datasets" / "VisDrone" / "val" / "images"
    sizes, beyond, max_rel = {}, 0, 0.0
    for d in preds:
        stem = str(d["image_id"])
        if stem not in sizes:
            with Image.open(val_dir / f"{stem}.jpg") as im:
                sizes[stem] = im.size
        W_, H_ = sizes[stem]
        x, y, bw, bh = d["bbox"]
        max_rel = max(max_rel, (x + bw) / W_, (y + bh) / H_)
        beyond += (x + bw > W_ + 1) or (y + bh > H_ + 1)
    dump(
        out,
        "val",
        {
            "weights": str(w),
            "wall_s": wall,
            "speed_ms_per_img": r.speed,
            "map50_95": float(r.box.map),
            "map50": float(r.box.map50),
            "predictions_json": str(pj),
            "n_predictions": len(preds),
            "n_images_with_preds": len(sizes),
            "image_id_example": preds[0]["image_id"] if preds else None,
            "bbox_max_rel_extent": max_rel,
            "bbox_beyond_image": beyond,
            "note": "1-2 epoch smoke model: mAP values are meaningless, only the pipeline is tested",
        },
    )


# ----------------------------------------------------------------------------------------- latency
def step_latency(out: Path, tag: str, n: int = 100, warmup: int = 20) -> None:
    import cv2
    from ultralytics import YOLO

    project = ROOT / "runs" / "smoke" / tag
    val_dir = ROOT / "datasets" / "VisDrone" / "val" / "images"
    imgs = [cv2.imread(str(p)) for p in sorted(val_dir.glob("*.jpg"))[: n + warmup]]
    cases = [("tp_y11s_640", 640), ("tp_y26s_640", 640), ("tp_y26s_960", 960)]
    res = {}
    for name, imgsz in cases:
        w = project / name / "weights" / "best.pt"
        for half in (False, True):
            key = f"{name}_{'fp16' if half else 'fp32'}"
            m = YOLO(str(w))
            pk = dict(imgsz=imgsz, half=half, conf=0.25, iou=0.7, max_det=300, device=0, verbose=False)
            for im in imgs[:warmup]:
                m.predict(im, **pk)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            rows = []
            for im in imgs[warmup:]:
                torch.cuda.synchronize()
                t = time.perf_counter()
                r = m.predict(im, **pk)[0]
                torch.cuda.synchronize()
                rows.append({**r.speed, "wall": (time.perf_counter() - t) * 1e3, "n_det": len(r.boxes)})
            res[key] = {
                "weights": str(w),
                "params": sum(p.numel() for p in m.model.parameters()),
                "end2end": bool(getattr(m.model, "end2end", False)),
                "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
                **{k: summarize([r[k] for r in rows]) for k in ("preprocess", "inference", "postprocess", "wall", "n_det")},
            }
            del m
            torch.cuda.empty_cache()
    dump(out, "latency", {"n_images": n, "warmup": warmup, "conf": 0.25, "unit": "ms", "results": res})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["env", "throughput", "resume", "val", "latency"])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out = ROOT / "results" / "smoke" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    {"env": step_env, "throughput": step_throughput, "resume": step_resume, "val": step_val, "latency": step_latency}[a.step](
        out, a.tag
    )


if __name__ == "__main__":
    main()
