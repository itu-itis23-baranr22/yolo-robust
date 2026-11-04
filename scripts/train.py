"""Train one experiment of configs/experiments.yaml. Run inside a SLURM GPU job (env.sh + YOLO_OFFLINE=true).

    python scripts/train.py --exp E3 --seed 0                       # full run -> runs/train/E3_s0
    python scripts/train.py --exp E3 --seed 0 --resume              # continue runs/train/E3_s0/weights/last.pt
    python scripts/train.py --exp E4 --short e2e_123 --epochs 2 --fraction 0.25   # short check -> runs/e2e/e2e_123/E4_s0

Writes results/train/<run>/meta.json (config, seed, git, packages, SLURM, GPU, dataset class, epoch timings) and
copies args.yaml / results.csv there at the end.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(os.environ["ROOT"]).resolve()
sys.path.insert(0, str(ROOT / "src"))
import runmeta  # noqa: E402
from ultralytics.utils import YAML  # noqa: E402


def load_experiment(exp: str) -> tuple[dict, dict]:
    cfg = YAML.load(ROOT / "configs" / "experiments.yaml")
    if exp not in cfg["experiments"]:
        raise SystemExit(f"unknown experiment {exp}; have {list(cfg['experiments'])}")
    common = dict(cfg["common"])
    common["data"] = str(ROOT / common["data"])
    return common, dict(cfg["experiments"][exp])


def trainer_for(spec: dict):
    if spec["augment"] == "corruption":
        import robust_data

        return robust_data.CorruptionAugTrainer
    assert spec["augment"] == "standard", spec["augment"]
    return None


def add_timing_callbacks(model, record: dict) -> None:
    def now():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    ep: dict = {}

    def on_train_start(tr):
        ds = tr.train_loader.dataset
        record["train_dataset_class"] = type(ds).__name__
        corr = getattr(ds, "corruption", None)
        record["train_corruption"] = corr.describe() if corr is not None else None
        record["train_images"] = len(ds)
        record["batch_actual"] = tr.batch_size

    def on_train_epoch_start(tr):
        torch.cuda.reset_peak_memory_stats()
        ep["t0"] = now()

    def on_train_epoch_end(tr):
        ep["t_train"] = now() - ep["t0"]
        ep["peak"] = torch.cuda.max_memory_reserved() / 1024**3

    def on_fit_epoch_end(tr):
        record["epochs"].append(
            {
                "epoch": tr.epoch + 1,
                "train_s": ep.get("t_train"),
                "train_plus_val_s": now() - ep["t0"],
                "img_per_s": len(tr.train_loader.dataset) / ep["t_train"] if ep.get("t_train") else None,
                "peak_reserved_gb": ep.get("peak"),
            }
        )

    for name, fn in (
        ("on_train_start", on_train_start),
        ("on_train_epoch_start", on_train_epoch_start),
        ("on_train_epoch_end", on_train_epoch_end),
        ("on_fit_epoch_end", on_fit_epoch_end),
    ):
        model.add_callback(name, fn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--short", metavar="TAG", help="short check run under runs/e2e/<TAG> (requires --epochs)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--fraction", type=float)
    a = ap.parse_args()

    from ultralytics import YOLO

    common, spec = load_experiment(a.exp)
    if spec.get("optional") and not os.environ.get("ALLOW_OPTIONAL_EXPERIMENTS"):
        raise SystemExit(f"{a.exp} is optional and needs explicit approval (set ALLOW_OPTIONAL_EXPERIMENTS=1)")
    name = f"{a.exp}_s{a.seed}"
    if a.short:
        if not a.epochs:
            raise SystemExit("--short requires --epochs")
        project, res_dir = ROOT / "runs" / "e2e" / a.short, ROOT / "results" / "e2e" / a.short / "train" / name
    else:
        if a.epochs or a.fraction:
            raise SystemExit("--epochs/--fraction overrides are only allowed with --short")
        project, res_dir = ROOT / "runs" / "train", ROOT / "results" / "train" / name
    run_dir = project / name

    kw = {**common, "imgsz": spec["imgsz"], "seed": a.seed}
    if a.epochs:
        kw["epochs"] = a.epochs
    if a.fraction:
        kw["fraction"] = a.fraction
    trainer = trainer_for(spec)

    record: dict = {"epochs": []}
    meta_path = res_dir / "meta.json"
    t0 = time.time()
    if a.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"nothing to resume: {last}")
        model = YOLO(str(last))
        add_timing_callbacks(model, record)
        meta = runmeta.collect(ROOT, config={"experiment": a.exp, "spec": spec, "train_args": kw}, seed=a.seed,
                               extra={"status": "resumed", "resume_from": str(last), "trainer": getattr(trainer, "__name__", None)})  # fmt: skip
        runmeta.write(res_dir / f"meta_resume_{int(t0)}.json", meta)
        model.train(resume=True, trainer=trainer)
    else:
        if run_dir.exists():
            raise SystemExit(f"{run_dir} exists; use --resume or a new seed/tag (refusing to overwrite)")
        w = ROOT / "weights"
        model = YOLO(spec["model"] if spec["model"].endswith(".yaml") else str(w / spec["model"]))
        if spec.get("pretrained"):
            kw["pretrained"] = str(w / spec["pretrained"])
        add_timing_callbacks(model, record)
        meta = runmeta.collect(ROOT, config={"experiment": a.exp, "spec": spec, "train_args": kw}, seed=a.seed,
                               extra={"status": "started", "trainer": getattr(trainer, "__name__", None)})  # fmt: skip
        runmeta.write(meta_path, meta)
        model.train(project=str(project), name=name, exist_ok=False, trainer=trainer, **kw)

    meta["extra"].update({"status": "completed", "wall_s": time.time() - t0, **record})
    meta["extra"]["save_dir"] = str(model.trainer.save_dir)
    runmeta.write(meta_path if not a.resume else res_dir / f"meta_resume_{int(t0)}.json", meta)
    for f in ("args.yaml", "results.csv"):
        src = Path(model.trainer.save_dir) / f
        if src.exists():
            shutil.copy2(src, res_dir / f)
    print(f"done: {model.trainer.save_dir}  meta: {res_dir}")


if __name__ == "__main__":
    main()
