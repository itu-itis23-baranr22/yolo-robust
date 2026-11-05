"""Stage 6 freeze: the exact checkpoints (sha256) and protocol files that are evaluated once on test-dev.

    python scripts/freeze_models.py --write            # results/manifests/final_models.json (login node)
    python scripts/freeze_models.py --verify E4_s0     # exit 1 unless best.pt matches the frozen sha256
    python scripts/freeze_models.py --specs [--reverse]  # NAME=weights:imgsz lines for scripts/latency.py

Frozen set = every completed training run (all experiments, all seeds): no checkpoint is chosen with test-dev.
The "best model" is declared here, before test-dev, by the approved criterion: highest mean val mPC (mAP50-95)
over seeds. Also recorded: sha256 of the evaluation code/configs and the test-dev GT counts (label/annotation
check only; no model is run here).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ["ROOT"]).resolve()
sys.path.insert(0, str(ROOT / "src"))
RUN_RE = re.compile(r"^(E\d+)_s(\d+)$")
MANIFEST = ROOT / "results" / "manifests" / "final_models.json"
PROTOCOL_FILES = ("scripts/evaluate.py", "scripts/latency.py", "src/coco_eval.py", "src/robust_data.py",
                  "src/corruptions.py", "src/robustness.py", "src/visdrone.py", "configs/corruptions.yaml",
                  "configs/experiments.yaml", "configs/visdrone.yaml", "requirements.lock")  # fmt: skip


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def completed_runs() -> list[dict]:
    import yaml

    exps = yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text())["experiments"]
    out = []
    for d in sorted((ROOT / "results" / "train").iterdir(), key=lambda p: p.name):
        m = RUN_RE.match(d.name)
        if not m or not (d / "meta.json").exists():
            continue
        if json.loads((d / "meta.json").read_text())["extra"].get("status") != "completed":
            continue
        w = ROOT / "runs" / "train" / d.name / "weights" / "best.pt"
        if not w.exists():
            raise SystemExit(f"completed run without best.pt: {w}")
        exp = m.group(1)
        out.append({"run": d.name, "exp": exp, "seed": int(m.group(2)), "model": exps[exp]["model"],
                    "imgsz": int(exps[exp]["imgsz"]), "augment": exps[exp]["augment"],
                    "weights": str(w.relative_to(ROOT))})  # fmt: skip
    out.sort(key=lambda r: (int(r["exp"][1:]), r["seed"]))
    return out


def write() -> None:
    import coco_eval

    runs = completed_runs()
    for r in runs:
        p = ROOT / r["weights"]
        r["sha256"], r["bytes"] = sha256(p), p.stat().st_size
        s = json.loads((ROOT / "results" / "eval" / r["run"] / "val" / "summary.json").read_text())
        r["val"] = {k: s[k][q] for k in ("mAP50_95", "AP_small") for q in ("clean", "mPC", "rPC")} | {
            "mAP50_95_clean": s["mAP50_95"]["clean"], "mAP50_95_mPC": s["mAP50_95"]["mPC"]}  # fmt: skip
    by_exp: dict[str, list[float]] = {}
    for r in runs:
        by_exp.setdefault(r["exp"], []).append(r["val"]["mAP50_95_mPC"])
    mean_mpc = {e: st.mean(v) for e, v in by_exp.items()}
    best = max(mean_mpc, key=mean_mpc.get)

    vd = ROOT / "datasets" / "VisDrone" / "test-dev"
    stems = sorted(p.stem for p in (vd / "images").glob("*.jpg"))
    gt = coco_eval.gt_counts(coco_eval.build_gt(vd, stems))
    git = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    man = {
        "purpose": "Stage 6: checkpoints and protocol frozen before the single test-dev evaluation",
        "git_commit_before_freeze": git,
        "selection": {"criterion": "highest mean val mPC (mAP50-95) over seeds (approved in Stage 5)",
                      "val_mean_mPC_by_exp": mean_mpc, "best_experiment": best,
                      "seeds_per_exp": {e: len(v) for e, v in by_exp.items()}},  # fmt: skip
        "protocol_sha256": {f: sha256(ROOT / f) for f in PROTOCOL_FILES},
        "test_dev": {"images": len(stems), "gt_counts": gt},
        "runs": runs,
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps({"runs": [r["run"] for r in runs], "best_experiment": best, "val_mean_mPC": mean_mpc,
                      "test_dev_gt": gt}, indent=1))  # fmt: skip


def verify(run: str) -> None:
    man = json.loads(MANIFEST.read_text())
    r = next((x for x in man["runs"] if x["run"] == run), None)
    if r is None:
        raise SystemExit(f"{run} is not in the frozen set")
    got = sha256(ROOT / r["weights"])
    bad = [f for f, h in man["protocol_sha256"].items() if sha256(ROOT / f) != h]
    if got != r["sha256"] or bad:
        raise SystemExit(f"FROZEN MISMATCH {run}: weights {'OK' if got == r['sha256'] else 'CHANGED'}, protocol files changed: {bad}")
    print(f"{run}: weights + protocol match the frozen manifest")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--verify", metavar="RUN")
    g.add_argument("--specs", action="store_true")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    if a.write:
        write()
    elif a.verify:
        verify(a.verify)
    else:
        runs = json.loads(MANIFEST.read_text())["runs"] if MANIFEST.exists() else completed_runs()
        for r in reversed(runs) if a.reverse else runs:
            print(f"{r['run']}={ROOT / r['weights']}:{r['imgsz']}")


if __name__ == "__main__":
    main()
