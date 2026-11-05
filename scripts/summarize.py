"""Aggregate per-run results into one table per split (no new measurement; reads committed result files only).

    python scripts/summarize.py --split val          # -> results/summary/val_runs.csv, val_experiments.csv, val.md
    python scripts/summarize.py --split test-dev

Per run (<EXP>_s<SEED>): results/eval/<run>/<split>/summary.json (+ clean.json for AP_medium/AP_large/AR),
results/train/<run>/results.csv (best epoch = max Ultralytics fitness 0.9*mAP50-95 + 0.1*mAP50, the rule
best.pt is saved by), results/train/<run>/meta.json (training wall time).
Per experiment: mean, sample standard deviation (n-1) and n over the available seeds. With n=3 the sd is a rough
estimate; differences are reported as mean differences only (no significance test).
Latency is NOT aggregated here: per-job latency files come from different nodes (~1 ms node-to-node spread);
the comparable numbers come from one job on one node (scripts/slurm/latency_all.sbatch).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics as st
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[1])).resolve()
RUN_RE = re.compile(r"^(E\d+)_s(\d+)$")
METRICS = ("mAP50_95", "mAP50", "AP_small")
TYPES = ("underexposed", "overexposed", "blur", "noise", "low_contrast")
CONTRASTS = [("E2", "E1", "model version @640"), ("E3", "E2", "input size 640 -> 960"),
             ("E4", "E3", "corruption-aware training"), ("E5", "E4", "P2 head (optional)")]  # fmt: skip


def best_epoch(results_csv: Path) -> tuple[int | None, float | None, int | None]:
    if not results_csv.exists():
        return None, None, None
    rows = list(csv.DictReader(open(results_csv)))
    fit = [(0.9 * float(r["metrics/mAP50-95(B)"].strip()) + 0.1 * float(r["metrics/mAP50(B)"].strip()),
            int(float(r["epoch"]))) for r in rows]  # fmt: skip
    f, e = max(fit)
    return e, f, len(rows)


def run_row(run: str, split: str) -> dict | None:
    m = RUN_RE.match(run)
    summ_p = ROOT / "results" / "eval" / run / split / "summary.json"
    if not m or not summ_p.exists():
        return None
    s = json.loads(summ_p.read_text())
    clean = json.loads((ROOT / "results" / "eval" / run / split / "clean.json").read_text())
    row: dict = {"run": run, "exp": m.group(1), "seed": int(m.group(2))}
    for k in METRICS:
        row[f"{k}_clean"] = s[k]["clean"]
        row[f"{k}_mPC"] = s[k]["mPC"]
        row[f"{k}_rPC"] = s[k]["rPC"]
    for t in TYPES:
        row[f"rPC_{t}"] = s["mAP50_95"]["rPC_by_type"][t]
    for sev in ("1", "2", "3"):
        row[f"mAP50_95_sev{sev}"] = s["mAP50_95"]["by_severity"][sev]
    for k in ("AP_medium", "AP_large", "AR_1000", "mAP50_95_maxdet100"):
        row[f"{k}_clean"] = clean[k]
    sp = s.get("val_speed_ms_per_img_clean", {})
    row["b16_inference_ms"] = sp.get("inference")
    row["b16_postprocess_ms"] = sp.get("postprocess")
    e, f, n = best_epoch(ROOT / "results" / "train" / run / "results.csv")
    row.update({"best_epoch": e, "best_fitness": f, "epochs_run": n})
    tm = ROOT / "results" / "train" / run / "meta.json"
    row["train_h"] = json.loads(tm.read_text())["extra"].get("wall_s", math.nan) / 3600 if tm.exists() else None
    return row


def agg(vals: list[float]) -> tuple[float, float | None, int]:
    vals = [v for v in vals if v is not None]
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else None), len(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test-dev"])
    ap.add_argument("--out", default=str(ROOT / "results" / "summary"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = sorted(p.name for p in (ROOT / "results" / "eval").iterdir() if p.is_dir())
    rows = [r for r in (run_row(x, a.split) for x in runs) if r]
    if not rows:
        raise SystemExit(f"no results for split {a.split}")
    fields = list(rows[0].keys())
    with open(out / f"{a.split}_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})

    num = [k for k in fields if k not in ("run", "exp", "seed")]
    exps = sorted({r["exp"] for r in rows}, key=lambda e: int(e[1:]))
    agg_rows, by_exp = [], {}
    for e in exps:
        rs = [r for r in rows if r["exp"] == e]
        d: dict = {"exp": e, "seeds": " ".join(str(r["seed"]) for r in rs), "n": len(rs)}
        for k in num:
            mu, sd, _ = agg([r[k] for r in rs])
            d[f"{k}_mean"], d[f"{k}_sd"] = mu, sd
        agg_rows.append(d)
        by_exp[e] = d
    with open(out / f"{a.split}_experiments.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        for r in agg_rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else ("" if v is None else v)) for k, v in r.items()})

    def ms(d, k, nd=4):
        mu, sd = d[f"{k}_mean"], d[f"{k}_sd"]
        return f"{mu:.{nd}f}" if sd is None else f"{mu:.{nd}f} ± {sd:.{nd}f}"

    lines = [f"Split `{a.split}`: mean ± sample sd over seeds (n per column header).", "",
             "| | " + " | ".join(f"{e} (n={by_exp[e]['n']})" for e in exps) + " |",
             "|---|" + "---|" * len(exps)]  # fmt: skip
    table = [("clean mAP50-95", "mAP50_95_clean"), ("clean mAP50", "mAP50_clean"), ("clean AP_small", "AP_small_clean"),
             ("mPC (mAP50-95)", "mAP50_95_mPC"), ("rPC (mAP50-95)", "mAP50_95_rPC"),
             ("AP_small mPC", "AP_small_mPC"), ("AP_small rPC", "AP_small_rPC"),
             *[(f"rPC {t}", f"rPC_{t}") for t in TYPES],
             ("mAP50-95 severity 3 (mean of 5 types)", "mAP50_95_sev3")]  # fmt: skip
    for label, k in table:
        lines.append(f"| {label} | " + " | ".join(ms(by_exp[e], k) for e in exps) + " |")
    if a.split == "val":
        lines.append("| best epoch | " + " | ".join(ms(by_exp[e], "best_epoch", 0) for e in exps) + " |")
        lines.append("| training time (h) | " + " | ".join(ms(by_exp[e], "train_h", 2) for e in exps) + " |")
    lines += ["", "Mean differences (A − B) for the studied variable:", "",
              "| contrast | Δ clean mAP50-95 | Δ mPC | Δ rPC | Δ clean AP_small | Δ AP_small mPC |", "|---|---|---|---|---|---|"]  # fmt: skip
    for x, y, what in CONTRASTS:
        if x in by_exp and y in by_exp:
            ks = ("mAP50_95_clean", "mAP50_95_mPC", "mAP50_95_rPC", "AP_small_clean", "AP_small_mPC")
            lines.append(f"| {x} − {y} ({what}) | " + " | ".join(f"{by_exp[x][k + '_mean'] - by_exp[y][k + '_mean']:+.4f}" for k in ks) + " |")
    md = "\n".join(lines) + "\n"
    (out / f"{a.split}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
