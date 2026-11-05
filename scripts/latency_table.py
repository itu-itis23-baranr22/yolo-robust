"""Latency table from the one-node job (scripts/slurm/latency_all.sbatch): both passes, per run and per experiment.

    python scripts/latency_table.py --job 556084     # reads results/latency/all_<job>_{fwd,rev}.json
                                                     # -> results/summary/latency_<job>.{csv,md}
Per run and setting: the mean of the two passes (forward / reverse model order) of batch-1 wall p50/p95,
Ultralytics' inference mean and the CUDA-event forward p50; |fwd − rev| shows order/thermal drift.
Per experiment: mean over seeds of the run values (the seeds share an architecture, so this mainly averages
measurement noise). Batch-16 numbers are not here: they come from the val runs (results/summary/val.md).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics as st
from pathlib import Path

ROOT = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[1])).resolve()
RUN_RE = re.compile(r"^(E\d+)_s(\d+)_(deploy|eval)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    a = ap.parse_args()
    passes = {p: json.loads((ROOT / "results" / "latency" / f"all_{a.job}_{p}.json").read_text()) for p in ("fwd", "rev")}
    host = {p: d["meta"]["host"] for p, d in passes.items()}
    gpu = passes["fwd"]["meta"]["gpus"][0]["name"]
    assert host["fwd"] == host["rev"], host
    rows = []
    for key, f in passes["fwd"]["results"].items():
        m = RUN_RE.match(key)
        r = passes["rev"]["results"][key]
        row = {"run": f"{m.group(1)}_s{m.group(2)}", "exp": m.group(1), "setting": m.group(3), "imgsz": f["imgsz"],
               "params_M": f["params"] / 1e6, "peak_vram_mb": max(f["peak_allocated_mb"], r["peak_allocated_mb"])}  # fmt: skip
        for name, path in (("wall_p50", ("wall", "p50")), ("wall_p95", ("wall", "p95")),
                           ("pre_mean", ("preprocess", "mean")), ("inf_mean", ("inference", "mean")),
                           ("post_mean", ("postprocess", "mean")), ("gpu_fwd_p50", ("gpu_forward_event", "p50")),
                           ("n_det_mean", ("n_det", "mean"))):  # fmt: skip
            x, y = f[path[0]][path[1]], r[path[0]][path[1]]
            row[name] = (x + y) / 2
            if name == "wall_p50":
                row["wall_p50_abs_fwd_rev"] = abs(x - y)
        rows.append(row)
    out = ROOT / "results" / "summary"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"latency_{a.job}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

    exps = sorted({r["exp"] for r in rows}, key=lambda e: int(e[1:]))
    lines = [f"Batch-1 latency, one job ({a.job}) on one GPU ({host['fwd']}, {gpu}), FP16, eager PyTorch, 300 val images "
             "after 50 warm-up; mean of two passes in opposite model order; per experiment = mean over seeds (n).", "",
             "| | " + " | ".join(f"{e} (n={len({r['run'] for r in rows if r['exp'] == e})})" for e in exps) + " |",
             "|---|" + "---|" * len(exps)]  # fmt: skip
    for setting, label in (("deploy", "deploy (conf 0.25, max_det 300)"), ("eval", "eval (conf 0.001, max_det 1000)")):
        for k, lab in (("wall_p50", "wall p50 (ms)"), ("wall_p95", "wall p95 (ms)"), ("pre_mean", "preprocess (ms)"),
                       ("inf_mean", "inference (ms)"), ("post_mean", "postprocess (ms)"), ("gpu_fwd_p50", "GPU forward p50 (ms)")):  # fmt: skip
            vals = []
            for e in exps:
                v = [r[k] for r in rows if r["exp"] == e and r["setting"] == setting]
                vals.append(f"{st.mean(v):.2f}")
            lines.append(f"| {label}: {lab} | " + " | ".join(vals) + " |")
    lines.append("| parameters (M) | " + " | ".join(
        f"{st.mean([r['params_M'] for r in rows if r['exp'] == e]):.2f}" for e in exps) + " |")  # fmt: skip
    drift = max(rows, key=lambda r: r["wall_p50_abs_fwd_rev"])
    lines += ["", f"Largest |forward − reverse| difference of a run's wall p50: {drift['wall_p50_abs_fwd_rev']:.2f} ms "
              f"({drift['run']}, {drift['setting']}). Peak VRAM is not reported from this multi-model process (earlier "
              "models' allocations inflate the peak); see the single-model per-run files results/latency/<run>.json."]  # fmt: skip
    md = "\n".join(lines) + "\n"
    (out / f"latency_{a.job}.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
