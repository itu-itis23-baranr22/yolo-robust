"""Robustness summaries: mPC (mean over 5 types x 3 severities) and rPC = mPC / clean, plus CSV breakdown."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import corruptions as C

CLEAN = "clean"
CORRUPTED_SETS = [f"{t}_s{s}" for t in C.TYPES for s in (1, 2, 3)]
ALL_SETS = [CLEAN, *CORRUPTED_SETS]
SUMMARY_METRICS = ("mAP50_95", "mAP50", "AP_small")


def parse_set(name: str) -> tuple[str, int | None]:
    if name == CLEAN:
        return CLEAN, None
    t, s = name.rsplit("_s", 1)
    assert t in C.TYPES and s in ("1", "2", "3"), name
    return t, int(s)


def summarize(per_set: dict[str, dict], metrics: tuple[str, ...] = SUMMARY_METRICS) -> dict:
    """per_set: {set_name: metric_dict}. Requires clean + all 15 corrupted sets (no partial mPC)."""
    missing = [s for s in ALL_SETS if s not in per_set]
    if missing:
        raise ValueError(f"missing sets for mPC: {missing}")
    out: dict = {}
    for m in metrics:
        clean = float(per_set[CLEAN][m])
        vals = [float(per_set[s][m]) for s in CORRUPTED_SETS]
        mpc = sum(vals) / len(vals)
        by_type = {t: sum(float(per_set[f"{t}_s{s}"][m]) for s in (1, 2, 3)) / 3 for t in C.TYPES}
        by_sev = {s: sum(float(per_set[f"{t}_s{s}"][m]) for t in C.TYPES) / len(C.TYPES) for s in (1, 2, 3)}
        out[m] = {
            "clean": clean,
            "mPC": mpc,
            "rPC": mpc / clean if clean > 0 else math.nan,
            "by_type": by_type,
            "rPC_by_type": {t: (v / clean if clean > 0 else math.nan) for t, v in by_type.items()},
            "by_severity": by_sev,
        }
    return out


BREAKDOWN_FIELDS = (
    "set", "type", "severity", "mAP50_95", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large",
    "AR_1000", "mAP50_95_maxdet100", "relative_to_clean",
)  # fmt: skip


def breakdown_rows(per_set: dict[str, dict]) -> list[dict]:
    clean = float(per_set[CLEAN]["mAP50_95"]) if CLEAN in per_set else math.nan
    rows = []
    for name in [s for s in ALL_SETS if s in per_set]:
        t, s = parse_set(name)
        r = per_set[name]
        row = {"set": name, "type": t, "severity": "" if s is None else s}
        row.update({k: r[k] for k in BREAKDOWN_FIELDS[3:-1]})
        row["relative_to_clean"] = float(r["mAP50_95"]) / clean if clean and clean > 0 else math.nan
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] = BREAKDOWN_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
