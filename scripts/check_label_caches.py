"""Check (default) or rebuild (--rebuild, login node, one process) the shared Ultralytics label caches.

    python scripts/check_label_caches.py            # exit 1 if datasets/VisDrone/{train,val}/labels.cache is stale
    python scripts/check_label_caches.py --rebuild  # rewrite stale caches (never while jobs are starting)
    python scripts/check_label_caches.py --test-dev # also datasets/VisDrone/test-dev (Stage 6 evaluation jobs)

Why: ultralytics 8.4.202 save_dataset_cache_file() does `if path.exists(): path.unlink()` before writing, so
jobs that start at the same time and all find a stale cache race on the unlink (FileNotFoundError; jobs 555219 and
555221). A cache goes stale whenever the image list differs, e.g. after a `fraction` run or a subset evaluation.
Check mode builds the dataset with Ultralytics' own validation logic but with cache_labels() disabled, so it
never writes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG

ROOT = Path(os.environ["ROOT"]).resolve()


class _CheckOnly(YOLODataset):
    def cache_labels(self, path=Path("./labels.cache")):
        raise RuntimeError(f"stale or missing label cache: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--data", default=str(ROOT / "configs" / "visdrone.yaml"))
    ap.add_argument("--test-dev", action="store_true", help="also check/rebuild the test-dev cache (Stage 6)")
    a = ap.parse_args()

    data = check_det_dataset(a.data)
    hyp = get_cfg(DEFAULT_CFG)
    cls = YOLODataset if a.rebuild else _CheckOnly
    # test-dev is not in visdrone.yaml (by design); same image dir string as scripts/evaluate.py builds
    paths = {"train": data["train"], "val": data["val"]}
    if a.test_dev:
        paths["test-dev"] = str(ROOT / "datasets" / "VisDrone" / "test-dev" / "images")
    bad = []
    for split, img_path in paths.items():
        try:
            ds = cls(img_path=img_path, imgsz=640, augment=False, hyp=hyp, data=data, task="detect", prefix=f"{split}: ")
            print(f"{split}: label cache OK ({len(ds)} images)")
        except RuntimeError as e:
            bad.append(split)
            print(f"{split}: {e}")
    if bad:
        print("rebuild on the login node with: python scripts/check_label_caches.py --rebuild" + (" --test-dev" if a.test_dev else ""))
        sys.exit(1)


if __name__ == "__main__":
    main()
