"""Organize extracted VisDrone2019-DET splits, write YOLO labels and verify them.

Input  (scripts/download_visdrone.sh): datasets/VisDrone/raw/VisDrone2019-DET-{train,val,test-dev}/{images,annotations}
Output (split dirs are moved out of raw/ by a directory rename, nothing is copied or deleted):
    datasets/VisDrone/<split>/images/*.jpg        original images
    datasets/VisDrone/<split>/annotations/*.txt   original VisDrone ground truth (kept for COCO-style eval)
    datasets/VisDrone/<split>/labels/*.txt        YOLO labels, project rules (src/visdrone.py)
    results/manifests/visdrone_stats.json

Every label file is checked with Ultralytics' own `verify_image_label`. For comparison, the labels the
stock Ultralytics converter (cfg/datasets/VisDrone.yaml) would produce are checked the same way.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image
from ultralytics.data.utils import verify_image_label

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from visdrone import (  # noqa: E402
    NUM_CLASSES,
    convert_boxes,
    format_rows,
    parse_annotation_text,
    ultralytics_reference_rows,
)

EXPECTED_IMAGES = {"train": 6471, "val": 548, "test-dev": 1610}


def ultra_verify(img: Path, lb: Path) -> tuple[int, int, str]:
    """Return (corrupt_flag, n_labels_kept, msg) from ultralytics.data.utils.verify_image_label."""
    r = verify_image_label((str(img), str(lb), "", False, NUM_CLASSES, 0, 0, False))
    labels, nc, msg = r[1], r[8], r[9]
    return int(nc), (0 if labels is None else len(labels)), msg or ""


def reason(msg: str) -> str:
    """Compact corruption/warning reason for counting."""
    for key in ("exceeds dataset class count", "non-normalized or out of bounds", "negative label values", "duplicate labels removed", "corrupt JPEG"):
        if key in msg:
            return key
    return msg[:80]


def process_split(vis_root: Path, split: str, tmp: Path) -> dict:
    d = vis_root / split
    img_dir, ann_dir, lb_dir = d / "images", d / "annotations", d / "labels"
    lb_dir.mkdir(exist_ok=True)
    images = sorted(img_dir.glob("*.jpg"))
    anns = sorted(ann_dir.glob("*.txt"))
    stem_imgs, stem_anns = {p.stem for p in images}, {p.stem for p in anns}

    total: Counter = Counter()
    sizes: Counter = Counter()
    empty_after = 0

    def work(img: Path):
        with Image.open(img) as im:
            w, h = im.size
        boxes = parse_annotation_text((ann_dir / f"{img.stem}.txt").read_text(encoding="utf-8"))
        rows, st = convert_boxes(boxes, w, h)
        lb = lb_dir / f"{img.stem}.txt"
        lb.write_text(format_rows(rows), encoding="utf-8")
        ours = ultra_verify(img, lb)
        ref_file = tmp / f"{split}_{img.stem}.txt"
        ref_file.write_text(format_rows(ultralytics_reference_rows(boxes, w, h)), encoding="utf-8")
        ref = ultra_verify(img, ref_file)
        ref_file.unlink()  # per-image scratch file created just above
        return (w, h), st, ours, ref

    ours_corrupt, ref_corrupt = 0, 0
    ours_reasons, ref_reasons = Counter(), Counter()
    ours_boxes, ref_boxes = 0, 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for (w, h), st, ours, ref in ex.map(work, images):
            sizes[f"{w}x{h}"] += 1
            total.update(st)
            empty_after += st["kept"] == 0
            ours_corrupt += ours[0]
            ref_corrupt += ref[0]
            ours_boxes += ours[1]
            ref_boxes += ref[1]
            if ours[2]:
                ours_reasons[reason(ours[2])] += 1
            if ref[2]:
                ref_reasons[reason(ref[2])] += 1

    return {
        "images": len(images),
        "expected_images": EXPECTED_IMAGES[split],
        "annotation_files": len(anns),
        "images_without_annotation": len(stem_imgs - stem_anns),
        "annotations_without_image": len(stem_anns - stem_imgs),
        "label_files": len(list(lb_dir.glob("*.txt"))),
        "images_with_no_kept_box": empty_after,
        "rows": dict(sorted(total.items())),
        "image_sizes_top5": dict(sizes.most_common(5)),
        "ultralytics_verify_project_labels": {
            "corrupt_images": ours_corrupt,
            "boxes_after_verify": ours_boxes,
            "messages": dict(ours_reasons),
        },
        "ultralytics_verify_stock_converter_labels": {
            "corrupt_images": ref_corrupt,
            "boxes_after_verify": ref_boxes,
            "messages": dict(ref_reasons),
        },
    }


def main() -> None:
    root = Path(os.environ["ROOT"]).resolve()
    vis_root = root / "datasets" / "VisDrone"
    raw = vis_root / "raw"
    for split in EXPECTED_IMAGES:
        src, dst = raw / f"VisDrone2019-DET-{split}", vis_root / split
        if not dst.exists():
            assert src.is_dir(), f"missing {src}"
            src.rename(dst)

    out = {}
    with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="visdrone_ref_") as tmp:
        for split in EXPECTED_IMAGES:
            out[split] = process_split(vis_root, split, Path(tmp))
            print(split, json.dumps(out[split], indent=1), flush=True)

    dst = root / "results" / "manifests" / "visdrone_stats.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    ok = all(out[s]["images"] == EXPECTED_IMAGES[s] == out[s]["annotation_files"] == out[s]["label_files"] for s in out)
    ok &= all(out[s]["ultralytics_verify_project_labels"]["corrupt_images"] == 0 for s in out)
    print("prepare_visdrone:", "OK" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
