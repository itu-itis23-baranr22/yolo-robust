"""VisDrone2019-DET annotation parsing and conversion to YOLO labels.

VisDrone-DET ground-truth rows:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
score: 1 = box is evaluated, 0 = box is ignored.
object_category: 0 ignored region, 1..10 the 10 evaluated classes, 11 "others".

Project rules (see README):
  * score == 0 or category 0  -> ignored region: not written to YOLO labels (kept in raw annotations).
  * category 11 ("others")    -> not written to YOLO labels (10-class setup; kept in raw annotations).
  * boxes are clipped to the image; boxes with no area after clipping are dropped.
  * class id = category - 1 (0..9), same mapping as ultralytics/cfg/datasets/VisDrone.yaml.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]
NUM_CLASSES = len(CLASS_NAMES)
IGNORED_CATEGORY = 0
OTHERS_CATEGORY = 11

# COCO area thresholds (pixels^2), applied in ORIGINAL image resolution.
COCO_SMALL_MAX = 32**2
COCO_MEDIUM_MAX = 96**2


@dataclass(frozen=True)
class VDBox:
    x: int
    y: int
    w: int
    h: int
    score: int
    category: int
    truncation: int
    occlusion: int


def parse_line(line: str) -> VDBox | None:
    """Parse one annotation row; returns None for blank lines. Tolerates a trailing comma."""
    parts = [p.strip() for p in line.strip().split(",")]
    while parts and parts[-1] == "":
        parts.pop()
    if not parts:
        return None
    if len(parts) < 8:
        raise ValueError(f"expected 8 fields, got {len(parts)}: {line!r}")
    v = [int(p) for p in parts[:8]]
    return VDBox(*v)


def parse_annotation_text(text: str) -> list[VDBox]:
    out = []
    for line in text.splitlines():
        b = parse_line(line)
        if b is not None:
            out.append(b)
    return out


def coco_area_bucket(area: float) -> str:
    if area < COCO_SMALL_MAX:
        return "small"
    if area < COCO_MEDIUM_MAX:
        return "medium"
    return "large"


def convert_boxes(boxes: list[VDBox], img_w: int, img_h: int) -> tuple[list[tuple[int, float, float, float, float]], Counter]:
    """Convert VisDrone boxes to YOLO (cls, xc, yc, w, h) normalized rows using the project rules."""
    assert img_w > 0 and img_h > 0
    stats: Counter = Counter()
    rows = []
    for b in boxes:
        stats["rows"] += 1
        if b.score == 0 or b.category == IGNORED_CATEGORY:
            stats["ignored_region"] += 1
            continue
        if b.category == OTHERS_CATEGORY:
            stats["others"] += 1
            continue
        if not 1 <= b.category <= NUM_CLASSES:
            raise ValueError(f"unknown category {b.category}")
        x1, y1 = max(0, b.x), max(0, b.y)
        x2, y2 = min(img_w, b.x + b.w), min(img_h, b.y + b.h)
        if (x1, y1, x2, y2) != (b.x, b.y, b.x + b.w, b.y + b.h):
            stats["clipped"] += 1
        if x2 <= x1 or y2 <= y1:
            stats["degenerate"] += 1
            continue
        w, h = x2 - x1, y2 - y1
        rows.append((b.category - 1, (x1 + w / 2) / img_w, (y1 + h / 2) / img_h, w / img_w, h / img_h))
        stats["kept"] += 1
        stats[f"cls_{b.category - 1}"] += 1
        stats[f"area_{coco_area_bucket(w * h)}"] += 1
    return rows, stats


def _clip_xywh(b: VDBox, img_w: int, img_h: int) -> tuple[int, int, int, int] | None:
    x1, y1 = max(0, b.x), max(0, b.y)
    x2, y2 = min(img_w, b.x + b.w), min(img_h, b.y + b.h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def eval_boxes(
    boxes: list[VDBox], img_w: int, img_h: int
) -> tuple[list[tuple[int, int, int, int, int]], list[tuple[int, int, int, int]]]:
    """Ground truth for COCO-style evaluation, in ORIGINAL image pixels.

    Returns (kept, ignore):
      kept   = [(cls, x, y, w, h)]: exactly the boxes `convert_boxes` writes as YOLO labels (same rules, same clipping).
      ignore = [(x, y, w, h)]: class-agnostic ignored areas = every row that is not evaluated (score == 0,
               category 0 "ignored region", category 11 "others"), clipped; zero-area rows dropped.
    """
    kept, ignore = [], []
    for b in boxes:
        xywh = _clip_xywh(b, img_w, img_h)
        if b.score == 0 or b.category in (IGNORED_CATEGORY, OTHERS_CATEGORY):
            if xywh is not None:
                ignore.append(xywh)
            continue
        if not 1 <= b.category <= NUM_CLASSES:
            raise ValueError(f"unknown category {b.category}")
        if xywh is not None:
            kept.append((b.category - 1, *xywh))
    return kept, ignore


def ultralytics_reference_rows(boxes: list[VDBox], img_w: int, img_h: int) -> list[tuple[int, float, float, float, float]]:
    """Rows exactly as `visdrone2yolo` in ultralytics 8.4.202 cfg/datasets/VisDrone.yaml would write them
    (skip score == 0, cls = category - 1, no clipping). Used only for comparison."""
    dw, dh = 1.0 / img_w, 1.0 / img_h
    rows = []
    for b in boxes:
        if b.score != 0:
            rows.append((b.category - 1, (b.x + b.w / 2) * dw, (b.y + b.h / 2) * dh, b.w * dw, b.h * dh))
    return rows


def format_rows(rows: list[tuple[int, float, float, float, float]]) -> str:
    return "".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for c, xc, yc, w, h in rows)
