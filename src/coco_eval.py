"""COCO-style box evaluation for VisDrone with ignored regions, in original image pixels.

Protocol (fixed before any evaluation):
  * GT boxes = the boxes used as training labels (`visdrone.eval_boxes`), category_id = class + 1.
  * Every non-evaluated VisDrone row (score 0 / ignored region / "others") becomes a crowd box
    (iscrowd=1) replicated for all 10 categories -> class-agnostic ignore: a detection whose
    intersection-over-detection-area with it is >= the IoU threshold is neither TP nor FP
    (standard COCOeval crowd rule; crowd boxes may absorb any number of detections).
  * Areas (COCO small < 32^2 <= medium < 96^2 <= large) use original-pixel box areas: GT areas come
    from the raw annotations, detection areas from predictions.json, which Ultralytics writes in
    original-image coordinates.
  * maxDets = (1, 10, 100, 1000); headline numbers use maxDets=1000 because VisDrone images hold up to
    902 objects and models are run with max_det=1000. AP@maxDets=100 is reported for reference.
  * Values are read directly from COCOeval.eval['precision'] / ['recall'] (pycocotools' summarize()
    hard-codes the maxDets index, so it is not used).
This is COCO-style evaluation, not the official VisDrone MATLAB toolkit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import visdrone as V

MAX_DETS = (1, 10, 100, 1000)
AREA_RNG = {"all": [0.0, 1e10], "small": [0.0, 32.0**2], "medium": [32.0**2, 96.0**2], "large": [96.0**2, 1e10]}
IOU_THRS = np.linspace(0.5, 0.95, 10)
CATEGORY_IDS = list(range(1, V.NUM_CLASSES + 1))


# ------------------------------------------------------------------------------------------------ ground truth
def gt_from_boxes(images: list[dict]) -> dict:
    """images: [{"stem", "width", "height", "kept": [(cls,x,y,w,h)], "ignore": [(x,y,w,h)]}] -> COCO dict.

    Image ids are 1..N in the given order; annotation ids start at 1 (pycocotools treats id 0 as unmatched).
    """
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": c, "name": V.CLASS_NAMES[c - 1]} for c in CATEGORY_IDS],
    }
    aid = 0
    for iid, im in enumerate(images, start=1):
        coco["images"].append({"id": iid, "file_name": f"{im['stem']}.jpg", "width": im["width"], "height": im["height"]})
        for cls, x, y, w, h in im["kept"]:
            aid += 1
            coco["annotations"].append(
                {"id": aid, "image_id": iid, "category_id": int(cls) + 1, "bbox": [x, y, w, h], "area": float(w * h), "iscrowd": 0}
            )
        for x, y, w, h in im["ignore"]:
            for c in CATEGORY_IDS:
                aid += 1
                coco["annotations"].append(
                    {"id": aid, "image_id": iid, "category_id": c, "bbox": [x, y, w, h], "area": float(w * h), "iscrowd": 1}
                )
    return coco


def build_gt(split_dir: Path, stems: list[str]) -> dict:
    """COCO GT for the given image stems of datasets/VisDrone/<split> (raw annotations + image sizes)."""
    from PIL import Image

    images = []
    for stem in stems:
        with Image.open(split_dir / "images" / f"{stem}.jpg") as im:
            w, h = im.size
        boxes = V.parse_annotation_text((split_dir / "annotations" / f"{stem}.txt").read_text())
        kept, ignore = V.eval_boxes(boxes, w, h)
        images.append({"stem": stem, "width": w, "height": h, "kept": kept, "ignore": ignore})
    return gt_from_boxes(images)


def gt_counts(gt: dict) -> dict:
    crowd = sum(a["iscrowd"] for a in gt["annotations"])
    return {
        "images": len(gt["images"]),
        "gt_boxes": len(gt["annotations"]) - crowd,
        "ignore_regions": crowd // len(CATEGORY_IDS),
    }


# ------------------------------------------------------------------------------------------------ predictions
def load_predictions(pred: str | Path | list, gt: dict) -> list[dict]:
    """Ultralytics predictions.json (image_id = file stem) -> list with the integer image ids of `gt`."""
    dets = json.loads(Path(pred).read_text()) if isinstance(pred, (str, Path)) else pred
    stem2id = {Path(im["file_name"]).stem: im["id"] for im in gt["images"]}
    out = []
    for d in dets:
        stem = str(d["image_id"])
        if stem not in stem2id:
            raise KeyError(f"prediction for unknown image {stem!r}")
        if d["category_id"] not in CATEGORY_IDS:
            raise ValueError(f"unexpected category_id {d['category_id']}")
        out.append({"image_id": stem2id[stem], "category_id": int(d["category_id"]), "bbox": list(d["bbox"]), "score": float(d["score"])})
    return out


# ------------------------------------------------------------------------------------------------ evaluation
def _cocoeval(gt: dict, dets: list[dict], backend: str):
    import copy

    if backend == "pycocotools":
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        g = COCO()
        g.dataset = copy.deepcopy(gt)
        g.createIndex()
        d = g.loadRes(copy.deepcopy(dets))
        E = COCOeval(g, d, iouType="bbox")
    elif backend == "faster":
        from faster_coco_eval import COCO, COCOeval_faster

        g = COCO(copy.deepcopy(gt))
        d = g.loadRes(copy.deepcopy(dets))
        E = COCOeval_faster(g, d, iouType="bbox", print_function=lambda *a, **k: None)
    else:
        raise ValueError(backend)
    E.params.imgIds = sorted(im["id"] for im in gt["images"])  # images without detections still count
    E.params.catIds = CATEGORY_IDS
    E.params.maxDets = list(MAX_DETS)
    E.params.areaRng = [AREA_RNG[k] for k in ("all", "small", "medium", "large")]
    E.params.areaRngLbl = ["all", "small", "medium", "large"]
    E.params.iouThrs = IOU_THRS.copy()
    E.evaluate()
    E.accumulate()
    return E


def _mean_valid(x: np.ndarray) -> float:
    x = x[x > -1]
    return float(x.mean()) if x.size else float("nan")


def summarize(precision: np.ndarray, recall: np.ndarray) -> dict:
    """precision [T,R,K,A,M], recall [T,K,A,M] (COCOeval layout; A = all/small/medium/large, M = MAX_DETS)."""
    A = {k: i for i, k in enumerate(("all", "small", "medium", "large"))}
    M = {m: i for i, m in enumerate(MAX_DETS)}
    t50, t75 = int(np.argmin(np.abs(IOU_THRS - 0.5))), int(np.argmin(np.abs(IOU_THRS - 0.75)))
    m = M[1000]
    out = {
        "mAP50_95": _mean_valid(precision[:, :, :, A["all"], m]),
        "mAP50": _mean_valid(precision[t50, :, :, A["all"], m]),
        "mAP75": _mean_valid(precision[t75, :, :, A["all"], m]),
        "AP_small": _mean_valid(precision[:, :, :, A["small"], m]),
        "AP_medium": _mean_valid(precision[:, :, :, A["medium"], m]),
        "AP_large": _mean_valid(precision[:, :, :, A["large"], m]),
        "AP50_small": _mean_valid(precision[t50, :, :, A["small"], m]),
        "AR_1000": _mean_valid(recall[:, :, A["all"], m]),
        "AR_small_1000": _mean_valid(recall[:, :, A["small"], m]),
        "mAP50_95_maxdet100": _mean_valid(precision[:, :, :, A["all"], M[100]]),
        "AR_100": _mean_valid(recall[:, :, A["all"], M[100]]),
    }
    out["per_class_mAP50_95"] = {
        V.CLASS_NAMES[k]: _mean_valid(precision[:, :, k, A["all"], m]) for k in range(precision.shape[2])
    }
    return out


def evaluate(gt: dict, dets: list[dict], backend: str = "faster") -> dict:
    """Return the metric dict for one prediction set. backend: 'faster' (faster-coco-eval) or 'pycocotools'."""
    if not dets:  # pycocotools.loadRes cannot handle an empty list; no detections -> zero precision/recall
        K, T, M = len(CATEGORY_IDS), len(IOU_THRS), len(MAX_DETS)
        res = summarize(np.zeros((T, 101, K, 4, M)), np.zeros((T, K, 4, M)))
    else:
        E = _cocoeval(gt, dets, backend)
        P, R = np.asarray(E.eval["precision"]), np.asarray(E.eval["recall"])
        assert P.shape == (len(IOU_THRS), 101, len(CATEGORY_IDS), 4, len(MAX_DETS)), P.shape
        res = summarize(P, R)
    res["counts"] = {**gt_counts(gt), "detections": len(dets)}
    res["backend"] = backend
    return res
