import math

import numpy as np
import pytest

import coco_eval as CE

BACKENDS = ["faster", "pycocotools"]


def img(stem, kept, ignore=(), w=1360, h=765):
    return {"stem": stem, "width": w, "height": h, "kept": list(kept), "ignore": list(ignore)}


def perfect(gt, score=0.9):
    stem = {im["id"]: im["file_name"][:-4] for im in gt["images"]}
    return [
        {"image_id": stem[a["image_id"]], "category_id": a["category_id"], "bbox": a["bbox"], "score": score}
        for a in gt["annotations"]
        if not a["iscrowd"]
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_perfect_predictions_give_ap_1(backend):
    gt = CE.gt_from_boxes([img("a", [(0, 10, 10, 50, 40), (3, 200, 100, 120, 80)]), img("b", [(9, 5, 5, 20, 20)])])
    r = CE.evaluate(gt, CE.load_predictions(perfect(gt), gt), backend)
    assert r["mAP50_95"] == pytest.approx(1.0) and r["mAP50"] == pytest.approx(1.0)
    assert r["counts"] == {"images": 2, "gt_boxes": 3, "ignore_regions": 0, "detections": 3}


@pytest.mark.parametrize("backend", BACKENDS)
def test_detection_inside_ignored_region_is_not_a_false_positive(backend):
    kept = [(0, 10, 10, 30, 30)]
    # same class as the GT, higher score, fully inside the ignored area
    det_in_ignore = {"image_id": "a", "category_id": 1, "bbox": [510, 310, 40, 40], "score": 0.99}
    no_ignore = CE.gt_from_boxes([img("a", kept)])
    with_ignore = CE.gt_from_boxes([img("a", kept, ignore=[(500, 300, 200, 200)])])
    d0 = CE.load_predictions(perfect(no_ignore) + [det_in_ignore], no_ignore)
    d1 = CE.load_predictions(perfect(with_ignore) + [det_in_ignore], with_ignore)
    assert CE.evaluate(no_ignore, d0, backend)["mAP50_95"] == pytest.approx(0.5)  # FP ranked first -> precision 1/2
    assert CE.evaluate(with_ignore, d1, backend)["mAP50_95"] == pytest.approx(1.0)
    assert with_ignore["annotations"][-1]["iscrowd"] == 1
    assert CE.gt_counts(with_ignore)["ignore_regions"] == 1  # replicated for 10 categories, counted once


@pytest.mark.parametrize("backend", BACKENDS)
def test_area_ranges_use_original_pixel_areas(backend):
    gt = CE.gt_from_boxes([img("a", [(0, 0, 0, 31, 31), (1, 100, 100, 33, 33)])])  # 961 px^2 small, 1089 medium
    r = CE.evaluate(gt, CE.load_predictions(perfect(gt), gt), backend)
    assert r["AP_small"] == pytest.approx(1.0) and r["AP_medium"] == pytest.approx(1.0)
    assert math.isnan(r["AP_large"])
    assert r["per_class_mAP50_95"]["pedestrian"] == pytest.approx(1.0)
    gt_small = CE.gt_from_boxes([img("a", [(0, 0, 0, 31, 31)])])
    r = CE.evaluate(gt_small, CE.load_predictions(perfect(gt_small), gt_small), backend)
    assert math.isnan(r["AP_medium"]) and r["AP_small"] == pytest.approx(1.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_max_dets_1000_counts_more_than_100_objects(backend):
    rng = np.random.default_rng(0)
    kept = [(3, int(x), int(y), 12, 12) for x, y in zip(rng.integers(0, 1340, 150), rng.integers(0, 750, 150))]
    gt = CE.gt_from_boxes([img("a", kept)])
    dets = [dict(d, score=float(s)) for d, s in zip(perfect(gt), rng.uniform(0.1, 1.0, 150))]
    r = CE.evaluate(gt, CE.load_predictions(dets, gt), backend)
    assert r["mAP50_95"] == pytest.approx(1.0)  # maxDets 1000
    assert r["mAP50_95_maxdet100"] < 0.75  # recall capped at 100/150


def test_backends_agree_on_random_data_with_crowds():
    rng = np.random.default_rng(1)
    images, dets = [], []
    for i in range(25):
        kept = []
        for _ in range(int(rng.integers(0, 40))):
            w, h = (int(v) for v in rng.integers(4, 120, 2))
            kept.append((int(rng.integers(0, 10)), int(rng.integers(0, 1200)), int(rng.integers(0, 700)), w, h))
        ignore = [(int(rng.integers(0, 1100)), int(rng.integers(0, 600)), 150, 100) for _ in range(int(rng.integers(0, 3)))]
        images.append(img(f"im{i}", kept, ignore))
        for cls, x, y, w, h in kept:
            if rng.random() < 0.8:
                j = rng.normal(0, 3, 4)
                dets.append({"image_id": f"im{i}", "category_id": cls + 1, "bbox": [x + j[0], y + j[1], max(1, w + j[2]), max(1, h + j[3])], "score": float(rng.random())})
        for _ in range(int(rng.integers(0, 30))):  # false positives, some inside ignore areas
            dets.append({"image_id": f"im{i}", "category_id": int(rng.integers(1, 11)), "bbox": [float(rng.integers(0, 1200)), float(rng.integers(0, 700)), 20.0, 20.0], "score": float(rng.random())})
    gt = CE.gt_from_boxes(images)
    d = CE.load_predictions(dets, gt)
    a, b = CE.evaluate(gt, d, "faster"), CE.evaluate(gt, d, "pycocotools")
    for k, v in a.items():
        if isinstance(v, float):
            assert v == pytest.approx(b[k], abs=1e-9, nan_ok=True), k
    for c, v in a["per_class_mAP50_95"].items():
        assert v == pytest.approx(b["per_class_mAP50_95"][c], abs=1e-9, nan_ok=True), c


def test_load_predictions_validates():
    gt = CE.gt_from_boxes([img("0000001_02999_d_0000005", [(0, 1, 1, 5, 5)])])
    ok = CE.load_predictions([{"image_id": "0000001_02999_d_0000005", "category_id": 1, "bbox": [1, 1, 5, 5], "score": 0.5}], gt)
    assert ok[0]["image_id"] == 1
    with pytest.raises(KeyError):
        CE.load_predictions([{"image_id": "nope", "category_id": 1, "bbox": [1, 1, 5, 5], "score": 0.5}], gt)
    with pytest.raises(ValueError):
        CE.load_predictions([{"image_id": "0000001_02999_d_0000005", "category_id": 0, "bbox": [1, 1, 5, 5], "score": 0.5}], gt)


def test_no_detections_gives_zero():
    gt = CE.gt_from_boxes([img("a", [(0, 1, 1, 50, 50)])])
    r = CE.evaluate(gt, [], "pycocotools")
    assert r["mAP50_95"] == 0.0 and r["counts"]["detections"] == 0


def test_annotation_ids_start_at_one():
    gt = CE.gt_from_boxes([img("a", [(0, 1, 1, 5, 5)], ignore=[(10, 10, 5, 5)])])
    ids = [a["id"] for a in gt["annotations"]]
    assert min(ids) == 1 and len(set(ids)) == len(ids) == 11
