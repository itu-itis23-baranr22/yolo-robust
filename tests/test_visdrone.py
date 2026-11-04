import os
from pathlib import Path

import pytest

from visdrone import (
    VDBox,
    convert_boxes,
    eval_boxes,
    format_rows,
    parse_annotation_text,
    parse_line,
    ultralytics_reference_rows,
)


def test_parse_line_trailing_comma_and_blank():
    assert parse_line("") is None
    b = parse_line("684,8,273,116,0,0,0,0,")
    assert b == VDBox(684, 8, 273, 116, 0, 0, 0, 0)


def test_parse_line_too_short():
    with pytest.raises(ValueError):
        parse_line("1,2,3")


def test_convert_rules():
    W, H = 100, 50
    boxes = parse_annotation_text(
        "\n".join(
            [
                "10,10,20,10,1,4,0,0",  # car -> cls 3, kept
                "0,0,30,30,0,0,0,0",  # ignored region
                "5,5,5,5,0,1,0,0",  # score 0 with a real class -> ignored
                "40,10,10,10,1,11,0,0",  # others -> dropped
                "90,40,20,20,1,1,0,0",  # pedestrian, clipped to 10x10
                "100,10,5,5,1,2,0,0",  # fully outside -> degenerate
            ]
        )
    )
    rows, st = convert_boxes(boxes, W, H)
    assert st["rows"] == 6
    assert st["ignored_region"] == 2 and st["others"] == 1
    assert st["clipped"] == 2 and st["degenerate"] == 1 and st["kept"] == 2
    assert rows[0][0] == 3 and rows[0][1:] == pytest.approx((0.2, 0.3, 0.2, 0.2))
    c, xc, yc, w, h = rows[1]
    assert c == 0 and (xc, yc, w, h) == pytest.approx((0.95, 0.9, 0.1, 0.2))
    # every kept row is normalized and inside the image
    for _, xc, yc, w, h in rows:
        assert 0 <= xc - w / 2 and xc + w / 2 <= 1 and 0 <= yc - h / 2 and yc + h / 2 <= 1


def test_unknown_category_raises():
    with pytest.raises(ValueError):
        convert_boxes([VDBox(0, 0, 1, 1, 1, 12, 0, 0)], 10, 10)


def test_reference_maps_others_to_class_10():
    # documents the behaviour of the Ultralytics converter we deliberately deviate from
    rows = ultralytics_reference_rows([VDBox(0, 0, 2, 2, 1, 11, 0, 0)], 10, 10)
    assert rows[0][0] == 10


def test_format_rows():
    assert format_rows([(3, 0.4, 0.3, 0.2, 0.2)]) == "3 0.400000 0.300000 0.200000 0.200000\n"


RULES_TEXT = "\n".join(
    [
        "10,10,20,10,1,4,0,0",
        "0,0,30,30,0,0,0,0",
        "5,5,5,5,0,1,0,0",
        "40,10,10,10,1,11,0,0",
        "90,40,20,20,1,1,0,0",
        "100,10,5,5,1,2,0,0",
        "95,45,20,20,0,0,0,0",
    ]
)


def _normalized(kept, W, H):
    return [(c, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H) for c, x, y, w, h in kept]


def test_eval_boxes_match_training_labels_and_collect_ignores():
    W, H = 100, 50
    boxes = parse_annotation_text(RULES_TEXT)
    kept, ignore = eval_boxes(boxes, W, H)
    rows, _ = convert_boxes(boxes, W, H)
    assert [r[0] for r in rows] == [k[0] for k in kept]
    for a, b in zip(rows, _normalized(kept, W, H)):
        assert a[1:] == pytest.approx(b[1:], abs=1e-12)
    assert kept == [(3, 10, 10, 20, 10), (0, 90, 40, 10, 10)]
    # ignored region, score-0 box, "others" box, and the clipped ignored region (95,45 -> 5x5)
    assert ignore == [(0, 0, 30, 30), (5, 5, 5, 5), (40, 10, 10, 10), (95, 45, 5, 5)]


VAL = Path(os.environ.get("ROOT", Path(__file__).resolve().parents[1])) / "datasets" / "VisDrone" / "val"


@pytest.mark.skipif(not (VAL / "annotations").is_dir(), reason="VisDrone val not present")
def test_eval_boxes_equal_written_labels_on_real_val():
    from PIL import Image

    files = sorted((VAL / "annotations").glob("*.txt"))[:60]
    for f in files:
        with Image.open(VAL / "images" / f"{f.stem}.jpg") as im:
            W, H = im.size
        kept, _ = eval_boxes(parse_annotation_text(f.read_text()), W, H)
        written = [tuple(float(v) for v in ln.split()) for ln in (VAL / "labels" / f.name).read_text().splitlines()]
        assert len(written) == len(kept), f.name
        for a, b in zip(written, _normalized(kept, W, H)):
            assert int(a[0]) == b[0] and a[1:] == pytest.approx(b[1:], abs=2e-6), f.name  # labels use 6 decimals
