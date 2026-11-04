import pytest

from visdrone import (
    VDBox,
    convert_boxes,
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
