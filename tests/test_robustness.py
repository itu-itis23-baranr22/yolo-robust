import csv

import pytest

import robustness as RB


def fake(clean=0.4, corrupted=0.2):
    per = {RB.CLEAN: {"mAP50_95": clean, "mAP50": clean + 0.2, "AP_small": clean / 2}}
    for s in RB.CORRUPTED_SETS:
        per[s] = {"mAP50_95": corrupted, "mAP50": corrupted + 0.2, "AP_small": corrupted / 2}
    for p in per.values():
        p.update({"mAP75": 0.0, "AP_medium": 0.0, "AP_large": 0.0, "AR_1000": 0.0, "mAP50_95_maxdet100": 0.0})
    return per


def test_sets():
    assert len(RB.CORRUPTED_SETS) == 15 and RB.ALL_SETS[0] == "clean" and len(set(RB.ALL_SETS)) == 16
    assert RB.parse_set("low_contrast_s2") == ("low_contrast", 2)
    assert RB.parse_set("clean") == ("clean", None)
    with pytest.raises(AssertionError):
        RB.parse_set("fog_s1")


def test_mpc_rpc():
    per = fake()
    per["noise_s3"]["mAP50_95"] = 0.05
    s = RB.summarize(per)["mAP50_95"]
    assert s["mPC"] == pytest.approx((14 * 0.2 + 0.05) / 15)
    assert s["rPC"] == pytest.approx(s["mPC"] / 0.4)
    assert s["by_type"]["noise"] == pytest.approx((0.2 + 0.2 + 0.05) / 3)
    assert s["by_severity"][3] == pytest.approx((4 * 0.2 + 0.05) / 5)
    assert s["clean"] == 0.4


def test_missing_set_refused():
    per = fake()
    del per["blur_s2"]
    with pytest.raises(ValueError, match="blur_s2"):
        RB.summarize(per)


def test_breakdown_csv(tmp_path):
    rows = RB.breakdown_rows(fake())
    assert len(rows) == 16 and rows[0]["set"] == "clean" and rows[1]["relative_to_clean"] == pytest.approx(0.5)
    RB.write_csv(tmp_path / "b.csv", rows)
    got = list(csv.DictReader(open(tmp_path / "b.csv")))
    assert len(got) == 16 and got[5]["type"] == "overexposed" and got[5]["severity"] == "2"
