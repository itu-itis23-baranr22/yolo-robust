"""On-the-fly corruption inside the real Ultralytics YOLODataset (tiny synthetic dataset in tmp_path)."""

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import ultralytics.data.base as ubase
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import DEFAULT_CFG

import corruptions as C
import robust_data as RD

CFG = C.load_config()
NAMES = {i: f"c{i}" for i in range(10)}
W, H = 320, 200  # imgsz = 320 = long side -> load_image does not resize


@pytest.fixture()
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for i in range(6):
        im = cv2.resize(rng.integers(0, 256, (20, 32, 3), dtype=np.uint8), (W, H), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(tmp_path / "images" / f"im{i}.png"), im)  # lossless: decoded == written
        (tmp_path / "labels" / f"im{i}.txt").write_text(f"{i % 10} 0.5 0.5 0.2 0.3\n3 0.25 0.25 0.1 0.1\n")
    return tmp_path


def build(data_dir, mode="val", **over):
    cfg = get_cfg(DEFAULT_CFG, {"imgsz": W, "workers": 0, "cache": False, **over})
    return build_yolo_dataset(cfg, str(data_dir / "images"), 2, {"names": NAMES, "nc": 10, "channels": 3}, mode=mode)


def test_dispatcher_installed_and_passthrough(data_dir):
    assert ubase.imread is RD._imread_dispatch
    f = str(data_dir / "images" / "im0.png")
    assert np.array_equal(ubase.imread(f), cv2.imread(f))  # no active context -> unchanged


def test_test_corruption_applied_at_original_resolution(data_dir):
    plain = build(data_dir)
    rob = RD.to_robust(build(data_dir), RD.TestCorruption("noise", 3, CFG))
    for i in range(len(plain)):
        f = plain.im_files[i]
        orig = cv2.imread(f)
        a, hw0, _ = plain.load_image(i)
        b, hw0b, _ = rob.load_image(i)
        assert np.array_equal(a, orig), "plain dataset must stay clean"
        assert np.array_equal(b, C.corrupt(orig, "noise", 3, Path(f).stem, CFG))
        assert hw0 == hw0b == (H, W)
    assert rob.corruption.counts["noise_s3"] == len(plain)
    assert ubase.imread(plain.im_files[0]) is not None and np.array_equal(ubase.imread(plain.im_files[0]), cv2.imread(plain.im_files[0]))


def test_labels_untouched(data_dir):
    plain = build(data_dir)
    rob = RD.to_robust(build(data_dir), RD.TestCorruption("blur", 3, CFG))
    for i in range(len(plain)):
        a, b = plain[i], rob[i]
        assert torch.equal(a["bboxes"], b["bboxes"]) and torch.equal(a["cls"], b["cls"])
        assert not torch.equal(a["img"], b["img"])


def test_other_threads_are_not_corrupted(data_dir):
    rob = RD.to_robust(build(data_dir), RD.TestCorruption("noise", 3, CFG))
    seen = {}

    def corr(im, filename):  # while the robust context is active, read the same file from another thread
        t = threading.Thread(target=lambda: seen.setdefault("other", ubase.imread(filename)))
        t.start()
        t.join()
        return C.corrupt(im, "noise", 3, "x", CFG)

    rob.corruption = corr
    rob.load_image(0)
    assert np.array_equal(seen["other"], cv2.imread(rob.im_files[0]))


def test_guards(data_dir):
    with pytest.raises(ValueError, match="cache"):
        RD.to_robust(build(data_dir, cache="ram"), RD.TestCorruption("noise", 1, CFG))
    np.save(str(data_dir / "images" / "im0.npy"), np.zeros((2, 2, 3), np.uint8))
    with pytest.raises(FileExistsError):
        RD.to_robust(build(data_dir), RD.TestCorruption("noise", 1, CFG))
    (data_dir / "images" / "im0.npy").unlink()
    ds = RD.to_robust(build(data_dir), RD.TestCorruption("noise", 1, CFG))
    with pytest.raises(TypeError):
        RD.to_robust(ds, RD.TestCorruption("noise", 1, CFG))  # already robust


def test_train_augment_in_mosaic_pipeline(data_dir, tmp_path):
    torch.manual_seed(0)
    aug = RD.TrainAugment(CFG, stats_dir=tmp_path / "stats", stats_every=2)
    ds = RD.to_robust(build(data_dir, mode="train"), aug)
    for k in range(12):
        out = ds[k % len(ds)]
        assert out["img"].shape[0] == 3 and out["bboxes"].shape[1] == 4
    # every image went through the hook once; afterwards Ultralytics' mosaic buffer (size 6 here) serves it
    assert sum(aug.counts.values()) >= len(ds)
    assert list((tmp_path / "stats").glob("train_augment_pid*.json"))


def test_train_augment_probability_and_types():
    torch.manual_seed(0)
    aug = RD.TrainAugment(CFG)
    im = np.full((40, 60, 3), 120, np.uint8)
    for _ in range(600):
        aug(im, "f.jpg")
    assert 0.42 < aug.counts["none"] / 600 < 0.58  # p = 0.5
    assert set(aug.counts) == {"none", *C.TYPES}


def test_train_augment_reproducible_for_same_torch_seed():
    im = np.full((60, 80, 3), 120, np.uint8)
    outs = []
    for _ in range(2):
        torch.manual_seed(123)
        aug = RD.TrainAugment(CFG)
        outs.append([aug(im, "f.jpg") for _ in range(12)])
    assert all(np.array_equal(a, b) for a, b in zip(*outs))
    torch.manual_seed(124)
    aug = RD.TrainAugment(CFG)
    assert not all(np.array_equal(a, aug(im, "f.jpg")) for a in outs[0])


def test_make_validator():
    assert RD.make_validator(None, None) is DetectionValidator
    V = RD.make_validator("low_contrast", 2, CFG)
    assert issubclass(V, DetectionValidator) and V.corruption_spec == ("low_contrast", 2)
