import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

import corruptions as C

CFG = C.load_config()
SETS = [(t, s) for t in C.TYPES for s in (1, 2, 3)]


@pytest.fixture(scope="module")
def img():
    rng = np.random.default_rng(0)
    # reference width (1360) so blur lengths are not scaled down to 1 px
    base = cv2.resize(rng.integers(0, 256, (24, 80, 3), dtype=np.uint8), (1360, 400), interpolation=cv2.INTER_CUBIC)
    cv2.rectangle(base, (400, 150), (440, 190), (255, 255, 255), -1)  # sharp edges for the blur test
    return base


def test_config_shape():
    assert tuple(CFG["types"]) == C.TYPES
    assert all(len(CFG["types"][t]["severities"]) == 3 for t in C.TYPES)
    assert len(SETS) == 15


@pytest.mark.parametrize("ctype,sev", SETS)
def test_deterministic_shape_dtype_and_input_untouched(img, ctype, sev):
    before = img.copy()
    a = C.corrupt(img, ctype, sev, "0000001_02999_d_0000005", CFG)
    b = C.corrupt(img, ctype, sev, "0000001_02999_d_0000005", CFG)
    assert np.array_equal(img, before), "input modified in place"
    assert a.shape == img.shape and a.dtype == np.uint8
    assert np.array_equal(a, b), "same seed must give identical output"
    assert not np.array_equal(a, img), "corruption had no effect"


def test_per_image_randomness_differs_between_stems(img):
    assert not np.array_equal(C.corrupt(img, "noise", 2, "a", CFG), C.corrupt(img, "noise", 2, "b", CFG))


def test_seed_is_independent_of_python_hash_seed(img):
    code = (
        "import sys, hashlib, numpy as np; sys.path.insert(0, %r); import corruptions as C; "
        "img = np.full((40, 60, 3), 128, np.uint8); "
        "print(hashlib.sha256(C.corrupt(img, 'noise', 3, 'x', C.load_config()).tobytes()).hexdigest())"
    ) % str(Path(C.__file__).parent)
    outs = set()
    for hs in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        outs.add(subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip())
    assert len(outs) == 1


def test_exposure_identity_and_monotonic(img):
    assert np.array_equal(C.exposure(img, 0.0), img)
    under = [C.corrupt(img, "underexposed", s, "x", CFG).mean() for s in (1, 2, 3)]
    over = [C.corrupt(img, "overexposed", s, "x", CFG).mean() for s in (1, 2, 3)]
    assert img.mean() > under[0] > under[1] > under[2]
    assert img.mean() < over[0] < over[1] < over[2]


def test_low_contrast_monotonic(img):
    stds = [C.corrupt(img, "low_contrast", s, "x", CFG).std() for s in (1, 2, 3)]
    assert img.std() > stds[0] > stds[1] > stds[2]


def test_blur_monotonic_sharpness(img):
    sharp = lambda x: cv2.Laplacian(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    v = [sharp(C.corrupt(img, "blur", s, "x", CFG)) for s in (1, 2, 3)]
    assert sharp(img) > v[0] > v[1] > v[2]


def test_noise_monotonic(img):
    d = [np.abs(C.corrupt(img, "noise", s, "x", CFG).astype(int) - img).mean() for s in (1, 2, 3)]
    assert d[0] < d[1] < d[2]


def test_motion_kernel():
    assert np.array_equal(C.motion_kernel(1, 30.0), np.ones((1, 1), np.float32))
    for L in (5, 9, 15, 21):
        for a in (0.0, 33.0, 90.0, 179.0):
            k = C.motion_kernel(L, a)
            assert k.shape[0] == k.shape[1] and k.shape[0] % 2 == 1
            assert abs(k.sum() - 1.0) < 1e-5 and (k >= 0).all()


def test_blur_length_scales_with_width():
    assert C.blur_length(9, 1360, 1360) == 9
    assert C.blur_length(9, 960, 1360) == 6
    assert C.blur_length(15, 1920, 1360) == 21


def test_train_sampler_probability_and_range(img):
    cfg = {**CFG, "train_augment": {"p": 1.0}}
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        out, t, p = C.sample_train_corruption(img, cfg, rng)
        s = CFG["types"][t]["severities"]
        assert min(s[0], s[2]) <= p <= max(s[0], s[2])
        assert out.shape == img.shape
        seen.add(t)
    assert seen == set(C.TYPES)
    cfg0 = {**CFG, "train_augment": {"p": 0.0}}
    out, t, p = C.sample_train_corruption(img, cfg0, rng)
    assert t is None and out is img


def test_jpeg_roundtrip_deterministic(img):
    assert np.array_equal(C.jpeg_roundtrip(img, 95), C.jpeg_roundtrip(img, 95))
