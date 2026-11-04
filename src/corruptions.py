"""Image corruptions for robustness evaluation and training-time augmentation.

Every function takes a BGR uint8 image (H, W, 3) and returns a new BGR uint8 image of the same shape.
Labels are never an input, so corruptions cannot change boxes. Parameters live in configs/corruptions.yaml.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils import YAML  # ultralytics 8.4.202 ships its own YAML parser (PyYAML is not installed)

TYPES = ("underexposed", "overexposed", "blur", "noise", "low_contrast")
CFG_PATH = Path(__file__).resolve().parents[1] / "configs" / "corruptions.yaml"


def load_config(path: str | Path = CFG_PATH) -> dict:
    cfg = YAML.load(Path(path))
    assert tuple(cfg["types"]) == TYPES, f"unexpected corruption types {list(cfg['types'])}"
    for t, spec in cfg["types"].items():
        assert len(spec["severities"]) == 3, f"{t}: expected 3 severities"
    return cfg


def image_seed(base_seed: int, ctype: str, severity: int, stem: str) -> int:
    """Stable (process- and platform-independent) per-image seed."""
    h = hashlib.sha256(f"{base_seed}|{ctype}|{severity}|{stem}".encode()).digest()
    return int.from_bytes(h[:8], "little")


# ----------------------------------------------------------------------------- primitives
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


@lru_cache(maxsize=64)
def _exposure_lut(ev: float) -> np.ndarray:
    x = np.arange(256, dtype=np.float64) / 255.0
    y = _linear_to_srgb(np.clip(_srgb_to_linear(x) * (2.0**ev), 0.0, 1.0))
    return np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)


def exposure(img: np.ndarray, ev: float) -> np.ndarray:
    return cv2.LUT(img, _exposure_lut(float(ev)))


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Normalized linear motion-blur kernel of the given length (px) and direction."""
    length = max(int(length), 1)
    if length == 1:
        return np.ones((1, 1), np.float32)
    k = length if length % 2 == 1 else length + 1
    c = (k - 1) / 2.0
    dx, dy = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
    half = (length - 1) / 2.0
    ker = np.zeros((k, k), np.float32)
    # sample the segment densely and splat with bilinear weights (sub-pixel accurate, deterministic)
    for t in np.linspace(-half, half, 4 * length + 1):
        x, y = c + t * dx, c + t * dy
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for xx, yy, w in ((x0, y0, (1 - fx) * (1 - fy)), (x0 + 1, y0, fx * (1 - fy)), (x0, y0 + 1, (1 - fx) * fy), (x0 + 1, y0 + 1, fx * fy)):
            if 0 <= xx < k and 0 <= yy < k:
                ker[yy, xx] += w
    return ker / ker.sum()


def motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    ker = motion_kernel(length, angle_deg)
    return cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_REFLECT_101)


def gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma * 255.0, size=img.shape)
    return np.clip(np.rint(img.astype(np.float64) + noise), 0, 255).astype(np.uint8)


def low_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    x = img.astype(np.float64)
    m = x.mean(axis=(0, 1), keepdims=True)
    return np.clip(np.rint((x - m) * factor + m), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- dispatch
def blur_length(length_ref: float, width: int, reference_width: int) -> int:
    return max(1, int(round(length_ref * width / reference_width)))


def apply(img: np.ndarray, ctype: str, param: float, rng: np.random.Generator, reference_width: int) -> np.ndarray:
    """Apply one corruption with an explicit parameter value."""
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3, (img.dtype, img.shape)
    if ctype in ("underexposed", "overexposed"):
        out = exposure(img, param)
    elif ctype == "blur":
        angle = float(rng.uniform(0.0, 180.0))
        out = motion_blur(img, blur_length(param, img.shape[1], reference_width), angle)
    elif ctype == "noise":
        out = gaussian_noise(img, param, rng)
    elif ctype == "low_contrast":
        out = low_contrast(img, param)
    else:
        raise ValueError(f"unknown corruption type {ctype!r}")
    assert out.shape == img.shape and out.dtype == np.uint8
    return out


def corrupt(img: np.ndarray, ctype: str, severity: int, stem: str, cfg: dict) -> np.ndarray:
    """Deterministic test-time corruption. severity is 1, 2 or 3."""
    assert severity in (1, 2, 3), severity
    param = cfg["types"][ctype]["severities"][severity - 1]
    rng = np.random.default_rng(image_seed(cfg["seed"], ctype, severity, stem))
    return apply(img, ctype, param, rng, cfg["reference_width"])


def jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def sample_train_corruption(img: np.ndarray, cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, str | None, float | None]:
    """Training augmentation: with prob p pick a type uniformly and a parameter ~ U[sev1, sev3]."""
    if rng.random() >= cfg["train_augment"]["p"]:
        return img, None, None
    ctype = TYPES[int(rng.integers(len(TYPES)))]
    s = cfg["types"][ctype]["severities"]
    lo, hi = sorted((float(s[0]), float(s[2])))
    param = float(rng.uniform(lo, hi))
    if ctype == "blur":
        param = float(round(param))
    return apply(img, ctype, param, rng, cfg["reference_width"]), ctype, param
