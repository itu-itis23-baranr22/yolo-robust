"""On-the-fly corruptions inside the Ultralytics 8.4.202 data pipeline (nothing is written to disk).

Mechanism (verified in ultralytics/data/base.py):
  * BaseDataset.load_image() reads the file with the module-global `ultralytics.data.base.imread`
    (imported from ultralytics.utils.patches) and only then resizes / letterboxes.
  * This module replaces that global ONCE with a pass-through dispatcher. The dispatcher corrupts the decoded
    image only while a RobustYOLODataset.load_image() call is active in the same thread (thread-local
    context). Any other dataset, or any other caller, gets exactly the original imread result.
  * Corruption therefore acts on the ORIGINAL-resolution BGR image, before resizing and before any geometric
    augmentation, and never sees labels.
  * A dataset is made robust by swapping its class after Ultralytics built it (to_robust); all Ultralytics state
    (labels, rect shapes, transforms) is kept unchanged.

Guards: RAM/disk image caches and pre-existing *.npy files are rejected, because load_image would then skip imread
(the corruption would silently not happen, or a corrupted image would be cached).

Training augmentation (E4/E5): a per-process np.random.Generator seeded from sha256(config seed, torch.initial_seed()).
DataLoader workers get torch.initial_seed() = base_seed + worker_id (base_seed drawn from the seeded torch RNG),
so the stream is reproducible for a fixed seed and worker count, and independent of the global np.random state
that Ultralytics' seed_worker sets. While an image sits in Ultralytics' mosaic buffer it is reused as loaded
(already corrupted or not), exactly like the rest of the load-time processing.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import ultralytics.data.base as _ubase
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import patches as _upatches

import corruptions as C

_ORIGINAL_IMREAD = _upatches.imread
_ctx = threading.local()


def _imread_dispatch(filename, flags: int = cv2.IMREAD_COLOR):
    """Same signature/defaults as ultralytics.utils.patches.imread (8.4.202)."""
    im = _ORIGINAL_IMREAD(filename, flags=flags)
    fn = getattr(_ctx, "fn", None)
    if fn is not None and im is not None:
        im = fn(im, str(filename))
    return im


def install() -> None:
    """Idempotently route ultralytics.data.base.imread through the dispatcher."""
    cur = _ubase.imread
    if cur is _imread_dispatch:
        return
    if cur is not _ORIGINAL_IMREAD:
        raise RuntimeError(f"ultralytics.data.base.imread is already patched by someone else: {cur!r}")
    _ubase.imread = _imread_dispatch


install()


# ------------------------------------------------------------------------------------------------ corruption modes
class TestCorruption:
    """Fixed (type, severity) for every image; per-image randomness from the image stem (corruptions.corrupt)."""

    def __init__(self, ctype: str, severity: int, cfg: dict | None = None):
        assert ctype in C.TYPES and severity in (1, 2, 3), (ctype, severity)
        self.ctype, self.severity, self.cfg = ctype, severity, cfg or C.load_config()
        self.counts: Counter = Counter()

    def __call__(self, im: np.ndarray, filename: str) -> np.ndarray:
        self.counts[f"{self.ctype}_s{self.severity}"] += 1
        return C.corrupt(im, self.ctype, self.severity, Path(filename).stem, self.cfg)

    def describe(self) -> dict:
        return {"mode": "test", "type": self.ctype, "severity": self.severity, "config_version": self.cfg["version"]}


class TrainAugment:
    """Random corruption with probability p (configs/corruptions.yaml: train_augment)."""

    def __init__(self, cfg: dict | None = None, stats_dir: str | Path | None = None, stats_every: int = 200):
        self.cfg = cfg or C.load_config()
        self.stats_dir = Path(stats_dir) if stats_dir else None
        self.stats_every = stats_every
        self._rng: np.random.Generator | None = None
        self._pid: int | None = None
        self._seed: int | None = None
        self.counts: Counter = Counter()

    def _rng_for_process(self) -> np.random.Generator:
        pid = os.getpid()
        if self._rng is None or self._pid != pid:  # first call in this process (DataLoader worker or main)
            import torch

            h = hashlib.sha256(f"{self.cfg['seed']}|train_augment|{torch.initial_seed()}".encode()).digest()
            self._seed = int.from_bytes(h[:8], "little")
            self._rng, self._pid, self.counts = np.random.default_rng(self._seed), pid, Counter()
        return self._rng

    def __call__(self, im: np.ndarray, filename: str) -> np.ndarray:
        out, ctype, _ = C.sample_train_corruption(im, self.cfg, self._rng_for_process())
        self.counts[ctype or "none"] += 1
        if self.stats_dir is not None and sum(self.counts.values()) % self.stats_every == 0:
            self.dump_stats()
        return out

    def dump_stats(self) -> None:
        self.stats_dir.mkdir(parents=True, exist_ok=True)
        p = self.stats_dir / f"train_augment_pid{os.getpid()}.json"
        p.write_text(json.dumps({"pid": os.getpid(), "rng_seed": self._seed, "counts": dict(self.counts)}) + "\n")

    def describe(self) -> dict:
        return {"mode": "train_augment", "p": self.cfg["train_augment"]["p"], "config_version": self.cfg["version"]}


# ------------------------------------------------------------------------------------------------ dataset
class RobustYOLODataset(YOLODataset):
    """YOLODataset whose images are corrupted at load time by `self.corruption`."""

    corruption = None  # TestCorruption | TrainAugment, set by to_robust()

    def load_image(self, i, *args, **kwargs):
        prev = getattr(_ctx, "fn", None)
        _ctx.fn = self.corruption
        try:
            return super().load_image(i, *args, **kwargs)
        finally:
            _ctx.fn = prev


def to_robust(ds: YOLODataset, corruption) -> RobustYOLODataset:
    """Turn an Ultralytics-built YOLODataset into a RobustYOLODataset in place (class swap, state unchanged)."""
    if type(ds) is not YOLODataset:
        raise TypeError(f"expected a plain YOLODataset, got {type(ds).__name__}")
    if ds.cache is not None:
        raise ValueError(f"image cache {ds.cache!r} is incompatible with on-the-fly corruption; use cache=False")
    if any(Path(f).exists() for f in ds.npy_files):
        raise FileExistsError("*.npy image caches exist next to the images; load_image would bypass imread")
    if _ubase.imread is not _imread_dispatch:
        raise RuntimeError("robust_data dispatcher is not installed")
    ds.__class__ = RobustYOLODataset
    ds.corruption = corruption
    return ds


# ------------------------------------------------------------------------------------------------ trainer / validator
def make_validator(ctype: str | None, severity: int | None, cfg: dict | None = None) -> type[DetectionValidator]:
    """DetectionValidator subclass evaluating on the (ctype, severity) corrupted copy; ctype=None -> clean."""
    if ctype is None:
        return DetectionValidator

    class CorruptedValidator(DetectionValidator):
        corruption_spec = (ctype, severity)

        def build_dataset(self, img_path, mode="val", batch=None):
            ds = super().build_dataset(img_path, mode=mode, batch=batch)
            return to_robust(ds, TestCorruption(ctype, severity, cfg))

    CorruptedValidator.__name__ = CorruptedValidator.__qualname__ = f"CorruptedValidator_{ctype}_s{severity}"
    return CorruptedValidator


class CorruptionAugTrainer(DetectionTrainer):
    """DetectionTrainer with on-the-fly corruption augmentation on the TRAIN split only.

    Validation during training (model selection / best.pt) stays on clean val via the default DetectionValidator.
    Set ROBUST_DATA_STATS_DIR to dump per-worker augmentation counts (used by the end-to-end check).
    """

    def build_dataset(self, img_path, mode="train", batch=None):
        ds = super().build_dataset(img_path, mode=mode, batch=batch)
        if mode == "train":
            to_robust(ds, TrainAugment(stats_dir=os.environ.get("ROBUST_DATA_STATS_DIR")))
        return ds
