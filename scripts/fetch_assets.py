"""Download pretrained weights and the plotting font on the LOGIN node; record sha256.

Compute nodes are offline: every file Ultralytics would otherwise fetch at runtime must exist beforehand.
Writes results/manifests/assets.json.
"""

import hashlib
import json
import os
from pathlib import Path

import ultralytics
from ultralytics.utils import USER_CONFIG_DIR
from ultralytics.utils.checks import check_font
from ultralytics.utils.downloads import attempt_download_asset

WEIGHTS = ["yolo11s.pt", "yolo26s.pt"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = Path(os.environ["ROOT"]).resolve()
    out = {"ultralytics": ultralytics.__version__, "files": {}}

    for name in WEIGHTS:
        p = Path(attempt_download_asset(root / "weights" / name))
        out["files"][name] = {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}

    font = Path(check_font("Arial.ttf"))  # VisDrone class names are ASCII -> Arial.ttf
    assert font.is_relative_to(Path(USER_CONFIG_DIR)), f"font resolved outside YOLO_CONFIG_DIR: {font}"
    out["files"]["Arial.ttf"] = {"path": str(font), "bytes": font.stat().st_size, "sha256": sha256(font)}

    dst = root / "results" / "manifests" / "assets.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
