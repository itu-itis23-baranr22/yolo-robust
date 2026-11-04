"""Reproducibility metadata for every run: git commit, packages, CUDA/GPU, SLURM job, config, seed."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

SLURM_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK",
    "SLURM_MEM_PER_NODE",
)
ENV_KEYS = ("YOLO_OFFLINE", "YOLO_AUTOINSTALL", "YOLO_CONFIG_DIR", "CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES")


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _sha256(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def gpu_info() -> list[dict]:
    import torch

    out = []
    try:
        import pynvml

        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
    except Exception:
        driver = None
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out.append(
            {
                "index": i,
                "name": p.name,
                "total_memory_gb": round(p.total_memory / 1024**3, 2),
                "capability": f"{p.major}.{p.minor}",
                "driver": driver,
            }
        )
    return out


def collect(root: Path, config: dict | None = None, seed: int | None = None, extra: dict | None = None) -> dict:
    import cv2
    import numpy
    import torch
    import torchvision
    import ultralytics

    # tracked files only: a job's own fresh outputs (untracked) must not mark the code as modified
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "python": sys.version.split()[0],
        "git": {"commit": _git(root, "rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None},
        "packages": {
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "ultralytics": ultralytics.__version__,
            "numpy": numpy.__version__,
            "opencv": cv2.__version__,
        },
        "requirements_lock_sha256": _sha256(root / "requirements.lock"),
        "cuda": {
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "available": torch.cuda.is_available(),
        },
        "gpus": gpu_info(),
        "slurm": {k: os.environ.get(k) for k in SLURM_KEYS},
        "env": {k: os.environ.get(k) for k in ENV_KEYS},
        "seed": seed,
        "config": config,
        "extra": extra,
    }


def write(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
