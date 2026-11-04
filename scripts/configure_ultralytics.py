"""Pin project-local Ultralytics settings.

Requires `source scripts/env.sh` (YOLO_CONFIG_DIR must point inside the project).
Disables analytics sync and third-party logger integrations; points all dirs into the project.
"""

import json
import os
from pathlib import Path

from ultralytics.utils import SETTINGS, SETTINGS_FILE, USER_CONFIG_DIR

root = Path(os.environ["ROOT"]).resolve()
assert Path(USER_CONFIG_DIR).resolve().is_relative_to(root), f"YOLO_CONFIG_DIR outside project: {USER_CONFIG_DIR}"

SETTINGS.update(
    {
        "datasets_dir": str(root / "datasets"),
        "weights_dir": str(root / "weights"),
        "runs_dir": str(root / "runs"),
        "sync": False,
        "clearml": False,
        "comet": False,
        "dvc": False,
        "mlflow": False,
        "raytune": False,
        "tensorboard": False,
        "wandb": False,
    }
)
print(SETTINGS_FILE)
print(json.dumps({k: v for k, v in json.loads(Path(SETTINGS_FILE).read_text()).items() if "key" not in k}, indent=2))
