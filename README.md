# yolo-robust

Corruption-robust small-object detection on drone imagery (VisDrone2019-DET) with Ultralytics YOLO,
run on the ITU UHeM cluster.

**Research question.** Under low light, blur and noise, what improves small-object accuracy most:
the model version (YOLO11s vs YOLO26s), the input resolution (640 vs 960), or corruption-aware training?

> Status: setup in progress. Methods, results, limitations and reproduction commands will be
> added as stages are completed. Only measured results will be reported here.

## Layout

| Path | Content | In git |
|---|---|---|
| `configs/` | dataset, experiment and corruption configs | yes |
| `scripts/` | environment, data preparation and SLURM scripts | yes |
| `src/` | project Python code | yes |
| `tests/` | unit tests | yes |
| `results/` | metrics, per-run metadata, figures | yes |
| `datasets/`, `weights/`, `runs/`, `logs/` | data, checkpoints, training runs, logs | no |

## Environment

```bash
source scripts/env.sh          # modules, TMPDIR, YOLO_CONFIG_DIR, venv on PATH
bash scripts/setup_env.sh      # login node only (needs internet)
```

Exact package versions: `requirements.lock` (Python 3.12.4, torch 2.6.0+cu124, torchvision 0.21.0+cu124,
ultralytics 8.4.202). `albumentations` is intentionally not installed: when importable, Ultralytics adds its own
blur/contrast augmentations, which would contaminate the "standard" training runs.

### Offline operation (compute nodes have no internet)

- `scripts/env.sh` sets `YOLO_CONFIG_DIR=$ROOT/.ultralytics`; `scripts/configure_ultralytics.py` pins
  `datasets_dir/weights_dir/runs_dir` inside the project and sets `sync=False` (no analytics).
- Jobs set `YOLO_OFFLINE=true`. In ultralytics 8.4.202 this makes `utils.is_online()` return `False`, so
  `ONLINE=False`: requirement auto-install, analytics events and the PyPI version check are skipped.
- Everything Ultralytics would otherwise download at runtime is fetched on the login node by
  `scripts/fetch_assets.py` (`yolo11s.pt`, `yolo26s.pt` from release v8.4.0, `Arial.ttf`); sha256 in
  `results/manifests/assets.json`. The AMP check builds `yolo26n.yaml` from config (no download).

## Data: VisDrone2019-DET

```bash
bash scripts/download_visdrone.sh        # login node; sizes + sha256 -> results/manifests/visdrone_zips.txt
python scripts/prepare_visdrone.py       # labels + verification -> results/manifests/visdrone_stats.json
```

Layout: `datasets/VisDrone/<split>/{images,annotations,labels}`; `annotations/` keeps the original ground truth
(needed for ignored regions in COCO-style evaluation). `configs/visdrone.yaml` lists **train/val only**;
test-dev is used once, after all choices are frozen.

Label conversion (`src/visdrone.py`, unit-tested): rows with `score=0` (ignored regions, and every
"others"/category-11 box in this release) are not written; class = category − 1; boxes are clipped to the image
and zero-area boxes dropped. All label files pass Ultralytics' `verify_image_label` with 0 corrupt images.
Compared with the stock Ultralytics converter, the only difference is 1 zero-area train box.

| split | images | kept boxes | ignored rows | small / medium / large (COCO, original px) |
|---|---|---|---|---|
| train | 6471 | 343,204 | 10,345 | 207,524 / 116,696 / 18,984 |
| val | 548 | 38,759 | 1,410 | 26,575 / 11,116 / 1,068 |
| test-dev | 1610 | 75,102 | 2,445 | 50,816 / 21,876 / 2,410 |

## License notes

- **Ultralytics** (YOLO11, YOLO26, training/validation code) is licensed under
  **AGPL-3.0** (https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
  This project uses it as a dependency; distributing this code or offering it as a network
  service brings AGPL-3.0 obligations. Ultralytics also offers a separate Enterprise License.
- **VisDrone2019-DET** (https://github.com/VisDrone/VisDrone-Dataset, Tianjin University) has its own
  terms of use; the dataset is not redistributed in this repository.
