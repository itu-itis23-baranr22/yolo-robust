# yolo-robust

Corruption-robust small-object detection on drone imagery (VisDrone2019-DET) with Ultralytics YOLO,
run on the ITU UHeM cluster.

**Research question.** Under low light, blur and noise, what improves small-object accuracy most:
the model version (YOLO11s vs YOLO26s), the input resolution (640 vs 960), or corruption-aware training?

> Status: evaluation infrastructure done (Stage 3); no experiment (E1–E5) has been trained yet.
> Methods, results, limitations and reproduction commands are added as stages are completed.
> Only measured results are reported here.

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

## Corruptions (fixed in commit 810edbd, before any evaluation)

Five types × three severities (`configs/corruptions.yaml`, `src/corruptions.py`, unit-tested): underexposed /
overexposed (exposure change in linear light), linear motion blur, additive Gaussian noise, low contrast. They act on
the original-resolution BGR image and never touch labels; per-image randomness comes from
`sha256(seed|type|severity|stem)`, so identical inputs give identical outputs.

**On-the-fly test sets (decision after the disk estimate).** Storing the 15 corrupted copies as JPEG q95 would take
2.77 GB (val) and 10.58 GB (test-dev) (`results/manifests/corruption_disk_estimate_*.json`). Instead, corrupted
copies are produced on the fly at load time; nothing is written. This also removes a confound: JPEG re-encoding
would partly smooth the added noise, and the clean baseline would need re-encoding too. Reproducibility is checked
with `scripts/corruption_manifest.py`: sha256 of every decoded array (548 × 16 for val: clean + 15 corrupted) in
`results/manifests/corruptions_val_sha256.csv`, one digest per set in `corruptions_val_digest.json`; `--check`
recomputes and compares.

**How corruption enters the Ultralytics 8.4.202 pipeline** (`src/robust_data.py`). `BaseDataset.load_image()` reads
the file through the module-global `ultralytics.data.base.imread` and only then resizes. That global is replaced once
by a pass-through dispatcher, which corrupts the decoded image only while a `RobustYOLODataset.load_image()` call is
active in the same thread; every other caller gets the unmodified image. A dataset is made robust by swapping the class
of the `YOLODataset` that Ultralytics built (labels, rect shapes and transforms unchanged). RAM/disk image caches and
pre-existing `*.npy` files are rejected because `load_image` would then bypass `imread`.

- Evaluation: `make_validator(type, severity)` returns a `DetectionValidator` subclass whose dataset is corrupted.
- Training (E4/E5): `CorruptionAugTrainer` corrupts the **train** split only, with probability 0.5, a uniformly
  chosen type and a parameter ~ U[severity-1, severity-3]. Validation during training (best.pt selection) stays clean.
  The per-process RNG is seeded from `sha256(seed, torch.initial_seed())` (reproducible for a fixed seed and
  worker count). Images held in Ultralytics' mosaic buffer are reused as loaded, like the rest of load-time processing.
- **Limitation:** training augmentation and test sets share the same corruption families and parameter ranges.

## Evaluation protocol

`scripts/evaluate.py` runs Ultralytics val for clean + 15 corrupted sets with one fixed configuration for every model:
`conf=0.001, iou=0.7, max_det=1000, nms=False` (YOLO26 NMS-free head; YOLO11 falls back to NMS), FP16, rect batches,
`save_json=True`. The predictions (original-image pixels) are scored by `src/coco_eval.py`:

- GT = exactly the boxes used as training labels; every non-evaluated VisDrone row (ignored region, score 0,
  "others") becomes a crowd box for all 10 classes, so a detection lying inside it is neither TP nor FP
  (standard COCOeval crowd rule).
- COCO area ranges (small < 32² ≤ medium < 96² ≤ large) on original-pixel areas.
- maxDets = (1, 10, 100, 1000); headline metrics use 1000 (VisDrone images hold up to 902 objects);
  AP@maxDets=100 is kept for reference. Values are read directly from `COCOeval.eval['precision']`.
- Backend: faster-coco-eval 1.8.0; it matches pycocotools 2.0.11 to 1e-9 in the unit tests (random data with
  crowd boxes) and on real predictions in the end-to-end check.
- This is COCO-style evaluation, not the official VisDrone MATLAB toolkit.

Robustness (`src/robustness.py`): **mPC** = mean mAP50-95 over the 15 corrupted sets, **rPC** = mPC / clean mAP50-95,
plus per-type and per-severity breakdowns (`breakdown.csv`). The same summaries are computed for mAP50 and AP_small.

Decisions taken after the Stage-3 check (approved): `max_det=1000` is kept for every model (its effect is quantified
below); raw predictions are kept gzip-compressed after scoring (verified read-back); latency is reported as batch-1
(primary, per the protocol) and, as a secondary number, Ultralytics' per-image val times at batch 16 on the clean set
(`summary.json: val_speed_ms_per_img_clean`), because batch-1 eager inference is launch-bound.

Training/evaluation jobs: `scripts/slurm/train_eval.sbatch` (train or resume → 16-set val evaluation → latency).

### Pipeline check (Stage 3, SLURM job 555207, 1× A100, 16 min; `results/e2e/e2e_555207/`)

Smoke-level only: the evaluated checkpoints are the 2-epoch Stage-2 models, so their mAP values say nothing about
the research question.

- 59 unit tests pass on the compute node. The corruption manifest (8,768 arrays) is reproduced exactly by `--check`.
- GT built from the raw annotations: 38,759 boxes and 1,410 ignored regions on val (same as the label statistics).
- faster-coco-eval and pycocotools give identical metrics on real predictions (max abs. difference 0.0 on three
  100-image sets, ~1.7 s vs ~24 s per set).
- Full val, 16 sets, one YOLO26s@960 model: ~18 s Ultralytics val + ~9 s COCO eval per set, 8.5 min in total;
  raw `predictions.json` ≈ 79 MB per set. mAP50-95 decreases monotonically with severity for every type.
- Corruption-augmented training costs ~1% throughput (2-epoch, 25% train runs: 123.9 vs 125.2 img/s); measured
  augmentation stats over 16 workers: 49.8% of loads clean, 9.8–10.3% per corruption type.
- `max_det=1000` is reached on 465/548 clean val images (conf 0.001). Truncating the same predictions to the top
  750 per image lowers mAP50-95 by 0.0008 (clean) / 0.0005 (noise s3) and AR@1000 by ~0.02, so the cap trims mainly
  the low-score recall tail. The cap is identical for every model.
- Batch-1 latency (FP16, eager PyTorch) is 8–9 ms for all three smoke models and almost independent of imgsz
  (YOLO26s inference 7.19 ms @640 vs 7.07 ms @960); CUDA-event time of the forward ≈ its CPU time, i.e. the
  measurement is kernel-launch-bound at batch 1.

## License notes

- **Ultralytics** (YOLO11, YOLO26, training/validation code) is licensed under
  **AGPL-3.0** (https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
  This project uses it as a dependency; distributing this code or offering it as a network
  service brings AGPL-3.0 obligations. Ultralytics also offers a separate Enterprise License.
- **VisDrone2019-DET** (https://github.com/VisDrone/VisDrone-Dataset, Tianjin University) has its own
  terms of use; the dataset is not redistributed in this repository.
