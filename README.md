# yolo-robust

Corruption-robust small-object detection on drone imagery (VisDrone2019-DET) with Ultralytics YOLO,
run on the ITU UHeM cluster.

**Research question.** Under low light, blur and noise, what improves small-object accuracy most:
the model version (YOLO11s vs YOLO26s), the input resolution (640 vs 960), or corruption-aware training?

> Status: complete. Stages 4–5 on val (E1, E2, E4, E5 × 3 seeds; E3 × 1 seed); Stage 6: checkpoints and protocol
> frozen (tag `stage6-freeze`), evaluated once on test-dev, latency of all models measured on one GPU.
> **Answer (test-dev):** corruption-aware training and 960-px input help about equally under corruption (mPC +0.038
> and +0.035 mAP50-95); resolution helps small objects a bit more (AP_small mPC +0.029 vs +0.022) and also clean
> accuracy, corruption training costs nothing at inference. YOLO26s vs YOLO11s at 640 changes mAP50-95 by ≈0
> (+0.008 clean AP_small). See [Final results](#final-results-test-dev).
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

## Stage 4 interim results: E1–E3, seed 0, VisDrone val

Jobs 555232 (E1), 555220 (E2), 555233 (E3); 1× A100 each; 100 epochs; `best.pt` = Ultralytics' best val fitness
(0.9·mAP50-95 + 0.1·mAP50, clean val). Metrics from `src/coco_eval.py` (ignored regions, maxDets 1000).
Files: `results/train/<run>/`, `results/eval/<run>/val/{summary.json,breakdown.csv}`, `results/latency/<run>.json`.

| | E1 YOLO11s@640 | E2 YOLO26s@640 | E3 YOLO26s@960 |
|---|---|---|---|
| clean mAP50-95 | 0.2266 | 0.2319 | 0.2956 |
| clean mAP50 | 0.3971 | 0.4041 | 0.4947 |
| clean AP_small | 0.1355 | 0.1428 | 0.2076 |
| mPC (mAP50-95) | 0.1806 | 0.1785 | 0.2269 |
| rPC (mAP50-95) | 0.797 | 0.770 | 0.768 |
| AP_small mPC / rPC | 0.1044 / 0.771 | 0.1056 / 0.739 | 0.1520 / 0.732 |
| rPC underexp. / overexp. / blur / noise / low-contrast | .970 / .900 / .732 / .611 / .771 | .979 / .896 / .694 / .547 / .734 | .985 / .911 / .606 / .513 / .822 |
| mAP50-95 at blur s3 / noise s3 | 0.0967 / 0.0858 | 0.0842 / 0.0707 | 0.0853 / 0.0777 |
| batch-1 latency p50 / p95 (ms, FP16, conf 0.25) | 7.89 / 8.18 | 8.72 / 9.01 | 9.29 / 9.65 |
| batch-1 pre / inference / post (ms, mean) | 1.12 / 5.81 / 0.78 | 1.11 / 7.12 / 0.29 | 1.39 / 7.14 / 0.30 |
| batch-16 val inference (ms/img) / postprocess | 0.76 / 0.92 | 0.81 / 0.08 | 1.27 / 0.08 |
| parameters | 9.43 M | 9.96 M | 9.96 M |
| peak VRAM batch-1 (MB) | 63 | 63 | 77 |
| best epoch / val fitness peak vs. epoch 100 (Ultralytics mAP50-95) | 68 / 0.221 vs 0.212 | 45 / 0.226 vs 0.219 | 48 / 0.288 vs 0.275 |
| training time (h) / throughput (img/s) | 0.79 / 255 | 1.05 / 189 | 1.57 / 123 |

Notes (single seed, so differences of a few 0.001 are not yet interpretable):
- val mAP peaked around epochs 45–68 and declined afterwards in all three runs; `best.pt` is chosen on the same
  val set that is reported here, so val numbers are optimistic (the same way for every model). test-dev (Stage 6)
  gives the unbiased estimate.
- Ultralytics drops 4 exact-duplicate box rows from the train labels (images 0000137_02220_d_0000163,
  0000140_00118_d_0000002, 9999945_00000_d_0000114, 9999987_00000_d_0000049); val is unaffected.
- E2 ran at commit ac551b5, E1/E3 at 6ceb1a4; the only difference is the label-cache guard (no training/eval code).

## Stage 5 results: seed repeats, E4 and E5, VisDrone val

Same protocol as Stage 4. E1/E2/E4/E5: seeds 0, 1, 2; E3: seed 0 only (seed repeats went to the best model by val mPC
at that point, E4; E5 was run with three seeds from the start). Generated by `python scripts/summarize.py --split val`
(`results/summary/val_runs.csv` per run, `val_experiments.csv` mean/sd, `val.md`). Jobs: E4 555258/555308/555309,
E1/E2 seeds 1–2 555259–555262, E5 555956–555958 (1× A100 each).

Split `val`: mean ± sample sd over seeds (n per column header).

| | E1 (n=3) | E2 (n=3) | E3 (n=1) | E4 (n=3) | E5 (n=3) |
|---|---|---|---|---|---|
| clean mAP50-95 | 0.2286 ± 0.0019 | 0.2317 ± 0.0006 | 0.2956 | 0.3007 ± 0.0019 | 0.3035 ± 0.0017 |
| clean mAP50 | 0.3995 ± 0.0030 | 0.4047 ± 0.0011 | 0.4947 | 0.5015 ± 0.0024 | 0.5053 ± 0.0032 |
| clean AP_small | 0.1355 ± 0.0027 | 0.1443 ± 0.0014 | 0.2076 | 0.2128 ± 0.0020 | 0.2188 ± 0.0021 |
| mPC (mAP50-95) | 0.1812 ± 0.0006 | 0.1789 ± 0.0004 | 0.2269 | 0.2746 ± 0.0018 | 0.2782 ± 0.0007 |
| rPC (mAP50-95) | 0.7925 ± 0.0049 | 0.7721 ± 0.0022 | 0.7675 | 0.9131 ± 0.0009 | 0.9165 ± 0.0034 |
| AP_small mPC | 0.1032 ± 0.0017 | 0.1068 ± 0.0011 | 0.1520 | 0.1882 ± 0.0027 | 0.1938 ± 0.0023 |
| AP_small rPC | 0.7620 ± 0.0089 | 0.7398 ± 0.0022 | 0.7322 | 0.8846 ± 0.0044 | 0.8858 ± 0.0024 |
| rPC underexposed | 0.9720 ± 0.0021 | 0.9746 ± 0.0036 | 0.9851 | 0.9941 ± 0.0030 | 0.9957 ± 0.0032 |
| rPC overexposed | 0.8951 ± 0.0068 | 0.8942 ± 0.0031 | 0.9107 | 0.9365 ± 0.0016 | 0.9384 ± 0.0054 |
| rPC blur | 0.7337 ± 0.0067 | 0.7013 ± 0.0089 | 0.6062 | 0.8063 ± 0.0022 | 0.8084 ± 0.0046 |
| rPC noise | 0.5994 ± 0.0104 | 0.5477 ± 0.0142 | 0.5134 | 0.8721 ± 0.0029 | 0.8767 ± 0.0050 |
| rPC low_contrast | 0.7624 ± 0.0125 | 0.7425 ± 0.0181 | 0.8222 | 0.9567 ± 0.0032 | 0.9631 ± 0.0030 |
| mAP50-95 severity 3 (mean of 5 types) | 0.1446 ± 0.0005 | 0.1402 ± 0.0005 | 0.1802 | 0.2525 ± 0.0014 | 0.2562 ± 0.0008 |
| best epoch | 60 ± 9 | 47 ± 3 | 48 | 58 ± 5 | 59 ± 11 |
| training time (h) | 0.80 ± 0.02 | 1.04 ± 0.01 | 1.57 | 1.58 ± 0.01 | 2.91 ± 0.01 |

Mean differences (A − B) for the studied variable:

| contrast | Δ clean mAP50-95 | Δ mPC | Δ rPC | Δ clean AP_small | Δ AP_small mPC |
|---|---|---|---|---|---|
| E2 − E1 (model version @640) | +0.0031 | -0.0023 | -0.0204 | +0.0088 | +0.0035 |
| E3 − E2 (input size 640 -> 960) | +0.0639 | +0.0480 | -0.0046 | +0.0632 | +0.0452 |
| E4 − E3 (corruption-aware training) | +0.0051 | +0.0477 | +0.1456 | +0.0052 | +0.0362 |
| E5 − E4 (P2 head (optional)) | +0.0028 | +0.0036 | +0.0033 | +0.0060 | +0.0056 |

Notes:
- Seed-to-seed sd is 0.0004–0.0019 for clean mAP50-95 and mPC. The mPC gains from resolution (E3−E2, +0.048) and
  from corruption-aware training (E4−E3, +0.048) are >25× the largest sd; E2−E1 and E5−E4 are only 1–3 sd (n=3,
  no significance test).
- Corruption-aware training (E4) raises mPC by +0.048 with almost no change in clean accuracy (+0.005): the gain is
  robustness, mostly under noise (rPC 0.51 → 0.87), blur (0.61 → 0.81) and low contrast (0.82 → 0.96).
  **Caveat:** training augmentation and test corruptions share families and parameter ranges, so this is an
  in-distribution robustness gain, not evidence of robustness to unseen corruptions.
- YOLO26s vs YOLO11s at 640 (E2−E1): +0.003 clean mAP50-95 and +0.009 clean AP_small, but −0.002 mPC / −0.020 rPC.
- E5 (P2 head, optional) vs E4: +0.003 clean, +0.004 mPC, +0.006 AP_small, at 1.84× training time (2.91 h vs
  1.58 h) and slower inference (batch-16 val inference 1.82 vs 1.27 ms/img). Confound: only 360/902 state-dict
  items of `yolo26s.pt` transfer into `yolo26s-p2.yaml`, so E5 differs from E4 in initialisation as well as in
  architecture.
- Model selection (approved criterion, fixed before test-dev): highest mean val mPC → **E5** (0.2782; E4 0.2746).
- Per-job batch-1 latencies differ by up to ~1 ms between nodes (same E5 architecture: 11.53 ms p50 on a108 vs
  10.50 ms on a112), so the final latency comparison comes from one job on one node
  (`scripts/slurm/latency_all.sbatch`), not from the per-run files.

## Stage 6: freeze and the single test-dev evaluation

Frozen in commit `285b0a3` (tag `stage6-freeze`) before any test-dev prediction was made:
`results/manifests/final_models.json` lists all 13 checkpoints (E1–E5, every seed; none is chosen with test-dev)
with sha256, the declared best experiment (E5, by val mPC) and sha256 of the evaluation code and configs.
The only test-dev check before evaluation used annotations, not models: 1610 images, 75,102 GT boxes, 2,445 ignored
regions (= Stage 1 statistics); the test-dev `labels.cache` was built once on the login node (0 corrupt).

- `scripts/slurm/eval_final.sbatch` (one job per checkpoint): `freeze_models.py --verify` (weights and protocol
  unchanged) → `evaluate.py --split test-dev --final` (same 16 sets and settings as val; refuses a dirty tree) and
  refuses to start when a complete test-dev result for that run already exists.
- `scripts/slurm/latency_all.sbatch`: all 13 checkpoints in one process on one GPU, forward and reverse order,
  then the sha256 manifest of the 15 on-the-fly corrupted test-dev sets (`results/manifests/corruptions_test-dev_*`).

### Final results (test-dev)

Jobs 556085–556097 (one per checkpoint, 15–19 min each on 1× A100). Every job verified its checkpoint and the
protocol files against `stage6-freeze` first; runs used a clean tree at `f051d16` (3 jobs) or `553926e` (10 jobs),
which differ only by `scripts/latency_table.py`. `python scripts/summarize.py --split test-dev`
→ `results/summary/test-dev{.md,_runs.csv,_experiments.csv}`.

Split `test-dev`: mean ± sample sd over seeds (n per column header).

| | E1 (n=3) | E2 (n=3) | E3 (n=1) | E4 (n=3) | E5 (n=3) |
|---|---|---|---|---|---|
| clean mAP50-95 | 0.1823 ± 0.0012 | 0.1825 ± 0.0035 | 0.2322 | 0.2312 ± 0.0009 | 0.2313 ± 0.0015 |
| clean mAP50 | 0.3217 ± 0.0024 | 0.3280 ± 0.0048 | 0.4033 | 0.4015 ± 0.0012 | 0.4039 ± 0.0020 |
| clean AP_small | 0.0910 ± 0.0021 | 0.0992 ± 0.0012 | 0.1402 | 0.1379 ± 0.0022 | 0.1414 ± 0.0007 |
| mPC (mAP50-95) | 0.1365 ± 0.0013 | 0.1364 ± 0.0014 | 0.1713 | 0.2092 ± 0.0012 | 0.2094 ± 0.0013 |
| rPC (mAP50-95) | 0.7489 ± 0.0036 | 0.7471 ± 0.0074 | 0.7377 | 0.9048 ± 0.0023 | 0.9055 ± 0.0011 |
| AP_small mPC | 0.0653 ± 0.0007 | 0.0699 ± 0.0004 | 0.0990 | 0.1208 ± 0.0010 | 0.1238 ± 0.0005 |
| AP_small rPC | 0.7180 ± 0.0097 | 0.7045 ± 0.0072 | 0.7062 | 0.8764 ± 0.0068 | 0.8757 ± 0.0022 |
| rPC underexposed | 0.9476 ± 0.0019 | 0.9599 ± 0.0033 | 0.9615 | 0.9823 ± 0.0012 | 0.9826 ± 0.0035 |
| rPC overexposed | 0.8889 ± 0.0042 | 0.8944 ± 0.0031 | 0.9085 | 0.9348 ± 0.0012 | 0.9336 ± 0.0050 |
| rPC blur | 0.7243 ± 0.0072 | 0.6923 ± 0.0126 | 0.5914 | 0.8137 ± 0.0060 | 0.8083 ± 0.0042 |
| rPC noise | 0.4934 ± 0.0044 | 0.4960 ± 0.0110 | 0.4826 | 0.8590 ± 0.0035 | 0.8629 ± 0.0023 |
| rPC low_contrast | 0.6904 ± 0.0182 | 0.6929 ± 0.0276 | 0.7442 | 0.9345 ± 0.0066 | 0.9400 ± 0.0029 |
| mAP50-95 severity 3 (mean of 5 types) | 0.1053 ± 0.0010 | 0.1043 ± 0.0009 | 0.1339 | 0.1915 ± 0.0013 | 0.1921 ± 0.0012 |

Mean differences (A − B) for the studied variable:

| contrast | Δ clean mAP50-95 | Δ mPC | Δ rPC | Δ clean AP_small | Δ AP_small mPC |
|---|---|---|---|---|---|
| E2 − E1 (model version @640) | +0.0002 | -0.0002 | -0.0018 | +0.0082 | +0.0046 |
| E3 − E2 (input size 640 -> 960) | +0.0497 | +0.0349 | -0.0094 | +0.0410 | +0.0291 |
| E4 − E3 (corruption-aware training) | -0.0010 | +0.0379 | +0.1672 | -0.0024 | +0.0218 |
| E5 − E4 (P2 head (optional)) | +0.0000 | +0.0002 | +0.0006 | +0.0035 | +0.0030 |

Batch-1 latency of the same checkpoints, all in one job (556084) on one GPU
(`python scripts/latency_table.py --job 556084` → `results/summary/latency_556084.{md,csv}`):

Batch-1 latency, one job (556084) on one GPU (a129, NVIDIA A100 80GB PCIe), FP16, eager PyTorch, 300 val images after 50 warm-up; mean of two passes in opposite model order; per experiment = mean over seeds (n).

| | E1 (n=3) | E2 (n=3) | E3 (n=1) | E4 (n=3) | E5 (n=3) |
|---|---|---|---|---|---|
| deploy (conf 0.25, max_det 300): wall p50 (ms) | 7.73 | 8.60 | 8.97 | 8.99 | 10.67 |
| deploy (conf 0.25, max_det 300): wall p95 (ms) | 8.09 | 8.80 | 9.01 | 9.25 | 10.93 |
| deploy (conf 0.25, max_det 300): preprocess (ms) | 1.11 | 0.97 | 1.10 | 1.17 | 1.34 |
| deploy (conf 0.25, max_det 300): inference (ms) | 5.72 | 7.09 | 7.01 | 7.01 | 8.51 |
| deploy (conf 0.25, max_det 300): postprocess (ms) | 0.76 | 0.29 | 0.29 | 0.29 | 0.29 |
| deploy (conf 0.25, max_det 300): GPU forward p50 (ms) | 5.60 | 7.02 | 6.94 | 6.94 | 8.44 |
| eval (conf 0.001, max_det 1000): wall p50 (ms) | 7.61 | 8.54 | 8.94 | 8.97 | 10.47 |
| eval (conf 0.001, max_det 1000): wall p95 (ms) | 7.82 | 8.65 | 8.99 | 9.14 | 10.54 |
| eval (conf 0.001, max_det 1000): preprocess (ms) | 0.90 | 0.88 | 1.09 | 1.14 | 1.09 |
| eval (conf 0.001, max_det 1000): inference (ms) | 5.67 | 7.10 | 6.99 | 6.99 | 8.53 |
| eval (conf 0.001, max_det 1000): postprocess (ms) | 0.81 | 0.29 | 0.29 | 0.29 | 0.29 |
| eval (conf 0.001, max_det 1000): GPU forward p50 (ms) | 5.59 | 7.03 | 6.93 | 6.92 | 8.47 |
| parameters (M) | 9.43 | 9.96 | 9.96 | 9.96 | 9.67 |

Largest |forward − reverse| difference of a run's wall p50: 1.45 ms (E1_s0, deploy). Peak VRAM is not reported from this multi-model process (earlier models' allocations inflate the peak); see the single-model per-run files results/latency/<run>.json.

Batch-16 val inference (ms/img, FP16; `results/summary/val.md` runs): E1 0.76, E2 0.80, E3 1.27, E4 1.27, E5 1.82.

**Findings (test-dev is the unbiased estimate; differences are means over seeds, sd 0.001–0.004):**
1. **Model version** (E2−E1, YOLO26s vs YOLO11s @640): clean mAP50-95 +0.0002, mPC −0.0002 — no difference; small
   objects favour YOLO26s (clean AP_small +0.008, AP_small mPC +0.005). Batch-1 latency: YOLO11s 7.73 vs YOLO26s 8.60 ms
   (YOLO11s forward 5.7 vs 7.1 ms; its NMS postprocess 0.76 vs 0.29 ms does not offset that).
2. **Input size** (E3−E2, 640→960): clean +0.050, mPC +0.035, AP_small mPC +0.029; rPC unchanged (0.747 → 0.738),
   i.e. the extra accuracy carries over to corrupted images proportionally rather than adding robustness.
   Cost: batch-16 inference 0.80 → 1.27 ms/img, training 1.5×; batch-1 latency only +0.4 ms (launch-bound).
3. **Corruption-aware training** (E4−E3): clean −0.001 (no clean cost), mPC +0.038, rPC 0.738 → 0.905, AP_small mPC
   +0.022; same network, so no inference cost. Largest gains: noise (rPC 0.48 → 0.86), blur (0.59 → 0.81),
   low contrast (0.74 → 0.93). Combined with 960 px (E4 vs E2): mPC +0.073, AP_small mPC +0.051.
4. **P2 head** (E5−E4, optional): the val advantage (+0.004 mPC) does not replicate: test-dev clean +0.0000, mPC
   +0.0002, AP_small +0.0035 / AP_small mPC +0.0030, at +1.7 ms batch-1 (+19%), +43% batch-16 inference and 1.84×
   training time. E5 was the best model declared before test-dev (val mPC); on test-dev it is equivalent to E4,
   so **E4 (YOLO26s@960 + corruption-aware training) is the better accuracy/cost choice**.
5. val → test-dev: every model loses 0.05–0.07 mAP50-95 (test-dev is harder and val also chose `best.pt`); the
   ordering of experiments is preserved (Spearman ρ = 0.89 between val and test-dev mPC over the 13 checkpoints;
   the swaps are within E1/E2 and within E4/E5).

**Limitations.** Training-time and test-time corruptions share families and parameter ranges (in-distribution
robustness only). E3 has one seed; n = 3 elsewhere, no significance tests. COCO-style evaluation with VisDrone
ignored regions, not the official VisDrone toolkit. Latency is eager PyTorch FP16 at batch 1 on an A100 (launch-bound),
not an optimised deployment engine. E5's initialisation differs from E4's (360/902 transferred items).

Compute: ~13 GPU-h on 2026-11-05 alone; project total 3,575 CpuSaat of 25,000 before today's last 6 h of jobs.
Raw test-dev predictions (gzip) take ~3.0 GB in `runs/eval/*/test-dev/` (not in git).

## License notes

- **Ultralytics** (YOLO11, YOLO26, training/validation code) is licensed under
  **AGPL-3.0** (https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
  This project uses it as a dependency; distributing this code or offering it as a network
  service brings AGPL-3.0 obligations. Ultralytics also offers a separate Enterprise License.
- **VisDrone2019-DET** (https://github.com/VisDrone/VisDrone-Dataset, Tianjin University) has its own
  terms of use; the dataset is not redistributed in this repository.
