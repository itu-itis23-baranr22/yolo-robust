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
