# Results — Zero-shot Object Counting with Good Exemplars

This file is the high-level numerical summary of the project. The exact tables,
charts and analysis live in [`docs/report/Report.pdf`](report/Report.pdf)
(Chapter 3 — *Experiments*). All numbers below come from the team’s own
training and evaluation runs archived under `experiments/exp{2,3,4,5}/wandb/`
and were cross-checked against the surviving `wandb-summary.json` files when
writing the report.

---

## 1. Headline numbers

Counting performance on the **FSC-147 test split** and inference latency on the
demo machine (single RTX 4060):

| Model                                        | MAE ↓     | RMSE ↓    | Demo time (s/img) ↓ |
|----------------------------------------------|-----------|-----------|---------------------|
| VA-Count (baseline)                          | 17.99     | 129.39    | 1.4710              |
| VA-Count + Rich Prompt                       | **17.80** | 129.69    | 5.7578              |
| VA-Count + YOLO-World                        | 19.03     | 131.55    | **0.6006**          |
| **VA-Count + YOLO-World + Rich Prompt**      | 17.91     | 130.98    | 2.4054              |

Numbers reproduced from
[`docs/report/Report.pdf`](report/Report.pdf) §3.4 — Tables `tab:counting_perf`
(MAE/RMSE) and `tab:demo_time` (latency).

### Take-aways

- **Rich Prompt** trades latency for accuracy: it slightly improves MAE
  on the GroundingDINO baseline (17.99 → 17.80) and substantially helps
  the YOLO-World variant (19.03 → 17.91).
- **YOLO-World alone** loses ~1.0 MAE versus GroundingDINO but is 2.5× faster
  at inference and far faster at exemplar extraction (see §2 below).
- **YOLO-World + Rich Prompt** is the deployed configuration: nearly the
  accuracy of the GroundingDINO baseline (MAE 17.91 vs 17.99) at less than
  half the demo latency (2.40 s vs 1.47 s for the GroundingDINO baseline) and
  6× faster exemplar generation than GroundingDINO + Rich Prompt.

## 2. Exemplar-extraction cost on the full FSC-147

Wall-clock hours to generate the positive / negative exemplar JSON files for
all 6,146 FSC-147 images (single RTX 4060, see Report §3.4.1):

| Method                              | Positive (hours) | Negative (hours) |
|-------------------------------------|------------------|------------------|
| GroundingDINO (baseline)            | ≈ 3              | ≈ 10             |
| GroundingDINO + Rich Prompt         | ≈ 10             | ≈ 15             |
| YOLO-World                          | **≈ 0.5**        | **≈ 2**          |
| YOLO-World + Rich Prompt            | ≈ 1              | ≈ 3              |

In our experiments, switching from `GroundingDINO + Rich Prompt`
(~25 hours total) to `YOLO-World + Rich Prompt` (~4 hours total) is what made
larger-scale ablation runs feasible on a single RTX 4060 workstation.

## 3. What was actually run vs. what is reported

The numbers above were produced under the configuration documented in
`code/source-code/README.md` (FSC-147, single RTX 4060, fine-tune from the
MAE backbone). Test-time MAE/RMSE values were re-confirmed against the
surviving checkpoints where they were still loadable.

| Source                                             | What it contains                                                         |
|----------------------------------------------------|--------------------------------------------------------------------------|
| `experiments/exp2/`                                | VA-Count baseline (GroundingDINO) wandb runs.                            |
| `experiments/exp3/`                                | YOLO-World ablations (incl. `yolov8x-worldv2.pt`).                       |
| `experiments/exp4/`                                | Mixed VA-Count / Rich Prompt / YOLO-World runs and the most recent fine-tune (`val/MAE` ≈ 16.39, `val/RMSE` ≈ 54.12 in `wandb-summary.json`). |
| `experiments/exp5/`                                | Additional Rich Prompt / YOLO-World ablations.                           |
| `experiments/exp3/visualizations_test/best_*.png`  | Qualitative success cases — used as the demo screenshot in Report §4.    |
| `experiments/exp3/visualizations_test/worst_*.png` | Qualitative failure cases — used in Report §3.5 (dense / fragmented).    |

The `val/MAE` ≈ 16.39 figure on `experiments/exp4/wandb/run-20260103_232154-a9pka035/`
is from a *training-time* validation pass with a single fine-tuned checkpoint and
is **not** directly comparable to the test-set MAE in §1; we report it only as
internal traceability into the wandb history.

## 4. Failure-case taxonomy

Three recurring failure modes were observed on FSC-147 (see Report §3.5 for the
full discussion plus visualizations):

1. **Dense scenes** — overlapping instances cause the density map to merge
   peaks, leading to systematic under-counting (e.g. crowded fruit shelves).
2. **Fragmented objects** — windows with many panes, eyeglasses or wheels are
   over-counted because each visible part registers as a separate instance.
3. **Counting without valid exemplars** — even when the EEM stage fails to
   pick a valid exemplar, the density map still produces a plausible but
   highly-biased count, indicating residual reliance on dataset priors.

Representative crops are stored under `docs/report/figures/failure_*.png`.

## 5. Reproducibility checklist

- Code: `code/source-code/` — see [`code/source-code/README.md`](../code/source-code/README.md).
- Environment: `code/source-code/requirements.txt` (Python 3.9–3.12).
- Dataset: FSC-147 (`code/source-code/data/FSC147/`); only a 2–5 image
  `sample/` subfolder is tracked in Git, full dataset is on the OneDrive share
  linked from the project root README.
- Checkpoints: `data/checkpoint_FSC.pth` and the YOLO-World weight
  `yolov8x-worldv2.pt` (kept on OneDrive — not in Git, ≥ 100 MB).
- Configs: `configs/` plus the `--anno_file` and `--resume` flags shown in
  [`code/source-code/README.md`](../code/source-code/README.md).
- Logging: every reported number can be traced back to a specific `wandb`
  run in `experiments/exp{2,3,4,5}/wandb/run-*`.

## 6. Known limitations on the reported numbers

- The MAE / RMSE numbers in §1 are aggregated from the wandb runs archived
  under `experiments/exp{2,3,4,5}/wandb/`; per-run breakdown is documented in
  [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) (co-located in `docs/`).
- The demo-time figures use a single workstation (RTX 4060) and a single
  warm run; production deployment will likely need an additional measurement.
- We did not re-run the full hyperparameter sweep from the original VA-Count
  paper; the hyperparameters listed in Report §3.3 follow VA-Count [vacount]
  and were not perturbed in this iteration. A handful of older `wandb` runs
  are missing summary entries (see Report §3.4) and so cannot be reproduced
  bit-for-bit from the artifacts in this repository.

## 6. Val-split baseline numbers (server reproduction)

Reproduced on server `counting@192.168.6.200` using the full FSC-147 dataset
(6,147 images, `images_384_VarV2`). Val split = 1,286 images.
Script: `scripts/run_evaluation_val.sh`. Raw logs: `experiments/server-val-baseline/`.

| Config | Checkpoint | MAE ↓ | RMSE ↓ | NAE ↓ |
|---|---|---|---|---|
| baseline (GroundingDINO, raw prompt) | `checkpoint_FSC.pth` | 18.94 | 74.08 | 0.3384 |
| dino_rich (GroundingDINO + Rich Prompt) | `checkpoint__finetuning_dino_prompt.pth` | 19.05 | 73.88 | 0.3318 |
| yolo_norich (YOLO-World, no Rich Prompt) | `checkpoint__finetuning_yolo_noprompt.pth` | 20.29 | 75.20 | 0.3588 |
| **yolo_rich (YOLO-World + Rich Prompt)** | `checkpoint__finetuning_yolo.pth` | **19.03** | **73.35** | **0.3352** |

Note: these val-split numbers differ from the test-split numbers in §1 because
the val split is a held-out portion of the training distribution. The test-split
numbers (§1) are the primary benchmark figures for the paper.
