### Experiments directory

This folder stores archived experiment runs and logs that support the results
reported in the project report (`docs/report/Report.pdf`). All runs were
produced by the CS338.Q21 team while training and evaluating the VA-Count
baseline and its Rich Prompt / YOLO-World extensions.

Each subfolder typically contains:

- `data/` – local copies or subsets of FSC147 (images, density maps, annotations)
- `GroundingDINO/` – installed GroundingDINO package and its weights
- `visualizations/`, `visualizations_test/` – qualitative result images
- `wandb/` – training and evaluation logs
- Additional assets such as YOLO-World weights

Current layout:

```text
experiments/
├── exp2/
├── exp3/
├── exp4/
└── exp5/
```

- **`exp2/`**  
  General VA-Count experiments (baseline and/or Rich Prompt), including notebooks,
  GroundingDINO installation, data and `wandb` logs.

- **`exp3/`**  
  Experiments involving **YOLO-World** (contains `yolov8x-worldv2.pt`), together with
  data, GroundingDINO, visualizations and `wandb` logs. These runs correspond to the
  YOLO-World and YOLO-World + Rich Prompt results in the report.

- **`exp4/`** and **`exp5/`**  
  Additional VA-Count / Rich Prompt / YOLO-World runs and ablations. Each folder
  includes its own `data/`, GroundingDINO environment, visualizations and `wandb`
  logs for reproducibility.

You generally do **not** need these folders to run the code in `code/`.  
They are kept for:

- Reproducing exact numbers in tables (MAE, RMSE, runtime)
- Inspecting qualitative examples (failure cases, dense scenes, fragmented objects)
- Debugging or extending the current experiments

