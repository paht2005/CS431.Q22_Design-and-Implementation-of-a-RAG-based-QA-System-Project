# `docs/`

Documentation, reports, and presentation materials for the CS338.Q21 project
*"Zero-shot Object Counting with Good Exemplars"*.

## Contents

| Path | Description |
|------|-------------|
| [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) | Per-member contribution split and responsibilities |
| [`RESULTS.md`](RESULTS.md) | Numerical results summary (MAE, RMSE, latency) with provenance |
| [`report/`](report/) | LaTeX source and compiled `Report.pdf` (26-page project report) |
| [`slide/`](slide/) | LaTeX presentation slides for in-class defense |
| [`references/`](references/) | Reference materials and example slides from other projects |
| `slides/` | (Reserved for additional slide formats — currently empty) |

## Building the report

```bash
cd docs/report
make pdf        # Builds Report.pdf via latexmk + XeLaTeX
make watch      # Continuously rebuild on changes
make clean      # Remove auxiliary files
```

See [`report/README.md`](report/README.md) for prerequisite packages.

## Building the slides

```bash
cd docs/cs338-slide
make            # Builds main.pdf via XeLaTeX
make clean      # Remove auxiliary files
```
