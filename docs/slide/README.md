# `docs/slide/`

LaTeX Beamer presentation slides for the in-class project defense.

## Files

| File | Description |
|------|-------------|
| `main.tex` | Beamer source for the presentation |
| `Makefile` | Build automation (`make` / `make clean`) |
| `main.pdf` | Compiled slide deck (generated) |

## Building

Requires a TeX Live distribution with XeLaTeX:

```bash
make            # Compile main.tex → main.pdf (runs xelatex twice)
make clean      # Remove auxiliary files (.aux, .log, .nav, etc.)
make distclean  # Also remove main.pdf
```

## Notes

- The slides summarize the project methodology, experimental results, and demo.
- Figures referenced in the slides come from `docs/report/figures/` and
  `experiments/exp*/visualizations_test/`.
