# STAT 4830 presentation

Slide deck and figure-generation pipeline for the STAT 4830 talk on
the Sidon autocorrelation lower-bound work.

## Contents

| File | Purpose |
|---|---|
| `STAT 4830 Presentation.pptx` | Primary deck (built by `build_deck.py`). |
| `STAT 4830 Presentation_alt.pptx` | Alternate variant of the deck (kept for reference). |
| `build_deck.py` | python-pptx script that assembles the deck from `figures/`. |
| `generate_figures.py` | Matplotlib script that produces the PNGs in `figures/`. |
| `figures/` | 7 PNGs consumed by `build_deck.py` (autoconv intuition, bounds number line, cascade schematic, correction gap, effective m, Lasserre ladder, three tracks). |
| `.gitignore` | Excludes `preview/` and `__pycache__/`. |

## Build

```
python generate_figures.py   # regenerates figures/*.png
python build_deck.py         # regenerates STAT 4830 Presentation.pptx
```

Both scripts share an editorial palette (paper background, serif type, three
per-track colours: navy = cascade, forest = SDP, brick = audit).
