# Documentation

Publication-facing documentation for the **Piterbarg--Bajaj--Vincent
Bound**, the rigorous lower bound $C_{1a} \ge 1292/1000 = 1.292$ on the
Sidon autocorrelation constant established in *Improving the Bounds on
the Supremum of Autoconvolutions*.

The headline artefacts (paper, Lean module, certifier driver) live at
the repository root and under [`../lean/`](../lean/) and
[`../delsarte_dual/`](../delsarte_dual/). Everything in this folder is
secondary documentation: math summary, Lean overview, build
instructions, audit specification, and an archive of dead-end attempts.

## Layout

| Path | Content |
|------|---------|
| [`proof_outline.md`](proof_outline.md) | Mathematical summary: the Matolcsi--Vinuesa master inequality, the three-scale arcsine kernel, the five certified anchors, and the strict-failure witness at $M = 1.292$. |
| [`formalization.md`](formalization.md) | Description of [`../lean/Sidon/MultiScale.lean`](../lean/Sidon/MultiScale.lean): headline theorem, the sole user axiom, and its correspondence with the paper. |
| [`reproducibility.md`](reproducibility.md) | Build instructions for the `flint.arb` certifier and the Lean formalisation; expected outputs and certificate hash. |
| [`verification.md`](verification.md) | Public audit specification: fourteen independent checks that exhaust the surface of the proof. |
| [`attempts/`](attempts/) | Archive of historical research directions (single-kernel sweeps, Lasserre/SOS, cascade estimators, interval branch-and-bound, GPU SCS, conditional Hölder, and others). See [`attempts/README.md`](attempts/README.md). |
| [`presentation/`](presentation/) | Slide deck and figure-generation scripts. |

## Conventions

- Math is rendered with `$...$` for inline and `$$...$$` for display.
- Cross-references use relative paths and point to files that exist.
- Reference values are quoted from
  [`../delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json`](../delsarte_dual/grid_bound_alt_kernel/certificates/reference_anchors.json).

## See also

- [`../README.md`](../README.md) -- repository-level overview.
- [`../lower_bound_proof.pdf`](../lower_bound_proof.pdf) /
  [`../lower_bound_proof.tex`](../lower_bound_proof.tex) -- the paper.
- [`../lean/README.md`](../lean/README.md) -- Lean toolchain notes and
  axiom inventory.
- [`../delsarte_dual/README.md`](../delsarte_dual/README.md) -- the
  productionised certifier pipeline.
