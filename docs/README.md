# Documentation

Publication-facing documentation for the **Piterbarg--Bajaj--Vincent
Bound**, the rigorous lower bound $C_{1a} \ge 1292/1000 = 1.292$ on the
Sidon autocorrelation constant established in *A New Lower Bound for
the Supremum of Autoconvolutions*.

The headline artefacts (paper, Lean module, certifier driver) live at
the repository root and under [`../lean/`](../lean/) and
[`../delsarte_dual/`](../delsarte_dual/). Everything in this folder is
secondary documentation: math summary, Lean overview, build
instructions, audit specification, and an archive of dead-end attempts.

## Layout

| Path | Content |
|------|---------|
| [`proof_outline.md`](proof_outline.md) | Mathematical summary: the Matolcsi--Vinuesa master inequality, the three-scale arcsine kernel, the five certified anchors, the strict-failure witness at $M = 1.292$, and the axiom-budget comparison with MV 2010. |
| [`formalization.md`](formalization.md) | Description of the fifteen-module Lean formalisation under [`../lean/Sidon/`](../lean/Sidon/): the three headline theorems (`autoconvolution_ratio_ge_1292_1000`, `_schwartz`, `_schwartz_residual`), the two verifiable-by-computation axioms (rigorously certified `flint.arb` numerical assertions), the `ExtremiserPrimitives` / `SchwartzAtomic` / `SchwartzAtomicResidual` records, and the correspondence with the paper. |
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
