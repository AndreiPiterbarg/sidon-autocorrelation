# cloninger-steinerberger/

Reproductions and probes of the Cloninger-Steinerberger 2017 lower bound
on C_{1a} (Proc. AMS 145(8), arXiv:1403.7988).

**Status (2026-05-11):** The CS17 LB of 1.2802 is **invalid** (matlab
mass/height bug). True current rigorous LB is MV's 1.2748. Material here
is retained as a reference reproduction of the (retracted) cascade
branch-and-prune approach; do not cite as a proof.

## Layout

Root (single-flat module — `core.py` re-exports from siblings):

- `core.py` — central import hub; pulls from the four sibling modules.
- `compositions.py` — composition enumeration (batched / canonical).
- `pruning.py` — `correction()`, asymmetry pruning, canonical masks.
- `test_values.py` — `compute_test_value*` (per-cell objective).
- `solvers.py` — `run_single_level`, `find_best_bound*`, generic provers.
- `cs_refined_lp.py` — refined LP probe over CS17 cells.
- `cs_refined_results.json` — output of the refined-LP probe.

`cpu/` — coarse cascade variants and CPU-side probes (flat namespace;
files cross-import each other by bare module name, so do not rename or
nest):

- `run_cascade.py`, `run_cascade_coarse{,_qp,_v2,_v3,_v4,_v5}.py` —
  successive cascade-coarse implementations; v5 is the latest.
- `cascade_opts.py`, `coarse_minimal_kernel.py`, `f_fn_fused.py`,
  `l_direct.py`, `post_filters.py`, `subdivision_cert.py` — cascade
  building blocks (F+FN fused, L direct, post-filters, subdivision
  certificates).
- `qp_bound.py`, `qp_bound_joint.py` — QP-style cell bounds.
- `benchmark.py` — timing harness.
