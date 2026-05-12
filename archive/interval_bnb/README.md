# Interval Branch-and-Bound for val(d)

**STATUS: DEAD direction.** Hit a rigor-parity barrier (Phase B
T1/T2/T3 net-negative); needs dual-cert or exact rational LP to
unlock. Kept for historical reference; the working rigorous LB
pipeline is in `archive/farkas_lasserre/` (val(4) > 1.0963 certified).

A rigorous lower-bound method for

    val(d) := min_{mu in Delta_d} max_{W in W_d}  mu^T M_W mu

with no discretisation correction: partitions the continuous simplex
into dyadic boxes and bounds each with interval arithmetic.

## Layout (post-reorg)

Top level (importable as `interval_bnb.<module>` from external code):

* `bnb.py` — best-first BnB driver
* `box.py` — dyadic-rational Box, exact simplex intersection
* `bound_eval.py` — natural / autoconv / McCormick (float + Fraction)
* `bound_anchor.py`, `bound_cctr.py`, `bound_epigraph.py` — extra cuts
* `bound_sdp_escalation{,_fast,_z2}.py` — SDP escalation cache
* `cctr_setup.py`, `lasserre_cert.py` — Lasserre / CCTR setup
* `windows.py` — wrapper over `lasserre.core.build_window_matrices`
* `symmetry.py` — half-simplex (Option C) cuts
* `parallel.py` — multiprocessing driver
* `eta.py` — ETA estimator
* `rigorous_check.py` — leaf-level Fraction verification
* `tree_d10.json` — saved BnB tree (d=10 pipeline check)

Subfolders (not externally imported):

* `runners/` — `run_d4`, `run_d10`, `run_d14`, `run_pod`, `bench_d12`
* `tests/` — `test_*.py`
* `diagnostics/` — `diagnose_*.py`

## Approach (summary)

1. Symmetry-reduced search on `H_d := { mu in Delta_d : mu_0 <= mu_{d-1} }`.
2. Best-first BnB; box certified if some W has `min mu^T M_W mu >= target_c`.
3. Three bounds combined (max): natural interval, autoconvolution
   complement, McCormick greedy LP.
4. Exact-arithmetic rigor replay: every leaf re-verified in
   `fractions.Fraction` with the same formula.

## Running

```
python -m interval_bnb.runners.run_d10 --target 1.24 --time_budget_s 3600
python -m interval_bnb.runners.run_d14 --target 1.2802
```

See git history for empirical status table (d=4 OK, d=10 in progress,
d>=14 blocked by rigor-parity barrier).
