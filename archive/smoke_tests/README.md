# smoke_tests/

Ad-hoc smoke runs (`_smoke_*` prefix). Each topic typically has a `.py`
script paired with a `.json` result and/or `.log` file from the same run.
Most document validated dead-ends or speedup probes for the cascade pipeline.

## Subfolders

### sdp/  (28 files)
SDP-based pruners and audits: split-cell SDP variants
(`_smoke_split_cell_SDP*`, `_smoke_split_cell_optionC_*`),
cluster/symmetric/sparse SDP, DSOS filter, direct-MOSEK,
Lasserre timing, and RLT cut audits.

### lp_solver/  (15 files)
LP-stage and L-stage solver experiments:
batched/warm/numba LP, GPU L-tier, KKT-rigorous L,
active-constraints L, L1-vs-L2 study, solver_compare.

### chain_perf/  (24 files)
Cascade chain performance and convergence:
M-chain (windowed), Q joint-spectral, QN iter, exact-Q,
profile_chain, fused F+FN, persistent pool,
c_target / converge sweeps, cascade-vs-split compare,
d=10 BnB smoke, prescreen.

### continuous_bounds/  (22 files)
Continuous-f and TV bounds checks:
TV_continuous_check (v1/v2/v3), step_continuous_gap,
sos_continuous_max, linfty_continuity, Fourier check,
bin-avg, max_t lower bound, tv-vs-maxt, Bochner test,
Boyer-Li probe, moment-LP logs.

## Notes
- No files were deleted; no duplicates or empty files were found.
- Filenames preserved (incl. `_smoke_` prefix) for grep/history continuity.
