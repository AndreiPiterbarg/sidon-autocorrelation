# lasserre/

Lasserre / SOS / Farkas-certified SDP attempts (archived; the working
rigorous pipeline is documented in `project_farkas_certified_lasserre.md`).

- `lasserre_solvers/` — solver-specific drivers: Mosek (cheby / dual /
  cliques / preelim / sublevel / tuned / z2block), SCS, Clarabel, DSOS-
  HiGHS, SDPNAL+, fusion, col-gen, scalable, sweeps; plus `lasserre_tssos.jl`.
- `run_d16_d32/` — the d=16 / d=32 production runners (l2, l3, adaptive,
  experiments, top10 cascade, BW31 Mosek, l3 estimators).
- `sweeps/` — hyperparameter / spectral / cg / aa-rho / d32-full /
  bisection / transfer sweeps + `parallel_bisect.py`.
- `push_val_knot/` — `push_val_knot_*` ladder (basinhop / fast / refine
  / sdp / upper) and `sanity_val_knot.py`.
- `tests/` — `test_*.py` for cell-min / refined-LP / dual-cliques /
  preelim / sos-dual / lasserre-d64 / improvements / sdpnal / trajectory,
  plus `val16_test.py`.
- `threepoint/` — three-point pilot / full / l3/2 level-2 / moment tests
  (see `project_threepoint_sdp_dead.md`, `project_path_a_l32_attempt.md`).
- `diagnostics/` — audit / interpret / investigate-feas / probe-mem /
  sdp-proof / sion-proof / box-cert-tightness / formula-b counterexample
  / aa-precond tuning.
- `farkas_bench/` — `_bench_farkas_fast`, `_bench_progressive`,
  `_farkas_profile`, `_atomic_nu_diagnostic`.
