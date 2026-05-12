# archive/legacy_tests/

Old test/bench/driver Python files for archived research directions on
the Sidon autocorrelation constant `C_{1a}`. Code here is not active —
kept for provenance.

## Subfolders

- `benchmarks/` — generic SDP solver / timing / profiling benchmarks
  (`bench_*`, `benchmark_*`, `profile_*`, MATLAB `sdpnal*`).
- `cloninger_steinerberger/` — Cloninger–Steinerberger reproduction:
  tests, verification scripts, GPU/ADMM, CPU↔GPU compare, vacuity probes,
  plus saved survivor/parent npy data in `cpu_gpu_data/`.
- `coarse_bnb/` — coarse branch-and-bound / cascade attempts: tests,
  prove/sweep drivers, parameter studies, coarse-grid drivers.
- `lasserre/` — Lasserre / Farkas SDP attempts: solver wrappers
  (`lasserre_*`), `d16/d32` runners, sweeps, push-val-knot ladder,
  three-point pilots, and diagnostics.

Each subfolder has its own README with the next layer of grouping.
