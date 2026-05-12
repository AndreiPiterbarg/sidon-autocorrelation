# archive/misc_scripts/

Catch-all for support material that did not fit any single attack folder.

## External / reference

- `original_baseline_matlab.m` — Cloninger-Steinerberger 2017 reference
                                 prover (MATLAB, branch-and-prune over
                                 binned mass distributions on $[-1/4,1/4]$).
- `cones.c`, `scs.c`, `scs_matrix.c` — vendored SCS solver C source
                                       (used during a pinned-build experiment).
- `autoconv_nb.ipynb`           — auto-convolution exploration notebook.

## Bundles (frozen snapshots of dead branches)

- `bnb_bundle_v3.tar.gz`        — branch-and-bound v3 source bundle.
- `interval_bnb_bundle.tar.gz`  — interval-arithmetic BnB bundle.
- `interval_bnb_full.tar.gz`    — full interval-BnB tree dump.

## Helpers

- `_an_star_check.py`     — probe heuristic $a_n^*$ (cascade discrete optimum)
                            at $n = 4..128$; sanity check on cascade ceiling.
- `_an_star_results.log`  — output from the above.
- `_probe_c_sparsity.py`  — measure $c_\beta$ sparsity at LP optima to
                            justify active-set restriction.
- `_monitor_estimator.sh` — bash watchdog for `cascade_estimate` runs
                            (kill / restart on per-config timeout).
- `_kill_v2v3v4.ps1`      — PowerShell helper: kill stray
                            `_v2_v3_v4_bench` python processes on Windows.
