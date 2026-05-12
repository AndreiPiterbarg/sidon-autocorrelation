# phase_sweeps

Archived Phase-1 / Phase-2 sweep runs (`.py` driver + `.json` result + `.log`
output triples) from the C_{1a} lower-bound exploration.

## Subfolders

- `phase1/` — Phase-1 run logs
  - `_phase1_high_res.log` — high-resolution Phase-1 sweep

- `phase2_N1/` — Phase-2 N1 filter sweep (large JSON outputs ~40 MB each)
  - `_phase2_N1_all_off.json`, `_phase2_N1_only_F7.json`,
    `_phase2_N1_only_F8.json` — filter ablation results
  - `_phase2_N1_filter_diag.log`, `_phase2_N1_run.log`

- `phase2_N2/` — Phase-2 N2 parallel sweep
  - `_phase2_N2_parallel.py` — driver
  - `_phase2_N2_parallel_{272,274,2749,275,275_5M}.log` — runs at different
    target thresholds
  - `_phase2_N2_smoke_{175,27485}.log`, `_phase2_N2_lower_M_test.log`,
    `_phase2_N2_run.log` — smoke + diagnostic logs

- `phase2_marg/` — Phase-2 marginalisation diagnostics
  - `_phase2_marg_{smoke,fixed,sweep_unbuf}.log`, `_phase2_bug_diag.log`

- `phase_general/` — Phase-agnostic probes
  - `_phase_rotation_probe.py`, `_phase_rotation_v{2,3}.py` — phase-rotation
    probes
  - `_phase_sdp_m4.py` + `_phase_sdp_m4_results.json` +
    `_phase_sdp_m4_rigorous.json` — m=4 phase-SDP relaxation

All files retain their original `_phase*` prefix.
