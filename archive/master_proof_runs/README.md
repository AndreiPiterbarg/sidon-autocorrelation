# master_proof_runs/

Pod / cluster proof-run artifacts: deploy scripts, dimension-d proof drivers,
diagnostics, KKT / mu-star searches, and saved proof-phase NPZ certificates.

## Layout

- `deploy_scripts/` — `deploy_*.py/.sh` (bench, cascade, clarabel, d16/d18 pods,
  farkas, gap-accel, GPU-ADMM, interval-BnB, MOSEK, prime-intellect, SCS,
  sublevel, two-pods, z2-block, d8-L3 pilot) plus `launch_d16_on_pod.py`,
  `pod_setup.sh`, `setup_scs_gpu.sh`, `deploy_and_run.sh`.
- `run_d_dimension/` — per-dimension drivers (`run_d10` … `run_d30`) and
  matching logs (`d10_*log`, `d14_validate.log`, `d22_t1281*`, `d22_t1p2805*`,
  `d24_t1281`, `d30_t1281`, `epigraph_d10_log{1..5}`).
- `diagnose/` — `diagnose_*.py` + `*_output.log/json` for boundary-gap, cheap
  tests, d14 stuck, d30, full, SDP, stall (d10_t1208 / joint / root-cause),
  stuck-boxes investigations.
- `mu_star_kkt/` — `find_mu_star_d{10,22,24,30}.py`, `mu_star_d{10,20,22,24,30}.npz`,
  `mu_star_optimal.py`, `kkt_correct_mu_star.py`, `kkt_refinement.py`, KKT logs,
  `regen_mu_d20.py`, `km_gap_audit.py`.
- `pod_misc/` — `pod_bench_*`, `pod_d14_*`, `pod_d16_full.py`, `pod_feasibility.py`,
  `_pod_d*S*.json`, `_orch_*.json`, `_pod_v2v3*`, `_run_*.sh`, `_v5_pod_run.sh`.
- `proof_artifacts/` — `_master_pd_*` (compute/correct/structure), `_proof_*`
  (push v2/v3, sweep, launch shells), `_prove_*` (125, 1275, v5-compare, pool),
  `proof_d10_phase1_w0..w7.npz` + `master_queue.npz`, `prove_d10_t1p2.py`,
  `prove_dN_tT.py`, `verify_d16.py`, `run_proof.py`, `run_scaling_{test,v2,v7}.py`,
  `active_window_*`, `phase_rotation_results.json`.
- `cert_131_pod/`, `cpupod/`, `pod_results/` — pre-existing pod outputs; kept as is.
