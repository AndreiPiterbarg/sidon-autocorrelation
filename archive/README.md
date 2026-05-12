# Archive

Historical **code** for every research direction explored during the project. The working result lives in `delsarte_dual/` and `lean/` at the repo root; everything in here is either superseded, a dead-end, or auxiliary scaffolding kept for provenance.

> The matching **writeups, derivations, audits, and proof drafts** for these directions were archived prior to publication; this folder is code-only.

## Layout

| Folder | What's in it |
|---|---|
| `multiscale_arcsine/` | Multi-scale arcsine kernel exploration — the line that produced the working `C_{1a} >= 1.28984` result. Loose `_M*`, `_K20`–`_K32`, `_N*`, `_V*` probes, sweeps, and intermediate certificates. The productionised pipeline is in `delsarte_dual/grid_bound_alt_kernel/`. |
| `cohn_elkies/` | Cohn–Elkies / single-kernel arcsine experiments (`_cohn_elkies_125`, `_cohn_elkies_128_v*`). Capped at ~1.276 single-kernel; superseded by the multi-scale lift. |
| `audits/` | Independent verification probes — `_audit_*`, `_AUDIT_*`, `_b35_*`, `_bl_witness_*`, `_check_*`, `_verify_*`. Includes the 14-agent audit of the 1.28984 claim. |
| `coarse_lp_bnb/` | Coarse LP / branch-and-bound cascade work — `_coarse_*`, `_bnb_*`, `_cell_split_*`, `_diag_lp_*`, `_hybrid_*`, `_joint_all_window_sdp_*`, `_putinar_*`, `_Q_*`, `_QN_*`, `_palindromic_*`, plus `cert_pipeline/` and `cascade_no_split/`. The user later forbade large BnB / SDP routes. |
| `cascade_estimator/` | Cascade throughput / estimator runs — `_cascade_*`, `COARSE_CASCADE*`, `_pruning_proto*`, `_sdp_hardcell_*`, `_sonine_*`. |
| `polya_lp/` | Pólya-class LP attempts (`polya_*`). |
| `tier_cascades/` | Tiered cascade probes (`_tier_*`, `_tier4_*`). |
| `master_proof_runs/` | Pod / cluster proof runs — `deploy_*`, `run_d*`, `pod_*`, `_pod_*`, `_remote_*`, `find_mu_star_*`, `kkt_*`, `diagnose_*`, `proof_d10_phase1_*.npz`, plus `cert_131_pod/`, `cpupod/`, `pod_results/`, `proof/`. |
| `agent_experiments/` | Per-agent scratch from research sub-agents — `_agent_a/b/c/d/d8/d16/min_d/pod_run_synthesis_*`, `_agent9_*`, `_remote_*`, plus `_novel_agents/`. |
| `qp_solver/` | QP / Fejér-side `G` optimisation scratch — `_F_*`, `_FN_*`, `_FQL_*`, `_L_*`, `_phi_M_scan*`, `_pointeval_*`, `_mosek_ipm_*`, `_mv_*`. |
| `path_a_holder/` | Path-A Hölder-inequality lane — `_hausdorff*`, `_l3*`, `_asym_*`, `_mo_conj_29_*`, `_mo214_*`, `_krein_markov_*`, `_route_c_*`, `_theorem4_atomic_nu`, `_option_A_*`. Superseded by the multi-scale arcsine route. |
| `phase_sweeps/` | Phase-1 / phase-2 sweep runs (`_phase_*`, `_phase1_*`, `_phase2_*`). |
| `smoke_tests/` | Ad-hoc smoke runs (`_smoke_*`). Many document validated dead-ends — see project memory. |
| `lean_exports/` | Misc Lean export scripts (`_lean_*`). The live Lean formalisation is at the repo root in `lean/`. |
| `lasserre_farkas/` | Lasserre-hierarchy + Farkas-certified routes — `lasserre/`, `farkas_fast/`, `certified_lasserre/`, `polya_lp_mps/`. |
| `sdp_attempts/` | SDP-based attempts that did not produce the working bound — `bochner_sos/`, `simplex_window_dual/`, `chebyshev_dual/`. |
| `gpu_attempts/` | GPU / large-pod scaling attempts (`gpu/`, `gpupod/`). |
| `cloninger-steinerberger/` | Reproductions and probes of the (later-retracted) CS17 1.2802 claim. |
| `interval_bnb/` | Interval branch-and-bound scaffolding. |
| `parametric/`, `path_b_kbk/` | Smaller dead-end directions. |
| `data_runs/` | Bulk run output — old `data/`, `data_experiment/`, `experiments/`, `results/`, `runs/`, `runs_local/`. Largest folder by file count. |
| `scripts/`, `misc_scripts/` | Loose utility scripts (sweep drivers, pod helpers, C source for SCS, original CS17 MATLAB, an `.ipynb`, etc.). |

## Provenance

Each subfolder is roughly one direction or one phase of the project. File-name prefixes (`_M*`, `_K*`, `_V*`, etc.) are how the working session indexed parallel agent runs; the prefix often identifies the direction even where the folder doesn't.

For why most of these are here rather than at root, see the project memory in `~/.claude/projects/.../memory/`; the historical progress notes were archived prior to publication.
