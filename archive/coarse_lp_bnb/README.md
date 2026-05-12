# archive/coarse_lp_bnb/

Scratch code from the coarse-LP / branch-and-bound cascade line of attack
on $C_{1a}$. Mostly `_`-prefixed session artifacts (kept verbatim). Files
were re-grouped 2026-05-11; nothing was renamed, content unchanged.

## Layout

| Subfolder            | Files | Contents |
|----------------------|------:|----------|
| `bnb/`               |    32 | Branch-and-bound: `_bnb_shor_*`, `_coarse_bnb_v2..v8`, validators, refiners, BnB epigraph probes. |
| `coarse_benches/`    |    41 | Coarse-LP bench runs `_coarse_*` (J/L/L2/N/NO/O/2W/KKT/v4/headroom/cascade-estimate) with logs/results. |
| `cascade_versions/`  |    26 | Cascade prover iterations: `_v2..v6`, `_hybrid_*`, `_coord_descent_v4plus`, `_deep_split_cascade`, `_rigorous_cascade_v5`, `coarse_cascade_prover{,_v2}.py`, `cascade_no_split/`. |
| `q_residue/`         |    14 | Q / QN residue-stage benches and JSON snapshots, plus `_qsweep_125.sh`. |
| `sdp_certs/`         |    37 | SDP / SOS / Lasserre / Putinar / palindromic / split-cell / joint-window certs and the `cert_pipeline/` package (BnB+SDP+Krawczyk+k-ladder+saddle-KKT). |
| `stuck_diagnostics/` |    29 | Stuck-box artifacts at d=10/d=14: `stuck_d1*` npz/pkl/json, `audit/dump/inspect/analyze` scripts, `stall_*`, `local_qp_*`, `contract_box.py`, `_dbg_d22_*`. |
| `sweeps/`            |    19 | Parameter sweeps: `sweep_K_d10/d22*`, `_sweep_125*`, `_scan_c_*`, `_k_sweep_iter4`, `_kernel_probe_helper`. |
| `smoke_and_tests/`   |    11 | `_test_*` and `test_*` smoke / integration scripts (PDLP, Schur, slack-elim, multistage). |
| `lp_diag/`           |    17 | LP-side diagnostics: `_diag_lp_gap*`, `_lp_*`, `_d14/d16/d2_*`, `_l0/_l1`, duality dumps, `_lake_*` log. |
| `bench_misc/`        |    64 | Misc benches/probes/refines: `_bench_*`, `_borderline_*`, `_el_extremiser_*`, `_fit_close_rate_*`, `_quick_*`, `_refine_*`, `_rigor_*`, `_step1_*`, `_struct_prune_*`, `_throughput/_tier1/_unified_pruning`, `_theorem4_atomic_nu`, `tube_method_v2`, `schnorr_euchner_tube`, `extrapolate`, `estimate_*`, `_cs17_paper.txt`, etc. |
| `_maybe_dup/`        |     1 | `_hybrid_v2_work/` whose `run.py` is byte-identical to `cascade_versions/_hybrid_rigorous_v2.py` (`cmp` confirmed). |

## Cleanup notes (2026-05-11)

- Deleted `__pycache__/` (1 stale `.pyc`, untracked, regenerable).
- Quarantined `_hybrid_v2_work/` to `_maybe_dup/` (exact byte duplicate of `_hybrid_rigorous_v2.py`).
- No content edits, no renames.
- Before: 292 entries (1 trash). After: 291 files in 11 subfolders.
