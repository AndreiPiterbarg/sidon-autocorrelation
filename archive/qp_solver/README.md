# qp_solver/

QP / Fejer-side `G` optimisation scratch — benchmarks, soundness probes,
and per-N solver caches used while pushing the QP ceiling and verifying
pointwise/cell certificates.

## Subfolders

- **F_benchmarks/** — `_F_*` Fejer-block sweeps: full / live / publication
  benchmarks, plus the verification driver (`_F_verify.py`).
- **FN_bench/** — Fejer-with-N-cap benchmark (`_FN_bench*`) and the
  combined F/Q/L cascade sweep at C=1.25 (`_FQL_sweep_125.sh`).
- **L_optimization/** — `_L_*` lower-bound / order-2 local search:
  bench driver, order-2 local test, and per (n,m) result JSONs.
- **mv_solves/** — Matolcsi–Vinuesa primal/extremal/ceiling `.npz`
  caches for N in {16,24,32,40,60,80,120}.
- **pointeval/** — Pointwise-evaluation certificates: 1.281 proof,
  axiom verification, correctness test, and push-max search.
- **phi_M_scan/** — `_phi_M_scan{,_v2,_v3}.py` scans of the phi-vs-M
  envelope used by the QP master.
- **kkt/** — Exact-QP and per-cell KKT certifiers
  (`_kkt_exact_qp.py`, `_kkt_cell_cert*`) with d4/d6 result JSONs.
- **S1_diag/** — Stage-1 diagnostics: enumeration, soundness, SDP/MOSEK
  assessment, tight-threshold and timing probes.
- **S3_active/** — Stage-3 active-set LP and assessment plus pod/local
  run logs (d8, d14).
- **qp_ceiling/** — QP-ceiling search, joint-bench drivers/results,
  soundness check, MOSEK IPM probe, and the full d=6/S=12 cascade log.

## Notes

All filenames keep their original `_X_…` prefixes; nothing renamed.
No deletions performed (no empty / dup / cache files were present).
