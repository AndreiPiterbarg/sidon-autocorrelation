# archive/cohn_elkies

Cohn-Elkies / single-kernel arcsine + 119-cosine experiments for the lower bound
on the Sidon autocorrelation constant `C_{1a}`. Direction is **DEAD**: single-kernel
CE ceiling is ~1.276 (see `MEMORY: project_cohn_elkies_status`,
`project_125_proven_ce`).

## Subfolders

- **v1_125_base/** — Foundational rigorous certs.
  - `_cohn_elkies_125.py` — rigorous `flint.arb` cert giving `C_{1a} >= 1.27428679`
    in ~30 s. Pivotal file.
  - `_cohn_elkies_128.py` — initial 1.28-target push.
- **greopt_v2/** — QP-reoptimised `G` line (`_cohn_elkies_128_greopt*`)
  plus the v2 iteration that fed it.
- **v3_v4_w11/** — v3 sweep + normalization, v4 results, plus the related
  `_w11_verify` and `_v4_w11_threescale` runs and a `_hybrid_v2_results.json`
  snapshot. Includes the v3 sweep-best Lean coefficient export
  (`_cohn_elkies_128_v3_sweepbest_coeffs.lean`).
- **v5_v6/** — v5 baseline + v6 (focused / N500 / sweep-all / three-scale
  N500 & N1000 finepoint coefficient JSONs).
- **v7_v8/** — v7 (incl. `_finalize`, `_finalize2`, `_recert`) and v8
  (`_tight` with xi=1e5/1e6 logs and final).

## Cleanup notes

- Deleted `__pycache__/` (3 untracked `.pyc`).
- Deleted `_cohn_elkies_128_v7_run_partial.log` (byte-identical to
  `_cohn_elkies_128_v7_run.log`, confirmed via `cmp`).

Before: 50 files flat (incl. pycache + 1 dup). After: 47 files in 5 subfolders.
