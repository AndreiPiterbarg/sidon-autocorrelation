# tier_cascades

Tiered cascade probes for the Sidon C_{1a} lower-bound pipeline.

## Subfolders

### tier_dual/
Dual-cascade tier probes: epigraph relaxations, OR-tools/PDLP LP backends,
pod benchmarks, and verifier scripts (with paired `.log` and `_results.json`
output).

Files: `_tier_dual_{epi_test, ortools, pdlp, pod_bench, pod_validate, verify}.{py,log,json}`

### tier4/
Tier-4 specific probes: alpha-elimination, coarse smoke/compare runs,
end-to-end (e2e + e2e_large + v2_e2e), debug/diagnose harnesses, and
PDHG sweep.

Files: `_tier4_{alpha_elim_test, coarse_compare, coarse_smoke, debug,
diagnose, e2e, e2e_large, pdhg_sweep, smoke, v2_e2e}.{py,json}`

## Convention

Each `.py` script is paired (where applicable) with `_results.json` and/or
`.log` from its most recent run.
