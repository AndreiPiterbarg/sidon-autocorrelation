"""Probe c_beta sparsity at the LP optimum.

If a small fraction of c_beta are positive (>tol) at the optimum, then the
"active row set" Sigma_R^* is a small subset of the full Sigma_R, and we
can shrink subsequent LPs by restricting to that active set + cutting-plane
verification.

This probe runs a sequence of (d, R) at full LP and counts:
  - n_active_c    = #{beta : c_beta > tol}
  - n_active_c_strict = #{beta : c_beta > 1e-6}  (large enough to matter)
  - distribution of c_beta magnitudes
  - n_active_q    = #{K : |q_K| > tol}
  - n_active_lambda
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.solve import solve_lp


def probe(d, R):
    print(f"\n=== d={d} R={R} ===", flush=True)
    _, M_mats = build_window_matrices(d)
    M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d)
    d_eff = z2_dim(d)
    opts = BuildOptions(R=R, use_z2=True, eliminate_c_slacks=False,
                        use_q_polynomial=True)
    t0 = time.time()
    build = build_handelman_lp(d_eff, M_mats_eff, opts)
    t_build = time.time() - t0
    print(f"  build: {t_build:.1f}s rows={build.A_eq.shape[0]} "
          f"vars={build.n_vars}", flush=True)
    t0 = time.time()
    sol = solve_lp(build, solver='mosek', verbose=False)
    t_solve = time.time() - t0
    if sol.alpha is None:
        print(f"  SOLVE FAILED status={sol.status}", flush=True)
        return None
    print(f"  solve: {t_solve:.1f}s alpha={sol.alpha:.7f}", flush=True)

    x = sol.x
    lam = x[build.lambda_idx]
    q = x[build.q_idx]
    c = x[build.c_idx]
    n_le_R = len(build.monos_le_R)
    n_q = len(build.monos_le_Rm1)
    n_W = build.n_windows

    # Active-set counts at multiple thresholds
    thresholds = [1e-9, 1e-6, 1e-4, 1e-2]
    out = dict(d=d, R=R, alpha=float(sol.alpha), t_solve=t_solve,
               n_W=n_W, n_q=n_q, n_le_R=n_le_R)
    for tol in thresholds:
        n_active_c = int((np.abs(c) > tol).sum())
        n_active_q = int((np.abs(q) > tol).sum())
        n_active_lam = int((np.abs(lam) > tol).sum())
        out[f"c_active@{tol:.0e}"] = n_active_c
        out[f"q_active@{tol:.0e}"] = n_active_q
        out[f"lam_active@{tol:.0e}"] = n_active_lam
        print(f"  tol={tol:.0e}: c_active={n_active_c}/{n_le_R} "
              f"({100.0*n_active_c/n_le_R:.1f}%), "
              f"q_active={n_active_q}/{n_q} "
              f"({100.0*n_active_q/n_q:.1f}%), "
              f"lam_active={n_active_lam}/{n_W} "
              f"({100.0*n_active_lam/n_W:.1f}%)",
              flush=True)

    # Distribution percentiles
    c_abs = np.abs(c)
    c_sorted = np.sort(c_abs)[::-1]  # descending
    pcts = [50, 75, 90, 95, 99, 99.9]
    for p in pcts:
        idx = max(0, int(len(c_sorted) * p / 100) - 1)
        out[f"c_p{p}"] = float(c_sorted[idx])
        print(f"  c at top-{100-p:.1f}% percentile (idx {idx}): "
              f"{c_sorted[idx]:.3e}", flush=True)
    out["c_max"] = float(c_abs.max())

    return out


SCHEDULE = [
    (16, 6),
    (16, 8),
    (16, 10),
    (16, 12),
]
results = []
for d, R in SCHEDULE:
    r = probe(d, R)
    if r is not None:
        results.append(r)
    with open('c_sparsity_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)


print("\n\n=== SUMMARY: c_beta active-set fraction at LP optimum ===\n", flush=True)
print(f"{'(d,R)':>10} {'n_le_R':>10} "
      f"{'c_act@1e-9':>12} {'%':>5} "
      f"{'c_act@1e-6':>12} {'%':>5} "
      f"{'c_act@1e-4':>12} {'%':>5}", flush=True)
print("-" * 80)
for r in results:
    case = f"d{r['d']}R{r['R']}"
    n = r['n_le_R']
    a9 = r['c_active@1e-09']; a6 = r['c_active@1e-06']; a4 = r['c_active@1e-04']
    print(f"{case:>10} {n:>10} {a9:>12} {100*a9/n:>5.1f} "
          f"{a6:>12} {100*a6/n:>5.1f} {a4:>12} {100*a4/n:>5.1f}", flush=True)
