"""Soundness verification: dual LP optimum should match primal LP optimum.

By LP duality this is a math identity. If alpha_dual_mosek != alpha_primal_mosek
to numerical precision, the build is wrong.

We test on a grid (d, R) of small instances. For each:
  1. Build primal LP, solve via MOSEK 1e-9.
  2. Build dual LP, solve via MOSEK 1e-9.
  3. Compare alpha values.
  4. Report dual LP shape (vars, eq, ub, free count) for sanity.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.tier_dual.build_dual import build_dual_lp, summarize
from lasserre.polya_lp.tier_dual.solve_dual import solve_dual_mosek


def run_one(d, R):
    print(f"\n=== d={d} R={R} ===", flush=True)
    _, M_full = build_window_matrices(d)
    M_eff, _ = project_window_set_to_z2_rescaled(M_full, d)
    d_eff = z2_dim(d)
    print(f"  d_eff={d_eff}, n_W={len(M_eff)}", flush=True)

    # Primal
    t = time.time()
    primal_build = build_handelman_lp(d_eff, M_eff, BuildOptions(R=R, use_z2=True))
    sol_primal = solve_lp(primal_build, solver="mosek", tol=1e-9)
    t_primal = time.time() - t
    n_free_primal = sum(
        1 for (lo, hi) in primal_build.bounds if lo is None and hi is None
    )
    print(f"  PRIMAL: alpha={sol_primal.alpha:.10f}  "
          f"n_vars={primal_build.n_vars}  n_eq={primal_build.A_eq.shape[0]}  "
          f"n_free={n_free_primal}  wall={t_primal*1000:.1f}ms", flush=True)

    # Dual
    t = time.time()
    dual_build = build_dual_lp(d_eff, M_eff, R)
    print(f"  {summarize(dual_build)}", flush=True)
    sol_dual = solve_dual_mosek(dual_build, tol=1e-9)
    t_dual = time.time() - t
    n_free_dual = sum(
        1 for (lo, hi) in dual_build.bounds if lo is None and hi is None
    )
    print(f"  DUAL  : alpha={sol_dual.alpha:.10f}  "
          f"n_vars={dual_build.n_vars}  n_eq={dual_build.n_eq}  "
          f"n_ub={dual_build.n_ub}  n_free={n_free_dual}  "
          f"wall={t_dual*1000:.1f}ms", flush=True)

    if sol_primal.alpha is None or sol_dual.alpha is None:
        diff = float("inf")
    else:
        diff = abs(sol_primal.alpha - sol_dual.alpha)
    print(f"  diff = {diff:.2e}  primal_free={n_free_primal} -> dual_free={n_free_dual}",
          flush=True)
    return dict(
        d=d, R=R, d_eff=d_eff, n_W=len(M_eff),
        alpha_primal=sol_primal.alpha,
        alpha_dual=sol_dual.alpha,
        diff=diff,
        primal_n_vars=primal_build.n_vars,
        primal_n_eq=primal_build.A_eq.shape[0],
        primal_n_free=n_free_primal,
        dual_n_vars=dual_build.n_vars,
        dual_n_eq=dual_build.n_eq,
        dual_n_ub=dual_build.n_ub,
        dual_n_free=n_free_dual,
        wall_primal_ms=t_primal * 1000,
        wall_dual_ms=t_dual * 1000,
    )


if __name__ == "__main__":
    grid = [
        (8, 4), (8, 6), (8, 8), (8, 10),
        (10, 4), (10, 6), (10, 8),
        (12, 4), (12, 6), (12, 8),
        (16, 4), (16, 6),
    ]
    rows = []
    for d, R in grid:
        try:
            rows.append(run_one(d, R))
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append(dict(d=d, R=R, error=str(e)))

    print("\n--- SUMMARY ---", flush=True)
    print(f"{'d':>3s} {'R':>3s} {'a_primal':>12s} {'a_dual':>12s} {'diff':>10s} "
          f"{'p_free':>7s} {'d_free':>7s} {'p_vars':>7s} {'d_vars':>7s} "
          f"{'wall_p':>8s} {'wall_d':>8s}", flush=True)
    for r in rows:
        if "error" in r:
            print(f"{r['d']:>3d} {r['R']:>3d}  ERROR: {r['error'][:70]}",
                  flush=True)
            continue
        print(f"{r['d']:>3d} {r['R']:>3d} "
              f"{r['alpha_primal']:>12.8f} {r['alpha_dual']:>12.8f} "
              f"{r['diff']:>10.2e} "
              f"{r['primal_n_free']:>7d} {r['dual_n_free']:>7d} "
              f"{r['primal_n_vars']:>7d} {r['dual_n_vars']:>7d} "
              f"{r['wall_primal_ms']:>7.1f}ms {r['wall_dual_ms']:>7.1f}ms",
              flush=True)

    with open("_tier_dual_verify_results.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("\nWrote _tier_dual_verify_results.json", flush=True)
