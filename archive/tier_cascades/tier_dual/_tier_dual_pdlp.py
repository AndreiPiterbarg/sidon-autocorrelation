"""Test PDLP on the dual LP.

The primal had ~n_q+1 free variables (alpha drift kills convergence).
The dual has only 1 free variable (y_simplex). If PDLP converges here,
it's a viable GPU coarse solver and Tier 4 becomes alive.

Tests:
  d=8 R=4 first (smallest). KKT target 1e-4 (active-set ID quality).
  Compare against MOSEK 1e-9 ground truth.
  Then scale up to d=12 R=8 if d=8 works.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.build import build_window_matrices
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.tier_dual.build_dual import build_dual_lp, summarize
from lasserre.polya_lp.tier_dual.solve_dual import (
    solve_dual_mosek, solve_dual_pdlp,
)


def run_one(d, R, max_outer=80, max_inner=2000, tol=1e-4,
            free_var_box=50.0, halpern=False, verbose=True):
    print(f"\n=== d={d} R={R}  tol={tol}  halpern={halpern} ===", flush=True)
    _, M_full = build_window_matrices(d)
    M_eff, _ = project_window_set_to_z2_rescaled(M_full, d)
    d_eff = z2_dim(d)
    build = build_dual_lp(d_eff, M_eff, R)
    print(f"  {summarize(build)}", flush=True)

    sol_mosek = solve_dual_mosek(build, tol=1e-9)
    print(f"  MOSEK dual: alpha={sol_mosek.alpha:.10f}  wall={sol_mosek.wall_s*1000:.1f}ms",
          flush=True)

    print(f"  --- PDLP on dual ---", flush=True)
    sol_pdlp = solve_dual_pdlp(
        build,
        max_outer=max_outer, max_inner=max_inner, tol=tol,
        free_var_box=free_var_box, use_halpern=halpern,
        verbose=verbose, log_every=4,
    )
    diff = (abs(sol_mosek.alpha - sol_pdlp.alpha)
            if sol_pdlp.alpha is not None and sol_mosek.alpha is not None
            else float("inf"))
    print(f"\n  PDLP dual : alpha={sol_pdlp.alpha:.6f}  "
          f"diff={diff:.2e}  kkt={sol_pdlp.kkt:.2e}  "
          f"converged={sol_pdlp.converged}  wall={sol_pdlp.wall_s*1000:.1f}ms",
          flush=True)
    return dict(
        d=d, R=R, halpern=halpern,
        alpha_mosek=sol_mosek.alpha, alpha_pdlp=sol_pdlp.alpha, diff=diff,
        kkt=sol_pdlp.kkt, primal_res=sol_pdlp.primal_res, dual_res=sol_pdlp.dual_res,
        converged=sol_pdlp.converged,
        wall_mosek_ms=sol_mosek.wall_s * 1000,
        wall_pdlp_ms=sol_pdlp.wall_s * 1000,
        n_vars=build.n_vars, n_eq=build.n_eq, n_ub=build.n_ub,
    )


if __name__ == "__main__":
    # Sanity: small first
    rows = []
    rows.append(run_one(8, 4, max_outer=60, max_inner=1000, tol=1e-4))
    rows.append(run_one(8, 4, max_outer=60, max_inner=1000, tol=1e-4, halpern=True))
    rows.append(run_one(8, 6, max_outer=80, max_inner=1500, tol=1e-4))
    rows.append(run_one(10, 4, max_outer=80, max_inner=1500, tol=1e-4))
    rows.append(run_one(12, 4, max_outer=80, max_inner=1500, tol=1e-4))

    print("\n--- SUMMARY ---", flush=True)
    print(f"{'d':>3s} {'R':>3s} {'halp':>5s} {'a_mosek':>10s} {'a_pdlp':>10s} "
          f"{'diff':>10s} {'kkt':>9s} {'pres':>9s} {'dres':>9s} "
          f"{'conv':>5s} {'w_mos':>7s} {'w_pdlp':>7s}", flush=True)
    for r in rows:
        if "error" in r:
            continue
        print(f"{r['d']:>3d} {r['R']:>3d} {str(r['halpern'])[:5]:>5s} "
              f"{r['alpha_mosek']:>10.6f} {r['alpha_pdlp']:>10.6f} "
              f"{r['diff']:>10.2e} {r['kkt']:>9.2e} "
              f"{r['primal_res']:>9.2e} {r['dual_res']:>9.2e} "
              f"{str(r['converged'])[:5]:>5s} {r['wall_mosek_ms']:>6.1f}ms "
              f"{r['wall_pdlp_ms']:>6.1f}ms",
              flush=True)

    with open("_tier_dual_pdlp_results.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("\nWrote _tier_dual_pdlp_results.json", flush=True)
