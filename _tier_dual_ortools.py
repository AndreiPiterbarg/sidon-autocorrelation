"""Test Google PDLP (ortools) on the Sidon dual epigraph LP.

This is the production-grade PDLP. If it converges here, the dual
reformulation path is validated and the only remaining question is GPU
vs CPU wall time.

Tests:
  1. d=8 R=4 epigraph dual : convergence + alpha vs MOSEK
  2. d=12 R=8, d=14 R=6, d=16 R=6, d=16 R=8 : scaling
  3. Wall-time vs MOSEK monolithic primal (the breakthrough baseline)
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.tier_dual.build_dual_epi import (
    build_dual_epi_lp, summarize_epi,
)
from lasserre.polya_lp.tier_dual.solve_ortools_pdlp import (
    solve_dual_ortools_pdlp,
)


def run_one(d, R, tol=1e-6, iter_limit=100000, time_limit=120.0,
            verbosity=2):
    print(f"\n=== d={d} R={R}  PDLP tol={tol:.0e} ===", flush=True)
    _, M_full = build_window_matrices(d)
    M_eff, _ = project_window_set_to_z2_rescaled(M_full, d)
    d_eff = z2_dim(d)

    # Primal MOSEK ground truth
    t = time.time()
    primal_build = build_handelman_lp(
        d_eff, M_eff, BuildOptions(R=R, use_z2=True),
    )
    sol_primal = solve_lp(primal_build, solver="mosek", tol=1e-9)
    t_primal = time.time() - t
    print(f"  PRIMAL MOSEK : alpha={sol_primal.alpha:.10f}  wall={t_primal*1000:.1f}ms",
          flush=True)

    # Build epigraph dual
    epi = build_dual_epi_lp(d_eff, M_eff, R)
    print(f"  {summarize_epi(epi)}", flush=True)

    # Run Google PDLP
    print(f"  --- Google PDLP on epi dual ---", flush=True)
    t0 = time.time()
    sol_pdlp = solve_dual_ortools_pdlp(
        epi, tol=tol, iter_limit=iter_limit,
        time_sec_limit=time_limit, verbosity=verbosity,
        is_epigraph=True,
    )
    t_pdlp = time.time() - t0
    diff = (abs(sol_pdlp.alpha - sol_primal.alpha)
            if sol_pdlp.alpha is not None and sol_primal.alpha is not None
            else float("inf"))
    speedup = t_primal / t_pdlp if t_pdlp > 0 else float("nan")
    print(f"  ortools PDLP : alpha={sol_pdlp.alpha:.10f}  diff={diff:.2e}  "
          f"status={sol_pdlp.raw_status}  conv={sol_pdlp.converged}",
          flush=True)
    print(f"               primal_res={sol_pdlp.primal_res:.2e} "
          f"dual_res={sol_pdlp.dual_res:.2e}  "
          f"wall={t_pdlp*1000:.1f}ms  speedup={speedup:.2f}x", flush=True)

    return dict(
        d=d, R=R,
        alpha_truth=sol_primal.alpha,
        alpha_pdlp=sol_pdlp.alpha,
        diff=diff,
        wall_truth_ms=t_primal * 1000,
        wall_pdlp_ms=t_pdlp * 1000,
        speedup=speedup,
        converged=sol_pdlp.converged,
        status=sol_pdlp.raw_status,
        primal_res=sol_pdlp.primal_res,
        dual_res=sol_pdlp.dual_res,
        n_vars=epi.n_vars, n_eq=epi.n_eq, n_ub=epi.n_ub,
    )


if __name__ == "__main__":
    grid = [
        (8, 4), (8, 8),
        (10, 6),
        (12, 6), (12, 8),
        (14, 6),
        (16, 4), (16, 6),
    ]
    rows = []
    for d, R in grid:
        try:
            rows.append(run_one(d, R, tol=1e-6, iter_limit=100000,
                                time_limit=180.0, verbosity=1))
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append(dict(d=d, R=R, error=str(e)))

    print("\n--- SUMMARY ---", flush=True)
    print(f"{'d':>3s} {'R':>3s} {'a_truth':>12s} {'a_pdlp':>12s} {'diff':>10s} "
          f"{'w_truth':>9s} {'w_pdlp':>9s} {'spd':>6s} {'conv':>5s}",
          flush=True)
    for r in rows:
        if "error" in r:
            print(f"{r['d']:>3d} {r['R']:>3d}  ERROR: {r['error'][:70]}", flush=True)
            continue
        print(f"{r['d']:>3d} {r['R']:>3d} {r['alpha_truth']:>12.8f} "
              f"{r['alpha_pdlp']:>12.8f} {r['diff']:>10.2e} "
              f"{r['wall_truth_ms']:>8.1f}ms {r['wall_pdlp_ms']:>8.1f}ms "
              f"{r['speedup']:>5.2f}x {str(r['converged'])[:5]:>5s}",
              flush=True)

    with open("_tier_dual_ortools_results.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print("\nWrote _tier_dual_ortools_results.json", flush=True)
