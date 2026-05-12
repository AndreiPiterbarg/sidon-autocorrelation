"""Test the epigraph dual: MOSEK first (soundness), then PDLP (the real test).

If PDLP converges on the epigraph dual, the structural reformulation
worked. If it still doesn't converge, the issue is deeper than free vars.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.tier_dual.build_dual_epi import (
    build_dual_epi_lp, summarize_epi, solve_epi_mosek, solve_epi_pdlp,
)


def run_one(d, R, do_pdlp=True, max_outer=80, max_inner=2000,
            tol=1e-4, halpern=False, verbose_pdlp=True):
    print(f"\n=== d={d} R={R} ===", flush=True)
    _, M_full = build_window_matrices(d)
    M_eff, _ = project_window_set_to_z2_rescaled(M_full, d)
    d_eff = z2_dim(d)

    # primal MOSEK ground truth
    t = time.time()
    primal_build = build_handelman_lp(d_eff, M_eff, BuildOptions(R=R, use_z2=True))
    sol_primal = solve_lp(primal_build, solver="mosek", tol=1e-9)
    print(f"  PRIMAL MOSEK: alpha={sol_primal.alpha:.10f}  "
          f"wall={(time.time()-t)*1000:.1f}ms", flush=True)

    # epi-dual MOSEK
    t = time.time()
    epi = build_dual_epi_lp(d_eff, M_eff, R)
    sol_em = solve_epi_mosek(epi, tol=1e-9)
    print(f"  {summarize_epi(epi)}", flush=True)
    print(f"  EPI MOSEK   : alpha={sol_em.alpha:.10f}  diff="
          f"{abs(sol_em.alpha-sol_primal.alpha):.2e}  "
          f"wall={(time.time()-t)*1000:.1f}ms", flush=True)

    if not do_pdlp:
        return

    # epi-dual PDLP
    print(f"  --- PDLP on epi dual (halpern={halpern}, tol={tol}) ---", flush=True)
    sol_ep = solve_epi_pdlp(
        epi, max_outer=max_outer, max_inner=max_inner, tol=tol,
        use_halpern=halpern, verbose=verbose_pdlp, log_every=4,
    )
    diff = (abs(sol_ep.alpha - sol_primal.alpha)
            if sol_ep.alpha is not None and sol_primal.alpha is not None
            else float("inf"))
    print(f"\n  EPI PDLP    : alpha={sol_ep.alpha:.6f}  diff={diff:.2e}  "
          f"kkt={sol_ep.kkt:.2e}  pres={sol_ep.primal_res:.2e}  "
          f"dres={sol_ep.dual_res:.2e}  conv={sol_ep.converged}  "
          f"wall={sol_ep.wall_s*1000:.1f}ms", flush=True)
    return sol_em, sol_ep


if __name__ == "__main__":
    # Sanity first
    run_one(8, 4, max_outer=50, max_inner=1000, tol=1e-4)
    run_one(8, 4, max_outer=50, max_inner=1000, tol=1e-4, halpern=True)
    run_one(8, 6, max_outer=80, max_inner=1500, tol=1e-4)
    run_one(10, 4, max_outer=80, max_inner=1500, tol=1e-4)
    run_one(12, 6, max_outer=100, max_inner=2000, tol=1e-4)
