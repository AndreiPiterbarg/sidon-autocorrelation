"""Compare coarse-solver backends on a fixed (d, R)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.tier4.coarse_solve import coarse_solve


def main():
    targets = [(10, 6), (12, 8), (14, 6), (14, 8), (16, 6)]
    for d, R in targets:
        _, M = build_window_matrices(d)
        M, _ = project_window_set_to_z2_rescaled(M, d)
        build = build_handelman_lp(z2_dim(d), M, BuildOptions(R=R, use_z2=True))
        sol = solve_lp(build, solver="mosek")
        print(f"\nd={d} R={R}: n_eq={build.A_eq.shape[0]} n_vars={build.n_vars} "
              f"  MOSEK 1e-9 alpha={sol.alpha:.8f}")

        for backend, kw in [
            ("highs_ipm", dict(tol=1e-5)),
            ("highs_simplex", dict()),
            ("mosek_ipm_low", dict(tol=1e-5)),
            ("mosek_ipm_low", dict(tol=1e-7)),
            ("mosek_simplex", dict()),
        ]:
            t = time.time()
            try:
                res = coarse_solve(build, backend=backend, **kw)
                wall = time.time() - t
                if res.alpha is None:
                    print(f"  {backend:<14s} {str(kw):>20s}: FAILED status={res.raw_status}  wall={wall*1000:.1f}ms")
                    continue
                diff = abs(sol.alpha - res.alpha)
                print(f"  {backend:<14s} {str(kw):>20s}: alpha={res.alpha:.8f}  "
                      f"diff={diff:.2e}  wall={wall*1000:.1f}ms")
            except Exception as e:
                print(f"  {backend:<14s} {str(kw):>20s}: ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
