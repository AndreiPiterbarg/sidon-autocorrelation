"""Quick smoke for the HiGHS-IPM coarse backend.

Confirms that coarse_solve(tol=1e-4) reproduces the MOSEK alpha within
~1e-3 on small instances and is faster than the MOSEK 1e-9 ground truth.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.build import BuildOptions, build_handelman_lp, build_window_matrices
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.tier4.coarse_solve import coarse_solve


def run(d, R):
    _, M = build_window_matrices(d)
    M, _ = project_window_set_to_z2_rescaled(M, d)
    build = build_handelman_lp(z2_dim(d), M, BuildOptions(R=R, use_z2=True))
    print(f"d={d} R={R}: n_eq={build.A_eq.shape[0]} n_vars={build.n_vars} "
          f"nnz={build.A_eq.nnz}")

    t = time.time()
    truth = solve_lp(build, solver="mosek")
    t_truth = time.time() - t
    print(f"  MOSEK 1e-9 : alpha={truth.alpha:.8f}  wall={t_truth*1000:.1f}ms")

    for tol in (1e-3, 1e-4, 1e-5):
        t = time.time()
        res = coarse_solve(build, tol=tol, backend="highs_ipm", verbose=False)
        wall = time.time() - t
        if res.alpha is None:
            print(f"  HiGHS-IPM tol={tol:.0e}: FAILED status={res.raw_status}  wall={wall*1000:.1f}ms")
            continue
        diff = abs(truth.alpha - res.alpha)
        print(f"  HiGHS-IPM tol={tol:.0e}: alpha={res.alpha:.8f}  diff={diff:.2e}  "
              f"kkt={res.kkt:.2e}  wall={wall*1000:.1f}ms  speedup={t_truth/wall:.1f}x")


if __name__ == "__main__":
    for (d, R) in [(8, 4), (8, 8), (10, 4), (10, 6), (12, 4), (12, 6)]:
        run(d, R)
        print()
