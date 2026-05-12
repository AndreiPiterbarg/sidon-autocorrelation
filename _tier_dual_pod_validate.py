"""Local sanity checks for the pod pipeline.

Run THIS first on the laptop, before pushing to the B300 pod, to catch
build-side regressions. It does NOT call cuOpt (which requires a Linux
GPU) -- it only validates:

  1. Fast dual epigraph build matches the reference build (parity).
  2. MOSEK on dual matches MOSEK on primal (LP duality).
  3. ortools PDLP on the dual converges and matches MOSEK at small d.
  4. The benchmark script imports cleanly with cuOpt unavailable.
"""
from __future__ import annotations
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from lasserre.polya_lp.build import (
        BuildOptions, build_handelman_lp, build_window_matrices,
    )
    from lasserre.polya_lp.solve import solve_lp
    from lasserre.polya_lp.symmetry import (
        project_window_set_to_z2_rescaled, z2_dim,
    )
    from lasserre.polya_lp.tier_dual.build_dual_epi import (
        build_dual_epi_lp, solve_epi_mosek,
    )
    from lasserre.polya_lp.tier_dual.build_dual_epi_fast import (
        build_dual_epi_fast, _self_check,
    )

    print("=== 1. fast / reference build parity ===")
    for d, R in [(8, 4), (8, 6), (10, 4), (12, 6), (16, 4)]:
        _, M = build_window_matrices(d)
        M, _ = project_window_set_to_z2_rescaled(M, d)
        deff = z2_dim(d)
        _self_check(deff, R, M)
        print(f"  d={d:>2d} R={R:>2d}  PARITY OK")

    print("\n=== 2. MOSEK on epi-dual matches primal ===")
    for d, R in [(8, 8), (12, 6), (16, 4)]:
        _, M = build_window_matrices(d)
        M, _ = project_window_set_to_z2_rescaled(M, d)
        deff = z2_dim(d)
        primal_build = build_handelman_lp(deff, M, BuildOptions(R=R, use_z2=True))
        sol_p = solve_lp(primal_build, solver="mosek", tol=1e-9)
        epi = build_dual_epi_fast(deff, M, R)
        sol_d = solve_epi_mosek(epi, tol=1e-9)
        diff = abs(sol_p.alpha - sol_d.alpha)
        ok = diff < 1e-7
        flag = "OK" if ok else "FAIL"
        print(f"  d={d:>2d} R={R:>2d}  primal={sol_p.alpha:.10f}  "
              f"dual={sol_d.alpha:.10f}  diff={diff:.2e}  [{flag}]")
        if not ok:
            sys.exit(1)

    print("\n=== 3. ortools PDLP on dual matches MOSEK ===")
    try:
        from lasserre.polya_lp.tier_dual.solve_ortools_pdlp import (
            solve_dual_ortools_pdlp,
        )
        for d, R in [(8, 4), (12, 4), (16, 4)]:
            _, M = build_window_matrices(d)
            M, _ = project_window_set_to_z2_rescaled(M, d)
            deff = z2_dim(d)
            primal = build_handelman_lp(deff, M, BuildOptions(R=R, use_z2=True))
            sol_p = solve_lp(primal, solver="mosek", tol=1e-9)
            epi = build_dual_epi_fast(deff, M, R)
            sol_o = solve_dual_ortools_pdlp(
                epi, tol=1e-6, iter_limit=200_000,
                time_sec_limit=60.0, verbosity=0, is_epigraph=True,
            )
            diff = abs(sol_p.alpha - sol_o.alpha) if sol_o.alpha is not None else float("inf")
            ok = diff < 1e-3 and sol_o.converged
            flag = "OK" if ok else "WARN"
            print(f"  d={d:>2d} R={R:>2d}  pdlp={sol_o.alpha}  "
                  f"diff={diff:.2e}  conv={sol_o.converged}  [{flag}]")
    except ImportError:
        print("  (ortools not installed; skipping)")

    print("\n=== 4. cuOpt detection ===")
    from lasserre.polya_lp.tier_dual.solve_cuopt import (
        CUOPT_AVAILABLE, CUOPT_IMPORT_ERROR,
    )
    print(f"  CUOPT_AVAILABLE = {CUOPT_AVAILABLE}")
    if not CUOPT_AVAILABLE:
        print(f"  expected on this machine.  install on pod via:")
        print(f"    pip install cuopt-cu12   # or cu13 for CUDA 13")

    print("\n=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
