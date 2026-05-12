"""Smoke test for the robust Halpern-PDHG in tier4/pdlp_robust.py.

Compares against MOSEK ground truth on a sequence of small instances.
Goal: PDHG should reach KKT < 1e-4 and recover alpha to within ~1e-3
of the MOSEK optimum on every instance. This is the prerequisite for
the Tier-4 active-set + polish pipeline to make sense.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.tier4.pdlp_robust import solve_buildresult


def run_one(d, R, max_outer=80, max_inner=2000, tol=1e-6,
            free_var_box=50.0, halpern=True):
    print(f"\n=== d={d} R={R}  halpern={halpern}  box={free_var_box}"
          f"  outer={max_outer} inner={max_inner} tol={tol} ===", flush=True)
    _, M_mats_orig = build_window_matrices(d)
    M_mats, _ = project_window_set_to_z2_rescaled(M_mats_orig, d)
    d_eff = z2_dim(d)
    build = build_handelman_lp(d_eff, M_mats, BuildOptions(R=R, use_z2=True))
    print(f"  LP: n_eq={build.A_eq.shape[0]} n_vars={build.n_vars} "
          f"nnz={build.A_eq.nnz}", flush=True)

    t0 = time.time()
    sol_truth = solve_lp(build, solver="mosek")
    t_mosek = time.time() - t0
    print(f"  MOSEK: alpha={sol_truth.alpha:.8f}  wall={t_mosek:.2f}s", flush=True)

    t0 = time.time()
    res, scaling, alpha_pdlp, x_orig, y_orig = solve_buildresult(
        build,
        max_outer=max_outer, max_inner=max_inner, tol=tol,
        free_var_box=free_var_box, use_halpern=halpern,
        log_every=4, print_log=True,
    )
    t_pdlp = time.time() - t0
    print(f"  PDLP: alpha={alpha_pdlp:.8f}  diff={abs(sol_truth.alpha-alpha_pdlp):.2e}  "
          f"kkt={res.kkt:.2e}  conv={res.converged}  outer={res.n_outer}  "
          f"inner={res.n_inner_total}  wall={t_pdlp:.2f}s", flush=True)

    return dict(
        d=d, R=R, halpern=halpern, free_var_box=free_var_box,
        n_eq=build.A_eq.shape[0], n_vars=build.n_vars,
        alpha_mosek=sol_truth.alpha, alpha_pdlp=alpha_pdlp,
        diff=abs(sol_truth.alpha - alpha_pdlp),
        kkt=res.kkt, primal_res=res.primal_res, dual_res=res.dual_res,
        wall_mosek=t_mosek, wall_pdlp=t_pdlp,
        converged=res.converged, n_outer=res.n_outer, n_inner=res.n_inner_total,
    )


if __name__ == "__main__":
    rows = []
    # Sanity: d=8 R=4 is the smallest instance. Halpern should converge.
    rows.append(run_one(8, 4, max_outer=50, max_inner=1000, tol=1e-5))
    # d=8 R=8: harder
    rows.append(run_one(8, 8, max_outer=80, max_inner=2000, tol=1e-5))
    # d=10 R=4: bigger ambient dim
    rows.append(run_one(10, 4, max_outer=80, max_inner=2000, tol=1e-5))

    print("\n--- SUMMARY ---", flush=True)
    print(f"{'d':>3s} {'R':>3s} {'n_vars':>8s} {'alpha_mos':>10s} {'alpha_pdlp':>10s} "
          f"{'diff':>9s} {'kkt':>9s} {'outer':>6s} {'wall_pdlp':>9s} {'wall_mos':>9s}",
          flush=True)
    for r in rows:
        print(f"{r['d']:>3d} {r['R']:>3d} {r['n_vars']:>8d} "
              f"{r['alpha_mosek']:>10.6f} {r['alpha_pdlp']:>10.6f} "
              f"{r['diff']:>9.2e} {r['kkt']:>9.2e} {r['n_outer']:>6d} "
              f"{r['wall_pdlp']:>9.1f} {r['wall_mosek']:>9.2f}",
              flush=True)

    with open("_tier4_smoke_results.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nWrote _tier4_smoke_results.json", flush=True)
