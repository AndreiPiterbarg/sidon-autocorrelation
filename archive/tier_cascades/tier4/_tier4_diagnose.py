"""Diagnose the existing pdlp.py at d=8 R=4 with a tighter free-var box.

The smoke run with free_var_box=1e3 diverges: alpha hits the box at 1000
instead of converging to the true ~1.067. Hypothesis: the free-variable
box is too loose for PDHG step sizes; alpha drifts faster than the dual
can react. Try free_var_box in {10, 100} and a smaller initial primal
weight to see whether the existing pdlp.py converges with sharper params.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.pdlp import build_gpu_lp, pdlp_solve, unscale_solution


def run_one(d, R, free_var_box, init_pw, max_outer=60):
    print(f"\n=== d={d} R={R} box={free_var_box} pw0={init_pw} ===", flush=True)
    _, M_mats_orig = build_window_matrices(d)
    M_mats, _ = project_window_set_to_z2_rescaled(M_mats_orig, d)
    d_eff = z2_dim(d)
    build = build_handelman_lp(d_eff, M_mats, BuildOptions(R=R, use_z2=True))
    print(f"  n_eq={build.A_eq.shape[0]} n_vars={build.n_vars} nnz={build.A_eq.nnz}", flush=True)

    sol_truth = solve_lp(build, solver="mosek")
    print(f"  MOSEK alpha = {sol_truth.alpha:.8f}", flush=True)

    lp, scaling = build_gpu_lp(
        build.A_eq, build.b_eq, build.c, build.bounds,
        ruiz_iter=20, free_var_box=free_var_box,
    )
    res = pdlp_solve(
        lp, max_outer=max_outer, max_inner=500, tol=1e-7,
        initial_primal_weight=init_pw, log_every=10, print_log=False,
    )
    x_orig, _ = unscale_solution(res.x, res.y, scaling)
    c_t = torch.from_numpy(build.c).to(x_orig.device).to(x_orig.dtype)
    alpha_pdlp = -float((c_t * x_orig).sum().item())

    print(f"  PDLP alpha = {alpha_pdlp:.6f}  diff={abs(sol_truth.alpha-alpha_pdlp):.2e}  "
          f"kkt={res.kkt:.2e}  pres={res.primal_res:.2e}  dres={res.dual_res:.2e}  "
          f"outers={res.n_outer}  wall={res.wall_s:.1f}s", flush=True)
    return res.kkt, alpha_pdlp, sol_truth.alpha


if __name__ == "__main__":
    for box in (10.0, 100.0, 1000.0):
        for pw in (1.0, 0.1, 10.0):
            try:
                run_one(8, 4, box, pw, max_outer=60)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
