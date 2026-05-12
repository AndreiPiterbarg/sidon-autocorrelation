"""Sweep PDHG configurations on d=8 R=4 to find one that actually works.

Runs: vanilla PDHG (no Halpern), fixed omega, different eta, tighter box.
Goal: find at least ONE configuration that gets KKT < 1e-3 at d=8 R=4
in <30s. If none does, the LP needs a structural fix (warm-start dual,
or eliminate alpha analytically).
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
from lasserre.polya_lp.tier4 import pdlp_robust as pr


def build_d8R4():
    _, M = build_window_matrices(8)
    M, _ = project_window_set_to_z2_rescaled(M, 8)
    d_eff = z2_dim(8)
    return build_handelman_lp(d_eff, M, BuildOptions(R=4, use_z2=True))


def run_config(label, build, alpha_truth, **kw):
    t0 = time.time()
    res, scaling, alpha_pdlp, x_orig, y_orig = pr.solve_buildresult(
        build, log_every=10, print_log=False, **kw,
    )
    wall = time.time() - t0
    diff = abs(alpha_truth - alpha_pdlp)
    # Did the primal hit the box?
    x_max = x_orig.abs().max().item()
    print(f"  {label:<40s}  alpha={alpha_pdlp:+.6f}  diff={diff:.2e}  "
          f"kkt={res.kkt:.2e}  pres={res.primal_res:.2e}  dres={res.dual_res:.2e}  "
          f"|x|max={x_max:.1f}  wall={wall:.1f}s", flush=True)
    return res, alpha_pdlp


def main():
    build = build_d8R4()
    sol = solve_lp(build, solver="mosek")
    print(f"d=8 R=4: n_eq={build.A_eq.shape[0]} n_vars={build.n_vars}  "
          f"MOSEK alpha={sol.alpha:.8f}\n", flush=True)

    # Run a battery of configs. Each gets a budget of 60 outer x 500 inner.
    common = dict(max_outer=60, max_inner=500, tol=1e-6)
    print("--- Halpern + adaptive omega (current default) ---", flush=True)
    run_config("halpern, adaptive omega, box=50", build, sol.alpha,
               use_halpern=True, free_var_box=50.0, initial_primal_weight=1.0,
               **common)
    run_config("halpern, adaptive omega, box=10", build, sol.alpha,
               use_halpern=True, free_var_box=10.0, initial_primal_weight=1.0,
               **common)

    print("\n--- Vanilla PDHG (no Halpern), adaptive omega ---", flush=True)
    run_config("vanilla, adaptive omega, box=50", build, sol.alpha,
               use_halpern=False, free_var_box=50.0, initial_primal_weight=1.0,
               **common)
    run_config("vanilla, adaptive omega, box=10", build, sol.alpha,
               use_halpern=False, free_var_box=10.0, initial_primal_weight=1.0,
               **common)
    run_config("vanilla, adaptive omega, box=2", build, sol.alpha,
               use_halpern=False, free_var_box=2.0, initial_primal_weight=1.0,
               **common)

    print("\n--- Vanilla PDHG, fixed omega (no adaptation) ---", flush=True)
    # Hack: set primal weight not to update by setting tight bounds via initial value
    # Actually the omega update is internal. Just probe a few static values.
    for omega in [0.1, 0.3, 1.0, 3.0, 10.0]:
        run_config(f"vanilla, omega0={omega}, box=10", build, sol.alpha,
                   use_halpern=False, free_var_box=10.0,
                   initial_primal_weight=omega, **common)


if __name__ == "__main__":
    main()
