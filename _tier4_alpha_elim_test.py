"""Test: rewrite the objective to eliminate alpha as the gradient driver.

Math: row beta=0 says  -alpha + q_0 - c_0 = 0  =>  alpha = q_0 - c_0.
So the objective  min -alpha  is equivalent to  min (c_0 - q_0).

Patch: in the LP we feed to PDHG, set c[alpha_idx] = 0 and add 1 to
c[c_0_idx], -1 to c[q_0_idx]. The row beta=0 then enforces the
identity alpha = q_0 - c_0, but the gradient driving alpha is gone.

This is a NUMERICALLY EQUIVALENT LP (same optimum, same duals up to a
shift on row beta=0) but PDHG no longer drives alpha to the box.
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
from lasserre.polya_lp.poly import index_map


def find_zero_index(monos):
    zero = tuple([0] * len(monos[0]))
    return monos.index(zero)


def patch_objective_for_alpha_elim(build):
    """Move the -1 cost off alpha and onto (c_0 - q_0)."""
    c_new = build.c.copy()
    alpha_idx = build.alpha_idx
    # Row beta=0 sits at row 0 in monos_le_R if zero is the first; locate it.
    monos = build.monos_le_R
    zero_pos = find_zero_index(monos)
    # The c-slack for beta=0 is at c_start + zero_pos; q at q_start + zero_pos.
    c_start = build.c_idx.start
    q_start = build.q_idx.start
    c0_idx = c_start + zero_pos
    q0_idx = q_start + zero_pos
    # Original c had c[alpha_idx] = -1 (we minimize -alpha).
    # Rewrite to: c[alpha] = 0, c[c0] += 1, c[q0] -= 1.
    cost_on_alpha = c_new[alpha_idx]
    c_new[alpha_idx] = 0.0
    c_new[c0_idx] += -cost_on_alpha   # i.e., +1
    c_new[q0_idx] += +cost_on_alpha   # i.e., -1
    # Confirm:
    assert abs(c_new[alpha_idx]) < 1e-15
    return c_new, alpha_idx, c0_idx, q0_idx


def main():
    _, M = build_window_matrices(8)
    M, _ = project_window_set_to_z2_rescaled(M, 8)
    build = build_handelman_lp(z2_dim(8), M, BuildOptions(R=4, use_z2=True))
    sol = solve_lp(build, solver="mosek")
    print(f"d=8 R=4: alpha_truth = {sol.alpha:.8f}", flush=True)

    c_new, alpha_idx, c0_idx, q0_idx = patch_objective_for_alpha_elim(build)
    print(f"  alpha_idx={alpha_idx}  c0_idx={c0_idx}  q0_idx={q0_idx}", flush=True)
    print(f"  c[alpha]={c_new[alpha_idx]}  c[c0]={c_new[c0_idx]}  c[q0]={c_new[q0_idx]}",
          flush=True)

    # Build a synthetic BuildResult-like object with patched c
    import copy
    build2 = copy.copy(build)
    build2.c = c_new

    print("\n--- PDHG on alpha-eliminated LP ---", flush=True)
    t0 = time.time()
    res, scaling, alpha_pdlp_obj, x_orig, y_orig = pr.solve_buildresult(
        build2,
        max_outer=80, max_inner=2000, tol=1e-6,
        free_var_box=50.0,
        use_halpern=False,
        log_every=4, print_log=True,
    )
    wall = time.time() - t0
    # The objective in the new LP equals (c_0 - q_0) at optimum, which
    # equals -alpha by the eliminated equation. So alpha_recovered = -obj.
    # solve_buildresult returns -obj as alpha_orig. But the *original* alpha
    # variable is also recoverable as x_orig[alpha_idx] (via the row
    # constraint). Let's report both.
    alpha_via_obj = alpha_pdlp_obj
    alpha_via_var = float(x_orig[alpha_idx].item())
    print(f"\n  alpha (via -obj)  = {alpha_via_obj:.8f}", flush=True)
    print(f"  alpha (via x_alpha) = {alpha_via_var:.8f}", flush=True)
    print(f"  truth             = {sol.alpha:.8f}", flush=True)
    print(f"  diff (obj)  = {abs(sol.alpha - alpha_via_obj):.2e}", flush=True)
    print(f"  diff (var)  = {abs(sol.alpha - alpha_via_var):.2e}", flush=True)
    print(f"  KKT={res.kkt:.2e}  pres={res.primal_res:.2e}  dres={res.dual_res:.2e}",
          flush=True)
    print(f"  |x|max={x_orig.abs().max().item():.3f}  wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
