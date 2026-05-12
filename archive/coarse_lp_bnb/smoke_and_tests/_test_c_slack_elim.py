"""Verify c_slack elimination produces same alpha as the original LP.

Before fix: alpha = NaN (inequality direction was flipped).
After fix:  alpha should match the equality-form LP exactly.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from lasserre.polya_lp.build import build_handelman_lp, BuildOptions, build_window_matrices
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.solve import solve_lp


def run(d, R, eliminate, use_z2=True):
    _, M_mats = build_window_matrices(d)
    if use_z2:
        M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d)
        d_eff = z2_dim(d)
    else:
        M_mats_eff = M_mats
        d_eff = d
    opts = BuildOptions(R=R, use_z2=use_z2, eliminate_c_slacks=eliminate, verbose=False)
    t0 = time.time()
    build = build_handelman_lp(d_eff, M_mats_eff, opts)
    t_build = time.time() - t0
    sol = solve_lp(build, solver="mosek", verbose=False)
    return dict(
        alpha=sol.alpha,
        n_vars=build.n_vars,
        n_eq=(build.A_eq.shape[0] if build.A_eq is not None else 0),
        n_ub=(build.A_ub.shape[0] if build.A_ub is not None else 0),
        nnz=(build.n_nonzero_A),
        t_build=t_build,
        t_solve=sol.wall_s,
        status=sol.status,
    )


print(f"{'R':>3} {'orig_alpha':>12} {'elim_alpha':>12} {'diff':>10} "
      f"{'orig_n_vars':>12} {'elim_n_vars':>12} {'reduction':>10} "
      f"{'orig_t':>8} {'elim_t':>8}")
print("-" * 110)
for R in [4, 6, 8, 10, 12]:
    o = run(8, R, eliminate=False)
    e = run(8, R, eliminate=True)
    diff = (None if o["alpha"] is None or e["alpha"] is None
            else abs(o["alpha"] - e["alpha"]))
    diff_str = f"{diff:.2e}" if diff is not None else "  N/A"
    oa = f"{o['alpha']:.6f}" if o["alpha"] is not None else "  N/A   "
    ea = f"{e['alpha']:.6f}" if e["alpha"] is not None else "  N/A   "
    red = f"{o['n_vars'] / e['n_vars']:.2f}x"
    print(f"{R:>3} {oa:>12} {ea:>12} {diff_str:>10} "
          f"{o['n_vars']:>12} {e['n_vars']:>12} {red:>10} "
          f"{o['t_solve']:>8.3f} {e['t_solve']:>8.3f}")
    if diff is not None and diff > 1e-6:
        print(f"   !!! MISMATCH: status orig={o['status']} elim={e['status']}")
