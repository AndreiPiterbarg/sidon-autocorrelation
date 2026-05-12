"""Confirm c_slack elimination scales: d=16, R in {6,8,10}."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    return dict(alpha=sol.alpha, n_vars=build.n_vars,
                t_build=t_build, t_solve=sol.wall_s, status=sol.status)


print(f"{'d':>3} {'R':>3} {'orig_alpha':>12} {'elim_alpha':>12} {'diff':>10} "
      f"{'orig_n_vars':>12} {'elim_n_vars':>12} {'red':>6} "
      f"{'orig_t':>8} {'elim_t':>8} {'speedup':>8}")
print("-" * 120)
for d, R in [(16, 4), (16, 6), (16, 8)]:
    o = run(d, R, eliminate=False)
    e = run(d, R, eliminate=True)
    diff = (None if o["alpha"] is None or e["alpha"] is None
            else abs(o["alpha"] - e["alpha"]))
    diff_str = f"{diff:.2e}" if diff is not None else "  N/A"
    oa = f"{o['alpha']:.6f}" if o["alpha"] is not None else "  N/A   "
    ea = f"{e['alpha']:.6f}" if e["alpha"] is not None else "  N/A   "
    red = f"{o['n_vars'] / e['n_vars']:.2f}x"
    sp = f"{o['t_solve'] / e['t_solve']:.2f}x" if e['t_solve'] > 0 else "N/A"
    print(f"{d:>3} {R:>3} {oa:>12} {ea:>12} {diff_str:>10} "
          f"{o['n_vars']:>12} {e['n_vars']:>12} {red:>6} "
          f"{o['t_solve']:>8.3f} {e['t_solve']:>8.3f} {sp:>8}")
