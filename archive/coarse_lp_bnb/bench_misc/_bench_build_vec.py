"""Benchmark build time at scale (d up to 32, R up to 8/10).

Measures only build wall-time (not solve), since vectorization targets build.
"""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.build import build_handelman_lp, BuildOptions, build_window_matrices
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled, z2_dim
from lasserre.polya_lp.solve import solve_lp


def bench(d, R):
    _, M_mats = build_window_matrices(d)
    M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d)
    d_eff = z2_dim(d)
    opts = BuildOptions(R=R, use_z2=True, eliminate_c_slacks=True)
    t0 = time.time()
    build = build_handelman_lp(d_eff, M_mats_eff, opts)
    t = time.time() - t0
    return dict(t=t, n_vars=build.n_vars, nnz=build.n_nonzero_A,
                n_eq=build.A_eq.shape[0],
                n_ub=(build.A_ub.shape[0] if build.A_ub is not None else 0))


print(f"{'d':>3} {'R':>3} {'n_vars':>10} {'nnz':>10} {'n_rows':>8} {'build_s':>10}")
print("-" * 60)
for d, R in [(8, 8), (8, 12), (16, 6), (16, 8), (16, 10), (24, 6), (24, 8), (32, 6), (32, 8)]:
    try:
        r = bench(d, R)
        print(f"{d:>3} {R:>3} {r['n_vars']:>10,} {r['nnz']:>10,} "
              f"{r['n_eq']+r['n_ub']:>8,} {r['t']:>10.3f}")
    except MemoryError:
        print(f"{d:>3} {R:>3}  OOM")
    except Exception as e:
        print(f"{d:>3} {R:>3}  err: {e}")
