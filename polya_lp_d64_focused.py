"""Focused d=64 R=4 solve with both simplex and IPM tried."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import traceback

import numpy as np

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.symmetry import (
    project_window_set_to_z2_rescaled, z2_dim,
)
from lasserre.polya_lp.solve import solve_lp
from lasserre.polya_lp.runner import VAL_D_KNOWN


def main():
    d = 64
    target = VAL_D_KNOWN.get(d)
    print(f"d={d} focused LP solve (target val(d)={target})", flush=True)

    t0 = time.time()
    print("  Building windows...", flush=True)
    _, M_mats = build_window_matrices(d)
    M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d)
    d_eff = z2_dim(d)
    print(f"  Z/2: {d}->{d_eff}, windows {len(M_mats)}->{len(M_mats_eff)} ({time.time()-t0:.1f}s)", flush=True)

    for R in (4, 6):
        print(f"\n--- R={R} ---", flush=True)
        t_b = time.time()
        try:
            opts = BuildOptions(R=R, use_z2=True, use_q_polynomial=True, verbose=False)
            build = build_handelman_lp(d_eff, M_mats_eff, opts)
            print(f"  Build: n_vars={build.n_vars:,}, n_eq={build.A_eq.shape[0]:,}, "
                  f"nnz={build.n_nonzero_A:,}, wall={time.time()-t_b:.1f}s", flush=True)
        except Exception as e:
            print(f"  BUILD FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue

        for method in ("ipm", "simplex"):
            print(f"  Solving with method={method}...", flush=True)
            t_s = time.time()
            try:
                sol = solve_lp(build, method=method, verbose=False)
                wall = time.time() - t_s
                print(f"  [{method}] status={sol.status}, alpha={sol.alpha}, "
                      f"wall={wall:.1f}s", flush=True)
                if sol.alpha is not None:
                    gap_to_val = target - sol.alpha
                    print(f"  [{method}] gap to val(d)={gap_to_val:.6f}, "
                          f"alpha-target(1.281)={sol.alpha-1.281:+.6f}", flush=True)
                    break  # use first method that works
            except Exception as e:
                wall = time.time() - t_s
                print(f"  [{method}] FAILED after {wall:.1f}s: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()

        # Free memory before next R
        del build


if __name__ == "__main__":
    main()
