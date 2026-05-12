"""Can we even BUILD the LP at d=64 (Z/2 d_eff=32) for low R?

Just measure construction sizes — don't solve yet.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
from math import comb

import numpy as np

from lasserre.polya_lp.build import (
    BuildOptions, build_handelman_lp, build_window_matrices,
)
from lasserre.polya_lp.symmetry import (
    project_window_set_to_z2_rescaled, z2_dim,
)


def main():
    print("d=64 LP build test (no solve)")
    print(f"{'R':>3} {'n_eq theory':>14} {'n_var theory':>14} "
          f"{'n_eq actual':>14} {'n_var actual':>14} "
          f"{'nnz':>14} {'build_s':>10}")

    d = 64
    d_eff = z2_dim(d)
    print(f"d={d}, Z/2 d_eff={d_eff}")

    t0 = time.time()
    _, M_mats = build_window_matrices(d)
    M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats, d)
    print(f"  Built {len(M_mats)} windows -> {len(M_mats_eff)} unique symmetric "
          f"in {time.time()-t0:.1f}s", flush=True)

    for R in (4, 6, 8):
        n_eq_theory = comb(d_eff + R, R)
        n_q_theory = comb(d_eff + R - 1, R - 1)
        n_var_theory = 1 + len(M_mats_eff) + n_q_theory + n_eq_theory
        print(f"  R={R}: theory n_eq={n_eq_theory:,}, "
              f"n_q={n_q_theory:,}, n_vars~={n_var_theory:,}", flush=True)

        t0 = time.time()
        try:
            opts = BuildOptions(R=R, use_z2=True, fixed_lambda=None,
                                use_q_polynomial=True, verbose=False)
            build = build_handelman_lp(d_eff, M_mats_eff, opts)
            wall = time.time() - t0
            print(f"  R={R}: actual n_eq={build.A_eq.shape[0]:,}, "
                  f"n_vars={build.n_vars:,}, nnz(A)={build.n_nonzero_A:,}, "
                  f"build={wall:.1f}s", flush=True)
        except Exception as e:
            wall = time.time() - t0
            print(f"  R={R}: FAILED after {wall:.1f}s: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
