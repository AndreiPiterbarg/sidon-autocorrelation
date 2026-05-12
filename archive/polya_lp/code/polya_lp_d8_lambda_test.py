"""Compare variable lambda (joint LP) vs fixed uniform lambda at d=8."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from lasserre.polya_lp.runner import run_one
from lasserre.polya_lp.build import build_window_matrices
from lasserre.polya_lp.symmetry import project_window_set_to_z2_rescaled


def main():
    print("d=8: variable vs uniform lambda")
    print(f"{'R':>3} {'mode':>10} {'alpha':>10} {'gap':>10} {'solve_s':>8}")

    # Build the rescaled M_mats once to know how many fixed lambdas to make
    _, M_mats_orig = build_window_matrices(8)
    M_mats_eff, _ = project_window_set_to_z2_rescaled(M_mats_orig, 8)
    n_W = len(M_mats_eff)
    uniform = np.full(n_W, 1.0 / n_W)

    for R in (6, 8, 10, 12, 14):
        # Variable lambda
        rec_v, _, _ = run_one(d=8, R=R, use_z2=True, fixed_lambda=None, verbose=False)
        # Fixed uniform lambda
        rec_u, _, _ = run_one(d=8, R=R, use_z2=True, fixed_lambda=uniform, verbose=False)
        print(f"{R:>3} {'variable':>10} {rec_v.alpha:>10.6f} {rec_v.gap_to_known:>10.6f} {rec_v.solve_wall_s:>8.2f}")
        print(f"{R:>3} {'uniform':>10} {rec_u.alpha:>10.6f} {rec_u.gap_to_known:>10.6f} {rec_u.solve_wall_s:>8.2f}")


if __name__ == "__main__":
    main()
