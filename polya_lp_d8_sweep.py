"""Convergence test: d=8, sweep R = 4, 6, 8, 10, 12, 14. Compare to val(8)=1.205.

Test variable lambda only. Z/2 mode.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.runner import run_one


def main():
    print("d=8 sweep: target val(8) = 1.205")
    print(f"{'R':>3} {'z2':>3} {'alpha':>10} {'gap':>10} {'n_vars':>10} {'n_eq':>8} {'nnz':>10} {'build_s':>8} {'solve_s':>8}")
    for R in (4, 6, 8, 10, 12, 14, 16):
        try:
            rec, _, _ = run_one(d=8, R=R, use_z2=True, use_q_polynomial=True,
                                verbose=False)
            print(f"{R:>3} {'T':>3} "
                  f"{rec.alpha if rec.alpha is not None else float('nan'):>10.6f} "
                  f"{rec.gap_to_known if rec.gap_to_known is not None else float('nan'):>10.6f} "
                  f"{rec.n_vars:>10} {rec.n_eq:>8} {rec.n_nonzero_A:>10} "
                  f"{rec.build_wall_s:>8.2f} {rec.solve_wall_s:>8.2f}")
        except Exception as e:
            print(f"{R:>3} ERROR: {e}")


if __name__ == "__main__":
    main()
