"""Convergence test: d=4, sweep R = 4, 6, 8, 10, 12. Compare to val(4)=1.102.

Tests both with and without Z/2 to confirm they agree.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.runner import run_one


def main():
    print("d=4 sweep: target val(4) = 1.102")
    print(f"{'R':>3} {'z2':>3} {'q':>2} {'alpha':>10} {'gap':>10} {'n_vars':>8} {'n_eq':>6} {'build_s':>8} {'solve_s':>8}")
    for use_z2 in (True, False):
        for R in (4, 6, 8, 10, 12):
            rec, _, _ = run_one(d=4, R=R, use_z2=use_z2, use_q_polynomial=True,
                                verbose=False)
            print(f"{R:>3} {str(use_z2)[0]:>3} {'Y':>2} "
                  f"{rec.alpha if rec.alpha is not None else float('nan'):>10.6f} "
                  f"{rec.gap_to_known if rec.gap_to_known is not None else float('nan'):>10.6f} "
                  f"{rec.n_vars:>8} {rec.n_eq:>6} "
                  f"{rec.build_wall_s:>8.2f} {rec.solve_wall_s:>8.2f}")


if __name__ == "__main__":
    main()
