"""Focused d=16, 20, 24 sweep with bounded R."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.runner import run_one, VAL_D_KNOWN


def main():
    schedule = [
        (16, [4, 6, 8]),
        (20, [4, 6, 8]),
        (24, [4, 6, 8]),
    ]
    for d, R_list in schedule:
        target = VAL_D_KNOWN.get(d)
        print(f"\nd={d}: target val(d) = {target}", flush=True)
        print(f"{'R':>3} {'alpha':>10} {'gap':>10} {'n_vars':>10} {'n_eq':>10} {'nnz':>12} {'build_s':>8} {'solve_s':>8}", flush=True)
        for R in R_list:
            try:
                rec, _, _ = run_one(d=d, R=R, use_z2=True, verbose=False)
                print(f"{R:>3} "
                      f"{rec.alpha if rec.alpha is not None else float('nan'):>10.6f} "
                      f"{rec.gap_to_known if rec.gap_to_known is not None else float('nan'):>10.6f} "
                      f"{rec.n_vars:>10} {rec.n_eq:>10} {rec.n_nonzero_A:>12} "
                      f"{rec.build_wall_s:>8.2f} {rec.solve_wall_s:>8.2f}", flush=True)
            except Exception as e:
                print(f"{R:>3} ERROR: {type(e).__name__}: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()
