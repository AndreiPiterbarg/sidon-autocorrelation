"""Solve the LP at d=64 R=4 (and try R=6 if memory permits)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.runner import run_one, VAL_D_KNOWN


def main():
    target = VAL_D_KNOWN.get(64)
    print(f"d=64 LP solve: target val(64) = {target}", flush=True)
    print(f"{'R':>3} {'alpha':>10} {'gap':>10} {'n_vars':>12} {'n_eq':>12} {'nnz':>12} {'build_s':>10} {'solve_s':>10}", flush=True)
    for R in (4, 6):
        try:
            rec, _, _ = run_one(d=64, R=R, use_z2=True, verbose=False)
            print(f"{R:>3} "
                  f"{rec.alpha if rec.alpha is not None else float('nan'):>10.6f} "
                  f"{rec.gap_to_known if rec.gap_to_known is not None else float('nan'):>10.6f} "
                  f"{rec.n_vars:>12} {rec.n_eq:>12} {rec.n_nonzero_A:>12} "
                  f"{rec.build_wall_s:>10.2f} {rec.solve_wall_s:>10.2f}", flush=True)
        except Exception as e:
            print(f"{R:>3} ERROR: {type(e).__name__}: {str(e)[:200]}", flush=True)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
