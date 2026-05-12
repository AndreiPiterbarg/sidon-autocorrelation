"""Test active-set acceleration at d=8."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lasserre.polya_lp.active_set import active_set_solve


def main():
    print("d=8 active-set acceleration test", flush=True)
    res = active_set_solve(
        d=8,
        R_schedule=[6, 8, 10, 12, 14, 16],
        initial_R=6,
        growth=0,
        verbose=True,
    )
    print(f"\nFinal alpha: {res.final_alpha:.6f} (target val(8)=1.205)", flush=True)
    print(f"Final |active|: {len(res.final_active_indices)}", flush=True)
    print(f"Total wall: {res.total_wall_s:.1f}s", flush=True)
    print(f"\nR vs alpha:", flush=True)
    for R, a, s, w in zip(res.R_history, res.alpha_history,
                          res.active_size_history, res.wall_history):
        print(f"  R={R}: alpha={a:.6f}, |W|={s}, solve+build={w:.1f}s", flush=True)


if __name__ == "__main__":
    main()
