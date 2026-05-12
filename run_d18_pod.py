"""Run KKT-correct mu* finder at d=18 on pod (intermediate point).
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kkt_correct_mu_star import (
    find_kkt_correct_mu_star, build_window_data, evaluate_tv_per_window,
    kkt_residual_qp,
)


def main():
    d = 18
    print(f"\n{'#'*72}", flush=True)
    print(f"# d={d} KKT-correct mu* on pod", flush=True)
    print(f"{'#'*72}", flush=True)

    t0 = time.perf_counter()
    result = find_kkt_correct_mu_star(
        d=d, x_cap=1.0, n_starts=6400, n_workers=64,
        top_K_phase2=100, top_K_phase3=20,
        target_residual=1e-6, verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTotal: {elapsed:.1f}s")
    if result['mu_star'] is None:
        print("FAILED")
        return

    print(f"\n{'='*72}")
    print(f"FINAL RESULT at d={d}")
    print(f"{'='*72}")
    print(f"  f(mu*) = val({d}) UB = {result['f_value']:.10f}")
    print(f"  KKT residual = {result['residual']:.6e}")
    print(f"  active windows: {len(result['active_idx'])}")
    print(f"  selected from phase {result.get('best_phase')} ({result.get('best_method')})")
    print(f"  margin to c=1.281: {result['f_value'] - 1.281:+.6f}")

    # Independent verification
    A_stack, c_W = build_window_data(d)
    T_W = evaluate_tv_per_window(result['mu_star'], A_stack, c_W)
    print(f"\nVerification: max TV_W = {T_W.max():.10f}")

    np.savez(f'mu_star_d{d}.npz',
             mu_star=result['mu_star'],
             alpha_star=result['alpha_star'],
             f_value=result['f_value'],
             residual=result['residual'])


if __name__ == "__main__":
    main()
