"""Run KKT-correct mu* finder at d=16 on pod (64 cores, 251GB).

n_starts=6400 with 64 workers in Phase 1, then top-100 to Phase 2,
then top-20 to Phase 3 (Newton-KKT). Expected residual ~1e-10.
"""
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kkt_correct_mu_star import (
    find_kkt_correct_mu_star, build_window_data, evaluate_tv_per_window,
    kkt_residual_qp,
)


def main():
    d = 16
    print(f"\n{'#'*72}", flush=True)
    print(f"# d=16 KKT-correct mu* on pod", flush=True)
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
        print("FAILED: no successful starts")
        return
    print(f"\n{'='*72}", flush=True)
    print(f"FINAL RESULT at d={d}", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  f(mu*) = val({d}) UB = {result['f_value']:.10f}")
    print(f"  KKT residual = {result['residual']:.6e}")
    print(f"  active windows: {len(result['active_idx'])}")
    print(f"  mu* = {result['mu_star']}")

    # Independent verification
    A_stack, c_W = build_window_data(d)
    T_W = evaluate_tv_per_window(result['mu_star'], A_stack, c_W)
    print(f"\nVerification: max TV_W = {T_W.max():.10f}")
    top10 = np.argsort(T_W)[::-1][:10]
    print(f"  top-10 TV: {T_W[top10]}")
    res_qp = kkt_residual_qp(result['mu_star'], A_stack, c_W, x_cap=1.0)
    print(f"  KKT residual (re-verify): {res_qp['residual']:.6e}")
    print(f"  active count (re-verify): {len(res_qp['active_idx'])}")

    # Save the result for later use
    np.savez('mu_star_d16.npz',
             mu_star=result['mu_star'],
             alpha_star=result['alpha_star'],
             active_idx=np.array(result['active_idx']),
             f_value=result['f_value'],
             residual=result['residual'])
    print("\nSaved mu_star_d16.npz")


if __name__ == "__main__":
    main()
