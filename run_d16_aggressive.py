"""Aggressive d=16 test: 25600 starts, save Phase 2 best, verify KKT residual.

Goal: nail down the true val(16). If UB drops below 1.281, d=16 fails.
If UB stays around 1.282, that's strong evidence val(16) is near 1.282.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kkt_correct_mu_star import (
    build_window_data, evaluate_tv_per_window, kkt_residual_qp,
    phase1_multistart, phase2_lse_newton,
)


def main():
    d = 16
    n_starts = 25600
    n_workers = 64
    top_K_phase2 = 200

    print(f"\n{'#'*72}")
    print(f"# d=16 AGGRESSIVE search: {n_starts} starts, {n_workers} workers")
    print(f"{'#'*72}", flush=True)

    A_stack, c_W = build_window_data(d)

    # Phase 1: massive multistart
    t1 = time.perf_counter()
    top_phase1, _, _ = phase1_multistart(
        d, x_cap=1.0, n_starts=n_starts, n_workers=n_workers,
        n_iters_nm=1200,  # longer NM
        top_K=top_K_phase2, verbose=True)
    t_p1 = time.perf_counter() - t1
    print(f"Phase 1: {t_p1:.1f}s", flush=True)

    # Phase 2: LSE-Newton continuation, with EXTENDED beta schedule
    print(f"\nPhase 2: LSE-Newton continuation on top-{top_K_phase2}", flush=True)
    extended_beta = [10.0 * (2.0 ** k) for k in range(15)]  # up to 163840
    t2 = time.perf_counter()
    phase2_results = []
    for i, (mu0, _) in enumerate(top_phase1):
        try:
            mu_p2, f_p2 = phase2_lse_newton(
                mu0, d, x_cap=1.0, A_stack=A_stack, c_W=c_W,
                beta_schedule=extended_beta, max_inner=80,
                verbose=False)
            # Get KKT residual
            res = kkt_residual_qp(mu_p2, A_stack, c_W, x_cap=1.0,
                                    tol_active=1e-3)
            phase2_results.append((mu_p2, f_p2, res['residual'], len(res['active_idx'])))
            if i % 20 == 0:
                print(f"  start {i}: f={f_p2:.7f}, residual={res['residual']:.4e}, "
                      f"active={len(res['active_idx'])}", flush=True)
        except Exception as e:
            if i < 3:
                print(f"  start {i} failed: {e}", flush=True)

    t_p2 = time.perf_counter() - t2
    print(f"\nPhase 2: {t_p2:.1f}s", flush=True)

    if not phase2_results:
        print("All Phase 2 starts failed!")
        return

    # Sort by f-value (lowest first)
    phase2_results.sort(key=lambda r: r[1])

    print(f"\n{'='*70}")
    print(f"TOP 30 BY f-VALUE")
    print(f"{'='*70}")
    print(f"{'rank':<6}{'f':<12}{'residual':<14}{'active':<8}{'mu summary':<30}")
    for rank, (mu, f, res, n_act) in enumerate(phase2_results[:30]):
        mu_summary = f"max={mu.max():.3f},min={mu.min():.4f},#zero={int((mu<1e-4).sum())}"
        print(f"{rank:<6}{f:<12.7f}{res:<14.4e}{n_act:<8}{mu_summary}")

    # Save best by f
    best_by_f = phase2_results[0]
    np.savez('mu_star_d16_aggressive.npz',
             mu=best_by_f[0],
             f=best_by_f[1],
             residual=best_by_f[2])

    print(f"\n{'='*70}")
    print(f"BEST BY f-VALUE")
    print(f"{'='*70}")
    print(f"  f(mu) = {best_by_f[1]:.10f}")
    print(f"  KKT residual = {best_by_f[2]:.4e}")
    print(f"  active windows: {best_by_f[3]}")
    print(f"  mu = {best_by_f[0]}")

    # Also report best by (residual + bonus for low f)
    print(f"\n{'='*70}")
    print(f"DOES IT HAVE A KKT POINT (residual < 1e-4) WITH f < 1.281?")
    print(f"{'='*70}")
    qualified = [r for r in phase2_results if r[2] < 1e-4 and r[1] < 1.281]
    if qualified:
        for r in qualified[:5]:
            print(f"  f={r[1]:.7f}, residual={r[2]:.4e}, active={r[3]}")
        print(f"  >>> val(16) < 1.281 PROVEN by counterexample mu")
    else:
        # Look at ones with low f even if residual not tight
        low_f = [r for r in phase2_results if r[1] < 1.281]
        if low_f:
            print(f"  Found {len(low_f)} with f < 1.281 but no clean KKT point.")
            for r in low_f[:5]:
                print(f"    f={r[1]:.7f}, residual={r[2]:.4e}")
        else:
            print(f"  No mu found with f < 1.281. Min f = {phase2_results[0][1]:.7f}")
            print(f"  >>> Strong evidence val(16) >= 1.281")


if __name__ == "__main__":
    main()
