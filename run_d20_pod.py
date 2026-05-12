"""Run KKT-correct mu* finder at d=20 on pod (64 cores, 251GB).

Uses the master pipeline find_kkt_correct_mu_star with 6400 starts, 64 workers.
Also extracts and verifies Phase 2 best (in case Phase 3 breaks at d=20 like d=16).
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kkt_correct_mu_star import (
    find_kkt_correct_mu_star, build_window_data, evaluate_tv_per_window,
    kkt_residual_qp, phase1_multistart, phase2_lse_newton,
)


def main():
    d = 20
    n_starts = 6400
    n_workers = 64
    top_K_phase2 = 100
    top_K_phase3 = 20

    print(f"\n{'#'*72}")
    print(f"# d=20 KKT-correct mu* finder")
    print(f"{'#'*72}", flush=True)

    # Run master pipeline
    t0 = time.perf_counter()
    result = find_kkt_correct_mu_star(
        d=d, x_cap=1.0, n_starts=n_starts, n_workers=n_workers,
        top_K_phase2=top_K_phase2, top_K_phase3=top_K_phase3,
        target_residual=1e-6, verbose=True,
    )
    elapsed_master = time.perf_counter() - t0

    if result['mu_star'] is None:
        print("Master pipeline FAILED. Running Phase 1+2 manually for diagnostic.")
    else:
        print(f"\n{'='*72}")
        print(f"MASTER PIPELINE RESULT at d={d}")
        print(f"{'='*72}")
        print(f"  f(mu*) = val({d}) UB = {result['f_value']:.10f}")
        print(f"  KKT residual = {result['residual']:.6e}")
        print(f"  active windows: {len(result['active_idx'])}")
        print(f"  total time: {elapsed_master:.1f}s")
        print(f"  mu* = {result['mu_star']}")

    # Independent verification using a fresh Phase 1 + 2 run
    print(f"\n{'='*72}")
    print(f"INDEPENDENT PHASE 1+2 RUN (no Phase 3) at d={d}")
    print(f"{'='*72}", flush=True)
    A_stack, c_W = build_window_data(d)
    t1 = time.perf_counter()
    top_phase1, _, _ = phase1_multistart(
        d, x_cap=1.0, n_starts=n_starts, n_workers=n_workers,
        n_iters_nm=1000, top_K=top_K_phase2, verbose=True)
    t_p1 = time.perf_counter() - t1

    print(f"\nPhase 2: LSE-Newton continuation on top-{top_K_phase2}", flush=True)
    t2 = time.perf_counter()
    phase2_results = []
    extended_beta = [10.0 * (2.0 ** k) for k in range(13)]
    for i, (mu0, _) in enumerate(top_phase1):
        try:
            mu_p2, f_p2 = phase2_lse_newton(
                mu0, d, x_cap=1.0, A_stack=A_stack, c_W=c_W,
                beta_schedule=extended_beta, max_inner=60, verbose=False)
            res = kkt_residual_qp(mu_p2, A_stack, c_W, x_cap=1.0,
                                    tol_active=1e-3)
            phase2_results.append((mu_p2, f_p2, res['residual'], len(res['active_idx'])))
        except Exception as e:
            if i < 3:
                print(f"  start {i} failed: {e}", flush=True)
    t_p2 = time.perf_counter() - t2
    print(f"Phase 2 done in {t_p2:.1f}s, {len(phase2_results)} successful starts", flush=True)

    if not phase2_results:
        print("ALL Phase 2 failed!")
        return

    phase2_results.sort(key=lambda r: r[1])

    print(f"\n{'='*72}")
    print(f"TOP 20 BY f-VALUE (Phase 2)")
    print(f"{'='*72}")
    print(f"{'rank':<6}{'f':<14}{'residual':<14}{'active':<8}")
    for rank, (mu, f, res, n_act) in enumerate(phase2_results[:20]):
        print(f"{rank:<6}{f:<14.7f}{res:<14.4e}{n_act:<8}")

    best = phase2_results[0]
    print(f"\n{'='*72}")
    print(f"BEST PHASE 2 mu*")
    print(f"{'='*72}")
    print(f"  f(mu*) = val({d}) UB = {best[1]:.10f}")
    print(f"  KKT residual = {best[2]:.4e}")
    print(f"  active windows: {best[3]}")
    print(f"  mu* = {best[0]}")
    print(f"  Phase 1 time: {t_p1:.1f}s, Phase 2 time: {t_p2:.1f}s")

    np.savez(f'mu_star_d{d}.npz',
             mu=best[0], f=best[1], residual=best[2])
    print(f"\nSaved mu_star_d{d}.npz")

    # Margin to c=1.281
    print(f"\n{'='*72}")
    print(f"VERDICT FOR c=1.281")
    print(f"{'='*72}")
    print(f"  val({d}) UB = {best[1]:.6f}")
    print(f"  c_target = 1.281000")
    margin = best[1] - 1.281
    print(f"  margin = val({d})_UB - c = {margin:+.6f}")
    if margin > 0:
        print(f"  >>> POSSIBLY FEASIBLE: val({d}) >= 1.281 if pipeline is at true minimum")
    else:
        print(f"  >>> INFEASIBLE: val({d}) < 1.281, d={d} cannot prove c=1.281")

    # Find lowest f attempting to demonstrate val(d) < 1.281
    below_target = [r for r in phase2_results if r[1] < 1.281]
    if below_target:
        print(f"  Found {len(below_target)} Phase 2 starts with f < 1.281")
        print(f"    -> these are counterexamples to val({d}) >= 1.281")


if __name__ == "__main__":
    main()
