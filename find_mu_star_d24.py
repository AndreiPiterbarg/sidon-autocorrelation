"""Run KKT-correct mu* finder at d=24 locally on a 16-core Windows machine.

Tuned for ~30-min budget (slightly larger than d=22 due to d^2 scaling
in window count):
  n_starts=1500, n_workers=8, top_K_phase2=50, top_K_phase3=10
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
    d = 24
    n_starts = 1500
    n_workers = 8
    top_K_phase2 = 50
    top_K_phase3 = 10
    target_residual = 1e-6

    print(f"\n{'#'*72}")
    print(f"# d={d} KKT-correct mu* finder (local, 16-core box)")
    print(f"# n_starts={n_starts}, n_workers={n_workers}, "
          f"top_K_phase2={top_K_phase2}, top_K_phase3={top_K_phase3}")
    print(f"{'#'*72}", flush=True)

    t0 = time.perf_counter()
    result = find_kkt_correct_mu_star(
        d=d, x_cap=1.0,
        n_starts=n_starts, n_workers=n_workers,
        top_K_phase2=top_K_phase2, top_K_phase3=top_K_phase3,
        target_residual=target_residual, verbose=True,
    )
    elapsed_master = time.perf_counter() - t0

    master_ok = (result['mu_star'] is not None)
    if master_ok:
        print(f"\n{'='*72}")
        print(f"MASTER PIPELINE RESULT at d={d}")
        print(f"{'='*72}")
        print(f"  f(mu*) = val({d}) UB = {result['f_value']:.10f}")
        print(f"  KKT residual = {result['residual']:.6e}")
        print(f"  active windows: {len(result['active_idx'])}")
        print(f"  selected from phase {result.get('best_phase')} "
              f"({result.get('best_method')})")
        print(f"  master total time: {elapsed_master:.1f}s")
    else:
        print("Master pipeline returned no candidate; falling back to Phase 1+2 only.",
              flush=True)

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
            phase2_results.append(
                (mu_p2, f_p2, res['residual'], len(res['active_idx'])))
        except Exception as e:
            if i < 3:
                print(f"  start {i} failed: {e}", flush=True)
    t_p2 = time.perf_counter() - t2
    print(f"Phase 2 done in {t_p2:.1f}s, {len(phase2_results)} successful starts",
          flush=True)

    phase2_results.sort(key=lambda r: r[1])

    if phase2_results:
        print(f"\n{'='*72}")
        print(f"TOP 20 BY f-VALUE (Phase 2)")
        print(f"{'='*72}")
        print(f"{'rank':<6}{'f':<14}{'residual':<14}{'active':<8}")
        for rank, (mu, f, res, n_act) in enumerate(phase2_results[:20]):
            print(f"{rank:<6}{f:<14.7f}{res:<14.4e}{n_act:<8}")

    candidates = []
    if master_ok:
        candidates.append({
            'mu': result['mu_star'],
            'f': result['f_value'],
            'residual': result['residual'],
            'active': len(result['active_idx']),
            'src': f"master(phase={result.get('best_phase')},"
                   f"method={result.get('best_method')})",
        })
    if phase2_results:
        best_p2 = phase2_results[0]
        candidates.append({
            'mu': best_p2[0],
            'f': best_p2[1],
            'residual': best_p2[2],
            'active': best_p2[3],
            'src': "phase1+2_independent",
        })

    if not candidates:
        print("\nNO CANDIDATES from any phase. ABORTING.", flush=True)
        sys.exit(2)

    qualified = [c for c in candidates if c['residual'] < 1e-3]
    if qualified:
        best = min(qualified, key=lambda c: c['f'])
    else:
        best = min(candidates, key=lambda c: c['residual'])

    print(f"\n{'='*72}")
    print(f"BEST mu* SELECTED at d={d}")
    print(f"{'='*72}")
    print(f"  source: {best['src']}")
    print(f"  f(mu*) = val({d}) UB = {best['f']:.10f}")
    print(f"  KKT residual = {best['residual']:.6e}")
    print(f"  active windows: {best['active']}")
    print(f"  mu* = {best['mu']}")
    print(f"  Phase 1 time: {t_p1:.1f}s, Phase 2 time: {t_p2:.1f}s, "
          f"master time: {elapsed_master:.1f}s")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f'mu_star_d{d}.npz')
    np.savez(out_path,
             mu=np.asarray(best['mu'], dtype=np.float64),
             f=float(best['f']),
             residual=float(best['residual']))
    print(f"\nSaved {out_path}")

    print(f"\n{'='*72}")
    print(f"VERDICT FOR c=1.281")
    print(f"{'='*72}")
    print(f"  val({d}) UB = {best['f']:.6f}")
    print(f"  c_target    = 1.281000")
    margin = best['f'] - 1.281
    print(f"  margin = val({d})_UB - c = {margin:+.6f}")
    if margin > 0:
        print(f"  >>> POSSIBLY FEASIBLE: val({d}) >= 1.281 if pipeline is at true minimum")
    else:
        print(f"  >>> INFEASIBLE: val({d}) < 1.281, d={d} cannot prove c=1.281")

    if phase2_results:
        below = [r for r in phase2_results if r[1] < 1.281]
        if below:
            print(f"  Found {len(below)} Phase 2 starts with f < 1.281 "
                  f"(counterexamples to val({d}) >= 1.281)")


if __name__ == "__main__":
    main()
