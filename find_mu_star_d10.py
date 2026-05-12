"""KKT-correct mu* finder for d=10 (fast — d=10 is small).

Mirrors find_mu_star_d22.py but tuned for d=10:
  n_starts=300, n_workers=8 (Phase 1+2 finishes in ~30 s).
Saves mu_star_d10.npz for the BnB anchor + centroid tiers to consume.
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kkt_correct_mu_star import (
    find_kkt_correct_mu_star, build_window_data,
    kkt_residual_qp, phase1_multistart, phase2_lse_newton,
)


def main():
    d = 10
    n_starts = 300
    n_workers = 8
    top_K_phase2 = 30
    top_K_phase3 = 8
    target_residual = 1e-6

    print(f"\n{'#'*72}")
    print(f"# d={d} KKT-correct mu* finder (local)")
    print(f"# n_starts={n_starts}, n_workers={n_workers}", flush=True)
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
        print(f"\n  f(mu*) = val({d}) UB = {result['f_value']:.10f}")
        print(f"  KKT residual = {result['residual']:.6e}")
        print(f"  active windows: {len(result['active_idx'])}")
        print(f"  master total time: {elapsed_master:.1f}s")

    # Phase 1+2 fallback for verification
    A_stack, c_W = build_window_data(d)
    top_phase1, _, _ = phase1_multistart(
        d, x_cap=1.0, n_starts=n_starts, n_workers=n_workers,
        n_iters_nm=1000, top_K=top_K_phase2, verbose=False)
    phase2_results = []
    extended_beta = [10.0 * (2.0 ** k) for k in range(13)]
    for mu0, _ in top_phase1:
        try:
            mu_p2, f_p2 = phase2_lse_newton(
                mu0, d, x_cap=1.0, A_stack=A_stack, c_W=c_W,
                beta_schedule=extended_beta, max_inner=60, verbose=False)
            res = kkt_residual_qp(mu_p2, A_stack, c_W, x_cap=1.0,
                                  tol_active=1e-3)
            phase2_results.append((mu_p2, f_p2, res['residual'],
                                   len(res['active_idx'])))
        except Exception:
            pass
    phase2_results.sort(key=lambda r: r[1])

    candidates = []
    if master_ok:
        candidates.append({
            'mu': result['mu_star'],
            'f': result['f_value'],
            'residual': result['residual'],
            'active': len(result['active_idx']),
            'src': 'master',
        })
    if phase2_results:
        best_p2 = phase2_results[0]
        candidates.append({
            'mu': best_p2[0], 'f': best_p2[1],
            'residual': best_p2[2], 'active': best_p2[3],
            'src': 'phase1+2',
        })

    qualified = [c for c in candidates if c['residual'] < 1e-3]
    best = (min(qualified, key=lambda c: c['f']) if qualified
            else min(candidates, key=lambda c: c['residual']))

    print(f"\n  selected: {best['src']}")
    print(f"  f(mu*) = val({d}) UB = {best['f']:.10f}")
    print(f"  KKT residual = {best['residual']:.6e}")
    print(f"  active windows: {best['active']}")
    print(f"  mu* nonzero count: {int((np.abs(best['mu']) > 1e-9).sum())}")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f'mu_star_d{d}.npz')
    np.savez(out_path,
             mu=np.asarray(best['mu'], dtype=np.float64),
             f=float(best['f']),
             residual=float(best['residual']))
    print(f"\nSaved {out_path}", flush=True)

    margin = best['f'] - 1.2
    print(f"\n  val({d}) UB - 1.2 = {margin:+.6f}  (slack for target 1.2)")


if __name__ == "__main__":
    main()
