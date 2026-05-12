"""Optimised multi-scale MV probe — enforce precondition, search jointly.

Honest analysis:
- The MV precondition u >= 1/2 + max(delta_i) is NECESSARY: K_{delta_i} must be
  supported within one period of length u, else the period-u Parseval identity
  used in MV's Lemma 3.1 fails.
- Multi-scale increases u (because we must pad for the largest delta), which
  HURTS the 2/u term.
- So multi-scale only wins if the (k1, K2, gain) gains outweigh the 2/u loss.

This script: (i) sanity check at single scale, (ii) joint global search over
(lambda, delta_1, delta_2, u) with the precondition enforced, (iii) try to
ALSO move u downward by setting all delta_i SMALL (since u_min = 1/2 + max d).
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mpmath as mp
import numpy as np
from mpmath import mpf
from scipy.optimize import minimize, differential_evolution

from delsarte_dual.mv_bound import MV_COEFFS_119, MV_DELTA, MV_U, MV_BOUND_FINAL
from delsarte_dual.mv_multiscale_probe import evaluate_mixture


def negative_M(x, a_coeffs=MV_COEFFS_119, dps=20):
    """x = [lam1, delta1, delta2].  u is set = 1/2 + max(delta) + 1e-6."""
    lam1, d1, d2 = x
    if not (0 <= lam1 <= 1 and 0.05 <= d1 <= 0.25 and 0.05 <= d2 <= 0.25):
        return 0.0
    u = 0.5 + max(d1, d2) + 1e-6
    try:
        r = evaluate_mixture([lam1, 1 - lam1], [d1, d2], u=u,
                             a_coeffs=a_coeffs, dps=dps, n_grid_minG=4001)
        if not r["valid"] or r["M"] is None:
            return 0.0
        return -float(r["M"])
    except Exception:
        return 0.0


def main():
    mp.mp.dps = 30
    print("=" * 72)
    print("Multi-scale MV — joint optimization with precondition u >= 1/2 + max d")
    print("=" * 72)

    # 1. Reference single-scale (delta = 0.138, u = 0.638).
    r0 = evaluate_mixture([1.0], [MV_DELTA], u=MV_U)
    print(f"\n  Single-scale baseline: M = {float(r0['M']):.6f} (MV: 1.27481)")

    # 2. Single-scale optimised over (delta, u): the existing scan reports
    #    (0.138, 0.638) as global max at M = 1.27481.  Let's re-verify
    #    using our evaluate_mixture (single-element list).
    print("\n[A] Single-scale baseline at varying delta (u = 1/2 + delta + eps)")
    print("-" * 72)
    print(f"  {'delta':>6}  {'u':>6}  {'M':>10}")
    for d in np.linspace(0.10, 0.20, 11):
        u_use = 0.5 + d + 1e-6
        r = evaluate_mixture([1.0], [d], u=u_use, dps=20, n_grid_minG=4001)
        if r["valid"] and r["M"] is not None:
            print(f"  {d:6.4f}  {u_use:6.4f}  {float(r['M']):10.6f}")

    # 3. Joint differential evolution over (lam1, d1, d2).
    print("\n[B] Differential evolution over (lam1, d1, d2)")
    print("-" * 72)
    bounds = [(0.0, 1.0), (0.05, 0.22), (0.05, 0.22)]
    res = differential_evolution(negative_M, bounds, maxiter=80, seed=42,
                                 tol=1e-7, workers=1, popsize=30, polish=True)
    lam1, d1, d2 = res.x
    M_best = -res.fun
    u_best = 0.5 + max(d1, d2) + 1e-6
    print(f"  Best DE: lam1={lam1:.4f}  d1={d1:.4f}  d2={d2:.4f}  u={u_best:.4f}")
    print(f"  M = {M_best:.6f}    vs MV {float(MV_BOUND_FINAL):.5f}    "
          f"diff = {M_best - float(MV_BOUND_FINAL):+.6f}")

    # 4. Pin d1 = d2 (degenerate) to verify DE recovers single-scale.
    print("\n[C] DE check: pin d1 = d2 (degenerate single-scale)")
    print("-" * 72)
    def neg_M_single(x):
        d, = x
        u = 0.5 + d + 1e-6
        r = evaluate_mixture([1.0], [d], u=u, dps=20, n_grid_minG=4001)
        if not r["valid"] or r["M"] is None:
            return 0.0
        return -float(r["M"])
    res2 = differential_evolution(neg_M_single, [(0.05, 0.22)], maxiter=80, seed=0)
    print(f"  Single-scale DE optimum: d={res2.x[0]:.5f}, "
          f"M={-res2.fun:.6f}")

    # 5. Try 3-scale.
    print("\n[D] Three-scale: DE over (lam1, lam2, d1, d2, d3)")
    print("-" * 72)
    def neg_M_three(x):
        l1, l2, d1, d2, d3 = x
        l3 = 1 - l1 - l2
        if l3 < 0 or l3 > 1:
            return 0.0
        u = 0.5 + max(d1, d2, d3) + 1e-6
        try:
            r = evaluate_mixture([l1, l2, l3], [d1, d2, d3], u=u,
                                 dps=20, n_grid_minG=4001)
            if not r["valid"] or r["M"] is None:
                return 0.0
            return -float(r["M"])
        except Exception:
            return 0.0
    bounds3 = [(0.0, 1.0), (0.0, 1.0),
               (0.05, 0.22), (0.05, 0.22), (0.05, 0.22)]
    res3 = differential_evolution(neg_M_three, bounds3, maxiter=80,
                                  seed=11, popsize=30, tol=1e-7)
    l1, l2, d1, d2, d3 = res3.x
    print(f"  Best 3-scale: lam=({l1:.3f},{l2:.3f},{1-l1-l2:.3f})  "
          f"d=({d1:.4f},{d2:.4f},{d3:.4f})")
    print(f"  M = {-res3.fun:.6f}   diff vs MV = "
          f"{-res3.fun - float(MV_BOUND_FINAL):+.6f}")

    # 6. Diagnostic: show why multi-scale fails.
    print("\n[E] Diagnostic: multi-scale loss decomposition (delta_1=0.10, "
          "delta_2=0.18, u=0.6810001)")
    print("-" * 72)
    print(f"  {'lam1':>5}  {'2/u':>7}  {'gain':>7}  {'k1':>7}  {'K2':>7}  "
          f"{'M':>9}")
    for lam1 in np.linspace(0, 1, 11):
        d1, d2 = 0.10, 0.18
        u_use = 0.5 + max(d1, d2) + 1e-6
        if lam1 == 0:
            r = evaluate_mixture([1.0], [d2], u=u_use, dps=20, n_grid_minG=4001)
        elif lam1 == 1:
            r = evaluate_mixture([1.0], [d1], u=u_use, dps=20, n_grid_minG=4001)
        else:
            r = evaluate_mixture([lam1, 1 - lam1], [d1, d2], u=u_use,
                                 dps=20, n_grid_minG=4001)
        if r["valid"] and r["M"] is not None:
            print(f"  {lam1:5.2f}  {2/u_use:7.4f}  {float(r['gain']):7.4f}  "
                  f"{float(r['k1']):7.4f}  {float(r['K2']):7.4f}  "
                  f"{float(r['M']):9.6f}")

    print("\n" + "=" * 72)
    if M_best > float(MV_BOUND_FINAL) + 1e-4:
        print(f"VERDICT: Multi-scale BEATS MV: M = {M_best:.6f}")
    else:
        print(f"VERDICT: Multi-scale does NOT beat MV.")
        print(f"  Best multi-scale M = {M_best:.6f}")
        print(f"  Best 3-scale M     = {-res3.fun:.6f}")
        print(f"  MV single-scale M  = {float(MV_BOUND_FINAL):.6f}")


if __name__ == "__main__":
    main()
