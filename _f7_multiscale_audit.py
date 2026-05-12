"""F7 audit: rigorous re-examination of multi-scale MV mixtures.

Hypothesis F7 claim: multi-scale MV mixtures fail because MV's precondition
    u >= 1/2 + max(delta_i)
forces u up, hurting the 2/u term, and existing scans (mv_multiscale_probe)
showed no mixture beats single-scale arcsine.

This script does an exhaustive re-examination:
 1. Single-scale baseline (delta jointly optimised with u = 1/2 + delta + eps).
 2. Two-scale: dense joint search + scipy DE with the precondition enforced.
 3. Three-scale and four-scale via DE.
 4. Allow u > 1/2 + max delta strictly to confirm the precondition is binding.
 5. Verify the precondition u >= 1/2 + max delta is satisfied for the reported best
    (no rounding leaks).

For each candidate (lambdas, deltas, u), we compute MV's bound via evaluate_mixture
exactly as in delsarte_dual/mv_multiscale_probe.py.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mpmath as mp
from mpmath import mpf
from scipy.optimize import differential_evolution, minimize
from itertools import product

from delsarte_dual.mv_bound import MV_DELTA, MV_U, MV_BOUND_FINAL, MV_COEFFS_119
from delsarte_dual.mv_multiscale_probe import evaluate_mixture


PRECOND_TOL = 1e-12  # numerical tolerance for u >= 1/2 + max delta (CLOSED)
DPS = 22

mp.mp.dps = 30


def safe_eval(lambdas, deltas, u, dps=DPS):
    """Evaluate mixture, return float M or None if invalid; enforce precondition.

    Precondition (MV Lemma 3.1): u >= 1/2 + max(delta_i)  (closed; MV uses equality).
    """
    if any(d <= 0 or d >= 0.25 for d in deltas):
        return None
    if min(lambdas) < 0:
        return None
    s = sum(lambdas)
    if s <= 0:
        return None
    lambdas = [l / s for l in lambdas]
    # precondition: u >= 1/2 + max(d) (closed; MV uses equality)
    if u + PRECOND_TOL < 0.5 + max(deltas):
        return None
    try:
        r = evaluate_mixture(lambdas, deltas, u=u, dps=dps, n_grid_minG=4001,
                             a_coeffs=MV_COEFFS_119)
        if not r["valid"] or r["M"] is None:
            return None
        return float(r["M"])
    except Exception:
        return None


def neg_M_n_scale(x, n):
    """x = [lam_1,...,lam_{n-1}, d_1,...,d_n, u].  lam_n = 1 - sum.
    Returns -M for minimisation; 0 if infeasible.
    """
    lams = list(x[:n - 1])
    last = 1.0 - sum(lams)
    if last < 0 or last > 1:
        return 0.0
    lams.append(last)
    deltas = list(x[n - 1: 2 * n - 1])
    u = float(x[2 * n - 1])
    M = safe_eval(lams, deltas, u)
    return -M if M is not None else 0.0


# -----------------------------------------------------------------------------
# 1. Single-scale baseline (sanity check, also optimised over delta)
# -----------------------------------------------------------------------------
print("=" * 76)
print("F7 AUDIT: rigorous multi-scale MV scan")
print("=" * 76)
print()

# baseline at MV's published parameters
M0 = safe_eval([1.0], [float(MV_DELTA)], float(MV_U))
print(f"[0] MV baseline (delta=0.138, u=0.638): M = {M0:.7f}  "
      f"(MV target: {float(MV_BOUND_FINAL):.5f})")

print("\n[1] Single-scale optimum: u = 1/2 + d + eps.")
print(f"  {'delta':>8}  {'u':>8}  {'M':>10}")
best_single = None
for d in np.linspace(0.10, 0.20, 41):
    u = 0.5 + d + 0.0
    M = safe_eval([1.0], [d], u)
    if M is None:
        continue
    if (best_single is None) or (M > best_single[0]):
        best_single = (M, d, u)
print(f"  Best single (precond):  d={best_single[1]:.5f}  u={best_single[2]:.5f}"
      f"  M={best_single[0]:.7f}")

# Also relax u upward (free u, free d)
def neg_M_single_free_u(x):
    d, u = x
    if u + PRECOND_TOL < 0.5 + d:
        return 0.0
    M = safe_eval([1.0], [d], u)
    return -M if M is not None else 0.0
res = differential_evolution(neg_M_single_free_u,
                             [(0.05, 0.22), (0.55, 0.85)], seed=0,
                             maxiter=120, tol=1e-9, popsize=30, polish=True)
print(f"  Free (d,u) DE single:   d={res.x[0]:.5f}  u={res.x[1]:.5f}  "
      f"M={-res.fun:.7f}")

baseline_best = max(best_single[0], -res.fun)
print(f"  ==> single-scale BEST (precondition enforced): M = {baseline_best:.7f}")


# -----------------------------------------------------------------------------
# 2. Two-scale
# -----------------------------------------------------------------------------
print("\n[2] Two-scale joint search (lam, d1, d2, u).")
def neg_M_two(x):
    lam, d1, d2, u = x
    return neg_M_n_scale([lam, d1, d2, u], n=2)

bounds_2 = [(0.0, 1.0), (0.05, 0.22), (0.05, 0.22), (0.55, 0.85)]
results_two = []
for seed in range(8):
    res = differential_evolution(neg_M_two, bounds_2, seed=seed,
                                 maxiter=200, tol=1e-10,
                                 popsize=40, polish=True)
    M = -res.fun
    results_two.append((M, res.x, seed))
results_two.sort(key=lambda t: -t[0])
M_best2, x_best2, seed_best2 = results_two[0]
lam, d1, d2, u = x_best2
print(f"  Best 2-scale (over 8 seeds): lam=({lam:.4f},{1-lam:.4f}), "
      f"d=({d1:.5f},{d2:.5f}), u={u:.5f}")
print(f"  M = {M_best2:.7f}    delta vs MV  = {M_best2 - float(MV_BOUND_FINAL):+.7f}")
# Verify the precondition (no rounding):
print(f"  Precondition check: u - (1/2 + max(d)) = "
      f"{u - 0.5 - max(d1, d2):+.6e}  (must be > 0)")


# Polish with Nelder-Mead from best
res_nm = minimize(neg_M_two, x_best2, method='Nelder-Mead',
                  options={'xatol': 1e-9, 'fatol': 1e-10, 'maxiter': 5000})
M_polished = -res_nm.fun
lam, d1, d2, u = res_nm.x
print(f"  After NM polish: lam=({lam:.5f},{1-lam:.5f}), "
      f"d=({d1:.5f},{d2:.5f}), u={u:.5f}  M={M_polished:.7f}")


# Also test: does the optimum collapse to single-scale (d1 == d2)?
print(f"  Scale separation |d1 - d2| = {abs(d1 - d2):.5f}")
print(f"  Mixing weight    min(lam, 1-lam) = {min(lam, 1-lam):.5f}")


# -----------------------------------------------------------------------------
# 2b. n=2 dense grid + analytic optimum
# -----------------------------------------------------------------------------
print("\n[2b] Two-scale dense grid (d1, d2 in (0.05, 0.22), 11x11; lam dense).")
best_grid2 = (0.0, None)
for d1 in np.linspace(0.06, 0.21, 16):
    for d2 in np.linspace(0.06, 0.21, 16):
        if d2 < d1:  # symmetry
            continue
        u = 0.5 + max(d1, d2) + 0.0
        for lam in np.linspace(0.0, 1.0, 21):
            M = safe_eval([lam, 1.0 - lam], [d1, d2], u)
            if M is None:
                continue
            if M > best_grid2[0]:
                best_grid2 = (M, (lam, d1, d2, u))
M_grid2, x_grid2 = best_grid2
lam_g, d1_g, d2_g, u_g = x_grid2
print(f"  Dense grid best 2-scale: lam=({lam_g:.4f},{1-lam_g:.4f}), "
      f"d=({d1_g:.5f},{d2_g:.5f}), u={u_g:.5f}, M={M_grid2:.7f}")
print(f"  delta vs MV: {M_grid2 - float(MV_BOUND_FINAL):+.7f}")


# -----------------------------------------------------------------------------
# 3. Three-scale
# -----------------------------------------------------------------------------
print("\n[3] Three-scale joint search (lam1, lam2, d1, d2, d3, u).")
def neg_M_three(x):
    return neg_M_n_scale(x, n=3)
bounds_3 = [(0.0, 1.0), (0.0, 1.0),
            (0.05, 0.22), (0.05, 0.22), (0.05, 0.22),
            (0.55, 0.85)]
results_three = []
for seed in range(6):
    res = differential_evolution(neg_M_three, bounds_3, seed=seed,
                                 maxiter=200, tol=1e-10,
                                 popsize=40, polish=True)
    results_three.append((-res.fun, res.x, seed))
results_three.sort(key=lambda t: -t[0])
M_best3, x_best3, _ = results_three[0]
l1, l2, d1, d2, d3, u = x_best3
l3 = 1 - l1 - l2
print(f"  Best 3-scale: lam=({l1:.4f},{l2:.4f},{l3:.4f}), "
      f"d=({d1:.5f},{d2:.5f},{d3:.5f}), u={u:.5f}")
print(f"  M = {M_best3:.7f}    delta vs MV = "
      f"{M_best3 - float(MV_BOUND_FINAL):+.7f}")
print(f"  Precond u - 1/2 - max(d) = {u - 0.5 - max(d1, d2, d3):+.6e}")


# -----------------------------------------------------------------------------
# 4. Four-scale (fast)
# -----------------------------------------------------------------------------
print("\n[4] Four-scale joint search (3 free lambdas, 4 deltas, u).")
def neg_M_four(x):
    return neg_M_n_scale(x, n=4)
bounds_4 = [(0.0, 1.0)] * 3 + [(0.05, 0.22)] * 4 + [(0.55, 0.85)]
res4 = differential_evolution(neg_M_four, bounds_4, seed=42,
                              maxiter=150, tol=1e-9,
                              popsize=40, polish=True)
M_best4 = -res4.fun
xs = res4.x
lams = list(xs[:3]) + [1 - sum(xs[:3])]
ds = list(xs[3:7])
u4 = xs[7]
print(f"  Best 4-scale: lams={[f'{l:.3f}' for l in lams]}")
print(f"  ds={[f'{d:.4f}' for d in ds]}, u={u4:.5f}")
print(f"  M = {M_best4:.7f}    delta vs MV = {M_best4 - float(MV_BOUND_FINAL):+.7f}")


# -----------------------------------------------------------------------------
# 5. Decomposition: probe how each ingredient changes when we mix scales.
# -----------------------------------------------------------------------------
print("\n[5] Why multi-scale fails (or wins): ingredient decomposition.")
print(f"   At single scale d=0.138, u=0.638: the 2/u + a target bound is")
print(f"   pushed up but k1, K2 also change.  We sweep d_2 around 0.138.")
print(f"   {'d2':>6}  {'u':>6}  {'lam_opt':>8}  {'2/u':>8}  {'gain':>8}  "
      f"{'k1':>8}  {'K2':>8}  {'M':>10}")
for d2 in [0.12, 0.13, 0.135, 0.138, 0.14, 0.145, 0.15, 0.16, 0.18, 0.20]:
    u_use = 0.5 + max(MV_DELTA, d2) + 0.0
    best = (0.0, None)
    for lam in np.linspace(0.0, 1.0, 41):
        if lam == 0:
            r = evaluate_mixture([1.0], [d2], u=u_use, dps=DPS, n_grid_minG=4001)
        elif lam == 1:
            r = evaluate_mixture([1.0], [float(MV_DELTA)], u=u_use, dps=DPS,
                                 n_grid_minG=4001)
        else:
            r = evaluate_mixture([lam, 1.0 - lam],
                                 [float(MV_DELTA), d2], u=u_use,
                                 dps=DPS, n_grid_minG=4001)
        if r["valid"] and r["M"] is not None:
            if float(r["M"]) > best[0]:
                best = (float(r["M"]), r)
    if best[1] is None:
        continue
    r = best[1]
    print(f"   {d2:6.3f}  {u_use:6.4f}  {0:8.3f}  "
          f"{2/u_use:8.4f}  {float(r['gain']):8.4f}  "
          f"{float(r['k1']):8.5f}  {float(r['K2']):8.4f}  {best[0]:10.6f}")


# -----------------------------------------------------------------------------
# 6. Final verdict
# -----------------------------------------------------------------------------
print("\n" + "=" * 76)
print("F7 AUDIT VERDICT")
print("=" * 76)
results = [
    ("Single-scale (free u)", baseline_best, None),
    ("Two-scale DE",          M_best2, x_best2),
    ("Two-scale dense grid",  M_grid2, x_grid2),
    ("Three-scale DE",        M_best3, x_best3),
    ("Four-scale DE",         M_best4, xs),
]
for name, M, x in results:
    delta = M - float(MV_BOUND_FINAL)
    flag = ""
    if delta > 1e-5:
        flag = "  *** sharper than MV ***"
    elif delta > -1e-5:
        flag = "  ~~ ties MV"
    print(f"  {name:30s}  M = {M:.7f}    diff vs MV = {delta:+.7f}{flag}")

print()
overall_best = max(results, key=lambda t: t[1])
print(f"  OVERALL BEST: {overall_best[0]}  M = {overall_best[1]:.7f}")
print(f"  Compared to MV 1.27481:   diff = {overall_best[1] - float(MV_BOUND_FINAL):+.6f}")
print(f"  Compared to CS 2017 1.2802: diff = {overall_best[1] - 1.2802:+.6f}")
if overall_best[1] > 1.2802 + 1e-5:
    print("  VERDICT: YES — rigorous multi-scale beats CS 2017's 1.2802")
elif overall_best[1] > float(MV_BOUND_FINAL) + 1e-5:
    print("  VERDICT: PARTIAL — beats MV's 1.2748 numerically but NOT CS 2017's 1.2802")
else:
    print("  VERDICT: NO — F7 confirmed; multi-scale does not beat single-scale MV")
