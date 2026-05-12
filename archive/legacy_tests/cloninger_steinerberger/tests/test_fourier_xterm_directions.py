"""Test the inequality directions carefully — what does the cross-term lemma
actually give us as bounds on C_{1a}?

For step function f_step with bin masses mu:
  ||f_step * f_step||_inf  =  max_s 4n * MC[s]  (EXACT, by cross-term lemma + piecewise linearity).

For f admissible: f_step is admissible, so C_{1a} <= ||f_step * f_step||_inf.
Hence  C_{1a} <= inf_mu max_s 4n MC[s] = val_knot(d), an UPPER BOUND.

For window TV_W (averages):  TV_W <= ||f*f||_inf, so val(d) := inf_mu max_W TV_W <= C_{1a},
a LOWER BOUND.

Question:
1. Compute val_knot(d) for several d. Does it beat 1.5029 (the known upper bound)?
2. Compute val(d) carefully and confirm val(d) <= val_knot(d).
3. As d grows, what does val_knot(d) approach?
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, linprog
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lasserre.fourier_xterm import (
    BinGeometry, step_autoconv_at_knot, step_autoconv_all_knots,
    step_autoconv_inf_norm,
)


def val_knot_brute(d: int, n_dirichlet: int = 5000, seed: int = 0) -> tuple:
    """Estimate val_knot(d) := inf_mu max_s 4n MC[s] by random Dirichlet search.

    Returns (best_value, best_mu).
    """
    rng = np.random.default_rng(seed)
    n = d // 2
    best_v = np.inf
    best_mu = None
    # Try a mix of concentrated and spread distributions
    for _ in range(n_dirichlet):
        # Random alpha drawn from a wide range
        alpha = rng.uniform(0.2, 5.0, size=d)
        mu = rng.dirichlet(alpha)
        v = step_autoconv_inf_norm(mu)
        if v < best_v:
            best_v = v
            best_mu = mu.copy()
    # Also try some specific candidates
    # Uniform
    mu_u = np.ones(d) / d
    v_u = step_autoconv_inf_norm(mu_u)
    if v_u < best_v:
        best_v = v_u
        best_mu = mu_u
    return float(best_v), best_mu


def val_knot_optimize(d: int, n_starts: int = 30, seed: int = 0) -> tuple:
    """Use scipy.optimize to minimize max_s 4n MC[s] over mu in simplex.

    Reformulate: minimize t s.t. mu in simplex and 4n MC[s] <= t for all s.
    This is non-convex (MC[s] is quadratic in mu). Use SLSQP from random starts.

    Returns (best_value, best_mu).
    """
    rng = np.random.default_rng(seed)
    n = d // 2

    def max_mc(mu):
        # Compute max_s 4n MC[s]
        max_v = 0.0
        for s in range(2 * d - 1):
            mc = 0.0
            i_lo = max(0, s - (d - 1))
            i_hi = min(d - 1, s)
            for i in range(i_lo, i_hi + 1):
                j = s - i
                mc += mu[i] * mu[j]
            v = 4.0 * n * mc
            if v > max_v:
                max_v = v
        return max_v

    def objective(x):
        # x has d - 1 free params; mu = (x[0], x[1], ..., x[d-2], 1 - sum x)
        mu = np.zeros(d)
        mu[:d - 1] = x
        mu[d - 1] = 1.0 - x.sum()
        if mu.min() < 0:
            return 100.0  # infeasible penalty
        return max_mc(mu)

    best_v = np.inf
    best_mu = None
    for trial in range(n_starts):
        # Random Dirichlet starting point
        alpha = rng.uniform(0.5, 3.0, size=d)
        mu_init = rng.dirichlet(alpha)
        x_init = mu_init[:d - 1]

        # Constraints: x[i] >= 0, sum x[i] <= 1
        bounds = [(0.0, 1.0)] * (d - 1)
        constraints = [{'type': 'ineq', 'fun': lambda x: 1.0 - x.sum()}]

        try:
            res = minimize(objective, x_init, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 500, 'ftol': 1e-9})
            if res.success or res.fun < best_v:
                v = res.fun
                if v < best_v and v >= 0:
                    best_v = v
                    best_mu = np.zeros(d)
                    best_mu[:d - 1] = res.x
                    best_mu[d - 1] = 1.0 - res.x.sum()
        except Exception:
            pass

    return float(best_v), best_mu


def val_window_brute(d: int, n_dirichlet: int = 2000, seed: int = 0) -> tuple:
    """Estimate val(d) := inf_mu max_W TV_W(mu) (the standard window TV).

    TV_W(mu) = (2d/ell) * sum_{k=s..s+ell-2} MC[k]
            = (4n/ell) * sum_{k} MC[k]
    (per CLAUDE.md notation where 2d/ell vs 4n/ell are interchangeable.)
    """
    rng = np.random.default_rng(seed)
    d2 = d  # alias

    def max_tv(mu):
        max_v = 0.0
        for ell in range(2, 2 * d + 1):
            for s in range(2 * d - ell + 1):
                mc_sum = 0.0
                for k in range(s, s + ell - 1):
                    if k < 0 or k > 2 * d - 2:
                        continue
                    i_lo = max(0, k - (d - 1))
                    i_hi = min(d - 1, k)
                    for i in range(i_lo, i_hi + 1):
                        j = k - i
                        mc_sum += mu[i] * mu[j]
                tv = (2.0 * d / ell) * mc_sum
                if tv > max_v:
                    max_v = tv
        return max_v

    best_v = np.inf
    best_mu = None
    for _ in range(n_dirichlet):
        alpha = rng.uniform(0.2, 5.0, size=d)
        mu = rng.dirichlet(alpha)
        v = max_tv(mu)
        if v < best_v:
            best_v = v
            best_mu = mu.copy()

    mu_u = np.ones(d) / d
    v_u = max_tv(mu_u)
    if v_u < best_v:
        best_v = v_u
        best_mu = mu_u
    return float(best_v), best_mu


def main():
    print("=" * 78)
    print("Bounds via cross-term lemma: directions and numerical values")
    print("=" * 78)
    print()
    print("Definitions:")
    print("  val_knot(d) := inf_mu max_s 4n MC[s]   "
          "[exact ||f_step * f_step||_inf]")
    print("  val(d)      := inf_mu max_W TV_W(mu)   "
          "[window-average TV, lower bound on C_{1a}]")
    print()
    print("Inequalities:")
    print("  val(d) <= C_{1a} <= val_knot(d)        "
          "[both <= 1.5029 known upper bound]")
    print()

    for d in [2, 4, 6, 8]:
        print("-" * 78)
        print(f"  d = {d}  (n = {d // 2})")
        v_knot, mu_knot = val_knot_optimize(d, n_starts=50, seed=0)
        v_window, mu_window = val_window_brute(d, n_dirichlet=2000, seed=0)
        print(f"    val_knot(d)  = {v_knot:.4f}  (UPPER bound on C_{{1a}})")
        print(f"      best mu:   {mu_knot}")
        print(f"      sum mu:    {mu_knot.sum():.6f}")
        print(f"    val(d)       = {v_window:.4f}  (LOWER bound on C_{{1a}}; "
              f"random search, may not be optimal)")
        print(f"    Gap:           val_knot - val = {v_knot - v_window:+.4f}")
        if v_knot < 1.5029:
            print(f"    *** val_knot beats current upper bound 1.5029 by "
                  f"{1.5029 - v_knot:.4f} ***")
        else:
            print(f"    val_knot does NOT beat 1.5029 (need to find better mu)")


if __name__ == "__main__":
    main()
