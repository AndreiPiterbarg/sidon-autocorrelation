#!/usr/bin/env python
"""Compute val(d) = min_mu max_W TV_W(mu) exactly via optimization.

val(d) is the minimum over all mass distributions mu on the d-simplex
of the maximum test value across all windows. This is a finite-dimensional
minimax problem: min over mu of max over (ell, s) of TV_W(mu).

We solve it as: min t  s.t. TV_W(mu) <= t for all W, mu on simplex.
This is a quadratically-constrained linear program (QCLP) — or equivalently,
we can use scipy.optimize.minimize on the smooth objective max_W TV_W(mu).
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def compute_all_tv(mu, d):
    """Compute TV_W(mu) for all windows (ell, s). Returns dict (ell,s)->TV."""
    conv_len = 2 * d - 1
    # Autoconvolution
    conv = np.zeros(conv_len)
    for i in range(d):
        for j in range(d):
            conv[i + j] += mu[i] * mu[j]

    results = {}
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        for s in range(conv_len - n_cv + 1):
            ws = sum(conv[s:s + n_cv])
            tv = (2.0 * d / ell) * ws
            results[(ell, s)] = tv
    return results


def max_tv(mu, d):
    """Compute max_W TV_W(mu)."""
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len)
    for i in range(d):
        for j in range(d):
            conv[i + j] += mu[i] * mu[j]

    best = 0.0
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        ws = sum(conv[:n_cv])
        for s in range(conv_len - n_cv + 1):
            if s > 0:
                ws += conv[s + n_cv - 1] - conv[s - 1]
            tv = (2.0 * d / ell) * ws
            if tv > best:
                best = tv
    return best


def neg_max_tv(mu_free, d):
    """Objective for minimization. mu_free has d-1 free variables."""
    # Map from unconstrained to simplex
    mu = np.zeros(d)
    mu[:d-1] = mu_free
    mu[d-1] = 1.0 - sum(mu_free)
    if mu[d-1] < 0 or np.any(mu < 0):
        return 1e10  # penalty
    return max_tv(mu, d)


def compute_val(d, n_restarts=50):
    """Compute val(d) via multistart optimization."""
    best_val = 1e10
    best_mu = None

    # Method 1: scipy minimize with multiple starts
    for trial in range(n_restarts):
        # Random starting point on simplex
        raw = np.random.exponential(1.0, d)
        mu0 = raw / raw.sum()

        try:
            res = minimize(
                lambda x: neg_max_tv(x, d),
                mu0[:d-1],
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            if res.fun < best_val:
                best_val = res.fun
                mu = np.zeros(d)
                mu[:d-1] = res.x
                mu[d-1] = 1.0 - sum(res.x)
                best_mu = mu.copy()
        except Exception:
            pass

    # Method 2: also try uniform as starting point
    mu0 = np.ones(d) / d
    res = minimize(
        lambda x: neg_max_tv(x, d),
        mu0[:d-1],
        method='Nelder-Mead',
        options={'maxiter': 50000, 'xatol': 1e-14, 'fatol': 1e-14}
    )
    if res.fun < best_val:
        best_val = res.fun
        mu = np.zeros(d)
        mu[:d-1] = res.x
        mu[d-1] = 1.0 - sum(res.x)
        best_mu = mu.copy()

    # Method 3: try COBYLA with explicit constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: sum(x) - 1.0},
    ] + [
        {'type': 'ineq', 'fun': lambda x, i=i: x[i]} for i in range(d)
    ]
    for trial in range(n_restarts // 2):
        raw = np.random.exponential(1.0, d)
        mu0 = raw / raw.sum()
        try:
            res = minimize(
                lambda x: max_tv(x, d),
                mu0,
                method='COBYLA',
                constraints=constraints,
                options={'maxiter': 20000, 'rhobeg': 0.01}
            )
            if res.fun < best_val and abs(sum(res.x) - 1.0) < 1e-6 and np.all(res.x >= -1e-8):
                best_val = res.fun
                best_mu = np.maximum(res.x, 0)
                best_mu /= best_mu.sum()
        except Exception:
            pass

    return best_val, best_mu


def main():
    print("=" * 60)
    print("COMPUTE val(d) = min_mu max_W TV_W(mu)")
    print("=" * 60, flush=True)

    for d in [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 32]:
        t0 = time.time()
        n_restarts = 100 if d <= 16 else 50
        val, mu = compute_val(d, n_restarts=n_restarts)
        elapsed = time.time() - t0

        # Verify
        tv_check = max_tv(mu, d)
        above_130 = "YES" if val > 1.30 else "NO"
        margin = val - 1.30

        print(f"\n  d={d:>2}: val(d) = {val:.8f}  "
              f"(margin over 1.30: {margin:+.6f})  "
              f"val>1.30? {above_130}  [{elapsed:.1f}s]")
        # Show the minimizing distribution (rounded)
        mu_str = ", ".join(f"{x:.4f}" for x in mu)
        print(f"        mu* = [{mu_str}]", flush=True)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60, flush=True)


if __name__ == '__main__':
    main()
