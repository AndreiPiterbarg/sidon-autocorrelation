"""Numerical extraction of ||f||_infty for the d-bin discretized Sidon optimum.

In the d-bin discretization, mu_i is the mass on bin i (width 1/(2d) since support
is [-1/4, 1/4] of total length 1/2).  The corresponding piecewise-constant density
is f(x) = 2d * mu_i on bin i, so ||f||_infty = 2d * max_i mu_i.

This gives an empirical estimate of ||f^*||_infty for the unconstrained Sidon
optimum (in the d -> infty limit).  If across d the value is bounded
(say, by 2.5), that's strong empirical evidence — though not a proof — that the
unconstrained optimum has ||f^*||_infty <= 2.5.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize


def max_tv(mu: np.ndarray, d: int) -> float:
    """max_W TV_W(mu) = (2d/ell) * sum-of-conv-window."""
    conv_len = 2 * d - 1
    conv = np.zeros(conv_len)
    for i in range(d):
        for j in range(d):
            conv[i + j] += mu[i] * mu[j]
    best = 0.0
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        # Sliding window sum
        ws = sum(conv[:n_cv])
        if (2.0 * d / ell) * ws > best:
            best = (2.0 * d / ell) * ws
        for s in range(1, conv_len - n_cv + 1):
            ws += conv[s + n_cv - 1] - conv[s - 1]
            v = (2.0 * d / ell) * ws
            if v > best:
                best = v
    return best


def compute_val_d_with_mu(d: int, n_restarts: int = 100, verbose: bool = False) -> tuple[float, np.ndarray]:
    """Returns (val_d, optimal_mu)."""
    best_val = float('inf')
    best_mu = None

    def obj_free(mu_free):
        mu = np.zeros(d)
        mu[:d-1] = mu_free
        mu[d-1] = 1.0 - sum(mu_free)
        if mu[d-1] < -1e-9 or np.any(mu < -1e-9):
            return 1e10
        mu = np.maximum(mu, 0)
        return max_tv(mu, d)

    rng = np.random.default_rng(seed=42 + d)
    for trial in range(n_restarts):
        raw = rng.exponential(1.0, d)
        mu0 = raw / raw.sum()
        try:
            res = minimize(obj_free, mu0[:d-1], method='Nelder-Mead',
                           options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_val:
                mu = np.zeros(d)
                mu[:d-1] = res.x
                mu[d-1] = 1.0 - sum(res.x)
                if mu[d-1] >= -1e-7 and np.all(mu >= -1e-7):
                    best_val = res.fun
                    best_mu = np.maximum(mu, 0)
                    best_mu /= best_mu.sum()
        except Exception:
            pass
    # Also uniform start
    mu0 = np.ones(d) / d
    try:
        res = minimize(obj_free, mu0[:d-1], method='Nelder-Mead',
                       options={'maxiter': 50000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res.fun < best_val:
            mu = np.zeros(d); mu[:d-1] = res.x; mu[d-1] = 1.0 - sum(res.x)
            best_val = res.fun
            best_mu = np.maximum(mu, 0)
            best_mu /= best_mu.sum()
    except Exception:
        pass
    return best_val, best_mu


def main():
    print("=" * 70)
    print("Empirical ||f^*||_infty extraction from val(d) optimum")
    print("=" * 70)
    print(f"{'d':>3} {'val(d)':>10} {'max mu_i':>10} {'||f||_inf':>10} {'mu_layout (sorted)':<40}")
    for d in [4, 6, 8, 10, 12, 14, 16, 20, 24, 32]:
        t0 = time.time()
        n_restarts = 200 if d <= 12 else (100 if d <= 20 else 50)
        val, mu = compute_val_d_with_mu(d, n_restarts=n_restarts)
        elapsed = time.time() - t0
        max_mu = np.max(mu)
        f_inf = 2 * d * max_mu
        # Sorted layout (decreasing)
        mu_sorted = np.sort(mu)[::-1]
        layout = " ".join(f"{m:.3f}" for m in mu_sorted[:8])
        if len(mu_sorted) > 8:
            layout += " ..."
        print(f"{d:>3} {val:>10.5f} {max_mu:>10.5f} {f_inf:>10.4f}  {layout}",
              flush=True)
        print(f"     (full mu, layout-not-sorted): {' '.join(f'{m:.3f}' for m in mu)}",
              flush=True)


if __name__ == "__main__":
    main()
