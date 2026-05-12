"""Fast val_knot(d) optimizer using analytical gradients.

For inf over simplex of max_s 4n MC[s], use:
  - Smooth log-sum-exp surrogate with annealing
  - Analytical gradient (no finite diff)
  - Projected gradient with simplex projection
  - Adam optimizer
"""
from __future__ import annotations

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_mc(mu, d):
    """Compute MC[s] = sum_{i+j=s} mu_i mu_j for s = 0..2d-2."""
    return np.array([
        sum(mu[i] * mu[s - i]
            for i in range(max(0, s - (d - 1)), min(d - 1, s) + 1))
        for s in range(2 * d - 1)
    ])


def compute_mc_grad(mu, d):
    """grad_{mu_k} MC[s] = 2 * mu_{s - k} (if 0 <= s - k < d) else 0."""
    grad = np.zeros((2 * d - 1, d))
    for s in range(2 * d - 1):
        for k in range(d):
            j = s - k
            if 0 <= j < d:
                grad[s, k] = 2.0 * mu[j]
    return grad


def project_simplex(v):
    """Project v onto the probability simplex (Duchi et al. 2008)."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u + (1 - cssv) / (np.arange(1, n + 1)) > 0)[0]
    if len(rho) == 0:
        # all negative; return uniform
        return np.ones(n) / n
    rho = rho[-1]
    lam = (1 - cssv[rho]) / (rho + 1)
    return np.maximum(v + lam, 0)


def smooth_max(g, T):
    """Smooth max via log-sum-exp at temperature T (large T = sharper)."""
    g_max = g.max()
    return g_max + np.log(np.sum(np.exp(T * (g - g_max)))) / T


def smooth_max_grad(g, grad_g, T):
    """Gradient of smooth max."""
    g_max = g.max()
    w = np.exp(T * (g - g_max))
    w /= w.sum()
    return w @ grad_g  # weighted sum of per-s gradients


def optimize_smooth(mu_init, d, n_iter=1000, lr=0.05, T_schedule=(10.0, 50.0, 200.0, 1000.0),
                     verbose=False):
    """Smooth-max optimization with Adam + temperature annealing."""
    n = d // 2
    mu = mu_init.copy()
    mu = project_simplex(mu)

    # Adam state
    m = np.zeros(d)
    v = np.zeros(d)
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    iter_count = 0
    for T in T_schedule:
        for it in range(n_iter):
            iter_count += 1
            mc = compute_mc(mu, d)
            mc_grad = compute_mc_grad(mu, d)
            g = 4.0 * n * mc
            grad_g = 4.0 * n * mc_grad

            sm = smooth_max(g, T)
            sm_grad = smooth_max_grad(g, grad_g, T)

            # Adam update
            m = beta1 * m + (1 - beta1) * sm_grad
            v = beta2 * v + (1 - beta2) * sm_grad ** 2
            m_hat = m / (1 - beta1 ** iter_count)
            v_hat = v / (1 - beta2 ** iter_count)
            step = lr * m_hat / (np.sqrt(v_hat) + eps)

            mu = mu - step
            mu = project_simplex(mu)

            if verbose and it % 200 == 0:
                hard_max = g.max()
                print(f"      T={T}, it={it}, smooth={sm:.5f}, hard={hard_max:.5f}")
        # Polishing: drop lr
        lr *= 0.5

    return mu


def evaluate(mu, d):
    """Hard max_s 4n MC[s]."""
    n = d // 2
    return float(np.max(4.0 * n * compute_mc(mu, d)))


def main():
    print("=" * 78)
    print("Fast val_knot(d) optimizer (smooth-max + Adam + cascade refine)")
    print("=" * 78)

    rng = np.random.default_rng(42)

    # Stage 1: d=8 with many random starts
    d = 8
    print(f"\n--- Stage 1: d={d} (50 random starts, smooth-max + Adam) ---")
    best_v = np.inf
    best_mu = None
    t0 = time.time()
    for trial in range(50):
        alpha = rng.uniform(0.5, 5.0, size=d)
        mu_init = rng.dirichlet(alpha)
        mu = optimize_smooth(mu_init, d, n_iter=500, lr=0.02)
        v = evaluate(mu, d)
        if v < best_v:
            best_v = v
            best_mu = mu.copy()
            print(f"  [trial {trial}, t={time.time() - t0:.1f}s] best = {best_v:.6f}")

    # Stages 2+: lift and refine
    while d < 64:
        d_new = 2 * d
        # Lift
        mu_lifted = np.zeros(d_new)
        for i in range(d):
            mu_lifted[2 * i] = best_mu[i] / 2
            mu_lifted[2 * i + 1] = best_mu[i] / 2
        v_lifted = evaluate(mu_lifted, d_new)

        print(f"\n--- Stage: d={d} -> {d_new} (lifted val = {v_lifted:.6f}) ---")
        # Refine: 30 perturbed starts
        best_v = v_lifted
        best_mu = mu_lifted.copy()
        t0 = time.time()
        for trial in range(30):
            sigma = 0.02 * (1.0 + trial / 30)
            mu_init = best_mu + sigma * rng.standard_normal(d_new)
            mu_init = project_simplex(mu_init)
            mu = optimize_smooth(mu_init, d_new, n_iter=500, lr=0.01)
            v = evaluate(mu, d_new)
            if v < best_v:
                best_v = v
                best_mu = mu.copy()
                print(f"  [trial {trial}, t={time.time() - t0:.1f}s] best = {best_v:.6f}")
        print(f"  d={d_new} final: {best_v:.6f}  ({time.time() - t0:.1f}s)")
        if best_v < 1.5029:
            print(f"\n  *** UPPER BOUND C_{{1a}} <= {best_v:.5f} BEATS 1.5029 ***")

        d = d_new


if __name__ == "__main__":
    main()
