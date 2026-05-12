"""Aggressive basin-hopping for val_knot(d) at fixed d."""
from __future__ import annotations

import sys, os, time
import numpy as np
from scipy.optimize import minimize, basinhopping

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def max_mc_4n(mu, d, n):
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


def run(d: int, n_iter: int = 200, seed: int = 0):
    n = d // 2

    def obj(x):
        mu = np.zeros(d)
        mu[:d - 1] = x
        mu[d - 1] = 1.0 - x.sum()
        if mu.min() < -1e-9:
            return 100.0
        mu = np.maximum(mu, 0.0)
        return max_mc_4n(mu, d, n)

    rng = np.random.default_rng(seed)

    # Several starting points
    bounds = [(0.0, 1.0)] * (d - 1)

    best_v = np.inf
    best_x = None
    best_mu = None
    print(f"  d = {d}, basin-hopping, {n_iter} iters")
    t0 = time.time()

    # Try multiple seeds
    for s_idx in range(5):
        rng_s = np.random.default_rng(seed + s_idx)
        alpha = rng_s.uniform(0.5, 5.0, size=d)
        x_init = rng_s.dirichlet(alpha)[:d - 1]

        minimizer_kwargs = {
            "method": "SLSQP",
            "bounds": bounds,
            "constraints": [{'type': 'ineq', 'fun': lambda x: 1.0 - x.sum()}],
            "options": {'maxiter': 500, 'ftol': 1e-10},
        }
        try:
            result = basinhopping(
                obj, x_init, minimizer_kwargs=minimizer_kwargs,
                niter=n_iter // 5, seed=int(seed + s_idx) % (2**31),
                stepsize=0.05,
            )
            if result.fun < best_v:
                best_v = result.fun
                best_x = result.x
                best_mu = np.zeros(d)
                best_mu[:d - 1] = best_x
                best_mu[d - 1] = 1.0 - best_x.sum()
                elapsed = time.time() - t0
                print(f"    [{elapsed:.1f}s, seed_idx={s_idx}] best = {best_v:.6f}")
        except Exception as e:
            print(f"    seed {s_idx} failed: {e}")

    elapsed = time.time() - t0
    print(f"  d={d} final: {best_v:.6f}  ({elapsed:.1f}s)")
    if best_mu is not None:
        # Sanity check non-negativity and sum
        assert best_mu.min() >= -1e-9
        assert abs(best_mu.sum() - 1.0) < 1e-6
        print(f"  best mu = {best_mu}")
    return best_v, best_mu


if __name__ == "__main__":
    print("=" * 70)
    print("Basin-hopping val_knot(d) — find UPPER bound on C_{1a}")
    print("=" * 70)
    print("Currently known: 1.2802 <= C_{1a} <= 1.5029")
    print()

    for d in [8, 12, 16, 20, 24]:
        run(d, n_iter=200 if d <= 16 else 100, seed=42)
        print()
