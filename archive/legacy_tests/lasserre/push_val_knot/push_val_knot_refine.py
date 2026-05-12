"""Cascade-style refinement: optimize at small d, then refine to d -> 2d
by splitting bins, then re-optimize. This avoids basin-hop local minima
at large d.

Key fact: any mu at d-bins lifts to d' = 2d bins by mu' := (mu_i / 2 each).
This refinement preserves the f_step (same continuous step function), so
val_knot(2d) <= val_knot(d) when initialized from a d-bin optimum.

After lifting, run local optimization at d' to find any improvement.
"""
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


def step_lift_d_to_2d(mu_d):
    """Refine d-bin mu to 2d-bin mu' such that the step function is preserved.

    Each old bin (mass mu_i, width h_old=1/(2d)) splits into two new bins of
    width h_new = h_old / 2 = 1/(4d). The step value 4n_old * mu_i must equal
    the step value in each child bin: 4n_new * mu'_{2i} = 4n_old * mu_i,
    so mu'_{2i} = mu'_{2i+1} = mu_i * (n_old / n_new) = mu_i * (1/2).
    """
    d_new = 2 * len(mu_d)
    mu_new = np.zeros(d_new)
    for i in range(len(mu_d)):
        mu_new[2 * i] = mu_d[i] / 2.0
        mu_new[2 * i + 1] = mu_d[i] / 2.0
    # Sanity: this lift should give the SAME val_knot
    return mu_new


def local_refine(mu_init, n_attempts: int = 100, seed: int = 0, verbose: bool = True):
    """Local refinement of mu via SLSQP."""
    d = len(mu_init)
    n = d // 2
    rng = np.random.default_rng(seed)

    def objective(x):
        mu = np.zeros(d)
        mu[:d - 1] = x
        mu[d - 1] = 1.0 - x.sum()
        if mu.min() < -1e-9:
            return 100.0
        mu = np.maximum(mu, 0.0)
        return max_mc_4n(mu, d, n)

    bounds = [(0.0, 1.0)] * (d - 1)
    constraints = [{'type': 'ineq', 'fun': lambda x: 1.0 - x.sum()}]

    best_v = objective(mu_init[:d - 1])
    best_mu = mu_init.copy()

    # First refine the init
    try:
        res = minimize(objective, mu_init[:d - 1], method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 2000, 'ftol': 1e-11})
        if res.fun < best_v:
            best_v = res.fun
            best_mu = np.zeros(d)
            best_mu[:d - 1] = res.x
            best_mu[d - 1] = 1.0 - res.x.sum()
    except Exception:
        pass

    if verbose:
        print(f"    after refining init: {best_v:.6f}")

    # Then do additional local searches with perturbations
    for trial in range(n_attempts):
        # Perturb the best
        sigma = 0.02 * (1.0 + trial / n_attempts)
        x_init = best_mu[:d - 1] + sigma * rng.standard_normal(d - 1)
        # Project to feasible
        x_init = np.maximum(x_init, 1e-6)
        s = x_init.sum()
        if s >= 1.0:
            x_init = x_init * (0.99 / s)

        try:
            res = minimize(objective, x_init, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 1000, 'ftol': 1e-10})
            v = objective(res.x)
            if v < best_v - 1e-7:
                best_v = v
                best_mu = np.zeros(d)
                best_mu[:d - 1] = res.x
                best_mu[d - 1] = 1.0 - res.x.sum()
                if verbose:
                    print(f"    [trial {trial}] new best = {best_v:.6f}")
        except Exception:
            pass

    return best_v, best_mu


def main():
    print("=" * 78)
    print("Cascade-refine val_knot(d): start small, lift bins, refine")
    print("=" * 78)
    print(f"Currently known: 1.2802 <= C_{{1a}} <= 1.5029")
    print()

    # Start from a good d=8 init (we know basin-hop gives ~1.579)
    d = 8
    print(f"--- Stage 1: d = {d} ---")
    rng = np.random.default_rng(0)
    # Start: known good d=8 from earlier
    mu_seed = np.array([0.11753558, 0.13361125, 0.16106661, 0.23675328,
                        0.0403568, 0.04950664, 0.06981174, 0.19135811])
    t0 = time.time()
    v, mu = local_refine(mu_seed, n_attempts=200, seed=0, verbose=False)
    print(f"  d={d}: val_knot = {v:.6f}  ({time.time() - t0:.1f}s)")

    # Lift and refine repeatedly
    for stage in range(4):
        d = len(mu)
        d_new = 2 * d
        if d_new > 64:
            break
        mu_lifted = step_lift_d_to_2d(mu)
        # Sanity: val_knot must be the same after lifting
        n_new = d_new // 2
        v_lifted = max_mc_4n(mu_lifted, d_new, n_new)
        n_old = d // 2
        v_old = max_mc_4n(mu, d, n_old)
        diff = abs(v_lifted - v_old)
        print(f"\n--- Stage {stage + 2}: lift d={d} -> {d_new} ---")
        print(f"  pre-lift val_knot = {v_old:.6f}")
        print(f"  post-lift val_knot = {v_lifted:.6f}  (diff = {diff:.2e}, "
              f"should be 0)")
        assert diff < 1e-9, "lift sanity check failed"
        # Refine
        t0 = time.time()
        v, mu = local_refine(mu_lifted, n_attempts=300, seed=stage, verbose=False)
        print(f"  after refining: val_knot(d={d_new}) = {v:.6f}  "
              f"({time.time() - t0:.1f}s)")
        if v < 1.5029:
            print(f"  *** UPPER BOUND IMPROVED: C_{{1a}} <= {v:.5f} ***")
            print(f"      (current published 1.5029, this beats by {1.5029 - v:.5f})")


if __name__ == "__main__":
    main()
