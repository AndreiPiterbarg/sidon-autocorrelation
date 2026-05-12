"""SDP relaxation + rounding for val_knot(d).

Solve the Lasserre-1 relaxation:
  min t
  s.t.  Y in S^d, Y >> 0, Y >= 0 (entrywise), sum_{ij} Y[i,j] = 1
        4n * sum_{i+j=s} Y[i,j] <= t   for all s = 0..2d-2

This gives a LOWER bound on val_knot(d). Extract mu via dominant eigenvector
of Y (or sqrt diag) — gives a candidate good mu. Then refine with SLSQP.
"""
from __future__ import annotations

import sys, os, time
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cvxpy as cp


def solve_sdp_relax(d: int):
    """Solve min t s.t. Y PSD, Y entrywise >=0, sum Y = 1, 4n*sum_{i+j=s} Y_ij <= t."""
    n = d // 2
    Y = cp.Variable((d, d), symmetric=True)
    t = cp.Variable()
    constraints = [Y >> 0, Y >= 0, cp.sum(Y) == 1]
    for s in range(2 * d - 1):
        s_term = 0
        for i in range(d):
            j = s - i
            if 0 <= j < d:
                s_term = s_term + Y[i, j]
        constraints.append(4.0 * n * s_term <= t)
    obj = cp.Minimize(t)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False, eps=1e-8)
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        return None, None
    return float(t.value), Y.value


def extract_mu_from_Y(Y, d):
    """Extract candidate mu vectors from PSD Y of rank possibly > 1."""
    eigs, vecs = np.linalg.eigh(Y)
    # Try top-k eigenvectors
    candidates = []
    for k in range(min(3, d)):
        v = vecs[:, -1 - k]  # eigenvector for top eigenvalue
        # Take sign to make sum positive
        if v.sum() < 0:
            v = -v
        v = np.maximum(v, 0)
        if v.sum() > 0:
            candidates.append(v / v.sum())
    # Also try sqrt diag (gives a symmetric rank-1 approximation)
    diag = np.sqrt(np.maximum(np.diag(Y), 0))
    if diag.sum() > 0:
        candidates.append(diag / diag.sum())
    return candidates


def max_mc_4n(mu, d, n):
    max_v = 0.0
    for s in range(2 * d - 1):
        mc = 0.0
        for i in range(max(0, s - (d - 1)), min(d - 1, s) + 1):
            j = s - i
            mc += mu[i] * mu[j]
        v = 4.0 * n * mc
        if v > max_v:
            max_v = v
    return max_v


def local_refine(mu_init, n_attempts=100, seed=0, verbose=False):
    d = len(mu_init)
    n = d // 2
    rng = np.random.default_rng(seed)

    def obj(x):
        mu = np.zeros(d)
        mu[:d - 1] = x
        mu[d - 1] = 1.0 - x.sum()
        if mu.min() < -1e-9:
            return 100.0
        return max_mc_4n(np.maximum(mu, 0.0), d, n)

    bounds = [(0.0, 1.0)] * (d - 1)
    constraints = [{'type': 'ineq', 'fun': lambda x: 1.0 - x.sum()}]

    best_v = obj(mu_init[:d - 1])
    best_mu = mu_init.copy()

    res = minimize(obj, mu_init[:d - 1], method='SLSQP',
                   bounds=bounds, constraints=constraints,
                   options={'maxiter': 2000, 'ftol': 1e-11})
    if res.fun < best_v:
        best_v = res.fun
        best_mu = np.zeros(d)
        best_mu[:d - 1] = res.x
        best_mu[d - 1] = 1.0 - res.x.sum()

    for trial in range(n_attempts):
        sigma = 0.05 * (1.0 + trial / n_attempts)
        x_init = best_mu[:d - 1] + sigma * rng.standard_normal(d - 1)
        x_init = np.maximum(x_init, 1e-6)
        if x_init.sum() >= 1.0:
            x_init = x_init * (0.99 / x_init.sum())
        try:
            res = minimize(obj, x_init, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 1000, 'ftol': 1e-10})
            v = res.fun
            if v < best_v - 1e-7:
                best_v = v
                best_mu = np.zeros(d)
                best_mu[:d - 1] = res.x
                best_mu[d - 1] = 1.0 - res.x.sum()
                if verbose:
                    print(f"      [trial {trial}] -> {best_v:.6f}")
        except Exception:
            pass
    return best_v, best_mu


def main():
    print("=" * 78)
    print("SDP relaxation + rounding for val_knot(d)")
    print("=" * 78)

    for d in [8, 12, 16, 20]:
        n = d // 2
        print(f"\n--- d = {d} ---")
        t0 = time.time()
        sdp_lb, Y = solve_sdp_relax(d)
        print(f"  SDP lower bound on val_knot:  {sdp_lb:.5f}  "
              f"({time.time() - t0:.1f}s)")
        if Y is None:
            continue

        # Extract candidate mu
        candidates = extract_mu_from_Y(Y, d)
        for c_idx, mu_c in enumerate(candidates):
            v_c = max_mc_4n(mu_c, d, n)
            print(f"  candidate {c_idx}: val_knot(extracted mu) = {v_c:.5f}")

        # Refine the best candidate
        best_init_idx = int(np.argmin([max_mc_4n(c, d, n) for c in candidates]))
        print(f"  Refining best candidate ({best_init_idx})...")
        t1 = time.time()
        v_refined, mu_refined = local_refine(candidates[best_init_idx],
                                              n_attempts=200, seed=0,
                                              verbose=False)
        print(f"  After SLSQP refinement:       {v_refined:.5f}  "
              f"({time.time() - t1:.1f}s)")
        if v_refined < 1.5029:
            print(f"  *** UPPER BOUND IMPROVED: C_{{1a}} <= {v_refined:.5f} ***")


if __name__ == "__main__":
    main()
