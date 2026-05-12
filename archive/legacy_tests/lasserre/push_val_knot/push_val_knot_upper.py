"""Push val_knot(d) at higher d to seek upper bound improvements on C_{1a}.

val_knot(d) := inf_mu max_s 4n MC[s]   where  MC[s] := sum_{i+j=s} mu_i mu_j.

For ANY mu in Delta_d, the step function f_step with bin masses mu is
admissible and ||f_step * f_step||_inf = max_s 4n MC[s] EXACTLY (cross-term
lemma + piecewise linearity). Therefore C_{1a} <= val_knot(d).

We aim to numerically minimise max_s 4n MC[s] for d = 8, 12, 16, 24, 32.

Optimization strategy:
  - Many random Dirichlet starts (varying alpha)
  - SLSQP local refinement
  - Subgradient steps from best-found
  - Symmetry breaking: try both symmetric and asymmetric inits
  - Track all-time best
"""
from __future__ import annotations

import sys, os, time
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lasserre.fourier_xterm import step_autoconv_inf_norm


def max_mc_4n(mu, d, n):
    """max_s 4n * MC[s]."""
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


def search_val_knot(d: int, n_starts: int = 200, seed: int = 0,
                    use_cvxpy_lp: bool = False, verbose: bool = True):
    """Aggressive numerical search for inf_mu max_s 4n MC[s]."""
    rng = np.random.default_rng(seed)
    n = d // 2

    def objective(x):
        mu = np.zeros(d)
        mu[:d - 1] = x
        mu[d - 1] = 1.0 - x.sum()
        if mu.min() < -1e-12:
            return 100.0
        mu = np.maximum(mu, 0.0)
        return max_mc_4n(mu, d, n)

    best_v = np.inf
    best_mu = None

    # Special starts to try
    special_inits = [
        np.ones(d) / d,                               # uniform
    ]
    # symmetric concentrated at endpoints
    if d >= 4:
        mu = np.zeros(d)
        mu[0] = 0.5
        mu[-1] = 0.5
        special_inits.append(mu)
    # symmetric trapezoid
    mu_t = np.linspace(1.0, d, d)
    mu_t = mu_t * (d - mu_t + 1)
    mu_t = mu_t / mu_t.sum()
    special_inits.append(mu_t)

    bounds = [(0.0, 1.0)] * (d - 1)
    constraints = [{'type': 'ineq', 'fun': lambda x: 1.0 - x.sum()}]

    t0 = time.time()
    for trial in range(n_starts + len(special_inits)):
        if trial < len(special_inits):
            mu_init = special_inits[trial]
        else:
            # Mix of distributions
            r = rng.random()
            if r < 0.3:
                alpha = rng.uniform(0.5, 5.0, size=d)
            elif r < 0.6:
                alpha = rng.uniform(0.1, 1.0, size=d)  # more concentrated
            else:
                alpha = rng.uniform(2.0, 10.0, size=d)  # more uniform
            mu_init = rng.dirichlet(alpha)

        x_init = mu_init[:d - 1]
        try:
            res = minimize(objective, x_init, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 1000, 'ftol': 1e-10})
            v = objective(res.x)
            if v < best_v:
                best_v = v
                best_mu = np.zeros(d)
                best_mu[:d - 1] = res.x
                best_mu[d - 1] = 1.0 - res.x.sum()
                if verbose:
                    elapsed = time.time() - t0
                    print(f"    [{elapsed:.1f}s, trial {trial}] new best = {best_v:.6f}")
        except Exception:
            pass

    return float(best_v), best_mu


def main():
    print("=" * 80)
    print("PUSH: val_knot(d) for upper bound on C_{1a}")
    print("=" * 80)
    print("Currently known: 1.2802 <= C_{1a} <= 1.5029")
    print("Target: find d, mu such that max_s 4n MC[s] < 1.5029")
    print()

    results = {}
    for d in [4, 6, 8, 10, 12, 16, 20, 24, 32]:
        # scale n_starts with d (need more for higher dim)
        n_starts = max(100, 50 * d)
        if d >= 24:
            n_starts = 200
        if d >= 32:
            n_starts = 100  # SLSQP becomes slow
        print(f"--- d = {d} (n_starts = {n_starts}) ---")
        t0 = time.time()
        v, mu = search_val_knot(d, n_starts=n_starts, seed=0, verbose=False)
        elapsed = time.time() - t0
        print(f"  d={d:3d}: val_knot(d) ~= {v:.5f}   ({elapsed:.1f}s)")
        if v < 1.5029:
            print(f"    *** BEATS 1.5029 by {1.5029 - v:.5f} -- new upper bound candidate ***")
            print(f"    mu = {mu}")
        results[d] = (v, mu)

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'d':>4} {'val_knot':>10} {'note':>30}")
    for d, (v, mu) in results.items():
        note = "BEATS 1.5029" if v < 1.5029 else ("MATCHES (tie)" if abs(v - 1.5029) < 0.001 else "")
        print(f"{d:>4} {v:>10.5f} {note:>30}")


if __name__ == "__main__":
    main()
