"""Witness search: lower-bound b_bar(M) := max ||f||_2^2 s.t. ||f*f||_inf <= M.

Idea
----
The Shor SDP gives b_bar_Shor = 2d = trivial UPPER bound (relaxation too
loose; admits spurious diagonal XX = I/d).  We need a TIGHTER upper bound.

But before sinking time in Lasserre level 2, we should answer the cheaper
question: is the TRUE b_bar(1.275) above or below the breakeven 2.08?

If we find ANY admissible mu (with max_W TV_W <= 1.275) such that
||f||_2^2 = 2d * sum mu_i^2 > 2.08, then the sharper-Markov path is DEAD:
the true b_bar > breakeven, no matter how tight the relaxation.

Method
------
Project-gradient ascent on the constrained nonconvex QP

    max  2d sum mu_i^2
    s.t. mu in Delta_d, max_W TV_W(mu) <= M

with multiple random restarts.  Each step:
  1. Compute gradient g_i = 2 * mu_i (of the objective) clipped by the
     active-set Lagrangian.
  2. Project onto the feasible set (intersection of simplex and the half-
     spaces TV_W(mu) <= M).
  3. Iterate.

Implementation:
  * Use scipy.optimize.minimize with SLSQP or trust-constr.
  * Multiple random Dirichlet starts, plus uniform start, plus existing
    val(d) optimizer mu_star (if available).
  * Each restart: ~few seconds.  Many restarts: ~minute.

Output
------
LOWER bound on b_bar(M), via the BEST witness found:
  b_bar(M) >= max over witnesses of 2d * sum mu_i^2.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from lasserre.core import build_window_matrices


@dataclass
class WitnessResult:
    M: float
    d: int
    best_b_bar_lo: float          # = 2d * max sum mu_i^2 found
    best_mu: np.ndarray
    n_restarts: int
    n_feasible_restarts: int


def find_witness(d: int, M: float,
                 n_restarts: int = 30,
                 max_iter: int = 500,
                 seed: int = 0,
                 verbose: bool = False) -> WitnessResult:
    """Find admissible mu with high ||f||_2^2 via project-gradient ascent.

    Returns the best witness found and the LOWER bound it implies on b_bar(M).
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    windows, M_mats = build_window_matrices(d)
    n_W = len(windows)

    # Stack window kernels into a (n_W, d, d) tensor for fast TV evaluation.
    M_stack = np.stack(M_mats, axis=0)   # (n_W, d, d)

    def neg_obj(mu):
        # We minimize -||f||_2^2 = -(2d) * sum mu_i^2
        return -2.0 * d * float(np.sum(mu ** 2))

    def neg_obj_grad(mu):
        return -4.0 * d * mu

    # Constraint: max_W TV_W(mu) <= M
    # Encoded as: M - mu^T M_W mu >= 0 for each W (= 1+ inequality constraints)
    def make_window_constr(W_idx):
        Mw = M_mats[W_idx]
        def fun(mu):
            return M - float(mu @ Mw @ mu)
        def jac(mu):
            return -2.0 * (Mw @ mu)
        return {"type": "ineq", "fun": fun, "jac": jac}

    constraints = [make_window_constr(w) for w in range(n_W)]
    # ALSO add pointwise junction constraints (l=1, continuum-tight)
    def make_junction_constr(k):
        def fun(mu):
            s = 0.0
            for i in range(d):
                j = k - i
                if 0 <= j < d:
                    s += mu[i] * mu[j]
            return M - 2.0 * d * s
        def jac(mu):
            g = np.zeros(d)
            for i in range(d):
                j = k - i
                if 0 <= j < d:
                    g[i] -= 2.0 * d * mu[j]
                    g[j] -= 2.0 * d * mu[i]
            return g
        return {"type": "ineq", "fun": fun, "jac": jac}
    for k in range(2 * d - 1):
        constraints.append(make_junction_constr(k))
    # plus simplex equality: sum mu = 1
    constraints.append({"type": "eq",
                        "fun": lambda mu: float(np.sum(mu) - 1.0),
                        "jac": lambda mu: np.ones(d)})
    # Bounds: mu_i >= 0, mu_i <= 1
    bounds = [(0.0, 1.0)] * d

    best_b_bar = -np.inf
    best_mu = None
    n_feasible = 0

    starts = []
    # Uniform start
    starts.append(np.ones(d) / d)
    # Random Dirichlet starts
    for _ in range(n_restarts - 1):
        alpha = float(rng.uniform(0.5, 5.0))
        starts.append(rng.dirichlet(alpha * np.ones(d)))

    for k, mu0 in enumerate(starts):
        try:
            res = minimize(
                fun=neg_obj, jac=neg_obj_grad,
                x0=mu0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": max_iter, "ftol": 1e-9},
            )
            mu_k = res.x
            # Verify feasibility (allow small slack)
            sum_mu = float(np.sum(mu_k))
            if abs(sum_mu - 1.0) > 1e-5:
                continue
            tv_max = float(np.max(np.einsum("wij,i,j->w", M_stack, mu_k, mu_k)))
            if tv_max > M + 1e-7:
                continue
            n_feasible += 1
            f_norm_sq = 2.0 * d * float(np.sum(mu_k ** 2))
            if f_norm_sq > best_b_bar:
                best_b_bar = f_norm_sq
                best_mu = mu_k.copy()
                if verbose:
                    print(f"  restart {k}: b_bar_lo = {f_norm_sq:.6f}  (max TV={tv_max:.4f})")
        except Exception as e:
            if verbose:
                print(f"  restart {k} failed: {e}")

    if best_mu is None:
        # Fallback: uniform mu (always feasible if M >= max TV at uniform)
        mu_uniform = np.ones(d) / d
        tv_max = float(np.max(np.einsum("wij,i,j->w", M_stack, mu_uniform, mu_uniform)))
        if tv_max <= M + 1e-7:
            best_mu = mu_uniform
            best_b_bar = 2.0 * d * float(np.sum(mu_uniform ** 2))
            n_feasible = 1

    return WitnessResult(
        M=M, d=d,
        best_b_bar_lo=float(best_b_bar) if best_b_bar > -np.inf else float("nan"),
        best_mu=best_mu if best_mu is not None else np.array([]),
        n_restarts=len(starts),
        n_feasible_restarts=n_feasible,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=22)
    p.add_argument("--M", type=float, nargs="*",
                   default=[1.275, 1.30, 1.40, 1.50])
    p.add_argument("--n_restarts", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"=== Witness search for b_bar(M, d={args.d}) ===")
    print(f"Breakeven at M=1.275 is b_bar = 2.08; at M=1.30 is 2.09")
    print()
    print(f"{'M':>7} {'breakeven':>11} {'b_bar_lo':>11} "
          f"{'feas/total':>11} {'verdict':>10}")
    print("-" * 70)
    for M in args.M:
        # breakeven
        import math
        mu_M = M * math.sin(math.pi / M) / math.pi
        bk = 1 + (M - 1) / mu_M
        res = find_witness(args.d, M, n_restarts=args.n_restarts,
                           seed=args.seed, verbose=args.verbose)
        verdict = "PATH ALIVE" if res.best_b_bar_lo < bk else "PATH DEAD"
        print(f"{M:>7.4f} {bk:>11.6f} {res.best_b_bar_lo:>11.6f} "
              f"{res.n_feasible_restarts:>4}/{res.n_restarts:<6} {verdict:>10}")
    print()
    print("If b_bar_lo > breakeven for any M of interest:")
    print("  -> there exists admissible mu with high ||f||_2^2,")
    print("     so the sharper-Markov path is structurally dead.")
