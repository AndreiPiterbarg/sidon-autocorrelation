"""Auxiliary SDP: upper bound on ||f||_2^2 given ||f*f||_infty <= M.

Discrete Shor (rank-1 lift, level-1 Lasserre) relaxation:

    Variables: Y = [1; mu] [1; mu]^T  in  R^{(d+1) x (d+1)}, PSD.
    Equivalently: Y_{00} = 1, mu_i = Y_{0, 1+i}, Y_{1+i, 1+j} = mu_i mu_j.

    Objective:  max  c_obj^T  *  vec(Y[1:, 1:])
        with c_obj built so that <c_obj, Y[1:,1:]> = ||f||_2^2 = (2d) sum mu_i^2
        i.e. c_obj = 2d * I  (diagonal scale-d).

    Constraints:
      Y[0, 0]          = 1
      sum_i Y[0, 1+i]  = 1                              (probability)
      Y[0, 1+i]       >= 0                              (mu_i >= 0)
      Y[1+i, 1+j]     >= 0                              (mu_i mu_j >= 0; entrywise RLT)
      sum_j Y[1+i, 1+j] = Y[0, 1+i]                     (linear RLT  mu_i sum mu_j = mu_i)
      <M_W, Y[1:, 1:]>  <=  M    for every window W      (||f*f||_infty proxy)

The optimum is an UPPER bound on max ||f||_2^2 over admissible f.  The
bound IS rigorous *given* the discretisation map between continuum f and
discrete mu.

Discretisation calibration
--------------------------
For a piecewise-constant f on d cells of width Dx covering [-1/4, 1/4],
mu_i = f(x_i) Dx (so sum mu_i = int f = 1, mu_i >= 0).  Then

    ||f||_2^2  =  int f^2  =  sum f(x_i)^2 Dx  =  sum (mu_i / Dx)^2 Dx
                =  (1/Dx) sum mu_i^2  =  (2d / 1)  sum mu_i^2 [if Dx = 1/(2d)]

i.e. ||f||_2^2 is (2d) sum mu_i^2 in our normalisation.  See
`lasserre/core.py:build_window_matrices` for the matching window matrices
M_W with scale 2d/ell encoding ||f*f||_infty.

Time / space optimality
-----------------------
* Variables: O(d^2) lifted entries (Y).  Sparse via PSD lift.
* Constraints: O(d^2) windows + O(d) linear RLT + O(d^2) entrywise.
* Solver: Clarabel (free, sparse, conic).  At d=22: ~50k constraints,
  ~500-var SDP; expected solve time well under a minute.
* Object dispatch: build M_W matrices ONCE (O(d^4) memory once), then
  re-use inside the SDP loop.

Tightness
---------
Shor (level 1) bounds the true max ||f||_2^2 from ABOVE, hence gives a
RIGOROUS upper bound b_bar(M).  Lasserre level >= 2 tightens further.
We start with Shor; if the bound is too loose (b_bar > 2.08 at M=1.275)
we escalate.
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
class BAuxResult:
    d: int
    M: float
    b_bar: float          # rigorous upper bound on ||f||_2^2
    mu_witness: Optional[np.ndarray]
    n_windows: int
    solver_status: str
    wall_s: float


def solve_b_aux_shor(d: int, M: float,
                     solver: str = "CLARABEL",
                     verbose: bool = False) -> BAuxResult:
    """Shor SDP for b_bar(M, d) := max (2d) sum mu_i^2 s.t. ...

    See module docstring for the constraint set.  Returns an UPPER BOUND
    on the true max ||f||_2^2 over admissible f at discretization d.

    Parameters
    ----------
    d : int
        Grid resolution; d points on [-1/4, 1/4].
    M : float
        Cap on ||f*f||_infty.
    solver : str
        cvxpy solver name; CLARABEL, MOSEK, SCS.
    verbose : bool
        Print solver progress.
    """
    import time
    import cvxpy as cp

    t0 = time.time()
    windows, M_mats = build_window_matrices(d)
    n_W = len(windows)

    # Lifted variable Y of size (d+1) x (d+1), symmetric PSD.
    Y = cp.Variable((d + 1, d + 1), PSD=True)
    mu = Y[0, 1:]                # shape (d,)
    XX = Y[1:, 1:]               # shape (d, d), == mu mu^T

    constraints = [
        Y[0, 0] == 1,
        cp.sum(mu) == 1,
        mu >= 0,
        XX >= 0,                 # entrywise nonneg
    ]
    # RLT: sum_j Y[1+i, 1+j] = Y[0, 1+i] for each i
    constraints.append(cp.sum(XX, axis=1) == mu)
    # Symmetry (already enforced by PSD=True for Y, but keep for clarity)
    # Window constraints: <M_W, XX> <= M for each W
    for W_idx in range(n_W):
        Mw = M_mats[W_idx]
        # cvxpy trace handles sparse-by-dense; M_w is dense float64
        constraints.append(cp.sum(cp.multiply(Mw, XX)) <= M)

    # Objective: max ||f||_2^2 = 2d * trace(XX)
    obj = cp.Maximize(2 * d * cp.trace(XX))

    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        return BAuxResult(
            d=d, M=M, b_bar=float("inf"),
            mu_witness=None, n_windows=n_W,
            solver_status=prob.status,
            wall_s=time.time() - t0,
        )

    b_bar = float(prob.value)
    mu_witness = np.asarray(mu.value).flatten()
    return BAuxResult(
        d=d, M=M, b_bar=b_bar,
        mu_witness=mu_witness,
        n_windows=n_W,
        solver_status=prob.status,
        wall_s=time.time() - t0,
    )


# ---------------------------------------------------------------------
# Sweep: compute b_bar at multiple M values
# ---------------------------------------------------------------------

def sweep_M(d: int, Ms, solver: str = "CLARABEL", verbose: bool = False):
    """Sweep b_bar(M) at multiple M values.  Returns a list of BAuxResult."""
    out = []
    for M in Ms:
        res = solve_b_aux_shor(d, M, solver=solver, verbose=verbose)
        out.append(res)
        if verbose:
            mark = "  <- BREAKEVEN" if res.b_bar < 1 + (M - 1) / mu_M(M) else ""
            print(f"  M={M:.4f}: b_bar={res.b_bar:.6f}, "
                  f"status={res.solver_status}, "
                  f"wall={res.wall_s:.1f}s{mark}")
    return out


def mu_M(M: float) -> float:
    """mu(M) = M * sin(pi/M) / pi  (Lemma 3.4 / Lemma 1 box)."""
    import math
    return M * math.sin(math.pi / M) / math.pi


def breakeven_b(M: float) -> float:
    """Breakeven b_bar value: b_bar = 1 + (M - 1) / mu(M) gives no improvement."""
    return 1 + (M - 1) / mu_M(M)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, default=22, help="discretisation; default 22")
    p.add_argument("--M", type=float, nargs="*",
                   default=[1.26, 1.27, 1.275, 1.28, 1.285, 1.29, 1.295, 1.30],
                   help="Cap M on ||f*f||_infty to evaluate.")
    p.add_argument("--solver", type=str, default="CLARABEL")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"=== B_aux Shor SDP at d={args.d} ===")
    print(f"Windows: built from lasserre.core.build_window_matrices(d={args.d})")
    print()
    print(f"{'M':>7} {'breakeven':>11} {'b_bar':>10} {'M_below_breakeven?':>20} {'wall(s)':>8}")
    print("-" * 70)
    results = []
    for M in args.M:
        bk = breakeven_b(M)
        res = solve_b_aux_shor(args.d, M, solver=args.solver, verbose=args.verbose)
        ok = "YES" if res.b_bar < bk else "NO"
        print(f"{M:>7.4f} {bk:>11.6f} {res.b_bar:>10.6f} {ok:>20} {res.wall_s:>8.1f}")
        results.append(res)

    print()
    print("Interpretation:")
    print("  b_bar < breakeven  =>  sharper Markov gives strict improvement.")
    print("  M=1.275: breakeven b ~ 2.08; if b_bar < 2.08 we beat MV's 1.276.")
