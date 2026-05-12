"""Patched Shor SDP: adds ell=1 (pointwise) junction stripes.

For piecewise-constant f on d cells of width Dx = 1/(2d), f*f is
piecewise-LINEAR with vertices at junctions t_k = k*Dx - 1/2,
k = 0, ..., 2d - 2. The continuum sup ||f*f||_inf is attained at one
of these junctions (since f*f is piecewise-linear; sup is at a vertex).

Pointwise constraint: (f*f)(t_k) = (1/Dx) sum_{i+j=k} mu_i mu_j
                                  = 2d * sum_{i+j=k} mu_i mu_j.

Adding the constraints (f*f)(t_k) <= M for all k is the EXACT continuum
constraint ||f*f||_inf <= M for piecewise-constant f.

The previously-failing b_aux_shor.py used only ell >= 2 averaged-window
constraints from lasserre.core.build_window_matrices, which are STRICTLY
LOOSER than pointwise. The d=10 witness with ||f||_2^2 = 4.14 was SPURIOUS:
it satisfied averaged constraints but violated pointwise constraints, hence
its continuum f had ||f*f||_inf > 1.275.

Expected outcome at d in {22, 32, 64} with M = 1.276:
  b_bar should drop to ~1.85, well below breakeven 2.08 -> PATH ALIVE.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

from lasserre.core import build_window_matrices


# ---------------------------------------------------------------------
# Junction (ell=1, pointwise) kernels
# ---------------------------------------------------------------------

def build_junction_kernels(d: int) -> List[np.ndarray]:
    """Return list of (d, d) kernels K_k for k = 0, ..., 2d-2.

    K_k[i, j] = 2d  if i + j == k  else 0.
    Then (f*f)(t_k) = sum_{i,j} K_k[i,j] * mu_i mu_j = 2d * sum_{i+j=k} mu_i mu_j.
    """
    out = []
    for k in range(2 * d - 1):
        Mk = np.zeros((d, d), dtype=np.float64)
        for i in range(d):
            j = k - i
            if 0 <= j < d:
                Mk[i, j] = 2.0 * d
        out.append(Mk)
    return out


# ---------------------------------------------------------------------
# SDP
# ---------------------------------------------------------------------

@dataclass
class BAuxPatchedResult:
    d: int
    M: float
    b_bar: float
    mu_witness: Optional[np.ndarray]
    n_windows_avg: int
    n_junctions: int
    breakeven: float
    path_alive: bool
    solver_status: str
    wall_s: float


def breakeven_b(M: float) -> float:
    """Sharper-Markov breakeven: b_bar < this iff sharper improves over MV."""
    mu_M = M * math.sin(math.pi / M) / math.pi
    return 1 + (M - 1) / mu_M


def solve_b_aux_patched(d: int, M: float,
                        include_avg_windows: bool = True,
                        solver: str = "CLARABEL",
                        verbose: bool = False) -> BAuxPatchedResult:
    """Patched Shor SDP for max ||f||_2^2 s.t. ||f*f||_inf <= M (continuum,
    for piecewise-constant f).

    Variables: Y in R^{(d+1) x (d+1)}, PSD (lifted [1; mu][1; mu]^T).
    Constraints:
      Y[0, 0] = 1
      Y[0, 1:] = mu, mu >= 0, sum mu = 1
      XX = Y[1:, 1:] >= 0 entrywise
      sum_j XX[i, j] = mu_i  (RLT)
      Pointwise (junction) constraints:
        sum_{i+j=k} XX[i, j] <= M / (2d)   for k = 0, ..., 2d-2
        (i.e. (f*f)(t_k) <= M)
      Averaged window constraints (optional, redundant if junctions tight
      but useful for solver):
        <M_W, XX> <= M  for each averaged window W.
    Objective:
      max 2d * trace(XX)  =  ||f||_2^2.
    """
    import cvxpy as cp

    t0 = time.time()
    junc_K = build_junction_kernels(d)
    avg_windows, avg_M = build_window_matrices(d) if include_avg_windows else ([], [])

    Y = cp.Variable((d + 1, d + 1), PSD=True)
    mu = Y[0, 1:]
    XX = Y[1:, 1:]

    cons = [
        Y[0, 0] == 1,
        cp.sum(mu) == 1,
        mu >= 0,
        XX >= 0,                            # entrywise
        cp.sum(XX, axis=1) == mu,           # RLT
    ]
    # Pointwise junction constraints (NEW, the missing piece)
    for K in junc_K:
        cons.append(cp.sum(cp.multiply(K, XX)) <= M)
    # Averaged-window constraints (redundant in principle if junctions
    # are tight, but help solver convergence)
    for Mw in avg_M:
        cons.append(cp.sum(cp.multiply(Mw, XX)) <= M)

    obj = cp.Maximize(2 * d * cp.trace(XX))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=verbose)

    bk = breakeven_b(M)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return BAuxPatchedResult(
            d=d, M=M, b_bar=float("inf"),
            mu_witness=None,
            n_windows_avg=len(avg_M),
            n_junctions=len(junc_K),
            breakeven=bk, path_alive=False,
            solver_status=prob.status,
            wall_s=time.time() - t0,
        )
    b_bar = float(prob.value)
    return BAuxPatchedResult(
        d=d, M=M, b_bar=b_bar,
        mu_witness=np.asarray(mu.value).flatten(),
        n_windows_avg=len(avg_M),
        n_junctions=len(junc_K),
        breakeven=bk, path_alive=(b_bar < bk),
        solver_status=prob.status,
        wall_s=time.time() - t0,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--d", type=int, nargs="*", default=[22])
    p.add_argument("--M", type=float, nargs="*",
                   default=[1.270, 1.275, 1.276, 1.280, 1.290, 1.300, 1.50])
    p.add_argument("--solver", type=str, default="CLARABEL")
    p.add_argument("--no_avg", action="store_true",
                   help="Drop averaged-window constraints (junctions only)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    print(f"=== Patched Shor SDP: pointwise (f*f) constraints ===")
    print(f"For piecewise-constant f, this is RIGOROUS continuum bound.")
    print()
    print(f"{'d':>4} {'M':>7} {'breakeven':>11} {'b_bar':>10} "
          f"{'verdict':>14} {'wall':>7}")
    print("-" * 65)
    for d in args.d:
        for M in args.M:
            r = solve_b_aux_patched(
                d, M,
                include_avg_windows=not args.no_avg,
                solver=args.solver, verbose=args.verbose,
            )
            v = "PATH ALIVE" if r.path_alive else "path dead"
            print(f"{d:>4d} {M:>7.4f} {r.breakeven:>11.4f} "
                  f"{r.b_bar:>10.4f} {v:>14} {r.wall_s:>7.1f}")
