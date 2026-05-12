"""Discretized SDP for the Sidon constant with L^{3/2} constraint.

================================================================================
MATHEMATICAL FORMULATION
================================================================================

Goal: lower-bound

    C_{1a}^{(B)} := inf  sup_{|t| <= 1/2} (f * f)(t)
                  s.t.  f >= 0,  supp f subset [-1/4, 1/4],  int f = 1,
                        ||f||_{3/2} <= B

via a discretization on uniform grid + Lasserre level-1 (Shor) relaxation.

Discretization
--------------
Grid: x_i = -1/4 + i * dx,  i = 0, ..., n-1,  dx = (1/2) / (n-1).
Variables: f_i ~ f(x_i) >= 0.

Relations (trapezoidal-style):
  int f       ~  dx * sum_i f_i = 1
  int f^{3/2} ~  dx * sum_i z_i,  with z_i >= f_i^{3/2}  (power cone, exponent 2/3)
  (f*f)(y_k)  ~  dx * sum_{i+j=k} f_i f_j,    y_k = -1/2 + k*dx,  k = 0..2n-2

Lasserre Level-1 lift
---------------------
Introduce symmetric Q in R^{n x n} relaxing q_{ij} = f_i f_j via
    [ 1   f^T ]
    [ f    Q  ]   PSD                                          ... (Shor)
    Q_{ij} >= 0, Q symmetric                                   (valid: f >= 0)

The convolution constraint becomes
    dx * sum_{i+j=k} Q_{ij} <= lambda   for each k = 0, ..., 2n-2
The L^{3/2} constraint:
    z_i >= f_i^{3/2}    via 3D power cone (z_i, 1, f_i) with exponents (2/3, 1/3)
    dx * sum_i z_i <= B^{3/2}

Objective: min lambda.

The relaxation gives lambda^SDP <= C_{1a,discrete}^{(B)} <= C_{1a}^{(B)} + O(dx^?).
The discretization error is examined empirically by sweeping n.

================================================================================
WHAT THIS DOES *NOT* PROVE
================================================================================

This SDP yields a lower bound on  C_{1a}^{(B)}, NOT on C_{1a}.  Converting to a
C_{1a} lower bound requires the analytical bridge

    sup(f*f) >= c0 * ||f||_{3/2}^3

(reverse-Young on [-1/4, 1/4]).  Empirical evidence: c0 ~ pi/8 ~= 0.393, with
Schinzel-Schmidt as the candidate extremizer in the alpha-family.  This bound
is CONJECTURED, not proved.  See REPORT for the full bridge analysis.

================================================================================
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np


# =====================================================================
# Discretized SDP builder
# =====================================================================

@dataclass
class L32SDPInfo:
    n: int
    dx: float
    B: float
    n_vars: int = 0
    n_psd_size: int = 0
    n_power_cones: int = 0
    n_conv_constraints: int = 0
    build_seconds: float = 0.0
    solve_seconds: float = 0.0
    status: str = ""
    objective: Optional[float] = None
    f_value: Optional[np.ndarray] = None
    Q_value: Optional[np.ndarray] = None
    L32_realized: Optional[float] = None  # actual ||f||_{3/2} of the optimum
    sup_ff_realized: Optional[float] = None  # actual sup f*f from the f


def build_l32_sdp(n: int, B: float, *,
                  add_lasserre_lift: bool = True,
                  enforce_reflection_symmetry: bool = False) -> Tuple[cp.Problem, L32SDPInfo, Dict]:
    """Build the L^{3/2}-constrained SDP.

    n: number of grid points.
    B: L^{3/2} bound (so int f^{3/2} <= B^{3/2}).
    add_lasserre_lift: if True, introduce Q with PSD lift; if False, treat
        f*f as quadratic in f (non-convex, not solvable) -- only used as a
        SANITY check via cp.Problem(...).is_dcp().  Always set True for solving.
    enforce_reflection_symmetry: if True, require f(-x) = f(x).  Halves the
        f-vector but assumes the optimum is symmetric (NOT proved; SS is not
        symmetric).  Default False.
    """
    if n < 3:
        raise ValueError(f"n must be >= 3, got {n}")
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    t0 = time.time()

    # Grid:  x_i = -1/4 + i * dx,  dx = (1/2)/(n-1)
    dx = 0.5 / (n - 1)
    # Convolution grid:  y_k = -1/2 + k * dx,  k = 0..2n-2

    # Variables
    f = cp.Variable(n, nonneg=True, name="f")
    z = cp.Variable(n, nonneg=True, name="z")  # z_i >= f_i^{3/2}
    lam = cp.Variable(name="lambda")

    constraints: List[cp.Constraint] = []

    # Normalization: int f = 1  (trapezoidal/midpoint -> sum_i f_i * dx = 1)
    constraints.append(cp.sum(f) * dx == 1.0)

    # L^{3/2} via power cone:  z_i >= f_i^{3/2}
    # Equivalent: z_i^{2/3} * 1^{1/3} >= |f_i|
    # CVXPY: cp.power(f_i, 1.5) <= z_i  -- CVXPY supports cp.power for constraints
    # via DCP atoms; cp.power(x, 1.5) is convex when x >= 0.
    for i in range(n):
        constraints.append(cp.power(f[i], 1.5) <= z[i])
    constraints.append(cp.sum(z) * dx <= B**1.5)

    # Lasserre lift
    if add_lasserre_lift:
        Q = cp.Variable((n, n), symmetric=True, name="Q")
        constraints.append(Q >= 0)  # entrywise nonneg
        # PSD lift: [[1, f^T], [f, Q]] >> 0
        big = cp.bmat([[cp.reshape(cp.Constant(1.0), (1,1)), cp.reshape(f, (1, n))],
                       [cp.reshape(f, (n, 1)), Q]])
        constraints.append(big >> 0)

        # Convolution constraint: dx * sum_{i+j=k} Q_{ij} <= lambda for each k
        n_conv = 2 * n - 1
        for k in range(n_conv):
            i_lo = max(0, k - (n - 1))
            i_hi = min(n - 1, k)
            terms = []
            for i in range(i_lo, i_hi + 1):
                j = k - i
                terms.append(Q[i, j])
            if not terms:
                continue
            constraints.append(cp.sum(cp.hstack(terms)) * dx <= lam)
    else:
        Q = None
        # Without lift, f*f is quadratic in f (non-convex).  Used only for testing.
        n_conv = 2 * n - 1
        for k in range(n_conv):
            i_lo = max(0, k - (n - 1))
            i_hi = min(n - 1, k)
            terms = []
            for i in range(i_lo, i_hi + 1):
                j = k - i
                terms.append(f[i] * f[j])
            if not terms:
                continue
            constraints.append(cp.sum(cp.hstack(terms)) * dx <= lam)

    if enforce_reflection_symmetry:
        for i in range(n // 2):
            constraints.append(f[i] == f[n - 1 - i])

    objective = cp.Minimize(lam)
    problem = cp.Problem(objective, constraints)

    info = L32SDPInfo(
        n=n, dx=dx, B=B,
        n_vars=int(np.prod(f.shape) + np.prod(z.shape) + (np.prod(Q.shape) if Q is not None else 0) + 1),
        n_psd_size=(n + 1) if add_lasserre_lift else 0,
        n_power_cones=n,
        n_conv_constraints=2 * n - 1,
        build_seconds=time.time() - t0,
    )
    handles = dict(f=f, z=z, lam=lam, Q=Q, dx=dx, B=B, n=n)
    return problem, info, handles


def solve_l32(problem: cp.Problem, info: L32SDPInfo, handles: Dict, *,
              solver: str = "MOSEK", verbose: bool = False,
              mosek_params: Optional[Dict] = None) -> L32SDPInfo:
    """Solve the SDP and populate info fields."""
    t0 = time.time()
    try:
        if mosek_params and solver == "MOSEK":
            problem.solve(solver=solver, verbose=verbose, mosek_params=mosek_params)
        else:
            problem.solve(solver=solver, verbose=verbose)
    except Exception as exc:
        info.solve_seconds = time.time() - t0
        info.status = f"ERROR: {type(exc).__name__}: {exc}"
        return info
    info.solve_seconds = time.time() - t0
    info.status = problem.status
    info.objective = float(problem.value) if problem.value is not None else None

    # Extract f-value and compute realized norms
    f = handles["f"]
    if f.value is not None:
        f_val = np.array(f.value).flatten()
        info.f_value = f_val
        dx = handles["dx"]
        info.L32_realized = (np.sum(f_val**1.5) * dx)**(2/3)
        # Compute actual sup f*f via discrete convolution
        # (f*f)(y_k) = dx * sum_{i+j=k} f_i f_j
        ff = np.convolve(f_val, f_val) * dx
        info.sup_ff_realized = float(np.max(ff))

    Q = handles.get("Q", None)
    if Q is not None and Q.value is not None:
        info.Q_value = np.array(Q.value)

    return info


__all__ = ["build_l32_sdp", "solve_l32", "L32SDPInfo"]
