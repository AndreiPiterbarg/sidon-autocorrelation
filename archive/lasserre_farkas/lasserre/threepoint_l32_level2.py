"""Discretized Lasserre LEVEL-2 SDP for the Sidon constant with L^{3/2} constraint.

================================================================================
MATHEMATICAL FORMULATION (Level-2 moment hierarchy)
================================================================================

Goal: lower-bound

    C_{1a}^{(B)} := inf  sup_{|t| <= 1/2} (f * f)(t)
                  s.t.  f >= 0,  supp f subset [-1/4, 1/4],  int f = 1,
                        ||f||_{3/2} <= B

via a discretization on uniform grid + Lasserre LEVEL-2 relaxation.

Discretization (same as level-1 in `threepoint_l32.py`):
    Grid: x_i = -1/4 + i * dx,  i = 0..n-1,  dx = (1/2)/(n-1).
    Variables: f_i ~ f(x_i) >= 0.

Level-2 moment lift
-------------------
Introduce moment variables y_alpha for every multi-index alpha over {0,..,n-1}
with |alpha| = sum_i alpha_i in {0, 1, 2, 3, 4}.  Interpretation:
    y_alpha = E[ prod_i f_i^{alpha_i} ].
By symmetry  y_alpha  is invariant under reordering of the index sequence;
internally we store the multi-index in CANONICAL FORM as a sorted tuple of
positions repeated by their multiplicity.  For example:
    y_{}                <-> 1                        (degree 0)
    y_{(i,)}            <-> f_i                       (degree 1)
    y_{(i,j)}, i<=j     <-> f_i f_j                   (degree 2)
    y_{(i,j,k)}, i<=j<=k <-> f_i f_j f_k              (degree 3)
    y_{(i,j,k,l)}       <-> f_i f_j f_k f_l           (degree 4)

Number of variables (multi-indices):
    deg <= 4 :  C(n+4,4)   -- e.g. n=15: 3876,  n=25: 23751.

PSD lift: the level-2 moment matrix M_2(y) is indexed by monomials of degree
<= 2 in the variables f_i.  Its rows/cols are alpha with |alpha| in {0,1,2}.
Number of such row/col indices:
    1 + n + n*(n+1)/2  =  N2.
For n=15: N2 = 136.   For n=25: N2 = 351.   For n=30: N2 = 496.
Entry M_2[alpha, beta] = y_{alpha + beta}  (where + is multiset union, deg<=4).

Localizer for f_i >= 0 (level 1 localizer at level-2 lift):
For each i, build a matrix L^{(i)} of size 1 + n indexed by monomials of degree
<= 1, with entry L^{(i)}[alpha, beta] = y_{alpha + beta + (i,)} (deg<=3 moment).
Constraint:  L^{(i)} >> 0.

Mass:                    y_{(0,)} + y_{(1,)} + ... + y_{(n-1,)} times dx = 1.
                         Equivalently: y_{(i,)} = f_i  ( we identify directly ).

L^{3/2}:                 z_i >= f_i^{3/2}  via 3D power cone, dx * sum_i z_i <= B^{3/2}.

Convolution (autocorrelation peak at y_k = -1/2 + k*dx):
    (f*f)(y_k) ~ dx * sum_{i+j=k} f_i f_j = dx * sum_{i+j=k} y_{(min(i,j), max(i,j))}.
    dx * sum_{i+j=k} q_{ij} <= lambda  for every k,  where  q_{ij} = y_{(i,j)} (i<=j).

Objective: min lambda.

Level-2 relaxation is at least as tight as level-1 (Shor).  In particular, the
level-1 PSD constraint  [[1, f^T], [f, Q]] >> 0  is the upper-left block of M_2.

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
# Multi-index utilities (canonical form: sorted tuple of positions)
# =====================================================================

def canon(idx: Tuple[int, ...]) -> Tuple[int, ...]:
    """Canonicalize a multi-index: sorted tuple of integer positions."""
    return tuple(sorted(idx))


def add_idx(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
    """Multiset sum (multiplication of monomials)."""
    return canon(a + b)


def enum_alphas(n: int, max_deg: int) -> List[Tuple[int, ...]]:
    """Enumerate canonical multi-indices over {0..n-1} with total degree <= max_deg.

    Returns a list of canonical (sorted) tuples, sorted by (degree, lex).
    """
    out = [tuple()]  # degree-0
    cur = [tuple()]
    for d in range(1, max_deg + 1):
        nxt: List[Tuple[int, ...]] = []
        for t in cur:
            start = t[-1] if t else 0
            for i in range(start, n):
                nxt.append(t + (i,))
        out.extend(nxt)
        cur = nxt
    return out


# =====================================================================
# Level-2 SDP builder
# =====================================================================

@dataclass
class L32Level2SDPInfo:
    n: int
    dx: float
    B: float
    n_moment_vars: int = 0
    n_psd_size_M2: int = 0
    n_psd_size_localizer: int = 0
    n_localizers: int = 0
    n_power_cones: int = 0
    n_conv_constraints: int = 0
    build_seconds: float = 0.0
    solve_seconds: float = 0.0
    status: str = ""
    objective: Optional[float] = None
    f_value: Optional[np.ndarray] = None
    Q_value: Optional[np.ndarray] = None
    L32_realized: Optional[float] = None
    sup_ff_realized: Optional[float] = None


def build_l32_level2_sdp(n: int, B: float, *,
                          add_localizers: bool = True,
                          enforce_reflection_symmetry: bool = False) -> Tuple[cp.Problem, L32Level2SDPInfo, Dict]:
    """Build the level-2 Lasserre SDP.

    n: number of grid points.
    B: L^{3/2} bound.
    add_localizers: if True, add the n localizer matrices for f_i >= 0
        (each PSD of size n+1).  These are key for tightness.
    enforce_reflection_symmetry: if True, require f(-x) = f(x).
    """
    if n < 3:
        raise ValueError(f"n must be >= 3, got {n}")
    if B <= 0:
        raise ValueError(f"B must be positive, got {B}")
    t0 = time.time()
    dx = 0.5 / (n - 1)

    # Enumerate canonical multi-indices up to degree 4
    alphas4 = enum_alphas(n, max_deg=4)
    alpha_to_id: Dict[Tuple[int, ...], int] = {a: i for i, a in enumerate(alphas4)}
    n_moments = len(alphas4)

    # Create one CVXPY variable per canonical multi-index
    y = cp.Variable(n_moments, name="y")

    # Convenience: fast lookup returning a CVXPY expression
    def Y(t: Tuple[int, ...]):
        c = canon(t)
        return y[alpha_to_id[c]]

    constraints: List[cp.Constraint] = []

    # Pin y_{} = 1
    constraints.append(y[alpha_to_id[tuple()]] == 1.0)

    # Convenience handles:
    f_idxs = [alpha_to_id[(i,)] for i in range(n)]
    f_vec = y[f_idxs]  # the n-vector of first moments

    # Normalization: dx * sum_i y_{(i,)} = 1
    constraints.append(cp.sum(f_vec) * dx == 1.0)

    # Nonneg first moments (cheap valid cuts; M_2 PSD already implies this
    # via the localizer trick at level-1, but explicit for robustness):
    constraints.append(f_vec >= 0)

    # Nonneg second moments (q_{ij} >= 0  for i,j  -- valid since f >= 0)
    # These are also implied by the localizers but explicit cuts help solvers.
    deg2_idxs = []
    for i in range(n):
        for j in range(i, n):
            deg2_idxs.append(alpha_to_id[(i, j)])
    constraints.append(y[deg2_idxs] >= 0)

    # ---- Build the level-2 moment matrix M_2(y) ----
    # Rows/cols indexed by alphas with |alpha| <= 2.  Entry M_2[a,b] = y_{a+b}.
    alphas_le2 = [a for a in alphas4 if len(a) <= 2]
    N2 = len(alphas_le2)
    M2_rows = []
    for a in alphas_le2:
        row = []
        for b in alphas_le2:
            row.append(Y(a + b))
        M2_rows.append(row)
    M2 = cp.bmat(M2_rows)
    constraints.append(M2 == M2.T)  # numerically enforce symmetry
    constraints.append(M2 >> 0)

    # ---- Localizers for f_i >= 0 ----
    # For each i: matrix L^{(i)}_{a,b} = y_{a+b+(i,)},  a,b in {alphas_le1}.
    n_loc = 0
    if add_localizers:
        alphas_le1 = [a for a in alphas4 if len(a) <= 1]
        Nloc = len(alphas_le1)
        for i in range(n):
            ei = (i,)
            rows = []
            for a in alphas_le1:
                row = []
                for b in alphas_le1:
                    row.append(Y(a + b + ei))
                rows.append(row)
            Li = cp.bmat(rows)
            constraints.append(Li == Li.T)
            constraints.append(Li >> 0)
            n_loc += 1
    else:
        Nloc = 0

    # ---- L^{3/2} via power cone ----
    z = cp.Variable(n, nonneg=True, name="z")
    for i in range(n):
        # z_i >= f_i^{3/2}  with f_i = y_{(i,)}
        constraints.append(cp.power(f_vec[i], 1.5) <= z[i])
    constraints.append(cp.sum(z) * dx <= B**1.5)

    # ---- Convolution -> lambda ----
    lam = cp.Variable(name="lambda")
    n_conv = 2 * n - 1
    for k in range(n_conv):
        i_lo = max(0, k - (n - 1))
        i_hi = min(n - 1, k)
        terms = []
        for i in range(i_lo, i_hi + 1):
            j = k - i
            terms.append(Y((i, j)))
        if not terms:
            continue
        constraints.append(cp.sum(cp.hstack(terms)) * dx <= lam)

    # ---- Reflection symmetry (optional) ----
    if enforce_reflection_symmetry:
        for i in range(n // 2):
            constraints.append(f_vec[i] == f_vec[n - 1 - i])

    objective = cp.Minimize(lam)
    problem = cp.Problem(objective, constraints)

    info = L32Level2SDPInfo(
        n=n, dx=dx, B=B,
        n_moment_vars=n_moments,
        n_psd_size_M2=N2,
        n_psd_size_localizer=Nloc,
        n_localizers=n_loc,
        n_power_cones=n,
        n_conv_constraints=n_conv,
        build_seconds=time.time() - t0,
    )
    handles = dict(
        y=y, z=z, lam=lam,
        alpha_to_id=alpha_to_id, alphas4=alphas4,
        f_idxs=f_idxs, deg2_idxs=deg2_idxs,
        dx=dx, B=B, n=n,
    )
    return problem, info, handles


def solve_l32_level2(problem: cp.Problem, info: L32Level2SDPInfo, handles: Dict, *,
                      solver: str = "MOSEK", verbose: bool = False,
                      mosek_params: Optional[Dict] = None) -> L32Level2SDPInfo:
    """Solve the level-2 SDP and populate info fields."""
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

    n = info.n
    dx = info.dx
    y = handles["y"]
    if y.value is not None:
        y_val = np.array(y.value).flatten()
        f_idxs = handles["f_idxs"]
        f_val = np.maximum(y_val[f_idxs], 0.0)
        info.f_value = f_val
        info.L32_realized = (np.sum(f_val**1.5) * dx) ** (2 / 3)
        ff = np.convolve(f_val, f_val) * dx
        info.sup_ff_realized = float(np.max(ff))

        # Build Q from second moments for diagnostics
        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                idx = handles["alpha_to_id"][(i, j)]
                Q[i, j] = Q[j, i] = y_val[idx]
        info.Q_value = Q

    return info


# =====================================================================
# Validation utility: inject a known feasible f and check PSD
# =====================================================================

def fill_moments_from_f(f_val: np.ndarray, n: int, alphas4: List[Tuple[int, ...]],
                         alpha_to_id: Dict[Tuple[int, ...], int]) -> np.ndarray:
    """Given a numerical density f, compute the corresponding y_alpha values
    (degenerate distribution: y_alpha = prod f_i^{alpha_i}).
    """
    y_val = np.zeros(len(alphas4))
    for a, idx in alpha_to_id.items():
        v = 1.0
        for i in a:
            v *= f_val[i]
        y_val[idx] = v
    return y_val


def validate_psd_for_uniform(n: int) -> Tuple[bool, float, float]:
    """Validation helper: build M_2 and the localizers from a UNIFORM density
    f_i = 2.0 (so that dx * sum f_i = 1 with dx = 0.5/(n-1) ... not quite,
    the uniform with int = 1 over [-1/4, 1/4] has f = 2, so f_i = 2).
    Verify that M_2 and all localizers come out PSD numerically.
    """
    alphas4 = enum_alphas(n, max_deg=4)
    alpha_to_id = {a: i for i, a in enumerate(alphas4)}
    f_val = np.full(n, 2.0)  # uniform
    y_val = fill_moments_from_f(f_val, n, alphas4, alpha_to_id)

    alphas_le2 = [a for a in alphas4 if len(a) <= 2]
    N2 = len(alphas_le2)
    M2 = np.zeros((N2, N2))
    for r, a in enumerate(alphas_le2):
        for c, b in enumerate(alphas_le2):
            M2[r, c] = y_val[alpha_to_id[canon(a + b)]]
    eigs_M2 = np.linalg.eigvalsh((M2 + M2.T) / 2)
    min_eig_M2 = float(eigs_M2.min())

    alphas_le1 = [a for a in alphas4 if len(a) <= 1]
    Nloc = len(alphas_le1)
    min_eig_loc = np.inf
    for i in range(n):
        L = np.zeros((Nloc, Nloc))
        ei = (i,)
        for r, a in enumerate(alphas_le1):
            for c, b in enumerate(alphas_le1):
                L[r, c] = y_val[alpha_to_id[canon(a + b + ei)]]
        eigs = np.linalg.eigvalsh((L + L.T) / 2)
        min_eig_loc = min(min_eig_loc, float(eigs.min()))

    # Numerical PSD: allow tiny negative slack
    ok = (min_eig_M2 >= -1e-9) and (min_eig_loc >= -1e-9)
    return ok, min_eig_M2, min_eig_loc


__all__ = [
    "build_l32_level2_sdp", "solve_l32_level2", "L32Level2SDPInfo",
    "enum_alphas", "canon", "fill_moments_from_f", "validate_psd_for_uniform",
]
