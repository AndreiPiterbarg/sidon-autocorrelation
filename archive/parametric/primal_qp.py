"""Inner moment SDP: lambda_N(m) = inf_{f in A} int p(t) (f*f)(t) dt.

We use a bivariate Lasserre-style moment relaxation on the box
    [-1/4, 1/4]^2
with variables w_{(i,j)} = E[X^i Y^j] where (X, Y) ~ f(x) f(y).

The admissibility of f reduces to:
    g_1(x, y) = 1/16 - x^2 >= 0,
    g_2(x, y) = 1/16 - y^2 >= 0,
    int 1 d(sigma) = 1,
    marginals symmetric: w_{(i, j)} = w_{(j, i)}.

Objective (given Chebyshev coefs m of p):
    int p(t) (f*f)(t) dt
      = int p(x+y) f(x) f(y) dx dy
      = sum_{l, i, j} m_l * K^(l)[i][j] * w_{(i,j)}                          (*)

At moment order N (N >= ceil((L-1)/2)), this is a valid LOWER bound on the
true inner infimum, since for any admissible f the moment sequence of f (x) f
satisfies all the relaxation's constraints.

Usage
-----
    from parametric.primal_qp import solve_inner_qp
    m = np.array([...])                 # Chebyshev coefs of p
    out = solve_inner_qp(m, N=4)
    out["lambda_N"]                     # rigorous lower bound on lambda(mu)
"""
from __future__ import annotations

from math import comb
from typing import Optional

import numpy as np

from .chebyshev_duality import (
    bivariate_basis,
    bivariate_pair_map,
    chebyshev_monomial_coefs,
    integrate_Tl_2t,
)

try:
    import cvxpy as cp
    HAVE_CVXPY = True
except ImportError:  # pragma: no cover
    HAVE_CVXPY = False


def _matrix_entries_via_pair_map(
    basis, pair_map, w_var
):
    """Build the (MN x MN) SDP matrix where entry [i, j] references the variable
    w[(basis[i] + basis[j])].  Returns a list-of-lists of CVXPY expressions."""
    MN = len(basis)
    rows = []
    for i in range(MN):
        a1, b1 = basis[i]
        row = []
        for j in range(MN):
            a2, b2 = basis[j]
            row.append(w_var[(a1 + a2, b1 + b2)])
        rows.append(row)
    return rows


def solve_inner_qp(
    m: np.ndarray,
    N: int,
    solver: str = "CLARABEL",
    verbose: bool = False,
    enforce_marginal_symmetry: bool = True,
    return_moments: bool = False,
) -> dict:
    """Lower bound on inf_{f in A} int p (f*f).

    Parameters
    ----------
    m : array of length L
        Chebyshev coefficients of p: p(t) = sum_l m_l * T_l(2 t).
    N : int
        Bivariate moment matrix order.  Requires 2 N >= L - 1.
    solver : str
        CVXPY solver.  "CLARABEL", "MOSEK", "SCS".
    enforce_marginal_symmetry : bool
        If True, add w_{(i,j)} == w_{(j,i)} (tightening; implied by f (x) f).
    return_moments : bool
        If True, also return the moment dict w.

    Returns
    -------
    dict with keys
        lambda_N : float             (rigorous LB on lambda(mu))
        status   : CVXPY status string
        w        : optional, dict {(i,j): float}
    """
    if not HAVE_CVXPY:
        raise RuntimeError("cvxpy is required for solve_inner_qp.")

    m = np.asarray(m, dtype=float).flatten()
    L = len(m)
    if 2 * N < L - 1:
        raise ValueError(f"Need 2 N >= L - 1, got N={N}, L={L}.")

    cheb = chebyshev_monomial_coefs(L)
    basis_N = bivariate_basis(N)
    basis_Nm1 = bivariate_basis(N - 1)
    MN = len(basis_N)

    # w-variables indexed by (A, B) with A + B <= 2 N.
    w = {}
    for A in range(2 * N + 1):
        for B in range(2 * N + 1 - A):
            w[(A, B)] = cp.Variable()

    constraints = [w[(0, 0)] == 1]

    # Moment matrix PSD.
    MM_rows = _matrix_entries_via_pair_map(basis_N, None, w)
    M_mat = cp.bmat(MM_rows)
    constraints.append(M_mat >> 0)

    # Localizing matrices for g_1 = 1/16 - x^2 and g_2 = 1/16 - y^2.
    if basis_Nm1:
        L1_rows = []
        L2_rows = []
        for i, (a1, b1) in enumerate(basis_Nm1):
            r1, r2 = [], []
            for j, (a2, b2) in enumerate(basis_Nm1):
                A0, B0 = a1 + a2, b1 + b2
                # g_1 localize: (1/16) * w[(A0, B0)] - w[(A0+2, B0)]
                r1.append((1.0 / 16.0) * w[(A0, B0)] - w[(A0 + 2, B0)])
                # g_2 localize: (1/16) * w[(A0, B0)] - w[(A0, B0+2)]
                r2.append((1.0 / 16.0) * w[(A0, B0)] - w[(A0, B0 + 2)])
            L1_rows.append(r1)
            L2_rows.append(r2)
        constraints.append(cp.bmat(L1_rows) >> 0)
        constraints.append(cp.bmat(L2_rows) >> 0)

    # Marginal symmetry: w_{(i,j)} = w_{(j,i)} for all i < j.
    if enforce_marginal_symmetry:
        for (A, B) in list(w.keys()):
            if A < B:
                constraints.append(w[(A, B)] == w[(B, A)])

    # Build objective (*).  K^(l)[i][j] = c_{l, i+j} * 2^{i+j} * C(i+j, i).
    obj_terms = []
    for l in range(L):
        for i in range(L):
            for j in range(L - i):
                k = i + j
                if k >= L:
                    continue
                coef = float(cheb[l][k]) * (2 ** k) * comb(k, i)
                if coef == 0.0:
                    continue
                obj_terms.append(m[l] * coef * w[(i, j)])
    objective = cp.Minimize(cp.sum(obj_terms))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    out = {
        "lambda_N": float(prob.value) if prob.value is not None else None,
        "status": prob.status,
    }
    if return_moments:
        out["w"] = {
            k: (float(v.value) if v.value is not None else None)
            for k, v in w.items()
        }
    return out


def evaluate_p_on_grid(m: np.ndarray, n_grid: int = 2001) -> dict:
    """Sanity check: evaluate p(t) = sum_l m_l T_l(2t) on a grid of
    [-1/2, 1/2] and report min/max.  Non-rigorous; for debugging.
    """
    L = len(m)
    ts = np.linspace(-0.5, 0.5, n_grid)
    # Chebyshev recurrence in 2 t = u on [-1, 1].
    u = 2.0 * ts
    T_prev = np.ones_like(u)
    T_curr = u.copy()
    p = m[0] * T_prev
    if L > 1:
        p = p + m[1] * T_curr
    for l in range(2, L):
        T_next = 2.0 * u * T_curr - T_prev
        p = p + m[l] * T_next
        T_prev, T_curr = T_curr, T_next
    return {"min": float(p.min()), "max": float(p.max()), "p": p, "t": ts}
