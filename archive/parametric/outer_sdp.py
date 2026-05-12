"""Joint (outer + inner) SDP via SOS duality.

Mathematical formulation
------------------------
Maximize tilde_lam subject to:

  (A) Outer (on mu):
        p(t) = sum_{l=0}^{L-1} m_l T_l(2 t)  is nonneg on [-1/2, 1/2],
              ==> p(t) = sigma_0_p(t) + (1 - 4 t^2) sigma_1_p(t),
                  with sigma_0_p, sigma_1_p SOS.     (Lukacs / Putinar)
        int_{-1/2}^{1/2} p(t) dt = 1.

  (B) Inner lower bound (SOS certificate on box [-1/4, 1/4]^2):
        p(x + y) - tilde_lam
          = sigma_0(x, y) + (1/16 - x^2) sigma_1(x, y) + (1/16 - y^2) sigma_2(x, y),
        with sigma_i bivariate SOS.

The feasibility of (B) implies that for every admissible f
    int f(x) f(y) [p(x+y) - tilde_lam] dx dy >= 0,
i.e.  int p (f*f) >= tilde_lam.  Combined with int p = 1, this yields
    max (f*f) >= int p (f*f) / int p >= tilde_lam,
hence C_{1a} >= tilde_lam.

SOS representation: we use monomial bases and PSD Gram matrices.

    sigma_0(x, y) = v_{D_0}(x, y)^T Sigma_0 v_{D_0}(x, y),   deg 2 D_0
    sigma_1(x, y) = v_{D_1}(x, y)^T Sigma_1 v_{D_1}(x, y),   deg 2 D_1
    sigma_2(x, y) = v_{D_2}(x, y)^T Sigma_2 v_{D_2}(x, y),   deg 2 D_2
    sigma_0_p(t)  = v_{k0}(t)^T   Q0p    v_{k0}(t),          deg 2 k0
    sigma_1_p(t)  = v_{k1}(t)^T   Q1p    v_{k1}(t),          deg 2 k1

with bases  v_D(x, y) = (x^a y^b)_{a + b <= D}.

We pick (for clean even-degree Lukacs):
    L odd, deg p = L - 1 = 2 k0,    k1 = k0 - 1.
    D_0 = N,  D_1 = D_2 = N - 1,    with N >= k0   (tightens as N grows).

Joint coefficient-matching gives a single SDP whose optimum tilde_lam* is a
rigorous (up to solver precision) lower bound on C_{1a}.
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


def _compute_rhs_coef_biv(
    A: int,
    B: int,
    Sigma0,
    Sigma1,
    Sigma2,
    pair0,
    pair12,
):
    """Build the CVXPY linear expression for coef of x^A y^B on the RHS of
    the bivariate SOS identity.

    RHS = sigma_0 + (1/16 - x^2) sigma_1 + (1/16 - y^2) sigma_2.
    """
    terms = []

    # sigma_0 contribution
    for (i, j) in pair0.get((A, B), []):
        terms.append(Sigma0[i, j])

    # (1/16) sigma_1 contribution
    for (i, j) in pair12.get((A, B), []):
        terms.append((1.0 / 16.0) * Sigma1[i, j])
    # -x^2 sigma_1 contribution
    if A >= 2:
        for (i, j) in pair12.get((A - 2, B), []):
            terms.append(-Sigma1[i, j])

    # (1/16) sigma_2 contribution
    for (i, j) in pair12.get((A, B), []):
        terms.append((1.0 / 16.0) * Sigma2[i, j])
    # -y^2 sigma_2 contribution
    if B >= 2:
        for (i, j) in pair12.get((A, B - 2), []):
            terms.append(-Sigma2[i, j])

    if not terms:
        return 0
    return cp.sum(terms)


def _compute_rhs_coef_univ(
    r: int,
    Q0p,
    Q1p,
    pair0_u,
    pair1_u,
):
    """Coef of t^r in sigma_0_p + (1 - 4 t^2) sigma_1_p, where Q0p, Q1p Gram matrices."""
    terms = []

    for (i, j) in pair0_u.get(r, []):
        terms.append(Q0p[i, j])

    if Q1p is not None:
        for (i, j) in pair1_u.get(r, []):
            terms.append(Q1p[i, j])
        if r >= 2:
            for (i, j) in pair1_u.get(r - 2, []):
                terms.append(-4.0 * Q1p[i, j])

    if not terms:
        return 0
    return cp.sum(terms)


def _univ_pair_map(size: int):
    """r -> [(i, j)] with 0 <= i, j < size and i + j = r."""
    from collections import defaultdict

    out = defaultdict(list)
    for i in range(size):
        for j in range(size):
            out[i + j].append((i, j))
    return out


def solve_outer_sdp(
    L: int,
    N: int,
    solver: str = "CLARABEL",
    verbose: bool = False,
    enforce_symmetry: bool = True,
) -> dict:
    """Solve the joint outer + inner SOS SDP.

    Parameters
    ----------
    L : int, odd
        Number of Chebyshev coefficients for mu;  deg p = L - 1 = 2 k0.
    N : int
        Order of SOS multipliers.  Need 2 N >= L - 1 and N >= 1.
    solver : str
        CVXPY solver name.
    enforce_symmetry : bool
        If True, also enforce m_l = 0 for odd l (mu symmetric about 0).
        The extremal mu is known to be symmetric, so this cuts variables
        without losing optimality for the parametric family considered.

    Returns
    -------
    dict with keys:
        bound     : float          (certified lower bound if solver honest)
        status    : str
        m         : np.ndarray     Chebyshev coefs of optimal p (= mu density)
        Sigma0, Sigma1, Sigma2     Gram matrices for bivariate SOS (float)
        Q0p, Q1p                   Gram matrices for univariate Lukacs
        basis_biv0, basis_biv12    Bivariate monomial bases
        basis_univ0, basis_univ1   Univariate exponent ranges (list)
        L, N                       echoed back
    """
    if not HAVE_CVXPY:
        raise RuntimeError("cvxpy is required for solve_outer_sdp.")

    if L % 2 == 0:
        raise ValueError("L must be odd (deg p = L - 1 even for Lukacs).")
    if 2 * N < L - 1:
        raise ValueError(f"Need 2 N >= L - 1, got N={N}, L={L}.")
    if N < 1:
        raise ValueError("Need N >= 1 for bivariate localizing constraints.")

    # Precompute.
    cheb = chebyshev_monomial_coefs(L)
    int_T = [float(v) for v in integrate_Tl_2t(L)]
    pow2 = [1.0]
    for _ in range(2 * N + 2):
        pow2.append(pow2[-1] * 2)

    # Bivariate SOS bases.
    basis0 = bivariate_basis(N)        # deg sigma_0 <= 2 N
    basis12 = bivariate_basis(N - 1)   # deg sigma_{1,2} <= 2 N - 2
    M0 = len(basis0)
    M12 = len(basis12)
    pair0 = bivariate_pair_map(basis0)
    pair12 = bivariate_pair_map(basis12)

    # Univariate Lukacs sizes.
    k0 = (L - 1) // 2
    k1 = k0 - 1
    Q0_size = k0 + 1
    Q1_size = k1 + 1 if k1 >= 0 else 0
    pair0_u = _univ_pair_map(Q0_size)
    pair1_u = _univ_pair_map(Q1_size) if Q1_size > 0 else {}

    # CVXPY variables.
    m = cp.Variable(L)
    tilde_lam = cp.Variable()
    Sigma0 = cp.Variable((M0, M0), symmetric=True)
    Sigma1 = cp.Variable((M12, M12), symmetric=True) if M12 > 0 else None
    Sigma2 = cp.Variable((M12, M12), symmetric=True) if M12 > 0 else None
    Q0p = cp.Variable((Q0_size, Q0_size), symmetric=True)
    Q1p = cp.Variable((Q1_size, Q1_size), symmetric=True) if Q1_size > 0 else None

    constraints = [Sigma0 >> 0, Q0p >> 0]
    if Sigma1 is not None:
        constraints.append(Sigma1 >> 0)
    if Sigma2 is not None:
        constraints.append(Sigma2 >> 0)
    if Q1p is not None:
        constraints.append(Q1p >> 0)

    # Normalization:  int p = 1.
    constraints.append(
        cp.sum([m[l] * int_T[l] for l in range(L)]) == 1
    )

    # Optional: mu symmetric about 0  =>  m_l = 0 for odd l.
    if enforce_symmetry:
        for l in range(1, L, 2):
            constraints.append(m[l] == 0)

    # --- (A) Univariate Lukacs: p(t) = sigma_0_p(t) + (1 - 4 t^2) sigma_1_p(t) ---
    # Match coef of t^r for r = 0..L-1.
    for r in range(L):
        lhs = cp.sum([m[l] * float(cheb[l][r]) * pow2[r] for l in range(L)])
        rhs = _compute_rhs_coef_univ(r, Q0p, Q1p, pair0_u, pair1_u)
        constraints.append(lhs == rhs)

    # --- (B) Bivariate SOS: p(x+y) - tilde_lam = sum of PSD terms ---
    # Match coef of x^A y^B for all (A, B) with A + B <= 2 N.
    for A in range(2 * N + 1):
        for B in range(2 * N + 1 - A):
            k = A + B
            # LHS coef of x^A y^B in  p(x+y) = sum_l m_l T_l(2(x+y)).
            if k <= L - 1:
                lhs = cp.sum(
                    [m[l] * float(cheb[l][k]) * pow2[k] * comb(k, A) for l in range(L)]
                )
            else:
                lhs = 0
            # Subtract tilde_lam from the (0, 0) constant term.
            if A == 0 and B == 0:
                lhs = lhs - tilde_lam
            rhs = _compute_rhs_coef_biv(A, B, Sigma0, Sigma1, Sigma2, pair0, pair12)
            constraints.append(lhs == rhs)

    # Solve.
    obj = cp.Maximize(tilde_lam)
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=solver, verbose=verbose)
    except cp.SolverError as e:  # pragma: no cover
        return {
            "bound": None,
            "status": f"SolverError: {e}",
            "L": L,
            "N": N,
        }

    def _val(x):
        if x is None or getattr(x, "value", None) is None:
            return None
        return np.asarray(x.value)

    return {
        "bound": float(prob.value) if prob.value is not None else None,
        "status": prob.status,
        "m": _val(m).flatten() if _val(m) is not None else None,
        "tilde_lam": float(tilde_lam.value) if tilde_lam.value is not None else None,
        "Sigma0": _val(Sigma0),
        "Sigma1": _val(Sigma1),
        "Sigma2": _val(Sigma2),
        "Q0p": _val(Q0p),
        "Q1p": _val(Q1p),
        "basis_biv0": basis0,
        "basis_biv12": basis12,
        "basis_univ0_size": Q0_size,
        "basis_univ1_size": Q1_size,
        "L": L,
        "N": N,
    }
