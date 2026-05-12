"""Fejer-Riesz SOS cone for real even cosine trig polynomials on T = R/Z.

For a real cosine polynomial

    p(t) = r_0 + 2 sum_{l=1}^{D} r_l cos(2 pi l t),

the correct characterization of  p >= 0 on T  (Fejer-Riesz, SOS form) is

    exists Q in S^{D+1}_+  with  r_l = sum_{k=0}^{D-l} Q_{k+l, k},  l = 0, ..., D.

Equivalently, via the trace inner product <S, Q>_F = tr(S^T Q):

    r_l = <S_l, Q>,   where  S_l is (D+1) x (D+1) symmetric with

        S_0 = I,
        S_l = (1/2) (J_l + J_l^T)  for l >= 1,  J_l having 1's on the l-th subdiagonal.

The mass normalization  integral p = 1  is  r_0 = 1,  i.e.  tr(Q) = 1.

Note on spec deviation
----------------------
The project prompt states the cone is the Caratheodory-Toeplitz condition
``R(r) := [r_{|i-j|}] >= 0''.  That condition is strictly weaker: it
characterizes when  (r_l)  are the first D+1 Fourier coefficients of SOME
nonneg measure on T, not when the density  p(t)  itself is nonneg.

Counter-example:  r = (1, 1).  R = [[1, 1], [1, 1]] has eigenvalues (0, 2)
and is PSD, but p(t) = 1 + 2 cos(2 pi t) = -1 at t = 1/2.  The correct
SOS cone rejects this r: the constraints tr(Q) = 1 and Q_{0,1} = 1 force
Q = [[a, 1], [1, 1 - a]] whose PSD condition a(1-a) >= 1 has no real
solution.

Using the Caratheodory condition in the primal-dual bound  < (f*f), p >
would make the bound UNSOUND (p could be negative somewhere).  Hence this
module uses the Fejer-Riesz SOS cone throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from flint import fmpq, fmpq_mat


# ---------------------------------------------------------------------
# Trace maps S_l
# ---------------------------------------------------------------------

def fejer_riesz_trace_maps(D: int) -> List[fmpq_mat]:
    """Return [S_0, ..., S_D] with S_l in S^{D+1} exact rational, such that

        r_l = <S_l, Q>_F = sum_{i, j} S_l[i, j] * Q[i, j]

    equals the l-th cosine-polynomial coefficient implied by Q >= 0 under
    Fejer-Riesz.  For symmetric Q this collapses to  r_l = sum_{k} Q_{k+l, k}.
    """
    if D < 0:
        raise ValueError("D must be >= 0")
    maps: List[fmpq_mat] = []
    for l in range(D + 1):
        S = fmpq_mat(D + 1, D + 1)
        if l == 0:
            for k in range(D + 1):
                S[k, k] = fmpq(1)
        else:
            half = fmpq(1, 2)
            for k in range(D + 1 - l):
                S[k + l, k] = half
                S[k, k + l] = half
        maps.append(S)
    return maps


def apply_trace_maps(
    Q: Sequence[Sequence],
    maps: Optional[List[fmpq_mat]] = None,
    D: Optional[int] = None,
) -> List[fmpq]:
    """Given Q (fmpq_mat, fmpq nested lists, or ndarray of floats/fmpq),
    compute r = (<S_0, Q>, ..., <S_D, Q>) in fmpq.

    If Q is an ndarray of floats, the result is still fmpq (via Fraction
    from_float + limit_denominator at 10^12) -- intended for diagnostics.
    """
    # Extract size and type
    if isinstance(Q, fmpq_mat):
        rows = Q.nrows()
        get = lambda i, j: Q[i, j]
    elif isinstance(Q, np.ndarray):
        rows = Q.shape[0]
        get = lambda i, j: fmpq(int(round(float(Q[i, j]) * 10**12)), 10**12)
    else:
        rows = len(Q)
        get = lambda i, j: Q[i][j] if isinstance(Q[i][j], fmpq) else fmpq(Q[i][j])

    if D is None:
        D = rows - 1
    if maps is None:
        maps = fejer_riesz_trace_maps(D)

    r: List[fmpq] = []
    for l in range(D + 1):
        S = maps[l]
        acc = fmpq(0)
        for i in range(rows):
            for j in range(rows):
                s_ij = S[i, j]
                if s_ij == 0:
                    continue
                acc = acc + s_ij * get(i, j)
        r.append(acc)
    return r


# ---------------------------------------------------------------------
# Numeric helpers: factorization + feasibility
# ---------------------------------------------------------------------

@dataclass
class FejerRieszFactorization:
    """Decomposition Q = sum_k q_k q_k^T with q_k the columns weighted
    by sqrt(eigenvalue)."""
    q_columns: np.ndarray    # shape (D+1, rank), each column a q vector
    eigenvalues: np.ndarray  # nonneg eigenvalues (rank,)
    residual: float          # ||Q - sum q_k q_k^T||_F


def fejer_riesz_factorize(
    Q: np.ndarray, rank_tol: float = 1e-10
) -> FejerRieszFactorization:
    """Spectral factorization Q = V diag(lam) V^T = sum_k q_k q_k^T with
    q_k = sqrt(lam_k) V[:, k].  Discards near-zero eigenvalues.

    Diagnostic only; the certified pipeline uses Q directly (not q).
    Yields  p(t) = sum_k |q_k(e^{2 pi i t})|^2 >= 0 on T.
    """
    Q = 0.5 * (Q + Q.T)
    eigvals, eigvecs = np.linalg.eigh(Q)
    mask = eigvals > rank_tol * max(1.0, eigvals.max())
    lams = eigvals[mask]
    V = eigvecs[:, mask]
    q_cols = V * np.sqrt(np.maximum(lams, 0.0))[None, :]
    reconstructed = q_cols @ q_cols.T
    residual = float(np.linalg.norm(Q - reconstructed, ord="fro"))
    return FejerRieszFactorization(
        q_columns=q_cols, eigenvalues=lams, residual=residual
    )


def is_fejer_riesz_feasible_numeric(
    r: Sequence[float], tol: float = 1e-8
) -> Tuple[bool, Optional[np.ndarray]]:
    """Numeric feasibility check via CVXPY: find PSD Q with r_l = <S_l, Q>.

    Returns (feasible, Q_opt or None).  Intended for tests and diagnostics.
    """
    import cvxpy as cp
    D = len(r) - 1
    Q = cp.Variable((D + 1, D + 1), symmetric=True)
    constraints = [Q >> 0]
    maps = fejer_riesz_trace_maps(D)
    for l in range(D + 1):
        S_l = np.zeros((D + 1, D + 1))
        for i in range(D + 1):
            for j in range(D + 1):
                sij = maps[l][i, j]
                if sij != 0:
                    S_l[i, j] = float(int(sij.p)) / float(int(sij.q))
        constraints.append(cp.trace(S_l @ Q) == float(r[l]))
    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver="CLARABEL")
    except Exception:
        return False, None
    if prob.status in ("optimal", "optimal_inaccurate"):
        return True, np.asarray(Q.value)
    return False, None


# ---------------------------------------------------------------------
# numpy view of the trace-map tensor for SDP driver use
# ---------------------------------------------------------------------

def trace_maps_as_numpy(D: int) -> np.ndarray:
    """Return an (D+1, D+1, D+1) float64 tensor M with
        M[l, i, j] = S_l[i, j]   (exact rationals 1 or 1/2 or 0).
    Used by the SDP driver to express the linear constraint  r_l = sum M[l, i, j] Q[i, j]
    as cvxpy linear functionals in Q.
    """
    maps = fejer_riesz_trace_maps(D)
    out = np.zeros((D + 1, D + 1, D + 1), dtype=np.float64)
    for l in range(D + 1):
        S = maps[l]
        for i in range(D + 1):
            for j in range(D + 1):
                sij = S[i, j]
                if sij != 0:
                    out[l, i, j] = float(int(sij.p)) / float(int(sij.q))
    return out


__all__ = [
    "fejer_riesz_trace_maps",
    "apply_trace_maps",
    "FejerRieszFactorization",
    "fejer_riesz_factorize",
    "is_fejer_riesz_feasible_numeric",
    "trace_maps_as_numpy",
]
