"""Shifted Chebyshev basis helpers for the Lasserre SDP.

The shifted Chebyshev polynomials on [0, 1] are

    T_0*(x) = 1
    T_1*(x) = 2x - 1
    T_{n+1}*(x) = (4x - 2) T_n*(x) - T_{n-1}*(x)

Key identities used throughout:

  (P1) Product formula:
           T_j*(x) T_k*(x) = (1/2) T_{j+k}*(x) + (1/2) T_{|j-k|}*(x).
       When min(j, k) = 0 the two summands coincide and the coefficient
       is exactly 1 (T_0* times T_k* = T_k*).

  (P2) Multiplication by x:
           x T_k*(x) = (1/4) T_{k+1}*(x) + (1/2) T_k*(x) + (1/4) T_{|k-1|}*(x).
       This follows from x = (T_1*(x) + 1) / 2 combined with (P1).
       For k = 0 two branches merge:
           x T_0*(x) = (1/2) T_1*(x) + (1/2) T_0*(x).

All coefficients below are computed with ``fractions.Fraction`` so that
the change of basis is exact; we only convert to float64 when we build
the sparse MOSEK input matrices.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy import sparse as sp


# ---------------------------------------------------------------------
# 1-D primitives
# ---------------------------------------------------------------------

def compute_b_table(max_deg: int) -> List[List[Fraction]]:
    """Integer-exact table ``b[n][k]`` such that

        mu^n = sum_{k = 0}^{n} b[n][k] * T_k*(mu),   n = 0, ..., max_deg.

    Derived from the recursion mu^{n+1} = mu * mu^n and identity (P2).
    Returns a (max_deg + 1) x (max_deg + 1) list-of-lists of ``Fraction``.
    """
    if max_deg < 0:
        raise ValueError("max_deg must be >= 0")
    sz = max_deg + 1
    b: List[List[Fraction]] = [[Fraction(0)] * sz for _ in range(sz)]
    b[0][0] = Fraction(1)
    for n in range(max_deg):
        for j in range(sz):
            cj = b[n][j]
            if cj == 0:
                continue
            # (P2): mu * T_j* distributes to T_{j+1}*, T_j*, T_{|j-1|}*.
            if j + 1 < sz:
                b[n + 1][j + 1] += cj * Fraction(1, 4)
            b[n + 1][j] += cj * Fraction(1, 2)
            b[n + 1][abs(j - 1)] += cj * Fraction(1, 4)
    return b


def cheb_prod_1d(a: int, b: int) -> List[Tuple[int, Fraction]]:
    """(P1) applied in a single coordinate.

    Returns the list of (k, coef) such that

        T_a*(x) T_b*(x) = sum coef * T_k*(x).

    The list has one entry when min(a, b) = 0 and two otherwise.
    """
    m_sum = a + b
    m_abs = abs(a - b)
    if m_sum == m_abs:  # iff min(a, b) == 0
        return [(m_sum, Fraction(1))]
    return [(m_sum, Fraction(1, 2)), (m_abs, Fraction(1, 2))]


def cheb_mul_x_1d(k: int) -> List[Tuple[int, Fraction]]:
    """(P2) in a single coordinate: x T_k*(x) as a list of (m, coef)."""
    if k == 0:
        return [(1, Fraction(1, 2)), (0, Fraction(1, 2))]
    return [(k + 1, Fraction(1, 4)),
            (k, Fraction(1, 2)),
            (abs(k - 1), Fraction(1, 4))]


# ---------------------------------------------------------------------
# Multi-variate expansions
# ---------------------------------------------------------------------

def mono_to_cheb(alpha: Sequence[int],
                 b_table: List[List[Fraction]]) -> Dict[Tuple[int, ...], Fraction]:
    """Expand mu^alpha into Chebyshev multi-indices.

        mu^alpha = prod_i ( sum_{k_i = 0}^{a_i} b[a_i][k_i] T_{k_i}*(mu_i) )
                 = sum_gamma ( prod_i b[a_i][gamma_i] ) T_gamma*(mu),

    where gamma ranges over multi-indices with 0 <= gamma_i <= a_i.
    Returns a dict {gamma: Fraction}.
    """
    d = len(alpha)
    current: Dict[Tuple[int, ...], Fraction] = {(): Fraction(1)}
    for i in range(d):
        ai = alpha[i]
        row = b_table[ai]
        nxt: Dict[Tuple[int, ...], Fraction] = {}
        for prefix, c_prefix in current.items():
            for k_i in range(ai + 1):
                c_i = row[k_i]
                if c_i == 0:
                    continue
                new_prefix = prefix + (k_i,)
                nxt[new_prefix] = nxt.get(new_prefix, Fraction(0)) + c_prefix * c_i
        current = nxt
    return current


def cheb_prod(alpha: Sequence[int],
              beta: Sequence[int]) -> Dict[Tuple[int, ...], Fraction]:
    """T_alpha*(mu) T_beta*(mu) in the Chebyshev basis.

    Applies (P1) coordinate-wise, merging when min(alpha_i, beta_i) = 0.
    """
    d = len(alpha)
    current: Dict[Tuple[int, ...], Fraction] = {(): Fraction(1)}
    for i in range(d):
        factors = cheb_prod_1d(alpha[i], beta[i])
        nxt: Dict[Tuple[int, ...], Fraction] = {}
        for prefix, c_prefix in current.items():
            for k_i, c_i in factors:
                new_prefix = prefix + (k_i,)
                nxt[new_prefix] = nxt.get(new_prefix, Fraction(0)) + c_prefix * c_i
        current = nxt
    return current


def cheb_mul_mu_i(expr: Dict[Tuple[int, ...], Fraction],
                  i: int) -> Dict[Tuple[int, ...], Fraction]:
    """Multiply a Chebyshev expansion by mu_i, expanding coordinate i via (P2)."""
    result: Dict[Tuple[int, ...], Fraction] = {}
    for gamma, c in expr.items():
        gi = gamma[i]
        for new_gi, c_i in cheb_mul_x_1d(gi):
            new_gamma = list(gamma)
            new_gamma[i] = new_gi
            t = tuple(new_gamma)
            result[t] = result.get(t, Fraction(0)) + c * c_i
    return result


def cheb_mu_prod(alpha: Sequence[int], beta: Sequence[int],
                 i: int) -> Dict[Tuple[int, ...], Fraction]:
    """mu_i * T_alpha* T_beta* expressed in the Chebyshev basis."""
    return cheb_mul_mu_i(cheb_prod(alpha, beta), i)


def cheb_mu_i_mu_j_prod(alpha: Sequence[int], beta: Sequence[int],
                        i: int, j: int) -> Dict[Tuple[int, ...], Fraction]:
    """mu_i * mu_j * T_alpha* T_beta* expressed in the Chebyshev basis.

    For i == j this computes mu_i^2 * T_alpha* T_beta*.
    Multiplication by mu commutes with itself so the order does not matter;
    we apply mu_j first, then mu_i, to make the direction explicit.
    """
    tmp = cheb_mul_mu_i(cheb_prod(alpha, beta), j)
    return cheb_mul_mu_i(tmp, i)


# ---------------------------------------------------------------------
# Pointwise evaluation (used by unit test A)
# ---------------------------------------------------------------------

def eval_t_star_1d(k: int, x: np.ndarray) -> np.ndarray:
    """Evaluate T_k*(x) on arbitrary arrays x in [0, 1] via the
    standard Chebyshev recurrence on 2x - 1."""
    y = 2.0 * x - 1.0
    if k == 0:
        return np.ones_like(y)
    if k == 1:
        return y
    T_prev = np.ones_like(y)
    T_curr = y
    for _ in range(2, k + 1):
        T_next = 2.0 * y * T_curr - T_prev
        T_prev = T_curr
        T_curr = T_next
    return T_curr


def eval_t_star_multi(gamma: Sequence[int], X: np.ndarray) -> np.ndarray:
    """Evaluate T_gamma*(mu) = prod_i T_{gamma_i}*(mu_i) at points X,
    where X has shape (N, d) and entries in [0, 1]."""
    out = np.ones(X.shape[0], dtype=np.float64)
    for i, g in enumerate(gamma):
        out *= eval_t_star_1d(int(g), X[:, i])
    return out


def eval_mono(alpha: Sequence[int], X: np.ndarray) -> np.ndarray:
    """Evaluate mu^alpha at points X of shape (N, d)."""
    out = np.ones(X.shape[0], dtype=np.float64)
    for i, a in enumerate(alpha):
        if a == 0:
            continue
        out *= X[:, i] ** int(a)
    return out


# ---------------------------------------------------------------------
# Sparse change-of-basis matrix B:  y_alpha = sum_gamma B[alpha, gamma] c_gamma
# ---------------------------------------------------------------------

def build_B_matrix(mono_list: Sequence[Tuple[int, ...]],
                   idx: Dict[Tuple[int, ...], int]) -> sp.csr_matrix:
    """Build the triangular change-of-basis matrix

        y_alpha = sum_gamma B[alpha, gamma] c_gamma,

    indexed over the full moment set ``mono_list`` of the precompute.
    Uses ``Fraction`` internally; final values are float64.

    Because gamma_i <= alpha_i and mono_list contains every multi-index
    of total degree <= 2k, every gamma encountered is in ``idx``.  We
    assert this to catch bugs.
    """
    n_y = len(mono_list)
    max_deg = max((sum(a) for a in mono_list), default=0)
    b_table = compute_b_table(max_deg)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for alpha in mono_list:
        a_idx = idx[alpha]
        for gamma, coef in mono_to_cheb(alpha, b_table).items():
            if coef == 0:
                continue
            if gamma not in idx:
                raise AssertionError(
                    f"Chebyshev coordinate {gamma} missing from mono_list "
                    f"(expansion of monomial {alpha}); change of basis "
                    f"cannot be represented exactly.")
            rows.append(a_idx)
            cols.append(idx[gamma])
            vals.append(float(coef))
    return sp.csr_matrix(
        (vals, (rows, cols)), shape=(n_y, n_y), dtype=np.float64)


# ---------------------------------------------------------------------
# Generic constructor for "picked" sparse maps in the Chebyshev basis
# ---------------------------------------------------------------------

def build_cheb_pick_matrix(entries: Iterable[Tuple[int,
                                                      Dict[Tuple[int, ...],
                                                           Fraction]]],
                            n_rows: int,
                            idx: Dict[Tuple[int, ...], int],
                            n_cols: int) -> sp.csr_matrix:
    """Convert a list of (row, {gamma: coef}) pairs into a sparse matrix
    of shape (n_rows, n_cols).

    Rows with no entries are kept as all-zero rows.
    """
    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    for row, exp in entries:
        for gamma, coef in exp.items():
            if coef == 0:
                continue
            if gamma not in idx:
                raise AssertionError(
                    f"Chebyshev coordinate {gamma} outside mono_list; "
                    f"precompute degree budget is too small.")
            rows.append(row)
            cols.append(idx[gamma])
            vals.append(float(coef))
    return sp.csr_matrix(
        (vals, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float64)


def build_moment_matrix_map(
        basis: Sequence[Tuple[int, ...]],
        idx: Dict[Tuple[int, ...], int],
        n_y: int) -> sp.csr_matrix:
    """Sparse map G of shape (n_basis^2, n_y) such that

        G @ c    (reshaped to n_basis x n_basis)  =  M_k^cheb(c).

    Entry (alpha, beta) of the Chebyshev moment matrix is
    ``E[T_alpha* T_beta*]`` = sum over cheb_prod(alpha, beta).
    """
    n = len(basis)
    entries: List[Tuple[int, Dict[Tuple[int, ...], Fraction]]] = []
    for i, alpha in enumerate(basis):
        row_i = i * n
        for j, beta in enumerate(basis):
            entries.append((row_i + j, cheb_prod(alpha, beta)))
    return build_cheb_pick_matrix(entries, n * n, idx, n_y)


def build_loc_matrix_map(
        loc_basis: Sequence[Tuple[int, ...]],
        idx: Dict[Tuple[int, ...], int],
        n_y: int,
        var_i: int) -> sp.csr_matrix:
    """Sparse map for the localizing matrix

        L_i = M_{k-1}^cheb(mu_i * c),  entry (a, b) = E[mu_i T_loc[a]* T_loc[b]*].
    """
    n = len(loc_basis)
    entries: List[Tuple[int, Dict[Tuple[int, ...], Fraction]]] = []
    for a, alpha in enumerate(loc_basis):
        row_a = a * n
        for b, beta in enumerate(loc_basis):
            entries.append((row_a + b, cheb_mu_prod(alpha, beta, var_i)))
    return build_cheb_pick_matrix(entries, n * n, idx, n_y)


def build_t_matrix_map(
        loc_basis: Sequence[Tuple[int, ...]],
        idx: Dict[Tuple[int, ...], int],
        n_y: int) -> sp.csr_matrix:
    """Sparse map for M_{k-1}^cheb(c) (the T-matrix used inside the
    window and upper-localizing cones)."""
    n = len(loc_basis)
    entries: List[Tuple[int, Dict[Tuple[int, ...], Fraction]]] = []
    for a, alpha in enumerate(loc_basis):
        row_a = a * n
        for b, beta in enumerate(loc_basis):
            entries.append((row_a + b, cheb_prod(alpha, beta)))
    return build_cheb_pick_matrix(entries, n * n, idx, n_y)


def build_window_Q_map(
        loc_basis: Sequence[Tuple[int, ...]],
        idx: Dict[Tuple[int, ...], int],
        n_y: int,
        Mw: np.ndarray) -> sp.csr_matrix:
    """Sparse map for Q_W^cheb(c), entry (a, b) =
    sum_{i, j} Mw[i, j] * E[mu_i mu_j T_loc[a]* T_loc[b]*].
    """
    n = len(loc_basis)
    nz_i, nz_j = np.nonzero(Mw)
    entries: List[Tuple[int, Dict[Tuple[int, ...], Fraction]]] = []
    for a, alpha in enumerate(loc_basis):
        row_a = a * n
        for b, beta in enumerate(loc_basis):
            acc: Dict[Tuple[int, ...], Fraction] = {}
            for ij_idx in range(len(nz_i)):
                i = int(nz_i[ij_idx])
                j = int(nz_j[ij_idx])
                # M_W[i, j] from build_window_matrices is 2*d/ell for a
                # small integer ell, so Fraction.from_float + limit_denominator
                # recovers the rational exactly.
                fw = Fraction(float(Mw[i, j])).limit_denominator(10**6)
                for gamma, coef in cheb_mu_i_mu_j_prod(alpha, beta, i, j).items():
                    acc[gamma] = acc.get(gamma, Fraction(0)) + fw * coef
            entries.append((row_a + b, acc))
    return build_cheb_pick_matrix(entries, n * n, idx, n_y)


__all__ = [
    'compute_b_table', 'cheb_prod_1d', 'cheb_mul_x_1d',
    'mono_to_cheb', 'cheb_prod', 'cheb_mul_mu_i',
    'cheb_mu_prod', 'cheb_mu_i_mu_j_prod',
    'eval_t_star_1d', 'eval_t_star_multi', 'eval_mono',
    'build_B_matrix', 'build_cheb_pick_matrix',
    'build_moment_matrix_map', 'build_loc_matrix_map',
    'build_t_matrix_map', 'build_window_Q_map',
]
