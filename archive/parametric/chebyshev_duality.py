"""Algebraic backbone for the parametric dual bound.

Contents
--------
chebyshev_monomial_coefs(L)    -> Fraction matrix c[l][k] = coef of t^k in T_l(t).
integrate_Tl_2t(L)             -> Fraction vector int_{-1/2}^{1/2} T_l(2t) dt.
build_Kl_table(L)              -> K^(l)[i][j] = coef of x^i y^j in T_l(2(x+y)).
bivariate_basis(D)             -> list of (a, b) with a+b <= D.
bivariate_pair_map(basis)      -> dict {(A, B): list of (i, j)} summing to (A, B).
build_lukacs_psd_block(L)      -> data for Lukacs representation p = q1^2 + (1-4t^2) q2^2.
build_hausdorff_moment_psd(N)  -> data for univariate Hausdorff moment problem on [-1/4, 1/4].

All coefficient tables use Fraction for exactness; downstream code converts to float.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from math import comb
from typing import Dict, List, Tuple


def chebyshev_monomial_coefs(L: int) -> List[List[Fraction]]:
    """c[l][k] = coefficient of t^k in T_l(t), for l = 0..L-1 and k = 0..L-1.

    Integer recurrence: T_0 = 1, T_1 = t, T_{l+1} = 2 t T_l - T_{l-1}.
    """
    if L <= 0:
        return []
    c = [[Fraction(0)] * L for _ in range(L)]
    c[0][0] = Fraction(1)
    if L > 1:
        c[1][1] = Fraction(1)
    for l in range(1, L - 1):
        for k in range(L - 1):
            c[l + 1][k + 1] += 2 * c[l][k]
        for k in range(L):
            c[l + 1][k] -= c[l - 1][k]
    return c


def integrate_Tl_2t(L: int) -> List[Fraction]:
    """int_{-1/2}^{1/2} T_l(2t) dt, for l = 0..L-1.

    Computed from int_{-1}^{1} T_l(s) ds:
        l = 0:         2
        l odd:         0
        l even, l>=2: -2 / (l^2 - 1)
    then multiplied by 1/2 (substitution s = 2t).
    """
    out: List[Fraction] = [Fraction(0)] * L
    for l in range(L):
        if l == 0:
            out[l] = Fraction(1)
        elif l % 2 == 1:
            out[l] = Fraction(0)
        else:
            out[l] = Fraction(-1, l * l - 1)
    return out


def build_Kl_table(L: int) -> List[List[List[Fraction]]]:
    """K^(l)[i][j] = coef of x^i y^j in T_l(2(x+y)).

    Derivation:
        T_l(2u) = sum_k c[l][k] 2^k u^k,  u = x + y,
        u^k = sum_{i+j=k} C(k, i) x^i y^j,
        hence K^(l)[i][j] = c[l][i+j] * 2^(i+j) * C(i+j, i) if i+j < L else 0.
    """
    c = chebyshev_monomial_coefs(L)
    pow2 = [Fraction(1)]
    for _ in range(L):
        pow2.append(pow2[-1] * 2)
    K: List[List[List[Fraction]]] = [
        [[Fraction(0)] * L for _ in range(L)] for _ in range(L)
    ]
    for l in range(L):
        for i in range(L):
            for j in range(L - i):
                k = i + j
                if k >= L:
                    continue
                K[l][i][j] = c[l][k] * pow2[k] * Fraction(comb(k, i))
    return K


def bivariate_basis(D: int) -> List[Tuple[int, int]]:
    """Monomial basis of bivariate polynomials of total deg <= D, ordered (a, b)."""
    if D < 0:
        return []
    return [(a, b) for a in range(D + 1) for b in range(D + 1 - a)]


def bivariate_pair_map(basis: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
    """(A, B) -> [(i, j)] such that basis[i] + basis[j] == (A, B).

    Used to match monomial coefficients on LHS vs quadratic-form RHS.
    """
    out: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    for i, (a1, b1) in enumerate(basis):
        for j, (a2, b2) in enumerate(basis):
            out[(a1 + a2, b1 + b2)].append((i, j))
    return out


def build_lukacs_psd_block(L: int) -> dict:
    """Sizes and index helpers for univariate Lukacs p(t) >= 0 on [-1/2, 1/2].

    Representation (L odd, deg p = L-1 even):
        p(t) = sigma_0(t) + (1 - 4 t^2) sigma_1(t)
        sigma_0(t) = v_{k0}(t)^T Q0 v_{k0}(t), Q0 in S_{k0+1}_+, k0 = (L-1)/2
        sigma_1(t) = v_{k1}(t)^T Q1 v_{k1}(t), Q1 in S_{k1+1}_+, k1 = k0 - 1

    For L = 1, Q1 is omitted (degenerate).

    Coefficient of t^r in p:
        p_r = sum_{i+j=r, 0<=i,j<=k0} Q0[i, j]
            + sum_{i+j=r, 0<=i,j<=k1} Q1[i, j]
            - 4 * sum_{i+j=r-2, 0<=i,j<=k1} Q1[i, j]
    """
    if L % 2 == 0:
        raise ValueError("build_lukacs_psd_block requires L odd (deg p = L-1 even).")
    k0 = (L - 1) // 2
    k1 = k0 - 1  # may be -1 for L == 1
    return {
        "k0": k0,
        "k1": k1,
        "Q0_size": k0 + 1,
        "Q1_size": k1 + 1 if k1 >= 0 else 0,
        "has_Q1": k1 >= 0,
    }


def build_hausdorff_moment_psd(N: int) -> dict:
    """Sizes for univariate Hausdorff moment problem on [-1/4, 1/4].

    For y = (y_0, ..., y_{2N}) to be a moment sequence of a nonneg measure on
    [-1/4, 1/4]:
        M_N(y)      = [y_{i+j}]_{0<=i,j<=N}                     PSD,
        L_{N-1}(y)  = [(1/16) y_{i+j} - y_{i+j+2}]_{0<=i,j<=N-1} PSD.

    Returns dict with index metadata (the actual LMI construction is done in the
    solver module, since it depends on CVXPY vs pure-numpy representation).
    """
    return {
        "N": N,
        "moment_size": N + 1,
        "localize_size": N if N >= 1 else 0,
        "max_moment": 2 * N,
    }


def cheb_expand_to_monomial_vector(m: List[Fraction]) -> List[Fraction]:
    """Given Chebyshev coefs m[l] of p(t) = sum_l m_l T_l(2t), return monomial
    coefs [p_0, p_1, ..., p_{L-1}] in the t basis.

        p_r = sum_l m_l * c[l][r] * 2^r.
    """
    L = len(m)
    c = chebyshev_monomial_coefs(L)
    pow2 = [Fraction(1)]
    for _ in range(L):
        pow2.append(pow2[-1] * 2)
    return [sum(m[l] * c[l][r] * pow2[r] for l in range(L)) for r in range(L)]
