"""Core helpers for simplex-window dual certificates."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np

from lasserre.core import build_window_matrices, enum_monomials

Monomial = Tuple[int, ...]


def _sort_monomials(monomials: Sequence[Monomial]) -> List[Monomial]:
    return sorted(monomials, key=lambda mono: (sum(mono), mono))


@lru_cache(maxsize=None)
def up_to_degree_monomials(d: int, degree: int) -> Tuple[Monomial, ...]:
    """Return all monomials of degree <= degree, sorted deterministically."""
    return tuple(_sort_monomials(tuple(enum_monomials(d, degree))))


@lru_cache(maxsize=None)
def exact_degree_monomials(d: int, degree: int) -> Tuple[Monomial, ...]:
    """Return all monomials of exact total degree, sorted deterministically."""
    return tuple(
        mono for mono in up_to_degree_monomials(d, degree) if sum(mono) == degree
    )


@lru_cache(maxsize=None)
def window_quadratic_coefficients(
    d: int,
) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[Dict[Monomial, float], ...]]:
    """Return quadratic polynomial coefficients for every window at dimension d.

    Each window polynomial is represented in the monomial basis:

        mu^T M_W mu = sum_{beta, |beta|=2} coeff_W[beta] * x^beta

    where off-diagonal entries of M_W are folded into the coefficient of
    x_i x_j via the symmetric factor 2*M_W[i,j].
    """
    windows, matrices = build_window_matrices(d)
    coeffs: List[Dict[Monomial, float]] = []

    for matrix in matrices:
        coeff: Dict[Monomial, float] = {}
        for i in range(d):
            diag = float(matrix[i, i])
            if diag:
                mono = [0] * d
                mono[i] = 2
                coeff[tuple(mono)] = coeff.get(tuple(mono), 0.0) + diag
            for j in range(i + 1, d):
                off = float(2.0 * matrix[i, j])
                if off:
                    mono = [0] * d
                    mono[i] += 1
                    mono[j] += 1
                    coeff[tuple(mono)] = coeff.get(tuple(mono), 0.0) + off
        coeffs.append(coeff)

    return tuple(windows), tuple(coeffs)


def evaluate_quadratic_polynomial(coeffs: Dict[Monomial, float], x: np.ndarray) -> float:
    """Evaluate a sparse quadratic polynomial at x."""
    return evaluate_polynomial(coeffs, x)


def evaluate_polynomial(coeffs: Dict[Monomial, float], x: np.ndarray) -> float:
    """Evaluate a sparse polynomial at x."""
    total = 0.0
    for mono, coeff in coeffs.items():
        term = coeff
        for idx, power in enumerate(mono):
            if power:
                term *= x[idx] ** power
        total += term
    return total
