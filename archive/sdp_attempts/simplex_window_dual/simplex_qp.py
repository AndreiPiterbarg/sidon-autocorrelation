"""Exact support-enumeration solver for quadratic minimization on the simplex."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SimplexQuadraticResult:
    value: float
    minimizer: np.ndarray
    support_mask: int


def _candidate_from_support(
    q: np.ndarray,
    support: np.ndarray,
    tol: float,
) -> Optional[np.ndarray]:
    """Solve the KKT system for a fixed support face."""
    s = len(support)
    kkt = np.zeros((s + 1, s + 1), dtype=np.float64)
    q_ss = q[np.ix_(support, support)]
    kkt[:s, :s] = 2.0 * q_ss
    kkt[:s, s] = -1.0
    kkt[s, :s] = 1.0
    rhs = np.zeros(s + 1, dtype=np.float64)
    rhs[s] = 1.0

    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError:
        sol, residuals, rank, _ = np.linalg.lstsq(kkt, rhs, rcond=None)
        if rank < s or (residuals.size and residuals[0] > tol**2):
            return None

    x_s = sol[:s]
    if np.any(x_s < -tol):
        return None

    x = np.zeros(q.shape[0], dtype=np.float64)
    x[support] = np.maximum(x_s, 0.0)
    total = x.sum()
    if total <= tol:
        return None
    x /= total

    grad = 2.0 * q @ x
    lagrange = float(2.0 * x @ q @ x)
    inactive = np.ones(q.shape[0], dtype=bool)
    inactive[support] = False
    if np.any(grad[inactive] < lagrange - tol):
        return None
    return x


def solve_simplex_quadratic(
    q: np.ndarray,
    tol: float = 1e-10,
) -> SimplexQuadraticResult:
    """Globally minimize x^T Q x over the probability simplex.

    The feasible set is the standard simplex {x >= 0, sum x = 1}. The solver
    enumerates all support faces, solves the KKT system on each face, and keeps
    the best feasible stationary point. This is exact for the simplex because
    every global minimizer lies either at a boundary face or at an interior KKT
    point of some face.
    """
    d = int(q.shape[0])
    best_value = float("inf")
    best_x = np.zeros(d, dtype=np.float64)
    best_mask = 0

    for i in range(d):
        x = np.zeros(d, dtype=np.float64)
        x[i] = 1.0
        value = float(q[i, i])
        if value < best_value:
            best_value = value
            best_x = x
            best_mask = 1 << i

    for mask in range(1, 1 << d):
        if mask & (mask - 1) == 0:
            continue
        support = np.array([idx for idx in range(d) if mask & (1 << idx)], dtype=np.int64)
        candidate = _candidate_from_support(q, support, tol)
        if candidate is None:
            continue
        value = float(candidate @ q @ candidate)
        if value + tol < best_value:
            best_value = value
            best_x = candidate
            best_mask = mask

    return SimplexQuadraticResult(best_value, best_x, best_mask)
