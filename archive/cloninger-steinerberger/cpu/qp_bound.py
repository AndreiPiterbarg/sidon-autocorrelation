"""Joint QP bound for coarse-grid box certification (the (D) proposal).

Computes the EXACT worst-case TV decrement over the cell

    Cell = { delta : |delta_i| <= h, sum delta_i = 0 },  h = 1/(2S)

for window W = (ell, s) at grid point mu* = c/S:

    B_QP(c, W) = max_{delta in Cell} [ -grad_W . delta - (2d/ell) delta^T A_W delta ]

where (A_W)_{i,j} = 1{ s <= i+j <= s+ell-2 }, grad_W = (4d/ell) A_W mu*.

This is an indefinite QP. The maximum of a quadratic over a polytope is attained
at a vertex. The vertices of Cell are obtained by:
  - choosing a "free" index f in 0..d-1
  - choosing signs +/-h for the other d-1 coordinates (2^(d-1) patterns)
  - the free coord is determined by sum=0; feasible iff |free| <= h

So total candidates: d * 2^(d-1). Practical for d <= 16.

The cascade certifies a composition c at window W when:

    TV_W(c) - c_target  >  B_QP(c, W).

This bound is sound (it's exact) and is provably tighter than the triangle
inequality `cell_var + quad_corr` (with strict tightening when the linear and
quadratic worst-case directions are anti-correlated).
"""
from __future__ import annotations

import numpy as np
import numba
from numba import njit


@njit(cache=True)
def build_window_matrix(d, ell, s):
    """A_{i,j} = 1{ s <= i+j <= s+ell-2 }."""
    A = np.zeros((d, d), dtype=np.float64)
    s_hi = s + ell - 2
    for i in range(d):
        for j in range(d):
            k = i + j
            if s <= k <= s_hi:
                A[i, j] = 1.0
    return A


@njit(cache=True)
def qp_bound_vertex(grad, A_W, scale, h, d):
    """Exact joint QP bound by vertex enumeration.

    Returns max over Cell of f(delta) := -grad . delta - scale * delta^T A_W delta.

    delta = 0 is always feasible and gives f(0) = 0, so the result is >= 0.

    Complexity: O(d * 2^(d-1) * d^2) per call. For d=8: ~65K ops. d=16: ~134M ops.
    """
    best = 0.0  # f(0) = 0
    delta = np.zeros(d, dtype=np.float64)
    tol = h * 1e-9

    for free_idx in range(d):
        n_pat = 1 << (d - 1)
        for mask in range(n_pat):
            # Build delta: free_idx coordinate is determined; others from mask.
            sum_others = 0.0
            bit_pos = 0
            for i in range(d):
                if i == free_idx:
                    continue
                bit = (mask >> bit_pos) & 1
                if bit == 0:
                    delta[i] = -h
                else:
                    delta[i] = h
                sum_others += delta[i]
                bit_pos += 1
            free_val = -sum_others
            if free_val < -h - tol or free_val > h + tol:
                continue
            # Clamp to box (handles rounding only)
            if free_val > h:
                free_val = h
            if free_val < -h:
                free_val = -h
            delta[free_idx] = free_val

            # Compute f(delta) = -grad . delta - scale * delta^T A delta
            lin = 0.0
            for i in range(d):
                lin += grad[i] * delta[i]
            quad = 0.0
            for i in range(d):
                Ai_dot_delta = 0.0
                for j in range(d):
                    Ai_dot_delta += A_W[i, j] * delta[j]
                quad += delta[i] * Ai_dot_delta
            val = -lin - scale * quad
            if val > best:
                best = val
    return best


@njit(cache=True)
def grad_for_window(c, A_W, S, d, ell):
    """grad_W = (4d/ell) A_W mu*, mu* = c/S."""
    grad = np.zeros(d, dtype=np.float64)
    factor = 4.0 * d / (ell * S)
    for i in range(d):
        s_acc = 0.0
        for j in range(d):
            s_acc += A_W[i, j] * c[j]
        grad[i] = factor * s_acc
    return grad


@njit(cache=True)
def qp_bound_for_composition(c_int, S, d, ell, s):
    """Convenience: compute B_QP for composition c_int, window (ell, s)."""
    A_W = build_window_matrix(d, ell, s)
    grad = grad_for_window(c_int, A_W, S, d, ell)
    h = 1.0 / (2.0 * S)
    scale = 2.0 * d / ell
    return qp_bound_vertex(grad, A_W, scale, h, d)


# --- Reference Python (no Numba) for cross-validation ---

def qp_bound_python(grad, A_W, scale, h, d):
    """Pure Python vertex enumeration, identical algorithm.
    Used only to validate the Numba kernel."""
    best = 0.0
    delta = np.zeros(d)
    tol = h * 1e-9
    for free_idx in range(d):
        for mask in range(1 << (d - 1)):
            sum_others = 0.0
            bit_pos = 0
            for i in range(d):
                if i == free_idx:
                    continue
                bit = (mask >> bit_pos) & 1
                delta[i] = h if bit else -h
                sum_others += delta[i]
                bit_pos += 1
            free_val = -sum_others
            if abs(free_val) > h + tol:
                continue
            free_val = max(-h, min(h, free_val))
            delta[free_idx] = free_val
            v = -float(grad @ delta) - scale * float(delta @ A_W @ delta)
            if v > best:
                best = v
    return best
