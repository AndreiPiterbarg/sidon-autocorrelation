"""Candidate dual functions g(t) for screening M(g) values.

A *useful* g must be strictly positive on [-1/2, 1/2] (or at least have
strict positivity on the support of f*f for any feasible f).  In
particular, g vanishing at the boundary t = +/- 1/2 gives M(g) = 0
because f can be chosen with mass concentrated near +/- 1/4 so that f*f
is concentrated near +/- 1/2, escaping g's support.

Function classes supported here:
  * `g_constant(c)`         : g(t) = c on [-1/2, 1/2].  M = c.
  * `g_pwc(values, breaks)` : piecewise-constant on user-given breakpoints.
  * `g_pwl(heights, breaks)`: piecewise-linear interpolant through (breaks, heights).
  * `g_cosine(coeffs)`      : g(t) = a_0 + sum_k a_k cos(2 pi k t)
                              (trigonometric polynomial).
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import numpy as np


def g_constant(c: float = 1.0) -> Callable:
    """Constant g = c on [-1/2, 1/2].  M(g) = c."""
    def g(t):
        if hasattr(t, 'shape'):
            return np.full_like(t, float(c), dtype=np.float64)
        return float(c)
    return g


def g_pwc(values: Sequence[float],
          breaks: Sequence[float] = None) -> Callable:
    """Piecewise-constant g on [-1/2, 1/2].

    `values` has length N; if `breaks` is None, use uniform partition into
    N intervals on [-1/2, 1/2].  Else `breaks` must have length N+1 and
    values[k] is the constant on [breaks[k], breaks[k+1]].
    """
    values = np.asarray(values, dtype=np.float64)
    N = len(values)
    if breaks is None:
        breaks = np.linspace(-0.5, 0.5, N + 1)
    else:
        breaks = np.asarray(breaks, dtype=np.float64)
        assert len(breaks) == N + 1

    def g(t):
        ta = np.asarray(t, dtype=np.float64) if hasattr(t, 'shape') \
            else np.array([t], dtype=np.float64)
        out = np.zeros_like(ta)
        for k in range(N):
            mask = (ta >= breaks[k]) & (ta <= breaks[k + 1])
            out[mask] = values[k]
        if hasattr(t, 'shape'):
            return out
        return float(out[0])
    return g


def g_pwl(heights: Sequence[float],
          breaks: Sequence[float] = None) -> Callable:
    """Piecewise-linear g on [-1/2, 1/2].

    `heights` has length N+1, values at break points.
    `breaks` is the list of N+1 break points; defaults to uniform.
    """
    heights = np.asarray(heights, dtype=np.float64)
    N1 = len(heights)
    if breaks is None:
        breaks = np.linspace(-0.5, 0.5, N1)
    else:
        breaks = np.asarray(breaks, dtype=np.float64)
        assert len(breaks) == N1
    breaks_arr = breaks
    heights_arr = heights

    def g(t):
        ta = np.asarray(t, dtype=np.float64) if hasattr(t, 'shape') \
            else np.array([t], dtype=np.float64)
        out = np.interp(ta, breaks_arr, heights_arr, left=0.0, right=0.0)
        if hasattr(t, 'shape'):
            return out
        return float(out[0])
    return g


def g_cosine(coeffs: Sequence[float]) -> Callable:
    """g(t) = c_0 + sum_{k=1..K} c_k cos(2 pi k t).  Periodic, supported
    on R but we only use it on [-1/2, 1/2].
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    K = len(coeffs) - 1

    def g(t):
        ta = np.asarray(t, dtype=np.float64) if hasattr(t, 'shape') \
            else np.array([t], dtype=np.float64)
        out = np.full_like(ta, coeffs[0])
        for k in range(1, K + 1):
            out = out + coeffs[k] * np.cos(2 * np.pi * k * ta)
        if hasattr(t, 'shape'):
            return out
        return float(out[0])
    return g


def g_central_bump(c0: float = 1.0, c1: float = 0.5) -> Callable:
    """Smooth nonneg g = c0 + c1 * (1 - 2|t|) on [-1/2, 1/2].

    Strictly positive on closed [-1/2, 1/2] iff c0 > 0.
    M(g) for uniform f computable in closed form.
    """
    def g(t):
        ta = np.asarray(t, dtype=np.float64) if hasattr(t, 'shape') \
            else np.array([t], dtype=np.float64)
        out = c0 + c1 * (1.0 - 2.0 * np.abs(ta))
        out = np.maximum(out, 0.0)   # guarantee >= 0
        if hasattr(t, 'shape'):
            return out
        return float(out[0])
    return g


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bochner_sos.m_g_eval import M_g

    print("=" * 70)
    print("g_candidates self-test")
    print("=" * 70)

    print("\n[1] g = 1 (constant)")
    res = M_g(g_constant(1.0), d=80, force_method='convex')
    print(f"    M = {res['M']:.6f}  (expected 1.0)")

    print("\n[2] g = central bump c0=1, c1=0.5")
    res = M_g(g_central_bump(1.0, 0.5), d=80, force_method='convex')
    print(f"    M = {res['M']:.6f}, int_g = {res['int_g']:.4f}, "
          f"is_psd = {res['is_psd']}")

    print("\n[3] g = piecewise-linear hat: heights [1, 2, 1] on uniform")
    res = M_g(g_pwl([1.0, 2.0, 1.0]), d=80, force_method='convex')
    print(f"    M = {res['M']:.6f}, int_g = {res['int_g']:.4f}")

    print("\n[4] g = cosine series c0=1, c1=-0.5 (g(0)=0.5, g(1/2)=1.5)")
    res = M_g(g_cosine([1.0, -0.5]), d=80, force_method='convex')
    print(f"    M = {res['M']:.6f}, int_g = {res['int_g']:.4f}")

    print("\n[5] g = cosine c0=1, c1=0.5 (g(0)=1.5, g(1/2)=0.5)")
    res = M_g(g_cosine([1.0, 0.5]), d=80, force_method='convex')
    print(f"    M = {res['M']:.6f}, int_g = {res['int_g']:.4f}")

    print("\n[6] g = pwc with values [1,1,1] (= constant 1)")
    res = M_g(g_pwc([1.0, 1.0, 1.0]), d=80, force_method='convex')
    print(f"    M = {res['M']:.6f}  (expected 1.0)")

    print("\nself-test OK")
