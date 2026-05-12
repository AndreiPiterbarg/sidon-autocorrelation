"""LP reformulation of F1 for fixed (T, omega).

For fixed (T, omega) the dual bound
    L(a) = [int ghat(xi; a) w(xi) dxi] / [int_{-1/2}^{1/2} g(t; a) dt]
has both numerator and denominator LINEAR in (1, a_1, ..., a_K). Hence for
fixed denominator, maximizing numerator over the cone
    C = { a : ghat(xi; a) >= 0 at breakpoints }  (a polyhedron)
  intersected with
    { a : g(t; a) >= 0 on [-1/2, 1/2] }  (discretized on a fine t-grid)
  and normalized to
    { a : int g = 1 }  (a hyperplane)
is a plain LP.

Algorithm:
  1. Compute coefficients: for each k in 0..K,
       num_k  = int [triangle contribution from a_k] * w(xi) dxi
       den_k  = int_{-1/2}^{1/2} sinc^2(pi T t) * cos(2 pi k omega t) dt
       (with den_0 the pure-sinc integral).
  2. Constraints:
       - ghat at each breakpoint xi_j >= 0:  A_ghat @ (1, a) >= 0
       - g at each t_i >= 0:  A_g @ (1, a) >= 0
       - Normalization: den_0 + sum_k a_k den_k = 1
       - Optional: |a_k| <= 1 (box).
  3. Maximize num_0 + sum_k a_k num_k  subject to the above.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

from . import family_f1_selberg as f1


def _sinc_sq(x):
    """sinc^2(x) = (sin(x)/x)^2 with sinc(0)=1."""
    return np.where(np.abs(x) < 1e-14, 1.0, (np.sin(x) / np.where(np.abs(x) < 1e-14, 1.0, x)) ** 2)


def _w_f64(xi):
    """Sharp signed lower bound L^sharp(xi) on A^2 - B^2 for general f:

        L^sharp(xi) = cos(pi |xi|)  for |xi| <= 1
        L^sharp(xi) = -1            for |xi| > 1

    See family_f1_selberg.weight_iv for derivation. Can be negative.
    """
    xi_arr = np.asarray(xi, dtype=float)
    a = np.abs(xi_arr)
    return np.where(a > 1.0, -1.0, np.cos(math.pi * a))


def _triangle(xi, T, shift=0.0):
    v = T - np.abs(xi - shift)
    return np.maximum(v, 0.0)


def f1_lp_for_fixed_Tomega(T: float, omega: float, K: int, *,
                           n_xi: int = 4001, n_t: int = 2001, a_box: float = 1.5):
    """Solve the LP in (a_1, ..., a_K) for fixed (T, omega).

    Returns (best_score, a_opt, diagnostics) where best_score is the float64
    value of the ratio.
    """
    # ----- numerator coefficients -----
    xi_max = T + K * omega
    xs = np.linspace(-xi_max, xi_max, n_xi)
    dx = xs[1] - xs[0]
    w = _w_f64(xs)
    # Only |xi| <= 1 has w > 0.
    # num_k = int [contribution of a_k to ghat] * w dxi
    # ghat = Delta_T + (1/2) sum a_k (Delta_T(-k*om) + Delta_T(+k*om))
    num0 = float(np.sum(_triangle(xs, T) * w) * dx)
    num_k = np.zeros(K)
    for k in range(1, K + 1):
        contrib = 0.5 * (_triangle(xs, T, -k * omega) + _triangle(xs, T, +k * omega))
        num_k[k - 1] = float(np.sum(contrib * w) * dx)

    # ----- denominator coefficients (int g on [-1/2, 1/2]) -----
    ts = np.linspace(-0.5, 0.5, n_t)
    dt = ts[1] - ts[0]
    sinc_sq_T = _sinc_sq(math.pi * T * ts) * T ** 2 if False else (T ** 2) * _sinc_sq(math.pi * T * ts)
    den0 = float(np.sum(sinc_sq_T) * dt)
    den_k = np.zeros(K)
    for k in range(1, K + 1):
        den_k[k - 1] = float(np.sum(sinc_sq_T * np.cos(2 * math.pi * k * omega * ts)) * dt)

    # ----- PD constraints (breakpoints) -----
    # ghat(xi) = tri(xi) + (1/2) sum a_k (tri(xi - k*om) + tri(xi + k*om)) >= 0
    bkpts_set = {0.0, -T, T}
    for k in range(1, K + 1):
        for s in (-k * omega, +k * omega):
            bkpts_set.update([s, s - T, s + T])
    bkpts = np.array(sorted(bkpts_set))
    # Build A_pd: rows for each breakpoint; entries are (tri@bkpt, 0.5*(tri(-k*om) + tri(+k*om)))
    n_pd = len(bkpts)
    A_pd = np.zeros((n_pd, K))
    b_pd_const = np.empty(n_pd)
    for j, xi in enumerate(bkpts):
        b_pd_const[j] = _triangle(np.array([xi]), T)[0]
        for k in range(1, K + 1):
            A_pd[j, k - 1] = 0.5 * (_triangle(np.array([xi]), T, -k * omega)[0] +
                                     _triangle(np.array([xi]), T, +k * omega)[0])

    # ----- g >= 0 constraints on t-grid -----
    # g(t) = sinc_sq(pi T t) * T^2 * (1 + sum a_k cos(2 pi k om t))
    # Factor sinc_sq(pi T t) * T^2 > 0 on [-1/2,1/2] (except at isolated zeros),
    # so we need 1 + sum a_k cos(2 pi k om t) >= 0 wherever sinc^2 > 0.
    cos_basis = np.stack([np.cos(2 * math.pi * k * omega * ts) for k in range(1, K + 1)], axis=0) if K > 0 else np.zeros((0, n_t))
    # constraint: 1 + a @ cos_basis(t_i) >= 0 for each i
    # equivalently: -cos_basis(t_i) @ a <= 1
    A_geq0 = -cos_basis.T  # shape (n_t, K)
    b_geq0 = np.ones(n_t)

    # ----- LP -----
    # Minimize -(num0 + a @ num_k)  subject to:
    #   A_pd @ a >= -b_pd_const              ==>   -A_pd @ a <= b_pd_const
    #   1 + cos_basis(t).T @ a >= 0          ==>    A_geq0 @ a <= b_geq0
    #   den0 + a @ den_k = 1                 ==>    equality
    #   -a_box <= a_k <= a_box
    if K == 0:
        # Trivially plain Fejer.
        score = num0 / den0 if den0 > 0 else -np.inf
        return score, np.zeros(0), {"num": num0, "den": den0, "T": T, "omega": omega}

    c = -num_k        # minimize -c^T a + const(num0)
    A_ub = np.vstack([-A_pd, A_geq0])
    b_ub = np.concatenate([b_pd_const, b_geq0])
    A_eq = den_k[None, :]
    b_eq = np.array([1.0 - den0])
    bounds = [(-a_box, a_box)] * K

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
    except Exception as e:
        return -np.inf, None, {"error": str(e)}

    if not res.success:
        return -np.inf, None, {"error": res.message, "status": res.status}

    a_opt = res.x
    num_opt = num0 + float(a_opt @ num_k)
    den_opt = den0 + float(a_opt @ den_k)   # = 1 by constraint
    score = num_opt / den_opt if den_opt > 1e-12 else -np.inf
    return score, a_opt, {
        "num": num_opt, "den": den_opt, "T": T, "omega": omega,
        "a": a_opt.tolist(),
    }


def f1_grid_search_lp(K: int, *,
                      T_grid=None, omega_grid=None,
                      n_xi: int = 4001, n_t: int = 2001) -> dict:
    """Grid-search over (T, omega); solve LP at each grid point."""
    if T_grid is None:
        T_grid = np.linspace(0.4, 2.0, 17)
    if omega_grid is None:
        omega_grid = np.linspace(0.3, 3.0, 28)

    best = {"score": -np.inf}
    for T in T_grid:
        for omega in omega_grid:
            s, a_opt, info = f1_lp_for_fixed_Tomega(
                float(T), float(omega), K=K, n_xi=n_xi, n_t=n_t
            )
            if s > best["score"]:
                best = {"score": s, "T": float(T), "omega": float(omega),
                        "a": a_opt, "info": info}
    if best["score"] == -np.inf:
        return best
    p = f1.F1Params(T=best["T"], omega=best["omega"], a=[float(v) for v in best["a"]])
    best["params"] = p
    return best


# -----------------------------------------------------------------------------
# Safety backoff: ensure g >= 0 by scaling coefficients a -> s * a.
# (Moved here from pod_rigorous_verify.py for reuse across the pipeline.)
# -----------------------------------------------------------------------------

def _check_g_nonneg_f64(p: f1.F1Params, n_t: int = 200001) -> Tuple[float, float]:
    """Return (min_g, int_g) sampled on a fine t-grid."""
    ts = np.linspace(-0.5, 0.5, n_t)
    T, om = p.T, p.omega
    sinc_sq = np.where(
        np.abs(ts) < 1e-14,
        T * T,
        (np.sin(np.pi * T * ts) / (np.pi * ts + 1e-300)) ** 2,
    )
    mod = np.ones_like(ts)
    for k, ak in enumerate(p.a, 1):
        mod = mod + ak * np.cos(2 * np.pi * k * om * ts)
    g = sinc_sq * mod
    return float(g.min()), float(g.sum() * (ts[1] - ts[0]))


def find_safe_scale(p: f1.F1Params, target_min_g: float = 1e-6,
                    n_iters: int = 40) -> float:
    """Binary-search the largest scale s in [0, 1] such that the scaled
    F1Params (a_k -> s * a_k) has f64 min g >= target_min_g on [-1/2, 1/2].

    Returns 1.0 if the raw params already satisfy the target.
    """
    a_raw = list(p.a)
    if not a_raw:
        return 1.0
    m_raw, _ = _check_g_nonneg_f64(p)
    if m_raw >= target_min_g:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(n_iters):
        mid = 0.5 * (lo + hi)
        p_mid = f1.F1Params(T=p.T, omega=p.omega, a=[mid * v for v in a_raw])
        m, _ = _check_g_nonneg_f64(p_mid)
        if m >= target_min_g:
            lo = mid
        else:
            hi = mid
    return lo


# -----------------------------------------------------------------------------
# Exact-rational conversion of LP output (for reproducibility).
# -----------------------------------------------------------------------------

def rationalize_params(p: f1.F1Params, denom_cap: int = 10 ** 12) -> f1.F1Params:
    """Convert each float field of an F1Params to Fraction.limit_denominator
    and back to float, producing a reproducible exact-rational canonical form.

    The returned F1Params has fields that are floats whose EXACT decimal
    expansions equal p/q for some p,q with q <= denom_cap. This removes
    platform-dependent float noise from LP output.
    """
    def rat_float(x: float) -> float:
        return float(Fraction(float(x)).limit_denominator(denom_cap))

    return f1.F1Params(
        T=rat_float(p.T),
        omega=rat_float(p.omega),
        a=[rat_float(ak) for ak in p.a],
    )


def verify_rational_lp_solution(p: f1.F1Params, *,
                                denom_cap: int = 10 ** 12,
                                rel_tol: float = 1e-10,
                                n_subdiv: int = 16384,
                                precision_bits: int = 200,
                                denom_kind: str = "int_g"):
    """Rationalize the LP output, then run verify_f1 on it.

    Returns the VerifiedBound from verify.py.
    """
    # Import locally to avoid a module-level circular import.
    from . import verify as _ver
    p_rat = rationalize_params(p, denom_cap=denom_cap)
    return _ver.verify_f1(
        p_rat,
        rel_tol=rel_tol,
        n_subdiv=n_subdiv,
        precision_bits=precision_bits,
        denom_kind=denom_kind,
    )
