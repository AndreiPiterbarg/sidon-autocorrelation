"""Parameter optimisation for the F1 and F2 admissible families.

Step 1 (this file, float64 / scipy): find good starting points.
Step 2 (verify.py, mpmath): certify a rigorous ball for the bound.

The objective to MAXIMISE is
    L(params) = numerator(params) / max_{t in [-1/2,1/2]} g(t; params)

where
    numerator(params) = int_{-xi_max}^{xi_max} ghat(xi; params) * w(xi) dxi
    w(xi) = max(0, 1 - pi|xi|/2)^2     (from theory.md Section 2)

subject to positive-definiteness of ghat.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
from scipy.optimize import minimize

from . import family_f1_selberg as f1
from . import family_f2_gauss_poly as f2


# -----------------------------------------------------------------------------
# Float64 scoring (fast, for search).
# -----------------------------------------------------------------------------

def _weight_f64(xi):
    """Sharp signed lower bound L^sharp(xi) on A^2 - B^2 for general f
    supported on [-1/4, 1/4], int f = 1:

        L^sharp(xi) = cos(pi |xi|)  for |xi| <= 1
        L^sharp(xi) = -1            for |xi| > 1

    See family_f1_selberg.weight_iv for derivation. The bound can be
    negative (e.g. for |xi| in (1/2, 1] or |xi| > 1) -- this is correct.
    """
    a = abs(xi)
    if a > 1.0:
        return -1.0
    return math.cos(math.pi * a)


def _f1_score_f64(params: f1.F1Params, n_grid: int = 4001) -> Tuple[float, dict]:
    """Float64 approximate L(params) for F1."""
    # Numerator.
    xi_max = f1.ghat_support(params)
    xs = np.linspace(-xi_max, xi_max, n_grid)
    dx = xs[1] - xs[0]
    num = 0.0
    T = params.T
    omega = params.omega

    def tri(x):
        v = T - abs(x)
        return v if v > 0 else 0.0

    for xi in xs:
        val = tri(xi)
        for k, ak in enumerate(params.a, start=1):
            val += ak / 2 * (tri(xi - k * omega) + tri(xi + k * omega))
        num += val * _weight_f64(xi) * dx

    # Denominator: int g over [-1/2, 1/2], with a g >= 0 sanity check.
    ts = np.linspace(-0.5, 0.5, 20001)
    dt = ts[1] - ts[0]
    pi = math.pi
    sinc_factor = np.where(
        np.abs(ts) < 1e-15,
        T ** 2,
        (np.sin(pi * T * ts) / (pi * ts + 1e-300)) ** 2,
    )
    mod = np.ones_like(ts)
    for k, ak in enumerate(params.a, start=1):
        mod += ak * np.cos(2 * pi * k * omega * ts)
    vals = sinc_factor * mod
    if vals.min() < -1e-8:
        return -1e9, {"error": "g negative on [-1/2, 1/2]"}
    int_g = float(vals.sum() * dt)
    if int_g <= 0:
        return -1e9, {"num": num, "int_g": int_g}
    return num / int_g, {"num": num, "int_g": int_g, "M": float(vals.max())}


def _f1_pd_penalty(params: f1.F1Params) -> float:
    """Penalty for positive-definiteness violation: -min(0, ghat(xi)) at
    breakpoints."""
    from mpmath import mpf
    pts = f1.breakpoints(params)
    penalty = 0.0
    for xi in pts:
        v = float(f1.ghat_mp(xi, params))
        if v < 0:
            penalty += -v
    return penalty


def optimise_f1(K: int = 1, n_restarts: int = 20, verbose: bool = False):
    """Search F1 parameters (T, omega, a_1, ..., a_K)."""
    best = {"score": -np.inf, "params": None, "info": None}

    rng = np.random.default_rng(42)
    for restart in range(n_restarts):
        if restart == 0:
            # Canonical start: plain Fejer (no modulation).
            x0 = np.concatenate([[1.0, 1.5], np.zeros(K)])
        else:
            T0 = float(rng.uniform(0.3, 2.0))
            omega0 = float(rng.uniform(max(0.1, T0 * 0.5), T0 * 2))
            a0 = rng.uniform(-0.4, 0.4, size=K)
            x0 = np.concatenate([[T0, omega0], a0])

        def neg_obj(x):
            T, omega, *a = x
            if T <= 0 or omega <= 0:
                return 1e9
            p = f1.F1Params(T=float(T), omega=float(omega), a=[float(v) for v in a])
            s, _ = _f1_score_f64(p, n_grid=1001)
            pen = _f1_pd_penalty(p)
            return -s + 100.0 * pen

        try:
            res = minimize(neg_obj, x0, method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-8, "fatol": 1e-10})
        except Exception:
            continue
        x = res.x
        T, omega, *a = x
        if T <= 0 or omega <= 0:
            continue
        p = f1.F1Params(T=float(T), omega=float(omega), a=[float(v) for v in a])
        pen = _f1_pd_penalty(p)
        if pen > 1e-8:
            continue
        s, info = _f1_score_f64(p, n_grid=4001)
        if verbose:
            print(f"[F1 restart {restart}] score={s:.6f} T={T:.4f} omega={omega:.4f} a={a} pen={pen:.1e}")
        if s > best["score"]:
            best = {"score": s, "params": p, "info": info}
    return best


# -----------------------------------------------------------------------------
# F2 scoring.
# -----------------------------------------------------------------------------

def _f2_score_f64(params: f2.F2Params, n_grid: int = 4001) -> Tuple[float, dict]:
    # Numerator: int ghat(xi) w(xi) dxi, sampled.
    from mpmath import mp, mpf
    xi_max = f2.ghat_support(params)
    xs = np.linspace(-xi_max, xi_max, n_grid)
    dx = xs[1] - xs[0]
    alpha = params.alpha
    # Compute Q coefs once (float).
    try:
        q = [float(qi) for qi in f2.Q_coefs(params)]
    except Exception:
        return -1e9, {"error": "Q_coefs failed"}
    pref = math.sqrt(math.pi / alpha)

    num = 0.0
    for xi in xs:
        u = xi * xi
        Qv = sum(qi * u ** i for i, qi in enumerate(q))
        ghat_v = pref * math.exp(-math.pi ** 2 * u / alpha) * Qv
        num += ghat_v * _weight_f64(xi) * dx

    # Denominator: int g on [-1/2, 1/2]; require g >= 0.
    ts = np.linspace(-0.5, 0.5, 20001)
    dt = ts[1] - ts[0]
    c = params.c
    vals = np.exp(-alpha * ts * ts) * np.polyval(list(reversed([float(ci) for ci in c])), ts * ts)
    if vals.min() < -1e-8:
        return -1e9, {"error": "g negative"}
    int_g = float(vals.sum() * dt)
    if int_g <= 0:
        return -1e9, {"num": num, "int_g": int_g}
    return num / int_g, {"num": num, "int_g": int_g, "M": float(vals.max())}


def _f2_pd_penalty(params: f2.F2Params) -> float:
    ok, _ = f2.positive_definite_certificate(params)
    if ok:
        return 0.0
    # Compute Q min on [0, xi_max^2].
    q = [float(qi) for qi in f2.Q_coefs(params)]
    us = np.linspace(0, (f2.ghat_support(params)) ** 2, 2001)
    Qv = np.zeros_like(us)
    for i, qi in enumerate(q):
        Qv += qi * us ** i
    return -float(min(0, Qv.min()))


def optimise_f2(N: int = 2, n_restarts: int = 30, verbose: bool = False):
    """Search F2 parameters (alpha, c_0, c_1, ..., c_N)."""
    best = {"score": -np.inf, "params": None, "info": None}

    rng = np.random.default_rng(7)
    for restart in range(n_restarts):
        if restart == 0:
            x0 = np.array([4.0] + [1.0] + [0.0] * N)
        else:
            alpha0 = float(rng.uniform(0.5, 20.0))
            c0 = [float(rng.uniform(0.1, 2.0))] + list(rng.uniform(-2, 2, size=N))
            x0 = np.array([alpha0] + c0)

        def neg_obj(x):
            alpha, *c = x
            if alpha <= 0:
                return 1e9
            p = f2.F2Params(alpha=float(alpha), c=[float(v) for v in c])
            s, _ = _f2_score_f64(p, n_grid=1001)
            pen = _f2_pd_penalty(p)
            return -s + 100.0 * pen

        try:
            res = minimize(neg_obj, x0, method="Nelder-Mead",
                           options={"maxiter": 3000, "xatol": 1e-8, "fatol": 1e-10})
        except Exception:
            continue
        x = res.x
        alpha = x[0]
        c = x[1:]
        if alpha <= 0:
            continue
        p = f2.F2Params(alpha=float(alpha), c=[float(v) for v in c])
        pen = _f2_pd_penalty(p)
        if pen > 1e-8:
            continue
        s, info = _f2_score_f64(p, n_grid=4001)
        if verbose:
            print(f"[F2 restart {restart}] score={s:.6f} alpha={alpha:.4f} c={c} pen={pen:.1e}")
        if s > best["score"]:
            best = {"score": s, "params": p, "info": info}
    return best


# -----------------------------------------------------------------------------
# Matolcsi-Vinuesa reference g (reproduce 1.2802 as G1 sanity gate).
# -----------------------------------------------------------------------------

def matolcsi_vinuesa_reference_f1() -> f1.F1Params:
    """Return an F1 parameter point that approximates the Matolcsi-Vinuesa
    construction. Exact reproduction requires the original Mathematica code;
    this is a close analog with one cosine modulation.

    This is used for the G1 gate: we expect the returned bound to be close
    (but not equal) to 1.2802 -- the exact MV g has more complex structure.
    Any value > 1.2 from this starting point is a reasonable sanity signal.
    """
    # Heuristic settings: T near 1, omega near 2T for disjoint triangles.
    # a_1 must be >= 0 for positive-definiteness (triangles don't overlap).
    return f1.F1Params(T=1.0, omega=2.0, a=[0.5])
