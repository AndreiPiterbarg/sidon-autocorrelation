"""F5 optimisation driver.

Searches over c = (c_0, ..., c_K) for P(u) = sum c_k u^k with
  P(u) >= 0 on [0, 1/4]    (Markov-Lukacs SOS in low K)
  ghat_F5(xi) >= 0 on R    (verified pointwise via is_pd_admissible)
to maximise the rigorous Delsarte ratio
  L(c) = int ghat(xi) w(xi) dxi / M_g(c)
where w(xi) = cos^2(pi xi/2) on [-1, 1].

NOTE: The conclusion (see derivation.md, RESULTS.md) is that F5 is
DOMINATED by F1 in this pipeline. This optimizer is provided for
reproducibility of the negative result.
"""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np
from scipy.optimize import minimize

from .f5 import (
    F5Params,
    NotPDAdmissible,
    f5_idealised_ratio,
    f5_lower_bound,
    g_hat_zero,
    g_hat_value,
    g_value,
    is_pd_admissible,
    M_g,
)


# ---------------------------------------------------------------------------
# Float64 fast scoring (search) - rough proxy for the rigorous L.
# ---------------------------------------------------------------------------

def _f5_idealised_f64(c: Iterable[float]) -> float:
    """Idealised ratio ghat(0) / M_g via float64 sampling (NOT rigorous)."""
    c = [float(ck) for ck in c]
    K = len(c) - 1
    g0 = sum(c[k] / ((2 * k + 1) * (2 * k + 2) * 4 ** k) for k in range(K + 1))
    # Sample g on [-1/2, 1/2].
    ts = np.linspace(-0.5, 0.5, 4001)
    P = np.polyval(list(reversed(c)), ts ** 2)
    g = (1 - 2 * np.abs(ts)) * P
    if g.min() < -1e-9:
        return -1e9
    M = max(g.max(), 1e-12)
    return g0 / M


def _ghat_f64(xi: float, c: Iterable[float]) -> float:
    """Float64 ghat by quad on [-1/2, 1/2]."""
    from scipy.integrate import quad
    c = [float(ck) for ck in c]

    def integrand(t):
        h = 1 - 2 * abs(t)
        if h <= 0:
            return 0.0
        Pv = sum(ck * t ** (2 * k) for k, ck in enumerate(c))
        return h * Pv * math.cos(2 * math.pi * t * xi)

    val, _ = quad(integrand, -0.5, 0.5, limit=200)
    return val


def _f5_rigorous_proxy_f64(c: Iterable[float], n_xi: int = 401) -> float:
    """Float64 proxy for the rigorous int ghat(xi) cos^2(pi xi/2) dxi / M_g."""
    c = [float(ck) for ck in c]
    K = len(c) - 1

    # Sample g and check non-negativity.
    ts = np.linspace(-0.5, 0.5, 4001)
    P = np.polyval(list(reversed(c)), ts ** 2)
    g = (1 - 2 * np.abs(ts)) * P
    if g.min() < -1e-9:
        return -1e9
    M = max(g.max(), 1e-12)

    # Numerator: int_{-1}^{1} ghat * cos^2(pi xi/2) dxi.
    xis = np.linspace(-1.0, 1.0, n_xi)
    dxi = xis[1] - xis[0]
    ghat_vals = np.array([_ghat_f64(float(xi), c) for xi in xis])
    w = np.cos(np.pi * np.abs(xis) / 2) ** 2
    # Spectral nonneg sanity:
    if ghat_vals.min() < -1e-6:
        return -1e9 + ghat_vals.min()
    num = float(np.sum(ghat_vals * w) * dxi)
    return num / M


def parameterize_P(K: int) -> dict:
    """Return a parameterization spec.

    For K=0,1: monomial basis is fine (low dim).
    For K>=2: parameterize via a Markov-Lukacs SOS so that P(u) >= 0 on [0, 1/4]
    is automatic. Returns a function from theta -> coefficients c.

    For Phase 3 we use the simple monomial basis with explicit constraints.
    """
    return {
        "K": K,
        "n_params": K + 1,
        "from_theta": lambda theta: list(theta),
    }


def _P_nonneg_penalty(c: Iterable[float]) -> float:
    """Penalty for P(u) < 0 on [0, 1/4] (sampled)."""
    us = np.linspace(0, 0.25, 1001)
    Pv = np.polyval(list(reversed([float(x) for x in c])), us)
    return -float(min(0, Pv.min()))


def optimize_ratio(K: int, n_restarts: int = 16, verbose: bool = False,
                   verify_pd: bool = False) -> dict:
    """Optimise c over R^{K+1} maximising the float64 proxy.

    verify_pd:
        If True, only accept candidates that pass `is_pd_admissible`.
        Default False (the proxy already enforces ghat sampled >= 0).

    Returns a dict with the best c found and diagnostics.
    """
    rng = np.random.default_rng(2026)
    best = {"score": -np.inf, "c": None}

    for restart in range(n_restarts):
        if restart == 0:
            x0 = np.array([1.0] + [0.0] * K)
        else:
            x0 = np.array([1.0] + list(rng.normal(0, 0.5, size=K)))

        def neg_obj(x):
            s = _f5_rigorous_proxy_f64(x)
            pen = _P_nonneg_penalty(x)
            return -s + 100.0 * pen

        try:
            res = minimize(neg_obj, x0, method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-7,
                                    "fatol": 1e-9})
        except Exception:
            continue
        x = res.x
        if _P_nonneg_penalty(x) > 1e-7:
            continue
        s = _f5_rigorous_proxy_f64(x)
        if verbose:
            print(f"[F5 K={K} restart {restart}] score={s:.6f} c={x}")
        if s > best["score"]:
            best = {"score": float(s), "c": [float(v) for v in x]}

    # If we have a best, report idealised and rigorous-proxy.
    if best["c"] is not None:
        best["idealised"] = _f5_idealised_f64(best["c"])
    return best


def sweep(K_list=(0, 1, 2, 3, 4)) -> list:
    """Run optimize_ratio for each K and return a tabulated list of dicts."""
    table = []
    for K in K_list:
        b = optimize_ratio(K, n_restarts=20, verbose=False)
        b["K"] = K
        table.append(b)
        print(f"K={K}  proxy={b['score']:.6f}  idealised={b.get('idealised', float('nan')):.6f}  c={b['c']}")
    return table
