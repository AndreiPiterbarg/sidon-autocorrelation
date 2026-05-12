"""Agent K30: direct parametric K_hat search for MV C_{1a} lower bound.

The MV master inequality only sees three numbers (k_1, K_2, S_1) computed from
K_hat (which equals phi_hat^2 when K = phi * phi).  Prior agents parameterised
phi(x) in physical space.  THIS agent parameterises *on the Fourier side*,
which is the natural surface for the Pareto frontier (k_1 high, K_2 low,
S_1 moderate).

Three parametrisations:

  (a) ``cos``       -- phi(x) = sum_{k=0}^{N-1} c_k cos(pi k x / (DELTA/2))
                       restricted to [-DELTA/2, DELTA/2].  Then
                       K_hat(xi) = phi_hat(xi)^2 = |P(xi)|^2 automatically,
                       where P(xi) is a finite sum of sincs centred at
                       0, +-1/(DELTA), +-2/DELTA, ...  This is the canonical
                       Fejer-Riesz / "Fourier side" parametrisation.

  (b) ``sinc``      -- phi(x) = sum c_k sinc(alpha (x - x_k)), x_k uniform on
                       [-DELTA/2, DELTA/2].  Equivalent to a band-limited P(xi)
                       with band ~ alpha pi.

  (c) ``cheb``      -- phi(x) = sum c_k T_k(2x/DELTA) (Chebyshev basis in
                       physical space).  Restricted to [-DELTA/2, DELTA/2].
                       K = phi*phi has K_hat = |phi_hat|^2 >= 0 automatically.
                       Note: a free-form Chebyshev K_hat parametrisation gives
                       K(x) that is generically sign-changing (admissible for
                       Bochner but not for our K>=0 constraint); putting the
                       Cheb expansion on phi side guarantees K>=0 when phi>=0.

For all three, we maximise M_cert (computed by mv_master_M_cert) over c in R^N
with N <= 6 using scipy.optimize.differential_evolution.

A precomputed-matrix fast evaluator hits ~11 ms / point so a full DE (popsize
14, maxiter 40 => 560 evals) runs in ~7 s.  We then re-evaluate the optimum
with the reference helper for an unbiased M_cert.

Output: _agent_K30_direct_Khat_result.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Sequence

import numpy as np
from scipy.optimize import differential_evolution

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import (  # noqa: E402
    DELTA,
    evaluate_phi,
    evaluate_K_directly,
    mv_master_M_cert,
    MV_COEFFS,
)

U = 0.5 + DELTA
N_QP = 119


# ============================================================
# Precomputed fast evaluator (theta-grid, identical to helper for k_1, S_1)
# K_2 uses N_XI=8001 (helper uses 16001); error in K_2 is ~1%.
# ============================================================

N_THETA = 4001
_THETA = np.linspace(-np.pi / 2.0, np.pi / 2.0, N_THETA)
_DTHETA = _THETA[1] - _THETA[0]
_COS_T = np.cos(_THETA)
_SIN_T = np.sin(_THETA)
_X_OF_T = (DELTA / 2.0) * _SIN_T

K_BASIS_MAX = 8

# Cosine basis on theta grid:  cos(pi k x / (DELTA/2)) for x = X_OF_T
_KS = np.arange(K_BASIS_MAX).reshape(-1, 1)
_COS_BASIS_T = np.cos(np.pi * _KS * _X_OF_T.reshape(1, -1) / (DELTA / 2.0))

# Fourier matrices on theta grid (Bochner check + S_1)
_NS = np.arange(1, 201)
_COS_NS_T = np.cos(2 * np.pi * _NS[:, None] * _X_OF_T[None, :])
_QP = np.arange(1, N_QP + 1) / U
_COS_QP_T = np.cos(2 * np.pi * _QP[:, None] * _X_OF_T[None, :])

# Xi grid for K_2 = int phi_hat^4 dxi
N_XI_FAST = 8001
_XI_FAST = np.linspace(0.0, 200.0 / DELTA, N_XI_FAST)
_DXI_FAST = _XI_FAST[1] - _XI_FAST[0]
_COS_XI_T = np.cos(2 * np.pi * _XI_FAST[:, None] * _X_OF_T[None, :])


def _eval_from_w_theta(w_theta: np.ndarray) -> dict | None:
    """Given probability-density w_theta (with int w_theta dtheta = 1 and
    w_theta = phi(x(theta)) * (DELTA/2) cos(theta) / Z), compute k_1, K_2, S_1
    and M_cert.  Returns None on Bochner / singularity failure.
    """
    ph_ns = (_COS_NS_T @ w_theta) * _DTHETA
    bochner_min = float(np.min(ph_ns ** 2))
    if bochner_min < -1e-12:
        return None
    k_1 = float(ph_ns[0]) ** 2
    ph_qp = (_COS_QP_T @ w_theta) * _DTHETA
    if np.any(ph_qp ** 2 < 1e-20):
        return None
    S_1 = float(np.sum((MV_COEFFS ** 2) / (ph_qp ** 2)))
    ph_xi = (_COS_XI_T @ w_theta) * _DTHETA
    K_2 = 2.0 * float(np.sum(ph_xi ** 4) * _DXI_FAST)
    M = mv_master_M_cert(k_1, K_2, S_1)
    return {"k_1": k_1, "K_2": K_2, "S_1": S_1, "M_cert": M,
            "bochner_min": bochner_min}


def fast_eval_cos(c: np.ndarray) -> dict | None:
    c = np.asarray(c, dtype=float)
    K = len(c)
    phi = c @ _COS_BASIS_T[:K]
    pmin = float(phi.min())
    if pmin < -1e-8:
        return None
    phi = np.maximum(phi, 0.0)
    w = phi * (DELTA / 2.0) * _COS_T
    Z = float(np.trapezoid(w, dx=_DTHETA))
    if Z <= 0 or not np.isfinite(Z):
        return None
    w = w / Z
    r = _eval_from_w_theta(w)
    if r is not None:
        r["phi_min"] = pmin
    return r


def make_sinc_basis_T(N_knots: int, alpha: float) -> np.ndarray:
    knots = np.linspace(-DELTA / 2.0, DELTA / 2.0, N_knots)
    arg = alpha * (_X_OF_T[None, :] - knots[:, None])
    return np.sinc(arg)


def fast_eval_sinc(c: np.ndarray, alpha: float) -> dict | None:
    c = np.asarray(c, dtype=float)
    basis = make_sinc_basis_T(len(c), alpha)
    phi = c @ basis
    pmin = float(phi.min())
    if pmin < -1e-8:
        return None
    phi = np.maximum(phi, 0.0)
    w = phi * (DELTA / 2.0) * _COS_T
    Z = float(np.trapezoid(w, dx=_DTHETA))
    if Z <= 0 or not np.isfinite(Z):
        return None
    w = w / Z
    r = _eval_from_w_theta(w)
    if r is not None:
        r["phi_min"] = pmin
    return r


# ============================================================
# Parametrisation (c):  phi(x) = sum c_k T_k(2x/DELTA)  on [-DELTA/2, DELTA/2]
# (Chebyshev basis in physical space).  K = phi*phi has K_hat >= 0 automatically.
# ============================================================

# Precompute Cheb basis on theta grid
def _chebT_arr_on_grid(N_basis: int, x_grid: np.ndarray) -> np.ndarray:
    z = 2.0 * x_grid / DELTA
    z = np.clip(z, -1.0, 1.0)
    return np.cos(np.arange(N_basis).reshape(-1, 1) * np.arccos(z).reshape(1, -1))


CHEB_N_BASIS_MAX = 8
_CHEB_BASIS_T = _chebT_arr_on_grid(CHEB_N_BASIS_MAX, _X_OF_T)   # (CHEB_N_BASIS_MAX, N_THETA)


def fast_eval_cheb(c: np.ndarray) -> dict | None:
    """Cheb in physical space (parametrisation (c))."""
    c = np.asarray(c, dtype=float)
    K_basis = len(c)
    phi = c @ _CHEB_BASIS_T[:K_basis]
    pmin = float(phi.min())
    if pmin < -1e-8:
        return None
    phi = np.maximum(phi, 0.0)
    w = phi * (DELTA / 2.0) * _COS_T
    Z = float(np.trapezoid(w, dx=_DTHETA))
    if Z <= 0 or not np.isfinite(Z):
        return None
    w = w / Z
    r = _eval_from_w_theta(w)
    if r is not None:
        r["phi_min"] = pmin
    return r


def _phi_cheb_at(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Evaluate Cheb-in-physical-space phi(x) on arbitrary x in [-DELTA/2, DELTA/2]."""
    N = len(c)
    z = 2.0 * x / DELTA
    z = np.clip(z, -1.0, 1.0)
    T_mat = np.cos(np.arange(N).reshape(-1, 1) * np.arccos(z).reshape(1, -1))
    return (c.reshape(1, -1) @ T_mat).ravel()


# ============================================================
# DE objectives
# ============================================================

INFEAS_COST = 1.0


def obj_cos(c: np.ndarray) -> float:
    r = fast_eval_cos(c)
    if r is None or r.get("M_cert") is None:
        return INFEAS_COST
    return -float(r["M_cert"])


def obj_sinc(c: np.ndarray, alpha: float) -> float:
    r = fast_eval_sinc(c, alpha)
    if r is None or r.get("M_cert") is None:
        return INFEAS_COST
    return -float(r["M_cert"])


def obj_cheb(c: np.ndarray) -> float:
    r = fast_eval_cheb(c)
    if r is None or r.get("M_cert") is None:
        return INFEAS_COST
    return -float(r["M_cert"])


# ============================================================
# Drivers (DE -> reference helper re-eval)
# ============================================================

def _phi_cos_at(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    N = len(c)
    return (c.reshape(1, -1) @
            np.cos(np.pi * np.arange(N).reshape(-1, 1) * x.reshape(1, -1)
                   / (DELTA / 2.0))).ravel()


def run_de_cos(N: int, maxiter: int = 50, popsize: int = 16,
               seed: int = 17) -> dict:
    print(f"\n--- (a) cos basis, N={N} ---")
    bounds = [(0.2, 2.0)] + [(-1.0, 1.0)] * (N - 1)
    t0 = time.time()
    try:
        res = differential_evolution(
            obj_cos, bounds=bounds,
            maxiter=maxiter, popsize=popsize, seed=seed,
            tol=1e-7, polish=False, workers=1, updating='deferred',
        )
    except Exception as e:
        return {"parametrisation": "cos", "N": N,
                "error": str(e), "M_cert": None}
    c = np.array(res.x, dtype=float)
    M_fast = -float(res.fun) if res.fun < 0 else None
    r = evaluate_phi(lambda x: _phi_cos_at(x, c),
                     label=f"K30_cos_N{N}", verbose=True) if M_fast else {}
    return {
        "parametrisation": "cos",
        "N": N,
        "c": c.tolist(),
        "M_cert_fast": M_fast,
        "M_cert": r.get("M_cert"),
        "k_1": r.get("k_1"),
        "K_2": r.get("K_2"),
        "S_1": r.get("S_1"),
        "phi_min": r.get("phi_min"),
        "bochner_min": r.get("bochner_min"),
        "beats_MV": r.get("beats_MV"),
        "beats_127": r.get("beats_127"),
        "wall_s": time.time() - t0,
    }


def _phi_sinc_at(x: np.ndarray, c: np.ndarray, alpha: float) -> np.ndarray:
    N = len(c)
    knots = np.linspace(-DELTA / 2.0, DELTA / 2.0, N)
    return (c.reshape(1, -1) @
            np.sinc(alpha * (x.reshape(1, -1) - knots.reshape(-1, 1)))).ravel()


def run_de_sinc(N: int, alpha: float, maxiter: int = 35, popsize: int = 14,
                seed: int = 23) -> dict:
    print(f"\n--- (b) sinc basis, N={N}, alpha={alpha:.2f} ---")
    bounds = [(-1.0, 2.0)] * N
    t0 = time.time()
    try:
        res = differential_evolution(
            lambda c, _a=alpha: obj_sinc(c, _a),
            bounds=bounds,
            maxiter=maxiter, popsize=popsize, seed=seed,
            tol=1e-7, polish=False, workers=1, updating='deferred',
        )
    except Exception as e:
        return {"parametrisation": "sinc", "N": N, "alpha": alpha,
                "error": str(e), "M_cert": None}
    c = np.array(res.x, dtype=float)
    M_fast = -float(res.fun) if res.fun < 0 else None
    r = evaluate_phi(lambda x: _phi_sinc_at(x, c, alpha),
                     label=f"K30_sinc_N{N}_a{alpha:.2f}",
                     verbose=True) if M_fast else {}
    return {
        "parametrisation": "sinc",
        "N": N,
        "alpha": float(alpha),
        "c": c.tolist(),
        "M_cert_fast": M_fast,
        "M_cert": r.get("M_cert"),
        "k_1": r.get("k_1"),
        "K_2": r.get("K_2"),
        "S_1": r.get("S_1"),
        "phi_min": r.get("phi_min"),
        "bochner_min": r.get("bochner_min"),
        "beats_MV": r.get("beats_MV"),
        "beats_127": r.get("beats_127"),
        "wall_s": time.time() - t0,
    }


def run_de_cheb(N: int, maxiter: int = 40, popsize: int = 16,
                seed: int = 29) -> dict:
    print(f"\n--- (c) Cheb phi (physical), N={N} ---")
    # c_0 dominates (DC of phi); allow Cheb coefficients to swing.
    bounds = [(0.2, 2.0)] + [(-1.0, 1.0)] * (N - 1)
    t0 = time.time()
    try:
        res = differential_evolution(
            obj_cheb,
            bounds=bounds,
            maxiter=maxiter, popsize=popsize, seed=seed,
            tol=1e-7, polish=False, workers=1, updating='deferred',
        )
    except Exception as e:
        return {"parametrisation": "cheb", "N": N,
                "error": str(e), "M_cert": None}
    c = np.array(res.x, dtype=float)
    M_fast = -float(res.fun) if res.fun < 0 else None
    r = evaluate_phi(lambda x: _phi_cheb_at(x, c),
                     label=f"K30_cheb_N{N}", verbose=True) if M_fast else {}
    return {
        "parametrisation": "cheb",
        "N": N,
        "c": c.tolist(),
        "M_cert_fast": M_fast,
        "M_cert": r.get("M_cert"),
        "k_1": r.get("k_1"),
        "K_2": r.get("K_2"),
        "S_1": r.get("S_1"),
        "phi_min": r.get("phi_min"),
        "bochner_min": r.get("bochner_min"),
        "beats_MV": r.get("beats_MV"),
        "beats_127": r.get("beats_127"),
        "wall_s": time.time() - t0,
    }


# ============================================================
# Sanity
# ============================================================

def sanity_cos_dc() -> dict:
    r_fast = fast_eval_cos(np.array([1.0]))
    r_helper = evaluate_phi(lambda x: np.ones_like(x),
                             label="K30_sanity_cos_DC", verbose=False)
    return {"fast": r_fast,
            "helper": {k: v for k, v in r_helper.items() if k != "label"}}


def sanity_cos_mix() -> dict:
    c = np.array([1.0, -0.5, 0.2])
    r_fast = fast_eval_cos(c)
    r_helper = evaluate_phi(lambda x: _phi_cos_at(x, c),
                             label="K30_sanity_cos_mix", verbose=False)
    return {"fast": r_fast,
            "helper": {k: v for k, v in r_helper.items() if k != "label"}}


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Agent K30: direct parametric K_hat = |P|^2 search")
    print(f"DELTA = {DELTA},  U = {U},  N_QP = {N_QP}")
    print("=" * 70)
    payload = {
        "delta": DELTA, "u": U, "n_qp": N_QP,
        "fast_evaluator": {
            "N_THETA": N_THETA, "N_XI_FAST": N_XI_FAST,
            "xi_max_over_delta": 200.0,
        },
    }

    print("\n=== Sanity: cos-basis DC (tophat phi -> triangle K) ===")
    s_dc = sanity_cos_dc()
    print(f"  fast:   {s_dc['fast']}")
    print(f"  helper: {s_dc['helper']}")
    payload["sanity_cos_dc"] = s_dc

    print("\n=== Sanity: cos-basis c=[1,-0.5,0.2] ===")
    s_mix = sanity_cos_mix()
    print(f"  fast:   {s_mix['fast']}")
    print(f"  helper: {s_mix['helper']}")
    payload["sanity_cos_mix"] = s_mix

    all_results = []

    print("\n" + "=" * 70)
    print("PARAMETRISATION (a): phi cosine series")
    print("=" * 70)
    cos_results = []
    for N in [2, 3, 4, 5, 6]:
        r = run_de_cos(N, maxiter=50, popsize=16, seed=17)
        cos_results.append(r)
        all_results.append(r)
    payload["cos_results"] = cos_results

    print("\n" + "=" * 70)
    print("PARAMETRISATION (b): phi sinc basis")
    print("=" * 70)
    sinc_results = []
    for N in [4, 5, 6]:
        nat = (N - 1) / DELTA
        for f in [0.5, 1.0, 1.5, 2.0]:
            alpha = nat * f
            r = run_de_sinc(N, alpha, maxiter=35, popsize=14, seed=23)
            sinc_results.append(r)
            all_results.append(r)
    payload["sinc_results"] = sinc_results

    print("\n" + "=" * 70)
    print("PARAMETRISATION (c): Chebyshev phi (physical space)")
    print("=" * 70)
    cheb_results = []
    for N in [2, 3, 4, 5, 6]:
        r = run_de_cheb(N, maxiter=50, popsize=16, seed=29)
        cheb_results.append(r)
        all_results.append(r)
    payload["cheb_results"] = cheb_results

    valid = [r for r in all_results if r.get("M_cert") is not None]
    valid.sort(key=lambda r: r["M_cert"], reverse=True)
    payload["all_valid_sorted_top10"] = valid[:10]
    payload["best"] = valid[0] if valid else None

    print("\n" + "=" * 70)
    print("TOP 10 across all parametrisations (re-evaluated via helper)")
    print("=" * 70)
    for r in valid[:10]:
        p = r.get("parametrisation", "?")
        N = r.get("N", "?")
        extras = []
        if "alpha" in r and r["alpha"] is not None:
            extras.append(f"a={r['alpha']:.2f}")
        extras_s = ("(" + ",".join(extras) + ")") if extras else ""
        beats = "MV" if r.get("beats_MV") else ("127" if r.get("beats_127") else "-")
        print(f"  {p:5s} N={N} {extras_s:18s}  "
              f"M_cert={r['M_cert']:.5f}  "
              f"k_1={(r.get('k_1') or 0):.4f}  "
              f"K_2={(r.get('K_2') or 0):.3f}  "
              f"S_1={(r.get('S_1') or 0):.2f}  "
              f"beats={beats}")

    out_path = os.path.join(REPO, "_agent_K30_direct_Khat_result.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"\nWrote: {out_path}")
    best = payload["best"]
    if best is not None:
        print(f"\nBEST M_cert = {best['M_cert']:.5f}  "
              f"(parametrisation={best['parametrisation']}, N={best['N']})")
        print(f"  beats MV (1.27481)? {best.get('beats_MV')}")
        print(f"  beats 1.27 numerical? {(best['M_cert'] or 0) > 1.27}")
    else:
        print("\nNo valid M_cert in sweep.")


if __name__ == "__main__":
    main()
