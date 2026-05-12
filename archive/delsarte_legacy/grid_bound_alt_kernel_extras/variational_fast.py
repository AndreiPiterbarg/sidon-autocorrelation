"""Fast (vectorised) variational search for the optimal kernel phi.

Key trick: substitute x = (delta/2) sin(theta) to regularise the envelope
    (1 - (2x/delta)^2)^{alpha - 1/2}  =  cos^{2 alpha - 1}(theta)
so that  phi(x) dx  =  (C delta/2) cos^{2 alpha}(theta) poly(sin theta) dtheta
with a SMOOTH integrand in theta on [-pi/2, pi/2] for every alpha in [0, 0.5].

This makes a fixed-mesh trapezoid rule exponentially convergent — about the
accuracy of Gauss-Chebyshev quadrature, computed via a single FFT-friendly
sum per candidate.

Parameterisation:
    phi(x)  =  Z^{-1} (1 - 4 x^2 / delta^2)^{alpha - 1/2}
            * (1 + sum_{k=1}^{K} c_k T_{2k}(2 x / delta))
alpha = 0, c = 0 recovers the MV arcsine (matching k_1 = 0.90928, K_2 = 4.164).
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
from scipy.optimize import differential_evolution, brentq
from scipy.special import eval_chebyt


# ---------- fixed MV parameters ----------
DELTA = 0.138
U = 0.5 + DELTA              # 0.638
N_QP = 119


def _mv_coeffs_float():
    from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq
    return np.array([float(c.p) / float(c.q) for c in mv_coeffs_fmpq()])

MV_COEFFS = _mv_coeffs_float()


# ---------- theta-grid (regularising substitution x = (delta/2) sin theta) ----------
N_THETA = 2001
_THETA = np.linspace(-np.pi / 2, np.pi / 2, N_THETA)
_DTHETA = _THETA[1] - _THETA[0]
_SIN_T = np.sin(_THETA)
_COS_T = np.cos(_THETA)
_X_OF_T = (DELTA / 2.0) * _SIN_T           # (N_THETA,)

# Precompute Chebyshev polynomials evaluated at u = sin(theta)
K_MAX = 6
# T_{2k}(u) with u = sin(theta)
_CHEB = np.stack([eval_chebyt(2 * k, _SIN_T) for k in range(1, K_MAX + 1)])  # (K_MAX, N_THETA)

# Precompute cos(2 pi n * x(theta)) for n = 1..5
_KN_J = np.arange(1, 6)
_KN_COS = np.cos(2 * np.pi * _KN_J[:, None] * _X_OF_T[None, :])   # (5, N_THETA)

# Precompute cos(2 pi (j/u) * x(theta)) for j = 1..N_QP
_QP_XI = np.arange(1, N_QP + 1) / U
_QP_COS = np.cos(2 * np.pi * _QP_XI[:, None] * _X_OF_T[None, :])   # (N_QP, N_THETA)


# ---------- xi grid for K_2 integration ----------
# K_2 = int phi_hat(xi)^4 dxi over R (doubled for negative).  The integrand
# is smooth and decays at least as fast as xi^{-4 alpha} (for alpha > 1/4
# even after Chebyshev modulation), so a uniform xi-grid is adequate.
N_XI = 8001
XI_MAX_OVER_DELTA = 100.0
_XI = np.linspace(0.0, XI_MAX_OVER_DELTA / DELTA, N_XI)
_DXI = _XI[1] - _XI[0]


def phi_dmu_on_theta(params: np.ndarray) -> np.ndarray:
    """Integrand in theta: cos^{2 alpha}(theta) * poly(sin theta) (unnormalised).

    Multiply by (delta / 2) and integrate over theta in [-pi/2, pi/2] to get
    int phi(x) dx (unnormalised) = Z (unnormalised).
    """
    alpha = params[0]
    c = np.asarray(params[1:])
    K = len(c)
    # cos^{2 alpha} — well-defined at theta = +/- pi/2 (= 0 for alpha > 0).
    # For alpha = 0, cos^0 = 1 (including at endpoints, since cos(0)=1 limit).
    # We use np.maximum to guard against floating-point negatives.
    cos2a = np.maximum(_COS_T, 0.0) ** (2 * alpha)
    poly = 1.0 + c @ _CHEB[:K]
    return cos2a * poly


def compute_quantities(params: np.ndarray) -> dict | None:
    """Compute k_1, K_2, S_1, a, M_cert for a candidate (alpha, c_1..c_K)."""
    alpha = float(params[0])
    if alpha < 0:
        return None

    integrand_theta = phi_dmu_on_theta(params)   # shape (N_THETA,)

    # Check pointwise non-negativity of the polynomial modulation: if the
    # poly part goes negative ANYWHERE, phi(x) < 0 there => reject.
    c = np.asarray(params[1:])
    K = len(c)
    poly = 1.0 + c @ _CHEB[:K]
    if np.any(poly < -1e-10):
        return None

    # Normalisation Z = int phi(x) dx = (delta/2) * int_{-pi/2}^{pi/2} integrand_theta dtheta.
    Z_scaled = np.trapz(integrand_theta, dx=_DTHETA)   # = (2/delta) Z
    Z = (DELTA / 2.0) * Z_scaled
    if Z <= 0 or not np.isfinite(Z):
        return None

    # weighted theta-integrand for Fourier transforms: (delta/2) * integrand_theta / Z
    w_theta = (DELTA / 2.0) * integrand_theta / Z    # trapezoidal weights to apply to cos matrices

    # k_n = (phi_hat(n))^2 = (int phi(x) cos(2 pi n x) dx)^2
    # phi_hat(n) = (delta/2) int cos(2 pi n x(theta)) * [cos^{2a} poly / Z] dtheta
    #           = int w_theta * cos(2 pi n x(theta)) dtheta
    # Trapezoid: sum over theta of w_theta_i * cos_i * DTHETA.
    phi_hat_kn = _KN_COS @ w_theta * _DTHETA          # shape (5,)
    k_1 = phi_hat_kn[0] ** 2

    # phi_hat at j/u for j = 1..N_QP
    phi_hat_qp = _QP_COS @ w_theta * _DTHETA          # shape (N_QP,)
    if np.any(phi_hat_qp ** 2 < 1e-18):
        return None
    S_1 = np.sum((MV_COEFFS ** 2) / (phi_hat_qp ** 2))
    if not np.isfinite(S_1) or S_1 <= 0:
        return None

    # K_2 = int |phi_hat(xi)|^4 dxi (over R; use 2x for [0, inf)).
    # phi_hat(xi) for arbitrary real xi: same formula with cos(2 pi xi x(theta)).
    # We evaluate at _XI grid and trapezoid.  Batch to avoid 64MB+ arrays.
    batch = 1000
    K2_pos = 0.0
    for s in range(0, N_XI, batch):
        xi_chunk = _XI[s:s + batch]
        cos_mat = np.cos(2 * np.pi * xi_chunk[:, None] * _X_OF_T[None, :])
        phi_hat_chunk = cos_mat @ w_theta * _DTHETA
        K2_pos += np.sum(phi_hat_chunk ** 4) * _DXI
    K_2 = 2.0 * K2_pos

    rad2 = K_2 - 1 - 2 * k_1 ** 2
    if rad2 <= 0 or not np.isfinite(K_2):
        return None

    a = (4.0 / U) / S_1   # m = 1 assumed (MV's G)
    target = 2.0 / U + a

    def sup_R(M):
        if M <= 1.0:
            return float('-inf')
        mu_ = M * np.sin(np.pi / M) / np.pi
        # y* where d/dy R = 0:  y^2 (K_2 - 1) = k_1^2 (M - 1)
        y_star_sq = (k_1 ** 2) * (M - 1) / (K_2 - 1)
        y_star = np.sqrt(max(0.0, y_star_sq))
        if y_star <= mu_:
            return M + 1 + np.sqrt((M - 1) * (K_2 - 1))
        rad1 = M - 1 - 2 * mu_ * mu_
        if rad1 < 0:
            return float('inf')
        return M + 1 + 2 * mu_ * k_1 + np.sqrt(rad1 * rad2)

    try:
        f = lambda M: sup_R(M) - target
        f_lo = f(1.0 + 1e-10)
        f_hi = f(2.0)
        if f_lo >= 0 or f_hi <= 0:
            return None
        M_cert = brentq(f, 1.0 + 1e-10, 2.0, xtol=1e-7)
    except Exception:
        return None

    return {
        "alpha": alpha,
        "c": [float(x) for x in params[1:]],
        "k_1": float(k_1),
        "K_2": float(K_2),
        "S_1": float(S_1),
        "a": float(a),
        "Z": float(Z),
        "M_cert": float(M_cert),
    }


def objective(params: np.ndarray) -> float:
    q = compute_quantities(params)
    if q is None:
        return 2.0
    return -q["M_cert"]


def run_sanity():
    print("-- sanity --")
    p = np.array([0.0])
    q = compute_quantities(p)
    if q:
        print(f"  pure arcsine (alpha=0, no poly):")
        print(f"    k_1 = {q['k_1']:.6f}  (MV 0.909277)")
        print(f"    K_2 = {q['K_2']:.4f}  (MV surrogate 4.164)")
        print(f"    S_1 = {q['S_1']:.3f}  (MV 87.86)")
        print(f"    a   = {q['a']:.6f}  (MV 0.07137)")
        print(f"    M_cert = {q['M_cert']:.6f}  (MV 1.27481)")
    p = np.array([0.5])    # "box" (alpha = 0.5) via the substitution
    q = compute_quantities(p)
    if q:
        print(f"  box (alpha=0.5, no poly):")
        print(f"    k_1 = {q['k_1']:.6f}   K_2 = {q['K_2']:.4f}  "
              f"S_1 = {q['S_1']:.3f}  M_cert = {q['M_cert']:.6f}  (triangle K2 ~ 1.231)")


def run_de(K_poly: int = 3,
           alpha_range: tuple = (0.0, 0.3),
           c_range: tuple = (-0.15, 0.15),
           maxiter: int = 40,
           popsize: int = 15,
           seed: int = 42,
           verbose: bool = True) -> dict:
    bounds = [alpha_range] + [c_range] * K_poly
    t0 = time.time()
    best = {"M_cert": -np.inf, "params": None, "q": None}

    eval_count = [0]
    def obj_with_log(params):
        eval_count[0] += 1
        q = compute_quantities(params)
        if q is None:
            return 2.0
        if q["M_cert"] > best["M_cert"]:
            best["M_cert"] = q["M_cert"]
            best["params"] = params.tolist()
            best["q"] = q
            if verbose:
                print(f"  [{time.time()-t0:.1f}s eval#{eval_count[0]}] "
                      f"NEW BEST M_cert = {q['M_cert']:.6f}  "
                      f"alpha={params[0]:.4f}  c={[round(float(x),4) for x in params[1:]]}")
        return -q["M_cert"]

    result = differential_evolution(
        obj_with_log,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        tol=1e-8,
        mutation=(0.5, 1.2),
        recombination=0.7,
        polish=True,
        disp=False,
    )
    return {
        "bounds": bounds,
        "result_x": result.x.tolist(),
        "result_fun": -float(result.fun),
        "result_nit": int(result.nit),
        "result_nfev": int(result.nfev),
        "best_seen": best,
        "wall_time_sec": time.time() - t0,
    }


if __name__ == "__main__":
    run_sanity()
    print()
    t0_total = time.time()
    print("=== DE K_poly = 3, alpha in [0, 0.3], c in [-0.15, 0.15], seed=42 ===")
    out3 = run_de(K_poly=3, maxiter=40, popsize=15, seed=42, verbose=True)
    print(f"K=3: best M_cert = {out3['result_fun']:.6f} in {out3['wall_time_sec']:.1f}s  ({out3['result_nfev']} evals)")

    print()
    print("=== DE K_poly = 5, wider c range, seed = 17 ===")
    out5 = run_de(K_poly=5, alpha_range=(0.0, 0.3), c_range=(-0.20, 0.20),
                   maxiter=50, popsize=20, seed=17, verbose=True)
    print(f"K=5: best M_cert = {out5['result_fun']:.6f} in {out5['wall_time_sec']:.1f}s  ({out5['result_nfev']} evals)")

    arc = compute_quantities(np.array([0.0]))
    summary = {
        "arcsine_M_cert": arc["M_cert"] if arc else None,
        "de_K3_best": out3["result_fun"],
        "de_K5_best": out5["result_fun"],
        "K3_delta_vs_arcsine": out3["result_fun"] - (arc["M_cert"] if arc else 0),
        "K5_delta_vs_arcsine": out5["result_fun"] - (arc["M_cert"] if arc else 0),
        "beats_MV": max(out3["result_fun"], out5["result_fun"]) > 1.27481,
    }
    data = {"de_K3": out3, "de_K5": out5, "arcsine": arc, "summary": summary}
    out_path = "delsarte_dual/grid_bound_alt_kernel/variational_fast_results.json"
    body = json.dumps(data, indent=2, sort_keys=True, default=str)
    digest = hashlib.sha256(body.encode()).hexdigest()
    with open(out_path, "w") as f:
        json.dump({"sha256": digest, "body": data}, f, indent=2, default=str)
    print()
    print(f"Saved: {out_path}  (SHA-256: {digest})")
    print()
    print("=== SUMMARY ===")
    print(f"Total wall time: {time.time() - t0_total:.1f}s")
    for k, v in summary.items():
        print(f"  {k}: {v}")
