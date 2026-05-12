"""Variational search for the kernel phi that maximises M_cert.

We parametrise phi >= 0 on [-delta/2, delta/2] as

    phi(x) = Z^{-1} * (1 - (2x/delta)^2)^{alpha - 1/2}
             * (1 + sum_{k=1}^{K} c_k * T_{2k}(2x/delta))

where T_{2k} are even Chebyshev-first-kind polynomials and alpha, c_1..c_K
are free parameters.  alpha = 0.5 recovers the pure arcsine (MV baseline);
the c_k modulations are even polynomial perturbations that keep phi symmetric.

Non-negativity of phi is enforced by pointwise checks on a dense grid
(rejected candidate otherwise).  Normalisation Z is computed numerically.

For speed, all per-candidate computation uses numpy floats.  The "surrogate"
||K||_2^2 for alpha = 0.5 is NOT used here — we compute the honest integral
||K||_2^2 = int |phi_hat|^4 dxi numerically, which for alpha = 0.5 + eps is
near-arcsine but finite.

Global search: scipy.optimize.differential_evolution over (alpha, c_1..c_K)
in a box around arcsine.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import differential_evolution, brentq
from scipy.integrate import quad
from scipy.special import eval_chebyt


DELTA = 0.138
U = 0.5 + DELTA      # 0.638
N_GRID_PHI = 1001    # sample points for phi positivity check & Z
N_QP = 119           # MV's number of QP coefficients


# Load MV's G-coefficients as floats
def _mv_coeffs_float() -> np.ndarray:
    from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq
    return np.array([float(c.p) / float(c.q) for c in mv_coeffs_fmpq()])


MV_COEFFS = _mv_coeffs_float()
assert len(MV_COEFFS) == 119


def _phi_values(params: np.ndarray, x_grid: np.ndarray) -> np.ndarray:
    """Evaluate phi(x) on x_grid (without normalisation).  params = (alpha, c_1, .., c_K)."""
    alpha = params[0]
    c = params[1:]
    u = 2 * x_grid / DELTA            # u in [-1, 1]
    envelope = (1.0 - u * u) ** (alpha - 0.5)
    # Polynomial perturbation
    poly = np.ones_like(u)
    for k, ck in enumerate(c, start=1):
        poly += ck * eval_chebyt(2 * k, u)
    return envelope * poly


def _phi_hat(params: np.ndarray, xi: float) -> float:
    """phi_hat(xi) = int_{-delta/2}^{delta/2} phi(x) cos(2 pi xi x) dx.

    We compute by quadrature on x in [-delta/2, delta/2].  For alpha = 0.5
    the envelope has endpoint singularity, so we use weighted quadrature via
    sub-intervals near endpoints (scipy.integrate.quad handles this).
    """
    alpha = params[0]
    c = params[1:]
    a = DELTA / 2.0
    def integrand(x):
        u = 2 * x / DELTA
        env = (1.0 - u * u) ** (alpha - 0.5)
        poly = 1.0
        for k, ck in enumerate(c, start=1):
            poly += ck * eval_chebyt(2 * k, u)
        return env * poly * np.cos(2 * np.pi * xi * x)
    # symmetric, so integrate 0..a and double
    val, _ = quad(integrand, 0.0, a, limit=200)
    return 2.0 * val


def _compute_kernel_quantities(params: np.ndarray) -> dict | None:
    """Compute Z (normalisation), k_1 = phi_hat(1)^2, K_2 = int phi_hat^4 dxi,
    S_1 = sum a_j^2 / phi_hat(j/u)^2 (MV's G coefficients), min_G, and M_cert.

    Returns None if phi is not non-negative or if any quantity is non-finite.
    """
    x_grid = np.linspace(-DELTA / 2, DELTA / 2, N_GRID_PHI)
    phi_unnorm = _phi_values(params, x_grid)
    # Check non-negativity (interior; endpoint singularity for alpha < 0.5 is OK)
    # Mask out near-endpoints (factor 1-u^2 dominates there); check interior
    mask = np.abs(x_grid) < DELTA / 2 - 1e-4
    if np.any(phi_unnorm[mask] < -1e-9):
        return None

    # Normalisation: Z = int_{-delta/2}^{delta/2} phi dx
    # We compute via quad for accuracy
    alpha = params[0]
    c = params[1:]
    def phi_scalar(x):
        u = 2 * x / DELTA
        env = (1.0 - u * u) ** (alpha - 0.5)
        poly = 1.0
        for k, ck in enumerate(c, start=1):
            poly += ck * eval_chebyt(2 * k, u)
        return env * poly
    try:
        Z, _ = quad(phi_scalar, 0.0, DELTA / 2, limit=200)
        Z *= 2.0
    except Exception:
        return None
    if Z <= 0 or not np.isfinite(Z):
        return None

    # Normalise params in-place? No — we'll divide by Z explicitly.
    # phi_hat(xi) = (1/Z) * int phi_unnorm(x) cos(2 pi xi x) dx
    def phi_hat_norm(xi):
        try:
            raw = _phi_hat(params, xi)
        except Exception:
            return np.nan
        return raw / Z

    # k_1 = phi_hat(1)^2  (period-1 Fourier coefficient for MV inequality)
    k1 = phi_hat_norm(1.0) ** 2
    # K_2 = int |phi_hat|^4 dxi  (||K||_2^2 of K = phi * phi)
    def integrand_K2(xi):
        v = phi_hat_norm(xi)
        return v ** 4
    try:
        K2_val, _ = quad(integrand_K2, 0.0, 200.0 / DELTA, limit=500)
        K2 = 2.0 * K2_val
    except Exception:
        return None
    if not np.isfinite(K2) or K2 < 1.0 + 2.0 * k1 * k1:
        # radicand in Phi must stay positive
        return None

    # S_1 = sum a_j^2 / (phi_hat(j/u))^2
    S1 = 0.0
    min_hat_sq = np.inf
    for j in range(1, N_QP + 1):
        ph_j = phi_hat_norm(j / U)
        if not np.isfinite(ph_j) or abs(ph_j) < 1e-12:
            # Division blows up — skip (equivalent to a_j = 0 penalty)
            return None
        w = ph_j * ph_j
        min_hat_sq = min(min_hat_sq, w)
        S1 += (MV_COEFFS[j - 1] ** 2) / w
    if not np.isfinite(S1):
        return None

    # a = (4/u) * m^2 / S_1, with m = 1 (active MV constraint)
    a = (4.0 / U) * 1.0 / S1

    # Solve 2/u + a = sup_y Phi-RHS(M, y) over y in [0, mu(M)] for M.
    # Interior critical point: R(M, y*) = M + 1 + sqrt((M-1)(K_2-1)).
    # Clipped case at y = mu(M): R(M, mu(M)) = M + 1 + 2 mu k_1 + sqrt((M-1-2mu^2)(K_2-1-2k_1^2)).
    # Sup = max of these two (when y* <= mu use interior; else clipped).
    # We bisect on M to find the largest M with sup = 2/u + a.
    target = 2.0 / U + a
    def sup_R(M):
        if M <= 1.0:
            return float('-inf')
        mu = M * np.sin(np.pi / M) / np.pi
        y_star = np.sqrt(k1 * (M - 1) / (K2 - 1)) if K2 > 1.0 else 0.0
        y_star = min(y_star, mu)
        # use interior formula when y_star < mu
        if y_star == mu:
            # clipped: y = mu
            rad1 = M - 1 - 2 * mu * mu
            rad2 = K2 - 1 - 2 * k1 * k1
            if rad1 < 0 or rad2 < 0:
                return float('inf')
            return M + 1 + 2 * mu * np.sqrt(k1) + np.sqrt(rad1 * rad2)
            # note: we used y = mu, so +2 mu k_1 should be +2 mu * k_1 (NOT sqrt k1)
        # interior:
        return M + 1 + np.sqrt((M - 1) * (K2 - 1))

    # Actually, the formula above has an error.  Let me rewrite cleanly:
    def sup_R_correct(M):
        if M <= 1.0:
            return float('-inf')
        mu = M * np.sin(np.pi / M) / np.pi
        rad2 = K2 - 1 - 2 * k1 * k1
        if rad2 <= 0:
            return float('inf')
        # Interior critical point y^* where partial_y R = 0:
        #   y^*^2 * (K_2 - 1) = k_1^2 * (M - 1)   (derivation in REPORT)
        # Wait that was wrong — the correct derivation:
        #   partial_y R = 2 k_1 - 2 y sqrt(K2-1-2k1^2) / sqrt(M-1-2y^2) = 0
        #   => y sqrt(K2-1-2k1^2) = k_1 sqrt(M-1-2y^2)
        #   => y^2 (K2-1-2k1^2) = k_1^2 (M-1-2y^2)
        #   => y^2 [K2-1-2k1^2 + 2k1^2] = k_1^2 (M-1)
        #   => y^2 (K2-1) = k_1^2 (M-1)
        #   => y^* = k_1 sqrt((M-1)/(K2-1)).
        y_star = k1 * np.sqrt((M - 1) / (K2 - 1))
        if y_star <= mu:
            # use interior
            # R(M, y*) = M + 1 + sqrt((M-1)(K2-1))  (derived in REPORT)
            return M + 1 + np.sqrt((M - 1) * (K2 - 1))
        else:
            # clipped at y = mu
            rad1 = M - 1 - 2 * mu * mu
            if rad1 < 0:
                return float('inf')
            return M + 1 + 2 * mu * k1 + np.sqrt(rad1 * rad2)

    # Bisect M in (1.0, 1.3)
    try:
        def f(M):
            return sup_R_correct(M) - target
        # f(1+) = 2 - target < 0 usually; f(large) = large > 0
        M_lo = 1.0 + 1e-8
        M_hi = 2.0
        f_lo = f(M_lo)
        f_hi = f(M_hi)
        if f_lo >= 0 or f_hi <= 0:
            return None
        M_cert = brentq(f, M_lo, M_hi, xtol=1e-6)
    except Exception:
        return None

    return {
        "k1": k1,
        "K2": K2,
        "S1": S1,
        "a": a,
        "M_cert": M_cert,
        "Z": Z,
        "alpha": alpha,
        "c": list(c),
    }


def objective(params: np.ndarray) -> float:
    """Negative M_cert — minimised by differential_evolution."""
    q = _compute_kernel_quantities(params)
    if q is None:
        return 1.0    # a large objective (maximising -(-) = minimising)
    return -q["M_cert"]


def sanity_arcsine() -> dict:
    """Compute M_cert for alpha = 0 (pure arcsine, MV baseline)."""
    p = np.array([0.0])
    q = _compute_kernel_quantities(p)
    return q


def run_differential_evolution(K_poly: int = 3,
                                alpha_range: tuple = (0.0, 0.5),
                                c_range: tuple = (-0.2, 0.2),
                                maxiter: int = 30,
                                popsize: int = 15,
                                seed: int = 42) -> dict:
    """Global DE search over (alpha, c_1, ..., c_K_poly).

    Defaults: 3 polynomial modes, ~450 function evaluations.
    """
    bounds = [alpha_range] + [c_range] * K_poly
    print(f"Bounds: {bounds}")
    t0 = time.time()
    best_so_far = {"M_cert": -np.inf, "params": None, "q": None}
    def callback(xk, convergence):
        q = _compute_kernel_quantities(xk)
        if q and q["M_cert"] > best_so_far["M_cert"]:
            best_so_far["M_cert"] = q["M_cert"]
            best_so_far["params"] = xk.tolist()
            best_so_far["q"] = q
            print(f"  [{time.time()-t0:.1f}s] new best M_cert = {q['M_cert']:.6f}  "
                  f"at alpha={xk[0]:.4f}, c={xk[1:].tolist()}")
        return False

    result = differential_evolution(
        objective,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        disp=False,
        callback=callback,
    )
    return {
        "result_x": result.x.tolist(),
        "result_fun": -result.fun,
        "result_nit": result.nit,
        "result_nfev": result.nfev,
        "result_message": result.message,
        "best_callback": best_so_far,
        "wall_time_sec": time.time() - t0,
    }


if __name__ == "__main__":
    print("--- sanity: pure arcsine (alpha=0.5, no perturbation) ---")
    sn = sanity_arcsine()
    if sn:
        print(f"  k1 = {sn['k1']:.6f}  (MV: 0.909277)")
        print(f"  K2 = {sn['K2']:.4f}")
        print(f"  S1 = {sn['S1']:.3f}  (MV: 87.86)")
        print(f"  a  = {sn['a']:.6f}  (MV: 0.0714)")
        print(f"  M_cert = {sn['M_cert']:.6f}  (MV: 1.27481)")

    print()
    print("--- differential evolution (K_poly=3) ---")
    out = run_differential_evolution(K_poly=3, maxiter=30, popsize=15)
    print()
    print("FINAL:")
    print(f"  best M_cert = {out['result_fun']:.6f}")
    print(f"  at params = {out['result_x']}")
    print(f"  callback best = {out['best_callback']['M_cert']:.6f}")
    print(f"  iterations = {out['result_nit']}, evaluations = {out['result_nfev']}")
    print(f"  wall time = {out['wall_time_sec']:.1f}s")

    # Save
    out_path = "delsarte_dual/grid_bound_alt_kernel/variational_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"arcsine_sanity": sn, "de_result": out}, f, indent=2, default=str)
    print(f"Saved: {out_path}")
