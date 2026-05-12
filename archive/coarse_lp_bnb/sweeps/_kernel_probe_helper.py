"""Shared kernel evaluation helper for parallel kernel-probe agents.

USAGE
-----
Each agent defines one of:
    (A) ``phi_on_grid(x_grid)``: returns unnormalised phi(x) on a 1D
        x-grid in [-DELTA/2, DELTA/2].  Caller will auto-conv to get
        K = phi*phi supported on [-DELTA, DELTA].
    (B) ``K_hat_fn(xi_grid)``: returns K_hat(xi) directly (so K is
        defined via inverse FT).  REQUIRES K_hat >= 0 (Bochner OK).

then calls ``evaluate_phi(...)`` or ``evaluate_K_hat(...)`` and prints
the result.

The MV master inequality uses the brentq-on-sup_R(M) trick to compute
M_cert (matches ``variational_pod.compute_quantities`` numerics).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import brentq

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)


DELTA = 0.138
U = 0.5 + DELTA      # 0.638
N_QP = 119

# MV's G coefficients (length 119)
def _mv_coeffs():
    from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq
    return np.array([float(c.p) / float(c.q) for c in mv_coeffs_fmpq()])

try:
    MV_COEFFS = _mv_coeffs()
except Exception:
    MV_COEFFS = None  # caller should handle

# Theta grid for the regularised x = (DELTA/2) sin(theta) substitution.
N_THETA = 4001
_THETA = np.linspace(-np.pi / 2, np.pi / 2, N_THETA)
_DTHETA = _THETA[1] - _THETA[0]
_SIN_T = np.sin(_THETA)
_COS_T = np.cos(_THETA)
_X_OF_T = (DELTA / 2.0) * _SIN_T

# Xi grid for K_2 = int phi_hat^4 dxi  (integrate on [0, XI_MAX/DELTA] then double)
N_XI = 16001
XI_MAX_OVER_DELTA = 200.0
_XI = np.linspace(0.0, XI_MAX_OVER_DELTA / DELTA, N_XI)
_DXI = _XI[1] - _XI[0]


def mv_master_M_cert(k_1: float, K_2: float, S_1: float) -> float | None:
    """Solve M_cert from k_1, K_2, S_1 via MV's sup-R(M) = 2/u + a inequality.

    Returns M_cert as a float, or None on failure (no bracket).
    """
    if K_2 <= 1 + 2 * k_1 * k_1:
        return None
    a = (4.0 / U) / S_1
    target = 2.0 / U + a

    rad2 = K_2 - 1 - 2 * k_1 * k_1

    def sup_R(M: float) -> float:
        if M <= 1.0:
            return float('-inf')
        mu_ = M * np.sin(np.pi / M) / np.pi
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
        return brentq(f, 1.0 + 1e-10, 2.0, xtol=1e-9)
    except Exception:
        return None


def evaluate_phi(phi_unnorm_fn, label: str = "kernel",
                 bochner_check_max: int = 200,
                 require_bochner: bool = True,
                 verbose: bool = True) -> dict:
    """Evaluate MV M_cert for a kernel K = phi * phi.

    phi_unnorm_fn(x_array)  returns  unnormalised phi(x) on the input grid.
    phi MUST be non-negative on [-DELTA/2, DELTA/2]; this is verified on the
    theta grid.  K = phi * phi has K_hat = (phi_hat)^2 >= 0 automatically.

    Returns
    -------
    dict with k_1, K_2, S_1, M_cert, bochner_min, and Z.
    """
    phi_vals = phi_unnorm_fn(_X_OF_T)
    phi_vals = np.asarray(phi_vals, dtype=float)
    if phi_vals.shape != _X_OF_T.shape:
        raise ValueError(
            f"phi_unnorm_fn returned shape {phi_vals.shape}, expected {_X_OF_T.shape}"
        )
    # phi >= 0 check (allow tiny numerical noise)
    pmin = float(phi_vals.min())
    if pmin < -1e-8:
        return {"label": label, "M_cert": None,
                "reason": f"phi negative on grid: min={pmin}"}
    phi_vals = np.maximum(phi_vals, 0.0)

    # weight = phi(x) * dx in theta-coords:  dx = (DELTA/2) cos(theta) dtheta
    # so phi_dx_theta(theta) = phi(x(theta)) * (DELTA/2) cos(theta)
    phi_dx_theta = phi_vals * (DELTA / 2.0) * _COS_T

    Z = np.trapz(phi_dx_theta, dx=_DTHETA)
    if Z <= 0 or not np.isfinite(Z):
        return {"label": label, "M_cert": None, "reason": f"Z={Z}"}

    w_theta = phi_dx_theta / Z   # normalised so int phi dx = 1

    # phi_hat(xi) = int phi(x) cos(2 pi xi x) dx
    def phi_hat(xi: np.ndarray | float) -> np.ndarray:
        xi = np.atleast_1d(xi)
        cos_mat = np.cos(2 * np.pi * xi[:, None] * _X_OF_T[None, :])
        return cos_mat @ w_theta * _DTHETA

    # k_n = (phi_hat(n))^2  for n = 1..bochner_check_max
    ns = np.arange(1, bochner_check_max + 1)
    ph_ns = phi_hat(ns)
    kn = ph_ns ** 2

    # Bochner check (always >= 0 since (phi_hat)^2; just a sanity check)
    bochner_min = float(kn.min())
    if require_bochner and bochner_min < -1e-12:
        return {"label": label, "M_cert": None,
                "reason": f"Bochner violated: min k_n = {bochner_min}"}

    k_1 = float(kn[0])

    # K_2 = int (phi_hat(xi))^4 dxi over R (double integral over [0, X_max])
    batch = 1000
    K2_pos = 0.0
    for s in range(0, N_XI, batch):
        xi_chunk = _XI[s:s + batch]
        ph_chunk = phi_hat(xi_chunk)
        K2_pos += np.sum(ph_chunk ** 4) * _DXI
    K_2 = 2.0 * K2_pos

    # S_1 = sum a_j^2 / (phi_hat(j/u))^2
    if MV_COEFFS is None:
        return {"label": label, "M_cert": None, "reason": "no MV coeffs"}
    qp_xi = np.arange(1, N_QP + 1) / U
    ph_qp = phi_hat(qp_xi)
    if np.any(ph_qp ** 2 < 1e-20):
        # If some QP frequency has near-zero FT, the term blows up
        # (Bochner-borderline).  Use a regularisation.
        return {"label": label, "M_cert": None,
                "reason": "phi_hat(j/u)^2 ~ 0 at some j"}
    S_1 = float(np.sum((MV_COEFFS ** 2) / (ph_qp ** 2)))

    M_cert = mv_master_M_cert(k_1, K_2, S_1)
    out = {
        "label": label,
        "k_1": k_1,
        "K_2": float(K_2),
        "S_1": S_1,
        "Z": float(Z),
        "phi_min": pmin,
        "bochner_min": bochner_min,
        "phi_hat_at_2": float(ph_ns[1]) if len(ph_ns) > 1 else None,
        "phi_hat_at_3": float(ph_ns[2]) if len(ph_ns) > 2 else None,
        "M_cert": M_cert,
        "beats_MV": (M_cert is not None and M_cert > 1.27481),
        "beats_127": (M_cert is not None and M_cert > 1.27),
    }
    if verbose:
        print(f"[{label}] k_1={k_1:.5f}  K_2={K_2:.4f}  S_1={S_1:.2f}  "
              f"M_cert={M_cert if M_cert is None else f'{M_cert:.5f}'}  "
              f"beats_MV={out['beats_MV']}")
    return out


def evaluate_K_directly(K_unnorm_fn, label: str = "kernel",
                        bochner_check_max: int = 200,
                        verbose: bool = True) -> dict:
    """Evaluate MV M_cert for a directly-defined K supported on [-DELTA, DELTA].

    K_unnorm_fn(x_array) returns K(x) for x in [-DELTA, DELTA].
    K MUST be even, >= 0, and have non-negative Fourier transform (Bochner).

    Same return signature as evaluate_phi.
    """
    # Theta grid for K is x = DELTA sin(theta)
    N_T = 4001
    th = np.linspace(-np.pi / 2, np.pi / 2, N_T)
    dth = th[1] - th[0]
    x_t = DELTA * np.sin(th)
    cos_t = np.cos(th)
    K_vals = np.asarray(K_unnorm_fn(x_t), dtype=float)
    if K_vals.shape != x_t.shape:
        raise ValueError("K_unnorm_fn returned wrong shape")
    Kmin = float(K_vals.min())
    if Kmin < -1e-8:
        return {"label": label, "M_cert": None,
                "reason": f"K negative on grid: min={Kmin}"}
    K_vals = np.maximum(K_vals, 0.0)

    K_dx_theta = K_vals * DELTA * cos_t   # dx = DELTA cos(theta) dtheta
    Z = np.trapz(K_dx_theta, dx=dth)
    if Z <= 0:
        return {"label": label, "M_cert": None, "reason": f"Z={Z}"}
    w_theta = K_dx_theta / Z

    def K_hat(xi: np.ndarray | float) -> np.ndarray:
        xi = np.atleast_1d(xi)
        cos_mat = np.cos(2 * np.pi * xi[:, None] * x_t[None, :])
        return cos_mat @ w_theta * dth

    ns = np.arange(1, bochner_check_max + 1)
    kn = K_hat(ns)
    bochner_min = float(kn.min())
    if bochner_min < -1e-10:
        return {"label": label, "M_cert": None,
                "reason": f"Bochner violated: min K_hat(j) = {bochner_min}"}

    k_1 = float(kn[0])

    # K_2 = int K(x)^2 dx (Parseval: = int K_hat^2 dxi)
    K_2 = float(np.trapz(K_vals ** 2 * DELTA * cos_t, dx=dth))

    # S_1 = sum a_j^2 / K_hat(j/u)
    if MV_COEFFS is None:
        return {"label": label, "M_cert": None, "reason": "no MV coeffs"}
    qp_xi = np.arange(1, N_QP + 1) / U
    kh_qp = K_hat(qp_xi)
    if np.any(kh_qp < 1e-20):
        return {"label": label, "M_cert": None,
                "reason": "K_hat(j/u) ~ 0 at some j"}
    S_1 = float(np.sum((MV_COEFFS ** 2) / kh_qp))

    M_cert = mv_master_M_cert(k_1, K_2, S_1)
    out = {
        "label": label,
        "k_1": k_1,
        "K_2": K_2,
        "S_1": S_1,
        "Z": float(Z),
        "K_min": Kmin,
        "bochner_min": bochner_min,
        "M_cert": M_cert,
        "beats_MV": (M_cert is not None and M_cert > 1.27481),
    }
    if verbose:
        print(f"[{label}] k_1={k_1:.5f}  K_2={K_2:.4f}  S_1={S_1:.2f}  "
              f"M_cert={M_cert if M_cert is None else f'{M_cert:.5f}'}  "
              f"beats_MV={out['beats_MV']}")
    return out


def reference_arcsine_value() -> dict:
    """Sanity: evaluate MV's arcsine = phi * phi where phi = arcsine density."""
    def phi(x):
        u = 2 * x / DELTA
        # Arcsine density on [-1, 1]: 1/(pi sqrt(1-u^2)), normalised so int = 1
        eps = 1e-10
        u_safe = np.clip(u, -1 + eps, 1 - eps)
        return 1.0 / (np.pi * np.sqrt(1.0 - u_safe ** 2)) * (2.0 / DELTA)
    return evaluate_phi(phi, "arcsine (MV)")


if __name__ == "__main__":
    print("Sanity: pure arcsine")
    print(json.dumps(reference_arcsine_value(), indent=2))
