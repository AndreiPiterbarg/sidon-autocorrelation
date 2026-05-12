"""Independent re-verification of the K26 multi-scale arcsine result.

Uses scipy.integrate.quad (adaptive) for K_2 and the bare MV master inequality
- no shared helpers, no shared xi grids. If we get a different answer, the
agent had a numerical issue. If we agree, the +0.005 is real.
"""
from __future__ import annotations

import numpy as np
from scipy.special import j0
from scipy.integrate import quad
from scipy.optimize import brentq

DELTA = 0.138
U = 0.5 + DELTA      # 0.638
N_QP = 119


def _mv_coeffs():
    """MV's 119 G-coefficients as floats. Independently load."""
    from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq
    return np.array([float(c.p) / float(c.q) for c in mv_coeffs_fmpq()])


MV_COEFFS = _mv_coeffs()
assert len(MV_COEFFS) == 119


def K_hat(xi, deltas, lambdas):
    """K_hat(xi) = sum_i lambdas[i] * J_0(pi * deltas[i] * xi)^2."""
    out = np.zeros_like(np.atleast_1d(xi), dtype=float)
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * xi) ** 2
    return out


def K_hat_scalar(xi, deltas, lambdas):
    """Scalar version for quad."""
    out = 0.0
    for lam, d in zip(lambdas, deltas):
        out += lam * j0(np.pi * d * xi) ** 2
    return out


def K_2_via_quad(deltas, lambdas, xi_split=10.0):
    """K_2 = ∫_{-inf}^{inf} K_hat(xi)^2 dxi via scipy.integrate.quad.

    K_hat is even, so K_2 = 2 ∫_0^inf K_hat^2 dxi.
    Use quad's adaptive integration on [0, xi_split] and [xi_split, inf).
    """
    def f(xi):
        return K_hat_scalar(xi, deltas, lambdas) ** 2
    # Split for accuracy: oscillatory tail decays like 1/xi^2 from J_0(z)^2 ~ 2/(pi z) cos(...)^2
    v1, _ = quad(f, 0.0, xi_split, limit=200, epsabs=1e-14, epsrel=1e-12)
    v2, _ = quad(f, xi_split, np.inf, limit=400, epsabs=1e-14, epsrel=1e-12)
    return 2.0 * (v1 + v2)


def mv_M_cert(k_1, K_2, S_1):
    """MV master inequality solver."""
    if K_2 <= 1 + 2 * k_1 * k_1:
        return None
    a = (4.0 / U) / S_1
    target = 2.0 / U + a
    rad2 = K_2 - 1 - 2 * k_1 * k_1

    def sup_R(M):
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

    f = lambda M: sup_R(M) - target
    try:
        return brentq(f, 1.0 + 1e-10, 2.0, xtol=1e-10)
    except Exception:
        return None


def evaluate(deltas, lambdas, label):
    deltas = np.asarray(deltas)
    lambdas = np.asarray(lambdas)
    assert abs(lambdas.sum() - 1.0) < 1e-10
    assert (lambdas >= 0).all()
    assert (deltas <= DELTA + 1e-12).all()

    k_1 = K_hat_scalar(1.0, deltas, lambdas)
    K_2 = K_2_via_quad(deltas, lambdas)
    qp_xi = np.arange(1, N_QP + 1) / U
    kh_qp = K_hat(qp_xi, deltas, lambdas)
    S_1 = float(np.sum(MV_COEFFS ** 2 / kh_qp))

    M_cert = mv_M_cert(k_1, K_2, S_1)
    print(f"[{label}]  k_1={k_1:.6f}  K_2={K_2:.6f}  S_1={S_1:.4f}  M_cert={M_cert:.6f}")
    return {"k_1": k_1, "K_2": K_2, "S_1": S_1, "M_cert": M_cert}


if __name__ == "__main__":
    print("INDEPENDENT VERIFICATION (scipy.quad adaptive integration)")
    print("=" * 70)

    print("\n--- Baseline: pure arcsine at DELTA=0.138 ---")
    r_base = evaluate([DELTA], [1.0], "pure-arcsine")

    print("\n--- Agent K26 best: (d1=0.138, d2=0.055, lam1=0.9312) ---")
    r_best = evaluate([DELTA, 0.055], [0.9312, 0.0688], "multi-scale-best")

    print("\n--- Agent K26 alternative: (d1=0.138, d2=0.0525, lam1=0.935) ---")
    r_alt = evaluate([DELTA, 0.0525], [0.935, 0.065], "multi-scale-alt")

    if r_best["M_cert"] is not None and r_base["M_cert"] is not None:
        print()
        print(f"IMPROVEMENT (best vs baseline): "
              f"{r_best['M_cert'] - r_base['M_cert']:+.6f}")
        print(f"NUMERICAL LB CANDIDATE: M_cert = {r_best['M_cert']:.5f}")
        print(f"Beats MV 1.27481?  {r_best['M_cert'] > 1.27481}")
        print(f"Beats CS17 1.2802? {r_best['M_cert'] > 1.2802}")

    # Refined scan around the agent's optimum (independent)
    print("\n--- Refined independent scan ---")
    best = (-np.inf, None, None)
    for d2 in [0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070]:
        for l1 in [0.90, 0.91, 0.92, 0.93, 0.94, 0.95]:
            r = evaluate([DELTA, d2], [l1, 1.0 - l1], f"d2={d2:.4f},l1={l1:.3f}")
            if r["M_cert"] is not None and r["M_cert"] > best[0]:
                best = (r["M_cert"], d2, l1)
    print(f"\nIndependent refined best: M_cert={best[0]:.6f} at d2={best[1]}, λ_1={best[2]}")
