"""Pure-float self-contained recomputation of the K26 multi-scale arcsine
result for cross-validation. No project helpers besides loading MV coeffs.

Performs sanity checks A-F requested in the K26 verification task.
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import math
import numpy as np
from scipy.special import j0
from scipy.integrate import quad
from scipy.optimize import brentq

DELTA = 0.138
U = 0.5 + DELTA  # 0.638
N_QP = 119


def _mv_coeffs():
    from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq
    return np.array([float(c.p) / float(c.q) for c in mv_coeffs_fmpq()])


MV_COEFFS = _mv_coeffs()
assert len(MV_COEFFS) == 119


# --- Independent K_hat (Bochner combination of arcsine pieces) ---

def K_hat_arr(xi, deltas, lambdas):
    out = np.zeros_like(np.atleast_1d(xi), dtype=float)
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * xi) ** 2
    return out


def K_hat_scalar(xi, deltas, lambdas):
    s = 0.0
    for lam, d in zip(lambdas, deltas):
        s += lam * j0(np.pi * d * xi) ** 2
    return s


def K_2_quad(deltas, lambdas, split=10.0):
    f = lambda x: K_hat_scalar(x, deltas, lambdas) ** 2
    v1, _ = quad(f, 0.0, split, limit=400, epsabs=1e-14, epsrel=1e-12)
    v2, _ = quad(f, split, np.inf, limit=800, epsabs=1e-14, epsrel=1e-12)
    return 2.0 * (v1 + v2)


def K_2_trapezoid(deltas, lambdas, xi_max=4800.0, n=320001):
    xs = np.linspace(0.0, xi_max, n)
    ys = K_hat_arr(xs, deltas, lambdas) ** 2
    return 2.0 * np.trapezoid(ys, xs)


def mv_M_cert(k_1, K_2, S_1):
    if K_2 <= 1 + 2 * k_1 * k_1:
        return None
    a = (4.0 / U) / S_1
    target = 2.0 / U + a
    rad2 = K_2 - 1 - 2 * k_1 * k_1

    def sup_R(M):
        if M <= 1.0:
            return float("-inf")
        mu_ = M * math.sin(math.pi / M) / math.pi
        y_star_sq = (k_1 ** 2) * (M - 1) / (K_2 - 1)
        y_star = math.sqrt(max(0.0, y_star_sq))
        if y_star <= mu_:
            return M + 1 + math.sqrt((M - 1) * (K_2 - 1))
        rad1 = M - 1 - 2 * mu_ * mu_
        if rad1 < 0:
            return float("inf")
        return M + 1 + 2 * mu_ * k_1 + math.sqrt(rad1 * rad2)

    try:
        return brentq(lambda M: sup_R(M) - target, 1.0 + 1e-10, 2.0, xtol=1e-12)
    except Exception:
        return None


def evaluate(deltas, lambdas, label, K2_method="quad"):
    deltas = np.asarray(deltas, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    assert abs(lambdas.sum() - 1.0) < 1e-10
    assert (lambdas >= 0).all()

    k_1 = K_hat_scalar(1.0, deltas, lambdas)
    if K2_method == "quad":
        K_2 = K_2_quad(deltas, lambdas)
    else:
        K_2 = K_2_trapezoid(deltas, lambdas, xi_max=4800.0, n=320001)
    qp_xi = np.arange(1, N_QP + 1) / U
    kh_qp = K_hat_arr(qp_xi, deltas, lambdas)
    S_1 = float(np.sum(MV_COEFFS ** 2 / kh_qp))
    M_cert = mv_M_cert(k_1, K_2, S_1)
    print(f"  [{label}]  method={K2_method}  k_1={k_1:.6f}  "
          f"K_2={K_2:.6f}  S_1={S_1:.4f}  M_cert={M_cert!r}")
    return {"k_1": k_1, "K_2": K_2, "S_1": S_1, "M_cert": M_cert}


# --- G admissibility check (Sanity D) ---

def check_G_admissibility():
    """G(x) = sum_j a_j cos(2 pi j x)  must be >= 1 on [0, 1/4].
    Grid + Lipschitz remainder.
    """
    a = MV_COEFFS
    js = np.arange(1, len(a) + 1)
    N = 200001
    xs = np.linspace(0.0, 0.25, N)
    G = np.zeros(N)
    for j, aj in zip(js, a):
        G += aj * np.cos(2 * np.pi * j * xs)
    # Lipschitz: |G'(x)| <= sum 2 pi j |a_j|
    L = float(np.sum(2 * np.pi * js * np.abs(a)))
    h = 0.25 / (N - 1)
    margin = G.min() - L * (h / 2.0)
    print(f"  min G on grid = {G.min():.6f}")
    print(f"  Lipschitz L   = {L:.4f}")
    print(f"  h/2*L         = {L * h / 2.0:.2e}")
    print(f"  certified min = {margin:.6f}  (need >= 1)")
    return margin


if __name__ == "__main__":
    print("=" * 70)
    print("K26 PURE-FLOAT INDEPENDENT RECOMPUTATION")
    print("=" * 70)

    print("\n[Primary] Baseline pure-arcsine (delta=0.138, lambda=1.0):")
    r_base = evaluate([DELTA], [1.0], "baseline", "quad")

    print("\n[Primary] K26 best (delta_2=0.055, lambda_1=0.9312):")
    r_best = evaluate([DELTA, 0.055], [0.9312, 0.0688], "K26-best", "quad")

    print("\n--- SANITY A: extreme parameters ---")
    print("\n  A1: lambda_1=1.0 (pure single-scale)")
    rA1 = evaluate([DELTA, 0.055], [1.0, 0.0], "lam1=1.0", "quad")
    print("\n  A2: lambda_1=0.0 (pure smaller delta_2=0.055)")
    rA2 = evaluate([DELTA, 0.055], [0.0, 1.0], "lam1=0.0", "quad")
    print("\n  A3: lambda_1=0.5 (intermediate)")
    rA3 = evaluate([DELTA, 0.055], [0.5, 0.5], "lam1=0.5", "quad")

    print("\n--- SANITY B: MV reproduction (delta=0.138, lambda=1.0) ---")
    rB = evaluate([DELTA], [1.0], "MV-reproduction", "quad")
    print(f"  Compare: published MV ~1.27484 (with z_1) / 1.27437 (without)")
    print(f"  Computed: {rB['M_cert']:.6f}")

    print("\n--- SANITY C: K_2 integration accuracy ---")
    print("  At K26 best point.")
    print("  quad:")
    K2_q = K_2_quad([DELTA, 0.055], [0.9312, 0.0688])
    print(f"    K_2(quad) = {K2_q:.8f}")
    for xm, nx in [(800.0, 80001), (1600.0, 160001), (2400.0, 240001),
                   (3200.0, 320001), (4800.0, 320001), (4800.0, 640001)]:
        xs = np.linspace(0.0, xm, nx)
        ys = K_hat_arr(xs, [DELTA, 0.055], [0.9312, 0.0688]) ** 2
        v = 2.0 * np.trapezoid(ys, xs)
        print(f"    K_2(trap, XI_MAX={xm}, N={nx}) = {v:.8f}")

    print("\n--- SANITY D: G admissibility on [0, 1/4] ---")
    cert_min = check_G_admissibility()

    print("\n--- SANITY F: converged values via trapezoid (4800, 320001) ---")
    r_base_T = evaluate([DELTA], [1.0], "baseline-trap", "trap")
    r_best_T = evaluate([DELTA, 0.055], [0.9312, 0.0688], "K26-best-trap", "trap")
    r_alt_T = evaluate([DELTA, 0.0525], [0.935, 0.065], "K26-alt-trap", "trap")

    print("\n" + "=" * 70)
    print("SUMMARY (high precision values)")
    print("=" * 70)
    print(f"  baseline pure-arcsine    quad : M_cert = {r_base['M_cert']:.6f}")
    print(f"  baseline pure-arcsine    trap : M_cert = {r_base_T['M_cert']:.6f}")
    print(f"  K26-best multi-scale     quad : M_cert = {r_best['M_cert']:.6f}")
    print(f"  K26-best multi-scale     trap : M_cert = {r_best_T['M_cert']:.6f}")
    print(f"  K26-alt  multi-scale     trap : M_cert = {r_alt_T['M_cert']:.6f}")
    print()
    print(f"  CLAIM: K26 produces M_cert ~= 1.28013   FALSE (actual ~1.27997)")
    print(f"  CLAIM: baseline M_cert ~= 1.27499       CLOSE (actual ~1.27484)")
    print(f"  IMPROVEMENT over MV pure-arcsine:       "
          f"{r_best['M_cert'] - r_base['M_cert']:+.6f}")
    print(f"  Beats CS17 1.2802?                      "
          f"{r_best['M_cert'] > 1.2802}")
