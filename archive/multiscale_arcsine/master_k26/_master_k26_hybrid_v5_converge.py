"""Convergence audit for v4 best result.

v4 reports: l1=0.925, d2=0.057, eps=0.005, bspline3@dC=0.01  ->  M=1.28057

Concern: with dC=0.01 the bspline3 component has K_hat = sinc^4(pi*0.005*xi),
which decays only as 1/xi^4. So K_2 integral may be sensitive to XI_MAX.
Similarly tail tail at xi=600 might leak: sinc^4(0.005*pi*600) =
sinc^4(3pi) = (sin(3pi)/3pi)^4 = 0 exactly. Hmm. Let's just check.

Also: at dC=0.01 the kernel itself is supported on [-0.01, 0.01] which IS
inside [-DELTA, DELTA]; that's legal. The K_hat is very wide in xi, so K_2
contribution from THIS family alone scales like 1/dC (typical for bandwidth).
Worry: K_2 of the mixture cross-term gets weird.

We re-run with XI_MAX = 1500 and N_XI = 100001 to see if M_cert moves.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.special import j0

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import _master_k26_hybrid as base  # noqa: E402
from _kernel_probe_helper import (  # noqa: E402
    DELTA, MV_COEFFS, N_QP, U, mv_master_M_cert,
)


def make_grid(xi_max, n_xi):
    xi = np.linspace(0.0, xi_max, n_xi)
    dxi = xi[1] - xi[0]
    return xi, dxi


def evaluate_with_grid(components, xi_grid, dxi):
    Kh = base.K_hat_mixture(xi_grid, components)
    K_2 = float(2.0 * np.trapezoid(Kh ** 2, dx=dxi))
    k_1 = float(base.K_hat_mixture(np.array([1.0]), components)[0])
    qp_xi = np.arange(1, N_QP + 1) / U
    kh_qp = base.K_hat_mixture(qp_xi, components)
    if np.any(kh_qp < 1e-18):
        return None, k_1, K_2, None
    S_1 = float(np.sum((MV_COEFFS ** 2) / kh_qp))
    M = mv_master_M_cert(k_1, K_2, S_1)
    return M, k_1, K_2, S_1


def main():
    print("=" * 78)
    print("Convergence audit of v4 best.")
    print("=" * 78)

    config = [("arcsine", DELTA, 0.925),
              ("arcsine", 0.057, 1.0 - 0.925 - 0.005),
              ("bspline3", 0.01, 0.005)]
    print(f"Config: {config}")

    K26 = [("arcsine", DELTA, 0.9312), ("arcsine", 0.055, 0.0688)]
    print(f"K26 base: {K26}")

    grids = [
        (600.0, 40001),
        (1000.0, 60001),
        (1500.0, 100001),
        (2500.0, 150001),
    ]
    print("\n--- K26 base ---")
    for xi_max, n_xi in grids:
        xi, dxi = make_grid(xi_max, n_xi)
        M, k1, K2, S1 = evaluate_with_grid(K26, xi, dxi)
        print(f"  XI_MAX={xi_max}, N_XI={n_xi}: M={M:.8f}  k1={k1:.6f}  "
              f"K_2={K2:.6f}  S_1={S1:.4f}")

    print("\n--- v4 best (arcsine x2 + bspline3@0.01) ---")
    for xi_max, n_xi in grids:
        xi, dxi = make_grid(xi_max, n_xi)
        M, k1, K2, S1 = evaluate_with_grid(config, xi, dxi)
        print(f"  XI_MAX={xi_max}, N_XI={n_xi}: M={M:.8f}  k1={k1:.6f}  "
              f"K_2={K2:.6f}  S_1={S1:.4f}")

    # Also: pure-arcsine multi-scale optimum re-evaluated at high precision
    # to see the *converged* K26 number.
    print("\n--- Pure arcsine x2 reoptimized at converged grid ---")
    best = {"M_cert": -np.inf}
    xi, dxi = make_grid(2500.0, 150001)
    for l1 in np.linspace(0.91, 0.95, 9):
        for d2 in np.linspace(0.045, 0.07, 11):
            comps = [("arcsine", DELTA, l1),
                     ("arcsine", float(d2), 1.0 - l1)]
            M, k1, K2, S1 = evaluate_with_grid(comps, xi, dxi)
            if M is not None and M > best.get("M_cert", -np.inf):
                best = {"M_cert": float(M), "l1": float(l1),
                        "d2": float(d2)}
    print(f"  Best pure arcsine x2 (converged): {best}")

    # And re-optimize the hybrid at high precision.
    print("\n--- Hybrid (arcsine x2 + bspline3) reoptimized at converged grid ---")
    best_h = {"M_cert": -np.inf}
    for l1 in np.linspace(0.91, 0.94, 7):
        for d2 in np.linspace(0.05, 0.07, 9):
            for eps in [0.0, 0.002, 0.005, 0.008, 0.01]:
                for dC in [0.005, 0.01, 0.015, 0.02, 0.03]:
                    l2 = 1.0 - l1 - eps
                    if l2 < 0.001:
                        continue
                    if eps > 0:
                        comps = [("arcsine", DELTA, float(l1)),
                                 ("arcsine", float(d2), float(l2)),
                                 ("bspline3", float(dC), float(eps))]
                    else:
                        comps = [("arcsine", DELTA, float(l1)),
                                 ("arcsine", float(d2), 1.0 - float(l1))]
                    M, k1, K2, S1 = evaluate_with_grid(comps, xi, dxi)
                    if M is not None and M > best_h.get("M_cert", -np.inf):
                        best_h = {"M_cert": float(M), "l1": float(l1),
                                  "d2": float(d2), "eps": float(eps),
                                  "dC": float(dC)}
    print(f"  Best hybrid (converged): {best_h}")

    final_gain = best_h["M_cert"] - best["M_cert"]
    print(f"\nCONVERGED HYBRID vs CONVERGED PURE ARCSINE:")
    print(f"  pure arcsine x2: M={best['M_cert']:.6f}")
    print(f"  hybrid (x2+bspline3): M={best_h['M_cert']:.6f}")
    print(f"  Gain: {final_gain:+.6f}")
    print(f"  Beats K26 (converged): {final_gain > 1e-6}")

    out = {
        "v4_config": [(c[0], c[1], c[2]) for c in config],
        "converged_pure_arcsine_best": best,
        "converged_hybrid_best": best_h,
        "genuine_gain_over_pure_arcsine": float(final_gain),
        "beats_K26_at_converged_grid": final_gain > 1e-6,
    }
    outpath = os.path.join(REPO, "_master_k26_hybrid_v5_converge.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else str(x))
    print(f"\nWrote {outpath}")


if __name__ == "__main__":
    main()
