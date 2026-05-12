"""Agent K25: 2-parameter asymmetric Beta-density kernel sweep.

Tests K = phi star phi (auto-correlation) for ASYMMETRIC phi:
    phi(x) = c * (1 - 2x/DELTA)^a * (1 + 2x/DELTA)^b   on x in [-DELTA/2, DELTA/2]
with a, b > -1.

For auto-correlation we have K_hat(xi) = |phi_hat(xi)|^2 >= 0 automatically,
and K is automatically even even when phi is asymmetric.

We compute phi_hat(xi) directly by adaptive quadrature over [-DELTA/2, DELTA/2]
(real & imaginary parts; |phi_hat|^2 is what we feed to the MV master).
This is the correct approach for asymmetric phi with integrable endpoint
singularities (no spatial-K interpolation artefacts).

Arcsine baseline is a = b = -0.5.  At a = b, phi is even so K_hat is real;
generally phi_hat is complex and |phi_hat|^2 is the right object.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import brentq

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

# Pull MV coeffs & master from helper
from _kernel_probe_helper import (  # noqa: E402
    MV_COEFFS,
    DELTA,
    U,
    N_QP,
    mv_master_M_cert,
)


# --- phi_hat via Gauss-Jacobi-like adaptive quadrature --------------------- #
# phi(x) = (1 - 2x/DELTA)^a (1 + 2x/DELTA)^b
# Sub u = 2x/DELTA, x = DELTA u / 2, dx = DELTA/2 du, u in [-1, 1]
# phi_hat(xi) = int_{-D/2}^{D/2} phi(x) e^{-2 pi i xi x} dx
#             = (DELTA/2) int_{-1}^{1} (1-u)^a (1+u)^b e^{-pi i xi D u} du
#
# The integrand has endpoint singularities (a, b > -1) so we use
# Gauss-Jacobi quadrature with weights w(u) = (1-u)^a (1+u)^b.
# numpy.polynomial.legendre.leggauss won't work; use scipy roots_jacobi.

from scipy.special import roots_jacobi  # noqa: E402


N_GJ = 800                  # Gauss-Jacobi nodes (converged at this resolution)


def make_phi_hat_evaluator(a: float, b: float):
    """Returns (Z, phi_hat_complex_fn) where Z = int phi dx (for normalisation).

    phi_hat_complex(xi)  returns phi_hat(xi) (complex array) WITHOUT
    normalisation by Z; user can divide by Z if normalised.
    """
    # Gauss-Jacobi nodes/weights for w(u) = (1-u)^a (1+u)^b
    nodes, weights = roots_jacobi(N_GJ, a, b)
    # Now int_{-1}^{1} f(u) (1-u)^a (1+u)^b du = sum_i w_i f(u_i)

    # Z (unnormalised) = (DELTA/2) * int (1-u)^a (1+u)^b du  = (DELTA/2) * sum w_i * 1
    Z = (DELTA / 2.0) * float(np.sum(weights))

    # x(u) = DELTA * u / 2
    x_nodes = DELTA * nodes / 2.0

    def phi_hat(xi):
        """phi_hat(xi) = int phi(x) e^{-2 pi i xi x} dx, complex."""
        xi = np.atleast_1d(np.asarray(xi, dtype=float))
        # phase: e^{-2 pi i xi x} = cos - i sin
        # For each xi, sum_k w_k e^{-2 pi i xi x_k} * (DELTA/2)
        phase = -2.0 * np.pi * xi[:, None] * x_nodes[None, :]
        real_part = (np.cos(phase) * weights[None, :]).sum(axis=1)
        imag_part = (np.sin(phase) * weights[None, :]).sum(axis=1)
        return (DELTA / 2.0) * (real_part + 1j * imag_part)

    return Z, phi_hat


# --- Compute K-statistics from phi_hat ------------------------------------- #
# K = phi star phi (auto-correlation), so K_hat(xi) = |phi_hat(xi)|^2.
# Need K to integrate to 1, i.e. K_hat(0) = 1.
# K_hat(0) = |phi_hat(0)|^2 = |Z|^2  (Z is real for our phi)
# So normalise phi -> phi / Z so K_hat(0) = 1.
# After normalisation K_hat(xi) = |phi_hat(xi) / Z|^2.

# Frequency grid for K_2 = int K(x)^2 dx = int K_hat(xi)^2 dxi (Parseval)
# But K = autocorr(phi), so K_hat = |phi_hat|^2, so K_hat^2 = |phi_hat|^4.
# K_2 = int |phi_hat(xi)/Z|^4 dxi.
#
# phi_hat decays fast enough for the integral to converge (since phi is
# compactly supported, phi_hat is entire with |phi_hat(xi)| ~ 1/|xi|^min(a,b)+1
# decay for large xi); same scale as helper: XI_MAX/DELTA, N_XI ~ 16001.

XI_MAX_OVER_DELTA = 200.0
N_XI = 16001


def compute_k1_K2_S1(a: float, b: float):
    """Compute k_1, K_2, S_1 for K = autocorr(phi_{a,b}) on [-DELTA, DELTA]."""
    Z, phi_hat = make_phi_hat_evaluator(a, b)

    # Bochner sanity: K_hat(j) for j = 1..200 should be >= 0 (it is by construction)
    ns = np.arange(1, 201, dtype=float)
    ph_ns = phi_hat(ns)
    kn = np.abs(ph_ns) ** 2 / Z ** 2     # = K_hat(n)
    bochner_min = float(kn.min())
    k_1 = float(kn[0])

    # K_2 = int K_hat(xi)^2 dxi = int |phi_hat/Z|^4 dxi (factor 2 for symmetry, but
    # phi_hat(-xi) = conjugate(phi_hat(xi)) so |phi_hat|^2 is even -> integrate over
    # [0, XI_MAX] then double).
    xi_grid = np.linspace(0.0, XI_MAX_OVER_DELTA / DELTA, N_XI)
    dxi = xi_grid[1] - xi_grid[0]

    # batch evaluation
    batch = 1000
    K2_pos = 0.0
    for s in range(0, N_XI, batch):
        chunk = xi_grid[s:s + batch]
        ph_c = phi_hat(chunk)
        K_hat_c = (np.abs(ph_c) ** 2) / (Z ** 2)
        K2_pos += np.sum(K_hat_c ** 2) * dxi
    K_2 = 2.0 * float(K2_pos)

    # S_1 = sum_j MV_COEFFS[j]^2 / K_hat(j/u)
    if MV_COEFFS is None:
        return None
    qp_xi = np.arange(1, N_QP + 1, dtype=float) / U
    ph_qp = phi_hat(qp_xi)
    kh_qp = np.abs(ph_qp) ** 2 / Z ** 2
    if np.any(kh_qp < 1e-20):
        return None
    S_1 = float(np.sum((MV_COEFFS ** 2) / kh_qp))

    return {
        "k_1": k_1,
        "K_2": K_2,
        "S_1": S_1,
        "Z": Z,
        "bochner_min": bochner_min,
    }


def evaluate_two_param(a: float, b: float, label: str, verbose: bool = True):
    try:
        stats = compute_k1_K2_S1(a, b)
    except Exception as e:
        return {"label": label, "a": a, "b": b, "M_cert": None,
                "reason": f"exception: {e}"}
    if stats is None:
        return {"label": label, "a": a, "b": b, "M_cert": None,
                "reason": "stats None"}
    M_cert = mv_master_M_cert(stats["k_1"], stats["K_2"], stats["S_1"])
    out = {
        "label": label,
        "a": a, "b": b,
        "k_1": stats["k_1"],
        "K_2": stats["K_2"],
        "S_1": stats["S_1"],
        "Z": stats["Z"],
        "bochner_min": stats["bochner_min"],
        "M_cert": M_cert,
        "beats_MV": (M_cert is not None and M_cert > 1.27481),
    }
    if verbose:
        m_str = f"{M_cert:.5f}" if M_cert is not None else "None"
        print(f"[{label}] a={a:+.3f} b={b:+.3f} "
              f"k1={stats['k_1']:.4f} K2={stats['K_2']:.3f} S1={stats['S_1']:.2f} "
              f"M_cert={m_str} beats_MV={out['beats_MV']}")
    return out


# --- Sweep ----------------------------------------------------------------- #

GRID = [-0.4, -0.25, -0.1, 0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0]


def run_sweep(grid_values, label_prefix="K25"):
    results = []
    # Baseline arcsine
    print("=== Baseline arcsine (a=b=-0.5) ===")
    res = evaluate_two_param(-0.5, -0.5, f"{label_prefix}_arcsine_a=-0.5_b=-0.5")
    results.append(res)

    print()
    print(f"=== 2D sweep, {len(grid_values)}^2 = {len(grid_values)**2} points ===")
    t0 = time.time()
    for ai, a in enumerate(grid_values):
        for bj, b in enumerate(grid_values):
            lab = f"{label_prefix}_a={a:+.3f}_b={b:+.3f}"
            res = evaluate_two_param(a, b, lab, verbose=True)
            results.append(res)
        print(f"-- row a={a} done [{time.time()-t0:.1f}s] --")
    return results


def main():
    results = run_sweep(GRID)

    valid = [r for r in results if r.get("M_cert") is not None]
    valid.sort(key=lambda r: r["M_cert"], reverse=True)

    print()
    print("=== Top 10 M_cert ===")
    for r in valid[:10]:
        print(f"  a={r['a']:+.3f}  b={r['b']:+.3f}  "
              f"M_cert={r['M_cert']:.5f}  "
              f"k1={r['k_1']:.4f}  K2={r['K_2']:.3f}  S1={r['S_1']:.2f}")

    if valid:
        top = valid[0]
        print()
        print(f"OPTIMUM (main grid): a*={top['a']}, b*={top['b']}, "
              f"M_cert={top['M_cert']:.6f}")
        print(f"  vs MV baseline 1.27481 -> beats_MV={top.get('beats_MV')}")
        print(f"  vs CS17 (invalid) 1.2802 -> beats_CS={top['M_cert'] > 1.2802}")

    out = {
        "delta": DELTA,
        "grid": GRID,
        "n_gj": N_GJ,
        "n_xi": N_XI,
        "results": results,
        "top10": valid[:10],
    }
    out_path = os.path.join(REPO, "_agent_K25_two_param_beta_result.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved {out_path}")

    return valid, out_path


def refine_around(a0, b0, out_path, width=0.1, n=7):
    print(f"\n=== Refined sweep around (a*={a0}, b*={b0}) ===")
    fine_a = sorted(set(round(a0 + (i - (n - 1) / 2) * (2 * width / (n - 1)), 4)
                        for i in range(n)))
    fine_b = sorted(set(round(b0 + (i - (n - 1) / 2) * (2 * width / (n - 1)), 4)
                        for i in range(n)))
    refined = []
    for a in fine_a:
        for b in fine_b:
            lab = f"K25R_a={a:+.4f}_b={b:+.4f}"
            refined.append(evaluate_two_param(a, b, lab, verbose=True))

    valid = [r for r in refined if r.get("M_cert") is not None]
    valid.sort(key=lambda r: r["M_cert"], reverse=True)

    print()
    print("=== Top 5 refined ===")
    for r in valid[:5]:
        print(f"  a={r['a']:+.4f}  b={r['b']:+.4f}  M_cert={r['M_cert']:.6f}")

    # Append to json
    with open(out_path, "r") as f:
        blob = json.load(f)
    blob["refined_results"] = refined
    blob["refined_top5"] = valid[:5]
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=2, default=str)
    return refined, valid


if __name__ == "__main__":
    main_top, out_path = main()
    if main_top:
        top = main_top[0]
        a_star, b_star = top["a"], top["b"]
        if not (abs(a_star + 0.5) < 1e-6 and abs(b_star + 0.5) < 1e-6):
            print("\nOptimum NOT at arcsine baseline -> refining grid.")
            refine_around(a_star, b_star, out_path)
        else:
            print("\nOptimum AT arcsine (a=b=-0.5) -> 2D local-max claim CONFIRMED.")
