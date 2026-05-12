"""Agent K20: Wendland C^{2k} compactly-supported PD kernels.

Tests the three Wendland dimension-1 RBF kernels (k = 1, 2, 3):
    phi_{1,1}(r) = (1-r)_+^3 * (3r + 1)              C^2
    phi_{1,2}(r) = (1-r)_+^5 * (8r^2 + 5r + 1)       C^4
    phi_{1,3}(r) = (1-r)_+^7 * (21r^3 + 19r^2 + 7r + 1)  C^6

With K(x) = c * phi_{1,k}(|x|/delta) on [-delta, delta], normalised so
int K = 1.  These are strictly PD on R^1 (Wendland 1995), so K_hat >= 0
automatically (no auto-conv step needed).

Goal: see whether any Wendland kernel beats the MV arcsine baseline
(M_cert ~= 1.2748) or the numerical arcsine value (~ 1.270).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import (  # noqa: E402
    DELTA,
    evaluate_K_directly,
)


def wendland_phi(r: np.ndarray, k: int) -> np.ndarray:
    """Wendland dimension-1 radial basis functions phi_{1,k} on [0, 1].

    These are the canonical W. Wendland (1995) compactly-supported PD
    radial functions on R^1.  Strictly positive definite on R^1.
    """
    r = np.clip(np.asarray(r, dtype=float), 0.0, 1.0)
    one_minus_r = 1.0 - r
    if k == 0:
        # box-tent; included only as a sanity reference (not C^k smooth)
        return one_minus_r
    if k == 1:
        # phi_{1,1}(r) = (1-r)^3 * (3r + 1)
        return (one_minus_r ** 3) * (3.0 * r + 1.0)
    if k == 2:
        # phi_{1,2}(r) = (1-r)^5 * (8 r^2 + 5 r + 1)
        return (one_minus_r ** 5) * (8.0 * r * r + 5.0 * r + 1.0)
    if k == 3:
        # phi_{1,3}(r) = (1-r)^7 * (21 r^3 + 19 r^2 + 7 r + 1)
        return (one_minus_r ** 7) * (21.0 * r * r * r + 19.0 * r * r + 7.0 * r + 1.0)
    raise ValueError(f"unsupported Wendland order k={k}")


def _wendland_mass(k: int) -> float:
    """Compute int_{-delta}^{delta} phi_{1,k}(|x|/delta) dx via fine quadrature.

    Used so we can pre-normalize K to int K = 1.
    """
    # x = delta sin(theta), dx = delta cos(theta) dtheta
    N_T = 8001
    th = np.linspace(-np.pi / 2, np.pi / 2, N_T)
    dth = th[1] - th[0]
    x_t = DELTA * np.sin(th)
    cos_t = np.cos(th)
    r = np.abs(x_t) / DELTA
    phi_vals = wendland_phi(r, k)
    return float(np.trapezoid(phi_vals * DELTA * cos_t, dx=dth))


def make_K_fn(k: int):
    """Return K(x) = phi_{1,k}(|x|/delta) / Z on [-delta, delta], normalised
    so int K = 1.

    IMPORTANT: evaluate_K_directly uses the *passed* K_vals (unnormalised
    inside its own scope) to compute K_2 = int K(x)^2 dx directly.  So we
    MUST pre-normalize K here so that K_2 matches the MV inequality.
    """
    Z = _wendland_mass(k)

    def K_fn(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        r = np.abs(x) / DELTA
        out = wendland_phi(r, k)
        out = np.where(np.abs(x) <= DELTA, out, 0.0)
        return out / Z

    return K_fn


def bochner_check(K_fn, j_max: int = 100) -> tuple[float, list[float]]:
    """Compute K_hat(j) for j = 1..j_max directly and return min and list."""
    # Build a normalised K on a fine x-grid (sin-substitution like helper)
    N_T = 4001
    th = np.linspace(-np.pi / 2, np.pi / 2, N_T)
    dth = th[1] - th[0]
    x_t = DELTA * np.sin(th)
    cos_t = np.cos(th)
    K_vals = K_fn(x_t)
    K_dx = K_vals * DELTA * cos_t  # dx = DELTA * cos(theta) dtheta
    Z = np.trapz(K_dx, dx=dth)
    if Z <= 0:
        return float("nan"), []
    w = K_dx / Z

    js = np.arange(1, j_max + 1)
    # K_hat(j) = int K(x) cos(2 pi j x) dx
    cos_mat = np.cos(2 * np.pi * js[:, None] * x_t[None, :])
    K_hat_js = cos_mat @ w * dth
    return float(K_hat_js.min()), K_hat_js.tolist()


def main():
    print("=" * 70)
    print("Agent K20: Wendland C^{2k} compactly-supported PD kernels")
    print(f"delta = {DELTA}, support [-delta, delta]")
    print("=" * 70)

    results = []

    for k in (1, 2, 3):
        label = f"Wendland_k={k}"
        K_fn = make_K_fn(k)

        # Direct Bochner check on j = 1..100
        bmin, K_hat_js = bochner_check(K_fn, j_max=100)
        n_neg = sum(1 for v in K_hat_js if v < 0)
        print(f"\n[{label}] Bochner check (j=1..100):")
        print(f"  min K_hat(j) = {bmin:.6e}    (negatives: {n_neg}/100)")
        print(f"  K_hat(1..5)  = {[f'{v:.4f}' for v in K_hat_js[:5]]}")
        print(f"  K_hat(96..100)= {[f'{v:.6e}' for v in K_hat_js[95:100]]}")

        # MV master inequality eval
        out = evaluate_K_directly(K_fn, label=label, bochner_check_max=200)
        out["wendland_k"] = k
        out["bochner_min_j100"] = bmin
        out["bochner_neg_count_j100"] = n_neg
        results.append(out)

    # Best numerical M_cert
    M_certs = [(r.get("M_cert"), r.get("wendland_k")) for r in results
               if r.get("M_cert") is not None]
    if M_certs:
        best_M, best_k = max(M_certs, key=lambda t: t[0])
    else:
        best_M, best_k = None, None

    summary = {
        "family": "Wendland",
        "delta": DELTA,
        "results": results,
        "best_M_cert": best_M,
        "best_k": best_k,
        "beats_MV_1_2748": (best_M is not None and best_M > 1.2748),
        "beats_arcsine_numerical_1_270": (best_M is not None and best_M > 1.270),
    }

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        k = r.get("wendland_k")
        M = r.get("M_cert")
        k1 = r.get("k_1")
        K2 = r.get("K_2")
        S1 = r.get("S_1")
        reason = r.get("reason", "")
        if M is None:
            print(f"  Wendland k={k}: M_cert=None  ({reason})")
        else:
            print(f"  Wendland k={k}: M_cert={M:.5f}  "
                  f"k_1={k1:.5f}  K_2={K2:.4f}  S_1={S1:.2f}")
    print(f"\nBest M_cert: {best_M}  (at k = {best_k})")
    print(f"Beats MV 1.2748?    {summary['beats_MV_1_2748']}")
    print(f"Beats arcsine 1.270? {summary['beats_arcsine_numerical_1_270']}")

    out_path = os.path.join(REPO, "_agent_K20_wendland_result.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
