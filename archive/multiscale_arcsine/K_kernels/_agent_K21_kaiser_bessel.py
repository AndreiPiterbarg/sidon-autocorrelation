"""Agent K21: Kaiser-Bessel kernel sweep for MV lower bound on C_{1a}.

Two variants:
  (A) auto-conv: K = phi * phi, with phi(x) = I_0(beta * sqrt(1 - (2x/DELTA)^2))
      on [-DELTA/2, DELTA/2].  K_hat = (phi_hat)^2 >= 0 automatically.
  (B) direct:    K(x) = I_0(beta * sqrt(1 - (x/DELTA)^2))     on [-DELTA, DELTA].
      Bochner verified numerically by helper.

beta sweep: {2, 4, 6, 8, 10, 12, 15, 20}.  beta = 0 reduces to a box (sanity).

If best M_cert beats 1.270 numerical, do a refined beta sweep around the optimum.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.special import i0  # modified Bessel I_0

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import (  # noqa: E402
    DELTA,
    evaluate_K_directly,
    evaluate_phi,
)


def make_phi_kb(beta: float):
    """Auto-conv variant: phi(x) = I_0(beta * sqrt(1 - (2x/DELTA)^2)) on [-DELTA/2, DELTA/2].

    Outside this support, returns 0.  Returns unnormalised callable for evaluate_phi.
    """
    def phi(x):
        x = np.asarray(x, dtype=float)
        u = 2.0 * x / DELTA
        mask = np.abs(u) <= 1.0
        out = np.zeros_like(x)
        if beta == 0.0:
            out[mask] = 1.0
            return out
        arg = np.sqrt(np.maximum(0.0, 1.0 - u[mask] ** 2))
        out[mask] = i0(beta * arg)
        return out
    return phi


def make_K_kb_direct(beta: float):
    """Direct variant: K(x) = I_0(beta * sqrt(1 - (x/DELTA)^2)) on [-DELTA, DELTA]."""
    def K(x):
        x = np.asarray(x, dtype=float)
        u = x / DELTA
        mask = np.abs(u) <= 1.0
        out = np.zeros_like(x)
        if beta == 0.0:
            out[mask] = 1.0
            return out
        arg = np.sqrt(np.maximum(0.0, 1.0 - u[mask] ** 2))
        out[mask] = i0(beta * arg)
        return out
    return K


def sweep(beta_values):
    variants = []
    best = {"M_cert": -np.inf, "beta": None, "variant": None}
    for beta in beta_values:
        # (A) auto-conv
        res_a = evaluate_phi(make_phi_kb(beta), f"KB-autoconv beta={beta}")
        rec_a = {"variant": "auto-conv", "beta": float(beta), **{k: v for k, v in res_a.items() if k != "label"}}
        variants.append(rec_a)
        if res_a.get("M_cert") is not None and res_a["M_cert"] > best["M_cert"]:
            best = {"M_cert": float(res_a["M_cert"]), "beta": float(beta), "variant": "auto-conv"}

        # (B) direct
        res_b = evaluate_K_directly(make_K_kb_direct(beta), f"KB-direct  beta={beta}")
        rec_b = {"variant": "direct", "beta": float(beta), **{k: v for k, v in res_b.items() if k != "label"}}
        variants.append(rec_b)
        if res_b.get("M_cert") is not None and res_b["M_cert"] > best["M_cert"]:
            best = {"M_cert": float(res_b["M_cert"]), "beta": float(beta), "variant": "direct"}

    return variants, best


def main():
    BETAS = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    print("=" * 78)
    print("Agent K21: Kaiser-Bessel kernel sweep")
    print("=" * 78)
    variants, best = sweep(BETAS)

    print("\n--- Coarse sweep summary ---")
    print(f"Best M_cert = {best['M_cert']:.5f} at beta={best['beta']}, variant={best['variant']}")
    print(f"Beats MV (1.27481): {best['M_cert'] > 1.27481}")
    print(f"Beats 1.270 numerical: {best['M_cert'] > 1.270}")

    refined = None
    if best["M_cert"] > 1.270:
        # Refined sweep ±2 around best beta with step 0.25
        b0 = best["beta"]
        lo = max(0.5, b0 - 2.0)
        hi = b0 + 2.0
        refined_betas = list(np.arange(lo, hi + 1e-9, 0.25))
        print(f"\n--- Refined sweep around beta={b0}: {refined_betas} ---")
        refined_variants, refined_best = sweep(refined_betas)
        refined = {"variants": refined_variants, "best": refined_best}
        print(f"\nRefined best: M_cert={refined_best['M_cert']:.5f} at beta={refined_best['beta']}, variant={refined_best['variant']}")
        if refined_best["M_cert"] > best["M_cert"]:
            best = refined_best

    out = {
        "family": "Kaiser-Bessel",
        "delta": DELTA,
        "variants": variants,
        "best_M_cert": float(best["M_cert"]),
        "best_beta": float(best["beta"]) if best["beta"] is not None else None,
        "best_variant": best["variant"],
        "beats_MV_1_2748": bool(best["M_cert"] > 1.27481),
        "beats_127_numerical": bool(best["M_cert"] > 1.270),
        "refined": refined,
    }
    outpath = os.path.join(REPO, "_agent_K21_kaiser_bessel_result.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {outpath}")

    return out


if __name__ == "__main__":
    main()
