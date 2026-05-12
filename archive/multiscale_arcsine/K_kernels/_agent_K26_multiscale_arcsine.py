"""Agent K26: Multi-scale convex combinations of arcsine kernels.

K(x) = sum_i lambda_i * K_arcsine(x; delta_i)
where K_arcsine(x; delta_i) is the MV arcsine auto-conv supported on
[-delta_i, delta_i], properly normalised (int = 1).

Closed form (Sonine):  K_arcsine(x; delta_i) =
  (2/(pi^2 * delta_i)) * EllipticK(sqrt(1 - (x/delta_i)^2))   for |x| < delta_i.
This has a logarithmic singularity at x = 0 but is in L^2 and L^1.

Admissibility:
  - K >= 0 even: each component is, convex combo preserves.
  - supp(K) subset [-DELTA, DELTA] iff max(delta_i) <= DELTA.
  - int K = sum lambda_i * int K_arcsine = sum lambda_i = 1 (each component
    has int = 1 since K_hat_i(0) = J_0(0)^2 = 1).
  - K_hat = sum lambda_i * J_0(pi delta_i xi)^2 >= 0 (Bochner OK).

Mathematical motivation:
  K_2 = sum_{ij} lambda_i lambda_j C_{ij}, C_{ij} = int J_0^2(pi delta_i xi)
       J_0^2(pi delta_j xi) dxi.  C_{ij} <= sqrt(C_{ii} C_{jj}) so K_2 can be
  strictly less than the convex combo of C_{ii}.  Since M_cert depends
  monotonically on K_2 (smaller K_2 helps), a multi-scale combo MAY beat the
  pure DELTA arcsine.

We carry out evaluation directly via a *custom* K_hat-side pipeline (because
evaluate_K_directly tries to integrate K(x)^2 dx in theta coords, which is
unstable for the log-singular arcsine kernel).  The custom pipeline mirrors
helper.evaluate_phi but for the multi-scale K_hat formula.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.special import j0  # Bessel J_0

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

from _kernel_probe_helper import (  # noqa: E402
    DELTA,
    MV_COEFFS,
    N_QP,
    U,
    mv_master_M_cert,
)


# Custom K_hat-side evaluator for multi-scale arcsine kernels.
# K_hat(xi) = sum_i lambda_i * J_0(pi * delta_i * xi)^2   >= 0.
# int K = K_hat(0) = sum lambda_i * 1 = 1 (assuming sum lambda_i = 1).
# k_1 = K_hat(1).
# K_2 = int K_hat(xi)^2 dxi  (Parseval, finite since each J_0^2 decays like 1/z).
# S_1 = sum_{j=1..119} a_j^2 / K_hat(j/U).
#
# We use a dense xi grid for K_2 (the integrand decays like 1/xi^2 for large xi).

# Xi grid for K_2 integration.
N_XI = 40001
XI_MAX = 600.0   # K_hat^2 ~ 1/(pi^2 delta^2 xi^2) for large xi; tail past XI_MAX ~ 1/xi
_XI = np.linspace(0.0, XI_MAX, N_XI)
_DXI = _XI[1] - _XI[0]


def K_hat_multiscale(xi, deltas, lambdas):
    """K_hat(xi) = sum_i lambdas[i] * J_0(pi * deltas[i] * xi)^2."""
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi)
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * xi) ** 2
    return out


def evaluate_multiscale(deltas, lambdas, label, verbose=True):
    """Compute M_cert for K = sum lambda_i K_arcsine(. ; delta_i)."""
    deltas = np.asarray(deltas, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    if not np.isclose(lambdas.sum(), 1.0):
        raise ValueError(f"lambdas must sum to 1, got {lambdas.sum()}")
    if np.any(lambdas < -1e-12):
        raise ValueError(f"lambdas must be non-negative, got {lambdas}")
    if np.any(deltas > DELTA + 1e-12):
        raise ValueError(f"all delta_i must be <= DELTA={DELTA}, got {deltas}")

    # k_1 = K_hat(1)
    k_1 = float(K_hat_multiscale(np.array([1.0]), deltas, lambdas)[0])

    # K_2 = int K_hat^2 dxi = 2 * int_0^infty K_hat^2 dxi (K_hat is even via cos)
    Kh = K_hat_multiscale(_XI, deltas, lambdas)
    K_2_pos = np.trapezoid(Kh ** 2, dx=_DXI)
    # Estimate tail past XI_MAX: K_hat(xi)^2 ~ (sum lambda_i * (2/(pi^2 delta_i xi)))^2
    # asymptotic of J_0(z)^2 ~ (1/(pi z))(1 + cos(2z - pi/2)); time-average -> 1/(pi z).
    # For tail estimate: just rely on XI_MAX large enough.  Check tail value:
    tail_val = Kh[-10:].max() ** 2
    K_2 = float(2.0 * K_2_pos)

    # S_1 = sum a_j^2 / K_hat(j/U)
    if MV_COEFFS is None:
        return {"label": label, "M_cert": None, "reason": "no MV coeffs"}
    qp_xi = np.arange(1, N_QP + 1) / U
    kh_qp = K_hat_multiscale(qp_xi, deltas, lambdas)
    if np.any(kh_qp < 1e-20):
        return {"label": label, "M_cert": None,
                "reason": "K_hat(j/U) ~ 0 at some j"}
    S_1 = float(np.sum((MV_COEFFS ** 2) / kh_qp))

    M_cert = mv_master_M_cert(k_1, K_2, S_1)
    out = {
        "label": label,
        "k_1": k_1,
        "K_2": K_2,
        "S_1": S_1,
        "tail_val_at_XI_MAX_sq": float(tail_val),
        "M_cert": M_cert,
        "beats_MV": (M_cert is not None and M_cert > 1.27481),
        "beats_127": (M_cert is not None and M_cert > 1.270),
    }
    if verbose:
        Mtxt = f"{M_cert:.5f}" if M_cert is not None else "None"
        print(f"[{label}] k_1={k_1:.5f}  K_2={K_2:.4f}  S_1={S_1:.2f}  "
              f"M_cert={Mtxt}  beats_MV={out['beats_MV']}")
    return out


def sweep_two_component(delta_1, delta_2_list, lambda_1_list):
    results = []
    best = {"M_cert": -np.inf, "delta_2": None, "lambda_1": None}
    for delta_2 in delta_2_list:
        for lam in lambda_1_list:
            label = f"d1={delta_1:.4f},d2={delta_2:.4f},l1={lam:.2f}"
            res = evaluate_multiscale([delta_1, delta_2], [lam, 1.0 - lam], label)
            rec = {
                "delta_1": float(delta_1),
                "delta_2": float(delta_2),
                "lambda_1": float(lam),
                **{k: v for k, v in res.items() if k != "label"},
            }
            results.append(rec)
            M = res.get("M_cert")
            if M is not None and M > best["M_cert"]:
                best = {
                    "M_cert": float(M),
                    "delta_2": float(delta_2),
                    "lambda_1": float(lam),
                }
    return results, best


def main():
    print("=" * 78)
    print("Agent K26: Multi-scale convex-combination of arcsine kernels")
    print(f"(custom K_hat pipeline; XI_MAX={XI_MAX}, N_XI={N_XI})")
    print("=" * 78)

    # --- Sanity: lambda_1 = 1.0 (pure delta_1 = DELTA) should give ~1.270. ---
    print("\n--- Sanity 1: lambda_1 = 1.0 (pure delta_1 = DELTA = 0.138) ---")
    sanity = evaluate_multiscale([DELTA, 0.05], [1.0, 0.0], "sanity-pure-DELTA")
    print(f"Sanity (pure DELTA) M_cert = {sanity.get('M_cert')}  (expect ~1.270)")

    # --- Sanity 2: lambda_1 = 0.0 (pure smaller delta_2). ---
    print("\n--- Sanity 2: lambda_1 = 0.0 (pure delta_2 < DELTA) ---")
    sanity_small = evaluate_multiscale([DELTA, 0.10], [0.0, 1.0], "sanity-pure-0.10")
    print(f"Sanity (pure 0.10) M_cert = {sanity_small.get('M_cert')}")

    # --- Coarse two-component sweep. ---
    delta_2_list = [0.05, 0.07, 0.09, 0.10, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135]
    lambda_1_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\n--- Coarse sweep: {len(delta_2_list)} x {len(lambda_1_list)} = "
          f"{len(delta_2_list) * len(lambda_1_list)} points ---")
    coarse_results, coarse_best = sweep_two_component(
        DELTA, delta_2_list, lambda_1_list
    )

    print("\n--- Coarse summary ---")
    print(f"Best M_cert = {coarse_best['M_cert']:.5f} at "
          f"delta_2={coarse_best['delta_2']}, lambda_1={coarse_best['lambda_1']}")
    sanity_M = sanity.get("M_cert")
    print(f"Baseline arcsine M_cert = {sanity_M}")
    print(f"Beats MV (1.27481): {coarse_best['M_cert'] > 1.27481}")
    print(f"Beats 1.270 numerical: {coarse_best['M_cert'] > 1.270}")

    # --- Refined if any improvement. ---
    refined_results, refined_best = None, None
    # Refine if best > sanity_M (any improvement, even sub-1.270, deserves a look).
    do_refine = (coarse_best["M_cert"] is not None and sanity_M is not None
                 and coarse_best["M_cert"] > sanity_M + 1e-5)
    if do_refine:
        d2_0 = coarse_best["delta_2"]
        l1_0 = coarse_best["lambda_1"]
        d2_lo = max(0.04, d2_0 - 0.02)
        d2_hi = min(DELTA - 1e-4, d2_0 + 0.02)
        d2_refined = list(np.round(np.linspace(d2_lo, d2_hi, 9), 5))
        l1_lo = max(0.05, l1_0 - 0.1)
        l1_hi = min(0.95, l1_0 + 0.1)
        l1_refined = list(np.round(np.linspace(l1_lo, l1_hi, 9), 4))
        print(f"\n--- Refined sweep around (delta_2={d2_0}, lambda_1={l1_0}) ---")
        print(f"  delta_2 in {d2_refined}")
        print(f"  lambda_1 in {l1_refined}")
        refined_results, refined_best = sweep_two_component(
            DELTA, d2_refined, l1_refined
        )
        print(f"\nRefined best: M_cert={refined_best['M_cert']:.5f} at "
              f"delta_2={refined_best['delta_2']}, lambda_1={refined_best['lambda_1']}")

    # --- Optional 3-component if 2-component shows improvement over baseline. ---
    three_component = None
    final_best = coarse_best if refined_best is None else (
        refined_best if refined_best["M_cert"] > coarse_best["M_cert"]
        else coarse_best
    )
    if (sanity_M is not None and final_best["M_cert"] > sanity_M + 1e-5):
        print("\n--- 3-component exploration ---")
        d2 = final_best["delta_2"]
        l1 = final_best["lambda_1"]
        # Add a third intermediate scale.
        d3_list = [d for d in [0.06, 0.08, 0.10, 0.115, 0.125] if abs(d - d2) > 0.005]
        l3_list = [0.05, 0.1, 0.15, 0.2, 0.3]
        three_results = []
        three_best = {"M_cert": final_best["M_cert"], "delta_3": None, "lambda_3": None}
        for d3 in d3_list:
            for l3 in l3_list:
                # Split (1 - l1) between delta_2 and delta_3.
                l2 = (1.0 - l1) - l3
                if l2 < 0.0:
                    continue
                label = f"3c:d1={DELTA:.4f},d2={d2:.4f},d3={d3:.4f},l1={l1:.2f},l2={l2:.2f},l3={l3:.2f}"
                res = evaluate_multiscale([DELTA, d2, d3], [l1, l2, l3], label)
                rec = {
                    "delta_1": float(DELTA),
                    "delta_2": float(d2),
                    "delta_3": float(d3),
                    "lambda_1": float(l1),
                    "lambda_2": float(l2),
                    "lambda_3": float(l3),
                    **{k: v for k, v in res.items() if k != "label"},
                }
                three_results.append(rec)
                M = res.get("M_cert")
                if M is not None and M > three_best["M_cert"]:
                    three_best = {"M_cert": float(M), "delta_3": float(d3),
                                  "lambda_3": float(l3),
                                  "delta_2": float(d2), "lambda_1": float(l1)}
        three_component = {"results": three_results, "best": three_best}
        print(f"3-component best M_cert = {three_best['M_cert']:.5f}")
        if three_best["M_cert"] > final_best["M_cert"]:
            final_best = three_best

    # --- Save. ---
    out = {
        "family": "multi-scale-arcsine-convex-combo",
        "delta_1": float(DELTA),
        "XI_MAX": XI_MAX,
        "N_XI": N_XI,
        "delta_2_list_coarse": delta_2_list,
        "lambda_1_list_coarse": lambda_1_list,
        "sanity_pure_DELTA": {k: v for k, v in sanity.items() if k != "label"},
        "sanity_pure_smaller": {k: v for k, v in sanity_small.items() if k != "label"},
        "coarse_results": coarse_results,
        "coarse_best": coarse_best,
        "refined_results": refined_results,
        "refined_best": refined_best,
        "three_component": three_component,
        "final_best_M_cert": float(final_best["M_cert"]),
        "baseline_M_cert": float(sanity_M) if sanity_M is not None else None,
        "improvement_over_baseline": (
            float(final_best["M_cert"] - sanity_M) if sanity_M is not None else None
        ),
        "beats_MV_1_2748": bool(final_best["M_cert"] > 1.27481),
        "beats_127_numerical": bool(final_best["M_cert"] > 1.270),
    }
    outpath = os.path.join(REPO, "_agent_K26_multiscale_arcsine_result.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"\nWrote {outpath}")
    print(f"\nFINAL BEST M_cert = {final_best['M_cert']:.5f}  "
          f"(baseline arcsine = {sanity_M})")
    return out


if __name__ == "__main__":
    main()
