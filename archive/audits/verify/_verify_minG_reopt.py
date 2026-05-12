"""Rigorously verify min_{x in [0, 1/4]} G(x) for:
  1) MV's published 119 coefficients (sanity check; should give >= 1.0).
  2) The re-optimized G's 119 coefficients at (delta_1=0.138, delta_2=0.045,
     lambda_1=0.85), produced by _K26_full_sweep_reopt.solve_QP.

Uses delsarte_dual.grid_bound.G_min.min_G_lower_bound (Taylor B&B + arb).

Procedure:
  - For each G, convert a_j (float) -> fmpq with denominator 10**8 (matches
    the existing pipeline's encoding precision in coeffs.py: MV is given to
    8 mantissa digits).
  - Call min_G_lower_bound(coeffs, u=fmpq(638,1000), n_cells=4096, prec_bits=192)
  - Report encl.lower() as the rigorous lower bound on min G.
"""
from __future__ import annotations

import json
import sys
import math

from flint import fmpq, arb

# Local imports
from delsarte_dual.grid_bound.G_min import min_G_lower_bound
from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq, MV_U

# Re-opt module (sweep)
import _K26_full_sweep_reopt as reopt


U_FMPQ = fmpq(638, 1000)        # 0.638 = 0.5 + 0.138 (same as MV_U)
DENOM = 10 ** 8                  # 8-digit fixed-point encoding
N_CELLS = 4096
PREC_BITS = 192


def floats_to_fmpq_8digits(a_float_list) -> list[fmpq]:
    """Round each float a_j to the nearest p/10**8 and return as fmpq exactly."""
    out = []
    for v in a_float_list:
        # Round to nearest 1e-8
        p = int(round(float(v) * DENOM))
        out.append(fmpq(p, DENOM))
    return out


def arb_lower_to_float(a: arb) -> float:
    """Cast an arb's lower endpoint to float (rigorous since float ops on
    arb endpoints round outward in mpfr by construction inside arb)."""
    return float(a.lower())


def report_min_G(label: str, coeffs_fmpq, u_fmpq):
    encl, center = min_G_lower_bound(
        coeffs_fmpq, u_fmpq, n_cells=N_CELLS, prec_bits=PREC_BITS
    )
    lo = arb_lower_to_float(encl)
    up = float(encl.upper())
    # fmpq -> float via numerator/denominator
    center_f = float(center.p) / float(center.q)
    print(f"[{label}]")
    print(f"  n_coeffs = {len(coeffs_fmpq)}")
    print(f"  argmin-cell center = {center_f:.6f}")
    print(f"  enclosure lower (CERTIFIED min_G LB) = {lo:.10f}")
    print(f"  enclosure upper                      = {up:.10f}")
    print(f"  min_G >= 1?                          = {lo >= 1.0}")
    if lo < 1.0:
        print(f"  VIOLATION amount (1 - lo)            = {1.0 - lo:.6e}")
    return lo, up, center_f


def main():
    # ----------- MV sanity -----------
    print("=" * 78)
    print("MV's published G (119 coeffs) -- sanity check (expected min_G >= 1)")
    print("=" * 78)
    mv_coeffs = mv_coeffs_fmpq()
    mv_lo, mv_up, mv_center = report_min_G("MV-published", mv_coeffs, MV_U)
    print()

    # ----------- Re-opt G at best params -----------
    delta_1 = 0.138
    delta_2 = 0.045
    lambda_1 = 0.85
    print("=" * 78)
    print(f"Re-opt G via solve_QP at (d1={delta_1}, d2={delta_2}, l1={lambda_1})")
    print("=" * 78)
    a_opt, S1, mG_num, status = reopt.solve_QP([delta_1, delta_2],
                                               [lambda_1, 1.0 - lambda_1])
    if status != "ok":
        print(f"QP failed: {status}")
        sys.exit(1)
    print(f"  QP status   = {status}")
    print(f"  S_1 (num)   = {S1:.6f}")
    print(f"  min_G (num, on 5001-pt grid) = {mG_num:.10f}")
    # Convert to fmpq
    reopt_coeffs = floats_to_fmpq_8digits(a_opt)
    # Use the same u as the QP (delta_1 = 0.138, so u = 0.638)
    reopt_lo, reopt_up, reopt_center = report_min_G("Re-opt", reopt_coeffs, U_FMPQ)
    print()

    # ----------- Impact analysis -----------
    print("=" * 78)
    print("Impact on M_cert claim")
    print("=" * 78)
    # The gain formula:  a = (4/u) * (min_G)^2 / S_1
    # If min_G_rigorous < min_G_numerical, then a_rigorous < a_numerical, so
    # the target (= 2/u + a) shrinks => sup_R = target solves to smaller M.
    u_f = 0.638
    # k_1, K_2 at the best params from JSON
    with open("_K26_full_sweep_reopt_result.json") as f:
        js = json.load(f)
    bp = js["best_params"]
    k_1 = bp["k_1"]
    K_2 = bp["K_2"]
    S_1 = bp["S_1"]
    mG_orig = bp["min_G"]
    M_cert_orig = js["best_M_cert"]
    print(f"  From JSON: k_1={k_1:.6f}, K_2={K_2:.6f}, S_1={S_1:.6f}")
    print(f"  Numerical min_G (5001-pt grid) reported in JSON: {mG_orig:.10f}")
    print(f"  Numerical M_cert reported in JSON                 : {M_cert_orig:.6f}")
    print()

    # Now recompute M_cert using min_G_rigorous (the certified lower bound)
    M_with_rig = reopt.M_cert(k_1, K_2, S_1, reopt_lo)
    M_with_num = reopt.M_cert(k_1, K_2, S_1, mG_orig)
    print(f"  Recomputed M_cert with rigorous min_G = {reopt_lo:.10f}:")
    print(f"    M_cert = {M_with_rig if M_with_rig is not None else 'None'}")
    print(f"  Recomputed M_cert with numerical min_G = {mG_orig:.10f}:")
    print(f"    M_cert = {M_with_num if M_with_num is not None else 'None'}")
    if M_with_rig is not None and M_with_num is not None:
        print(f"  Delta M_cert (rig - num) = {M_with_rig - M_with_num:+.6f}")
        print(f"  vs MV's 1.27481: rig delta = {M_with_rig - 1.27481:+.6f}")

    # Save report
    out = {
        "MV": {
            "min_G_certified_lower": mv_lo,
            "min_G_enclosure_upper": mv_up,
            "argmin_cell_center": mv_center,
        },
        "ReOpt": {
            "delta_1": delta_1, "delta_2": delta_2, "lambda_1": lambda_1,
            "S_1_numerical": S1,
            "min_G_numerical": mG_num,
            "min_G_certified_lower": reopt_lo,
            "min_G_enclosure_upper": reopt_up,
            "argmin_cell_center": reopt_center,
        },
        "ImpactOnMCert": {
            "k_1": k_1, "K_2": K_2, "S_1": S_1,
            "min_G_used_in_JSON": mG_orig,
            "M_cert_in_JSON": M_cert_orig,
            "M_cert_with_rigorous_min_G": M_with_rig,
            "M_cert_with_numerical_min_G": M_with_num,
        },
        "n_cells": N_CELLS,
        "prec_bits": PREC_BITS,
    }
    with open("_verify_minG_reopt_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print()
    print("Wrote _verify_minG_reopt_result.json")


if __name__ == "__main__":
    main()
