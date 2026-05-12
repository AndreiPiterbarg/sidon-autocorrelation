"""Agent M1: 3-component multi-scale arcsine kernel sweep.

K(x) = lam_1 * K_arc(d_1=0.138) + lam_2 * K_arc(d_2) + lam_3 * K_arc(d_3)
with lam_3 = 1 - lam_1 - lam_2, then QP-re-opt the 119 G-coeffs.

Phase 1: grid over (d_2, d_3, lam_1, lam_2).
Phase 2: scipy.differential_evolution refine around the best.
Phase 3: report S_1 drop and frequency-fill diagnostic.
"""
from __future__ import annotations

import json
import math
import time
import numpy as np
from scipy.special import j0
from scipy.optimize import differential_evolution

# Reuse the existing pipeline (uses MOSEK via cvxpy, MV's 119 G-coeffs implicit)
from _K26_full_sweep_reopt import (
    K_hat_ms,
    solve_QP,
    K_2_quad,
    M_cert,
    eval_at,
    DELTA,
    U,
    N_QP,
)

D1_FIXED = 0.138  # = DELTA
TWO_COMP_BASELINE = 1.29005  # numerical M_cert at d1=0.138, d2=0.045, lam1=0.85
MV_BASELINE = 1.27481


# ---------------------------------------------------------------------------
# Phase 1: coarse grid
# ---------------------------------------------------------------------------
def phase1_grid():
    delta2_vals = [0.015, 0.025, 0.035, 0.045, 0.055, 0.075, 0.10]
    delta3_vals = [0.015, 0.025, 0.035, 0.045, 0.055, 0.075, 0.10]
    lam1_vals = [0.75, 0.80, 0.85]
    lam2_vals = [0.05, 0.10, 0.12]

    results = []
    best = {"M_cert": -np.inf, "params": None}
    print("Phase 1: coarse grid")
    print(f"  d_1 fixed = {D1_FIXED}")
    print(f"  d_2 in {delta2_vals}")
    print(f"  d_3 in {delta3_vals}")
    print(f"  lam_1 in {lam1_vals}")
    print(f"  lam_2 in {lam2_vals}")
    n_total = (
        len(delta2_vals) * len(delta3_vals) * len(lam1_vals) * len(lam2_vals)
    )
    print(f"  total = {n_total} points")
    print("=" * 90)
    print(
        f"{'d_2':>7} {'d_3':>7} {'lam1':>6} {'lam2':>6} {'lam3':>6} "
        f"{'k_1':>8} {'K_2':>7} {'S_1':>9} {'min_G':>7} {'M_cert':>9}"
    )
    print("-" * 90)

    t0 = time.time()
    count = 0
    for d2 in delta2_vals:
        for d3 in delta3_vals:
            # Avoid degeneracy: d_3 != d_2 (else just a 2-comp with merged lam)
            if abs(d2 - d3) < 1e-6:
                continue
            for l1 in lam1_vals:
                for l2 in lam2_vals:
                    l3 = 1.0 - l1 - l2
                    if l3 < 0.01 or l3 > 0.5:
                        continue
                    count += 1
                    deltas = [D1_FIXED, d2, d3]
                    lambdas = [l1, l2, l3]
                    r = eval_at(deltas, lambdas)
                    Mc = r.get("M_cert")
                    if Mc is None:
                        continue
                    rec = {
                        "delta_1": D1_FIXED,
                        "delta_2": d2,
                        "delta_3": d3,
                        "lambda_1": l1,
                        "lambda_2": l2,
                        "lambda_3": l3,
                        "k_1": r["k_1"],
                        "K_2": r["K_2"],
                        "S_1": r["S_1"],
                        "min_G": r["min_G"],
                        "M_cert": Mc,
                    }
                    results.append(rec)
                    if Mc > best["M_cert"]:
                        best = {"M_cert": float(Mc), "params": rec}
                        marker = " *"
                    else:
                        marker = ""
                    print(
                        f"{d2:>7.4f} {d3:>7.4f} {l1:>6.3f} {l2:>6.3f} {l3:>6.3f} "
                        f"{r['k_1']:>8.5f} {r['K_2']:>7.4f} {r['S_1']:>9.4f} "
                        f"{r['min_G']:>7.5f} {Mc:>9.5f}{marker}"
                    )

    dt = time.time() - t0
    print(f"\nPhase 1: {len(results)} valid / {count} evaluated in {dt:.1f}s")
    if best["params"] is not None:
        print(
            f"BEST: M_cert={best['M_cert']:.5f} at "
            f"d2={best['params']['delta_2']}, d3={best['params']['delta_3']}, "
            f"l1={best['params']['lambda_1']}, l2={best['params']['lambda_2']}, "
            f"l3={best['params']['lambda_3']:.4f}"
        )
        print(f"  vs 2-comp 1.29005: {best['M_cert'] - TWO_COMP_BASELINE:+.5f}")
    return results, best


# ---------------------------------------------------------------------------
# Phase 2: DE refinement around the best (Phase 1)
# ---------------------------------------------------------------------------
def neg_M_cert_4param(x):
    """For DE: x = (d_2, d_3, l_1, l_2). Returns -M_cert (DE minimises)."""
    d2, d3, l1, l2 = float(x[0]), float(x[1]), float(x[2]), float(x[3])
    l3 = 1.0 - l1 - l2
    if l3 < 0.005 or l3 > 0.6:
        return 100.0
    if abs(d2 - d3) < 1e-5:
        return 100.0
    if d2 <= 0 or d3 <= 0 or d2 > D1_FIXED or d3 > D1_FIXED:
        return 100.0
    r = eval_at([D1_FIXED, d2, d3], [l1, l2, l3])
    Mc = r.get("M_cert")
    if Mc is None or not np.isfinite(Mc):
        return 100.0
    return -float(Mc)


def phase2_DE(best):
    p = best["params"]
    d2_c, d3_c = p["delta_2"], p["delta_3"]
    l1_c, l2_c = p["lambda_1"], p["lambda_2"]

    bounds = [
        (max(0.005, d2_c - 0.015), min(D1_FIXED - 0.005, d2_c + 0.015)),
        (max(0.005, d3_c - 0.015), min(D1_FIXED - 0.005, d3_c + 0.015)),
        (max(0.60, l1_c - 0.05), min(0.95, l1_c + 0.05)),
        (max(0.02, l2_c - 0.05), min(0.40, l2_c + 0.05)),
    ]
    print("\nPhase 2: differential evolution")
    print(f"  bounds: {bounds}")

    t0 = time.time()
    res = differential_evolution(
        neg_M_cert_4param,
        bounds,
        seed=0,
        maxiter=20,
        popsize=10,
        tol=1e-5,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=False,
        workers=1,
    )
    dt = time.time() - t0
    d2, d3, l1, l2 = res.x
    l3 = 1.0 - l1 - l2
    M_de = -res.fun
    print(f"  DE finished in {dt:.1f}s, nit={res.nit}, nfev={res.nfev}")
    print(
        f"  best: d2={d2:.5f}, d3={d3:.5f}, l1={l1:.5f}, l2={l2:.5f}, "
        f"l3={l3:.5f} -> M_cert={M_de:.6f}"
    )
    r = eval_at([D1_FIXED, float(d2), float(d3)], [float(l1), float(l2), float(l3)])
    return {
        "delta_1": D1_FIXED,
        "delta_2": float(d2),
        "delta_3": float(d3),
        "lambda_1": float(l1),
        "lambda_2": float(l2),
        "lambda_3": float(l3),
        "k_1": r.get("k_1"),
        "K_2": r.get("K_2"),
        "S_1": r.get("S_1"),
        "min_G": r.get("min_G"),
        "M_cert": r.get("M_cert"),
        "DE_runtime_s": dt,
        "DE_nit": res.nit,
        "DE_nfev": res.nfev,
    }


# ---------------------------------------------------------------------------
# Phase 3: frequency-fill mechanism analysis
# ---------------------------------------------------------------------------
def phase3_analysis(best_rec, two_comp_rec, mv_rec):
    """Compare K_hat at the 119 QP frequencies for MV, 2-comp, 3-comp."""
    xi_qp = np.arange(1, N_QP + 1) / U  # length 119

    def K_hat_at_qp(deltas, lambdas):
        return np.array([K_hat_ms(x, deltas, lambdas) for x in xi_qp])

    Kh_mv = K_hat_at_qp(mv_rec["deltas"], mv_rec["lambdas"])
    Kh_2 = K_hat_at_qp(two_comp_rec["deltas"], two_comp_rec["lambdas"])
    Kh_3 = K_hat_at_qp(best_rec["deltas"], best_rec["lambdas"])

    # j-indices where MV's K_hat is "near zero" (compared to mean)
    mv_thresh = 0.05 * Kh_mv.mean()
    near_zero_j = [int(j + 1) for j in range(N_QP) if Kh_mv[j] < mv_thresh]

    # Ratios at those j's:
    fill = []
    for j in near_zero_j:
        fill.append({
            "j": j,
            "xi": float(xi_qp[j - 1]),
            "K_hat_MV": float(Kh_mv[j - 1]),
            "K_hat_2comp": float(Kh_2[j - 1]),
            "K_hat_3comp": float(Kh_3[j - 1]),
            "ratio_3_to_MV": float(Kh_3[j - 1] / max(Kh_mv[j - 1], 1e-30)),
            "ratio_3_to_2": float(Kh_3[j - 1] / max(Kh_2[j - 1], 1e-30)),
        })
    return {
        "K_hat_MV_min": float(Kh_mv.min()),
        "K_hat_2comp_min": float(Kh_2.min()),
        "K_hat_3comp_min": float(Kh_3.min()),
        "K_hat_MV_mean": float(Kh_mv.mean()),
        "K_hat_2comp_mean": float(Kh_2.mean()),
        "K_hat_3comp_mean": float(Kh_3.mean()),
        "n_near_zero_MV": len(near_zero_j),
        "near_zero_fill": fill[:30],  # top 30
    }


def main():
    print("=" * 90)
    print("AGENT M1: 3-component multi-scale arcsine sweep")
    print(f"  D1 fixed = {D1_FIXED},  U = {U},  N_QP = {N_QP}")
    print(f"  MV baseline = {MV_BASELINE}, 2-comp baseline = {TWO_COMP_BASELINE}")
    print("=" * 90)

    # First, validate environment with a known point (2-comp recompute)
    r2 = eval_at([D1_FIXED, 0.045], [0.85, 0.15])
    print(f"\nValidation (2-comp): d2=0.045, l1=0.85")
    print(
        f"  k_1={r2['k_1']:.5f} K_2={r2['K_2']:.4f} S_1={r2['S_1']:.4f} "
        f"min_G={r2['min_G']:.5f} M_cert={r2['M_cert']:.5f}"
    )

    # Phase 1
    results, best_p1 = phase1_grid()

    out = {
        "D1": D1_FIXED,
        "U": U,
        "N_QP": N_QP,
        "MV_baseline": MV_BASELINE,
        "two_comp_baseline": TWO_COMP_BASELINE,
        "two_comp_validation": {
            "deltas": [D1_FIXED, 0.045],
            "lambdas": [0.85, 0.15],
            **r2,
        },
        "phase1_all_results": results,
        "phase1_best": best_p1["params"],
    }

    if best_p1["params"] is None:
        out["status"] = "phase1_no_valid_points"
        with open("_M1_3component_result.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        print("No valid points; exiting.")
        return

    # Phase 2: DE
    best_p2 = phase2_DE(best_p1)
    out["phase2_DE_best"] = best_p2

    # Choose overall best
    if best_p2 and best_p2["M_cert"] is not None and best_p2["M_cert"] > best_p1["M_cert"]:
        overall_best = best_p2
    else:
        overall_best = best_p1["params"]
    out["overall_best"] = overall_best
    out["improvement_over_2comp"] = (
        overall_best["M_cert"] - TWO_COMP_BASELINE if overall_best.get("M_cert") else None
    )
    out["improvement_over_MV"] = (
        overall_best["M_cert"] - MV_BASELINE if overall_best.get("M_cert") else None
    )

    # Phase 3: mechanism
    mv_rec = {"deltas": [D1_FIXED], "lambdas": [1.0]}
    two_rec = {
        "deltas": [D1_FIXED, 0.045],
        "lambdas": [0.85, 0.15],
    }
    best_rec = {
        "deltas": [
            D1_FIXED,
            overall_best["delta_2"],
            overall_best["delta_3"],
        ],
        "lambdas": [
            overall_best["lambda_1"],
            overall_best["lambda_2"],
            overall_best["lambda_3"],
        ],
    }
    mech = phase3_analysis(best_rec, two_rec, mv_rec)
    out["mechanism"] = mech
    out["S_1_drop"] = {
        "S_1_MV": None,  # we don't recompute; not strictly required
        "S_1_2comp": r2["S_1"],
        "S_1_3comp": overall_best["S_1"],
        "drop_from_2comp": r2["S_1"] - overall_best["S_1"],
    }

    with open("_M1_3component_result.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n" + "=" * 90)
    print("SUMMARY")
    print(f"  overall best M_cert = {overall_best['M_cert']:.6f}")
    print(f"  vs 2-comp 1.29005:  {overall_best['M_cert'] - TWO_COMP_BASELINE:+.6f}")
    print(f"  vs MV  1.27481:    {overall_best['M_cert'] - MV_BASELINE:+.6f}")
    print(
        f"  S_1: 2-comp={r2['S_1']:.3f} -> 3-comp={overall_best['S_1']:.3f}  "
        f"(drop {r2['S_1'] - overall_best['S_1']:+.3f})"
    )
    print(f"  K_hat min at QP freqs: MV={mech['K_hat_MV_min']:.4f}  "
          f"2-comp={mech['K_hat_2comp_min']:.4f}  3-comp={mech['K_hat_3comp_min']:.4f}")
    print("Wrote _M1_3component_result.json")


if __name__ == "__main__":
    main()
