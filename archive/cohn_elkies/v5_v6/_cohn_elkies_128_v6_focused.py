"""v6 focused: a few high-promise 3-scale points at N=1000, xi_max=1e5.

The partial v6 sweep at N=500 (xi_max=1e4) ranked:
    d3=0.022, l1=0.85, l2=0.10:  M_cert = 1.292454 (best at xi=1e4)
    d3=0.020, l1=0.85, l2=0.12:  M_cert = 1.292390
    d3=0.020, l1=0.82, l2=0.15:  M_cert = 1.292328

The v4-optimum centerpoint (d3=0.025, l1=0.85, l2=0.10) at N=1000, xi=1e5
already certifies M_cert >= 1.29302.

This script tests the three above candidates at N=1000 with xi_max=1e5 to
look for an improvement past 1.29302.  Each point takes ~25s.
"""
from __future__ import annotations

import json
import sys
import time
from fractions import Fraction
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import _cohn_elkies_128_v4 as v4
from _cohn_elkies_128_v4 import (
    DELTA1_Q,
    U_Q,
    configure_precision,
    certify,
    solve_qp_safe,
)


def _save(prefix, deltas_q, lambdas_q, coeffs_q, result):
    d1, d2, d3 = deltas_q
    l1, l2, _ = lambdas_q
    out = {
        "delta_1": [d1.numerator, d1.denominator],
        "delta_2": [d2.numerator, d2.denominator],
        "delta_3": [d3.numerator, d3.denominator],
        "lambda_1": [l1.numerator, l1.denominator],
        "lambda_2": [l2.numerator, l2.denominator],
        "N": len(coeffs_q),
        "M_cert_lower": result["M_cert_lower"],
        "xi_max": result["xi_max"],
        "prec_bits": result["prec_bits"],
        "coeffs_num_den": [[q.numerator, q.denominator] for q in coeffs_q],
    }
    path = _HERE / f"{prefix}_coeffs.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {path}")


def main():
    configure_precision(256)

    # (d2, d3, l1, l2)
    candidates = [
        (Fraction(55, 1000), Fraction(22, 1000), Fraction(85, 100), Fraction(10, 100)),
        (Fraction(55, 1000), Fraction(20, 1000), Fraction(85, 100), Fraction(12, 100)),
        (Fraction(55, 1000), Fraction(20, 1000), Fraction(82, 100), Fraction(15, 100)),
        (Fraction(55, 1000), Fraction(25, 1000), Fraction(85, 100), Fraction(10, 100)),  # v4 optimum baseline
    ]

    print("=" * 78)
    print("v6 FOCUSED: N=1000 + xi_max=1e5 + 4 top candidates")
    print("=" * 78)

    results = []
    best = None
    for (d2, d3, l1, l2) in candidates:
        deltas_q = [DELTA1_Q, d2, d3]
        lambdas_q = [l1, l2, Fraction(1) - l1 - l2]
        t0 = time.time()
        try:
            coeffs_q, _ = solve_qp_safe(deltas_q, lambdas_q, n_modes=1000)
            r = certify(deltas_q, lambdas_q, coeffs_q,
                        xi_max=10**5, prec_bits=256, verbose=False)
        except Exception as exc:
            print(f"  d2={float(d2)} d3={float(d3)} l1={float(l1)} l2={float(l2)}  ERROR: {exc}")
            continue
        el = time.time() - t0
        marker = ""
        if best is None or r["M_cert_lower"] > best[1]["M_cert_lower"]:
            best = ((deltas_q, lambdas_q, coeffs_q), r)
            marker = " *"
        print(f"  d2={float(d2):.3f} d3={float(d3):.3f} l1={float(l1):.3f} l2={float(l2):.3f} "
              f"M={r['M_cert_lower']:.8f} S1={r['S_1_upper']:.3f} minG={r['min_G_lower']:.4f} "
              f"K2hi={r['K_2_upper']:.4f} ({el:.1f}s){marker}")
        results.append({
            "params": {"d2": float(d2), "d3": float(d3),
                       "l1": float(l1), "l2": float(l2)},
            "M_cert_lower": r["M_cert_lower"],
            "min_G_lower": r["min_G_lower"],
            "S_1_upper": r["S_1_upper"],
            "K_2_upper": r["K_2_upper"],
            "xi_max": r["xi_max"],
            "n_modes": r["n_modes"],
            "prec_bits": r["prec_bits"],
            "G2_OK": r["G2_OK"],
            "a_gain_lower": r["a_gain_lower"],
            "k_1_mid": r["k_1_mid"],
            "k_1_rad": r["k_1_rad"],
            "lip_remainder_upper": r["lip_remainder_upper"],
            "K_2_lower": r["K_2_lower"],
            "K_2_bulk_lower": r["K_2_bulk_lower"],
            "K_2_bulk_upper": r["K_2_bulk_upper"],
            "K_2_tail_upper": r["K_2_tail_upper"],
        })

    if best is not None:
        (deltas_q, lambdas_q, coeffs_q), r_best = best
        print()
        print("BEST FOCUSED:")
        print(f"  deltas = {[float(d) for d in deltas_q]}")
        print(f"  lambdas = {[float(l) for l in lambdas_q]}")
        print(f"  M_cert_lower = {r_best['M_cert_lower']:.8f}")
        _save("_cohn_elkies_128_v6_focused_best", deltas_q, lambdas_q,
              coeffs_q, r_best)

    out = {
        "results": results,
        "best_M_cert_lower": (None if best is None
                              else best[1]["M_cert_lower"]),
        "best_params": (None if best is None
                        else {
                            "deltas": [float(d) for d in best[0][0]],
                            "lambdas": [float(l) for l in best[0][1]],
                        }),
    }
    with open(_HERE / "_cohn_elkies_128_v6_focused_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote _cohn_elkies_128_v6_focused_results.json")


if __name__ == "__main__":
    main()
