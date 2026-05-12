"""W11 verification: reproduce threescale_N200_best directly.

Calls v4's certify() at the known-best parameters
(deltas=[0.138, 0.055, 0.025], lambdas=[0.85, 0.10, 0.05])
with N=200 modes, xi_max=1e5, prec=256, and compares against the existing JSON.
"""
from __future__ import annotations

import json
import sys
import time
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import _cohn_elkies_128_v4 as v4

PREC = 256
XI_MAX = 10**5
ARB_GRID = 200001
N = 200

DELTAS_Q = [Fraction(138, 1000), Fraction(55, 1000), Fraction(25, 1000)]
LAMBDAS_Q = [Fraction(85, 100), Fraction(10, 100), Fraction(5, 100)]


def main():
    print("W11 reproduction of threescale_N200_best", flush=True)
    print(f"deltas_q = {DELTAS_Q}", flush=True)
    print(f"lambdas_q = {LAMBDAS_Q}", flush=True)

    v4.configure_precision(PREC)

    t0 = time.time()
    print("[Solving QP for G coefficients with N=200] ...", flush=True)
    coeffs_q, _ = v4.solve_qp_safe(DELTAS_Q, LAMBDAS_Q, N)
    print(f"  QP done in {time.time()-t0:.1f}s; coeffs[0..2]={[float(c) for c in coeffs_q[:3]]}", flush=True)

    t0 = time.time()
    print("[Calling certify() with xi_max=1e5] ...", flush=True)
    r = v4.certify(
        DELTAS_Q, LAMBDAS_Q, coeffs_q,
        xi_max=XI_MAX, prec_bits=PREC, arb_grid=ARB_GRID, verbose=True,
    )
    print(f"  Certify done in {time.time()-t0:.1f}s", flush=True)

    print("\n=== RESULT ===", flush=True)
    print(json.dumps(r, indent=2, default=str), flush=True)

    # Compare with existing JSON
    with open(HERE / "_cohn_elkies_128_v4_results.json", "r") as f:
        existing = json.load(f)
    ref = existing["summary_best_per_plan"]["threescale_N200_best"]

    print("\n=== COMPARISON ===", flush=True)
    keys = [
        "k_1_mid", "k_1_rad",
        "K_2_lower", "K_2_upper", "K_2_tail_upper",
        "min_G_lower", "lip_remainder_upper",
        "S_1_upper", "a_gain_lower",
        "M_cert_lower",
    ]
    all_match = True
    for key in keys:
        old = ref[key]
        new = r[key]
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            if old == 0:
                ok = (new == 0)
                reldiff = 0.0
            else:
                reldiff = abs(new - old) / abs(old)
                ok = reldiff < 1e-9
        else:
            ok = (str(old) == str(new))
            reldiff = float("nan")
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {key:>22}  old={old!r:>30}  new={new!r:>30}  reldiff={reldiff:.2e}", flush=True)
        if not ok:
            all_match = False

    print(f"\nALL_FIELDS_MATCH: {all_match}", flush=True)
    print(f"M_cert_lower reproduced: {r['M_cert_lower']}", flush=True)
    print(f"M_cert_lower ref:        {ref['M_cert_lower']}", flush=True)
    print(f"Claim threshold:         1.29215", flush=True)
    print(f"PASS (M >= 1.29215):     {r['M_cert_lower'] >= 1.29215}", flush=True)


if __name__ == "__main__":
    main()
