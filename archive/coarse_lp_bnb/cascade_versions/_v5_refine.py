"""Local refinement around the v5 4-scale optimum.

Best so far: (0.138, 0.046, 0.025, 0.080) with lambdas (0.82, 0.10, 0.04, 0.04)
giving rigorous M_cert = 1.29136 at XI_MAX = 1e5.

Sweep neighborhood and try 5-scale.
"""

from __future__ import annotations

import json
import time
from fractions import Fraction
from pathlib import Path

import _cohn_elkies_128_v5 as v5

_HERE = Path(__file__).parent


def _Q(x, den=10**6):
    return Fraction(int(round(x * den)), den)


def _norm(lams):
    return v5._norm_lambdas(lams)


def run(deltas_f, lambdas_f, xi_max, tag):
    deltas_q = [v5.DELTA1_Q] + [_Q(d, 10**6) for d in deltas_f[1:]]
    lambdas_q = _norm(lambdas_f)
    t0 = time.time()
    r = v5.certify_with_reopt(deltas_q, lambdas_q, xi_max=xi_max)
    el = time.time() - t0
    print(f"  {tag:<58}  M={r['M_cert_lower']:.6f}  S1={r['S_1_upper']:.3f}  "
          f"K2={r['K_2_upper']:.4f}  ({el:.1f}s)")
    r["tag"] = tag
    return r


def main():
    v5.configure_precision(256)
    print("=" * 76)
    print("v5 local refinement around 4-scale best")
    print("  base: (0.138, 0.046, 0.025, 0.080)  lambdas (0.82, 0.10, 0.04, 0.04)")
    print("=" * 76)

    runs = []
    # neighborhood in delta (slight shifts) - keep grid moderate
    grid = []
    for d2 in [0.044, 0.046, 0.048]:
        for d3 in [0.022, 0.025, 0.028]:
            for d4 in [0.070, 0.080, 0.090]:
                for lams in [
                    (0.82, 0.10, 0.04, 0.04),
                    (0.83, 0.09, 0.04, 0.04),
                    (0.81, 0.11, 0.04, 0.04),
                    (0.82, 0.09, 0.05, 0.04),
                    (0.80, 0.12, 0.04, 0.04),
                ]:
                    grid.append((d2, d3, d4, lams))
    print(f"  {len(grid)} 4-scale configurations at XI_MAX=1e4")
    best = None
    for (d2, d3, d4, lams) in grid:
        deltas_f = [0.138, d2, d3, d4]
        tag = f"4sc d=({d2:.3f},{d3:.3f},{d4:.3f}) l={lams}"
        try:
            r = run(deltas_f, list(lams), xi_max=10**4, tag=tag)
        except Exception as e:
            print(f"  {tag}: ERROR {e}")
            continue
        runs.append(r)
        if best is None or r["M_cert_lower"] > best["M_cert_lower"]:
            best = r
            print(f"    NEW BEST: {best['M_cert_lower']:.6f}")

    # try 5-scale
    print()
    print("=" * 76)
    print("5-scale exploration")
    print("=" * 76)
    five_configs = [
        (0.046, 0.025, 0.080, 0.018, (0.80, 0.10, 0.04, 0.04, 0.02)),
        (0.046, 0.025, 0.080, 0.015, (0.80, 0.10, 0.04, 0.04, 0.02)),
        (0.046, 0.025, 0.080, 0.012, (0.81, 0.10, 0.04, 0.04, 0.01)),
        (0.046, 0.022, 0.075, 0.012, (0.81, 0.10, 0.04, 0.04, 0.01)),
        (0.050, 0.028, 0.090, 0.018, (0.79, 0.10, 0.05, 0.04, 0.02)),
        (0.045, 0.025, 0.080, 0.100, (0.80, 0.10, 0.04, 0.03, 0.03)),  # higher delta_5
        (0.046, 0.025, 0.080, 0.110, (0.79, 0.10, 0.04, 0.04, 0.03)),
    ]
    for (d2, d3, d4, d5, lams) in five_configs:
        deltas_f = [0.138, d2, d3, d4, d5]
        tag = f"5sc d=({d2},{d3},{d4},{d5}) l={lams}"
        try:
            r = run(deltas_f, list(lams), xi_max=10**4, tag=tag)
        except Exception as e:
            print(f"  {tag}: ERROR {e}")
            continue
        runs.append(r)
        if best is None or r["M_cert_lower"] > best["M_cert_lower"]:
            best = r
            print(f"    NEW BEST: {best['M_cert_lower']:.6f}")

    # re-certify overall best at XI_MAX = 1e5
    print()
    print("=" * 76)
    print(f"Re-certify best at XI_MAX = 1e5: {best['tag']}")
    print("=" * 76)
    deltas_f_best = best["deltas"]
    lambdas_f_best = best["lambdas"]
    r_hi = run(deltas_f_best, lambdas_f_best, xi_max=10**5,
               tag=best["tag"] + " @xi=1e5")
    runs.append(r_hi)
    final_best = max(runs, key=lambda r: r["M_cert_lower"])

    print()
    print("=" * 76)
    print("Top 12 from refinement:")
    print("=" * 76)
    for r in sorted(runs, key=lambda r: -r["M_cert_lower"])[:12]:
        print(f"  {r.get('tag','?')[:60]:<60}  M={r['M_cert_lower']:.6f}")
    print()
    print(f"FINAL BEST: {final_best['M_cert_lower']:.8f}")
    print(f"  vs v3 (1.28984): {final_best['M_cert_lower'] - 1.28984:+.6f}")

    with open(_HERE / "_v5_refine_results.json", "w") as f:
        json.dump({"runs": runs, "best": final_best}, f, indent=2)


if __name__ == "__main__":
    main()
