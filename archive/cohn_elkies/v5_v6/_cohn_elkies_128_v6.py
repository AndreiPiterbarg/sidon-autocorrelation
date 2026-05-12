"""Cohn--Elkies multi-scale arcsine certifier v6 -- 3-scale at N in {500, 1000}.

Builds on v4 (3-scale rigorous certifier) and extends to larger N with a
finer sweep around the v4 optimum.

Reference (v4):
    Best 3-scale rigorous certificate at N=200:
        deltas = [0.138, 0.055, 0.025], lambdas = [0.85, 0.10, 0.05]
        M_cert >= 1.29215650

Targets:
    N=500, N=1000 should give M_cert >= 1.293 (parallel to 2-scale gain
    from N=200 -> N=500 -> N=1000: +0.0007 + +0.0001).

Every output is a rigorous arb (or float upper/lower bound from arb). G coeffs
are rationalized to Fraction with denom 10^12, then verified in arb.

Usage:
    python _cohn_elkies_128_v6.py --plan N500
    python _cohn_elkies_128_v6.py --plan N1000
    python _cohn_elkies_128_v6.py --plan all  # default
    python _cohn_elkies_128_v6.py --plan finepoint  # only the v4 optimum, N=500,1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

# Import v4's certified building blocks.  Every routine below pulls
# directly from v4 so the rigorous semantics (arb intervals, Bessel
# tail bound, Lipschitz remainder, etc.) are identical.
import _cohn_elkies_128_v4 as v4
from _cohn_elkies_128_v4 import (
    DELTA1_Q,
    U_Q,
    RATIONAL_DEN,
    ARB_GRID_DEFAULT,
    Q_arb,
    configure_precision,
    certify,
    solve_qp_safe,
)

import flint
from flint import arb


# ---------------------------------------------------------------------------
# Helpers for cleaner JSON output
# ---------------------------------------------------------------------------
def _fraction_list_to_json(fracs):
    return [[q.numerator, q.denominator] for q in fracs]


def _save_best_coeffs(prefix: str, best: dict) -> None:
    """Export best G coefficients as Fraction list + Lean-friendly JSON."""
    if best is None or "coeffs_q" not in best:
        return
    d1, d2, d3 = best["deltas_q"]
    l1, l2, _ = best["lambdas_q"]
    payload = {
        "delta_1": [d1.numerator, d1.denominator],
        "delta_2": [d2.numerator, d2.denominator],
        "delta_3": [d3.numerator, d3.denominator],
        "lambda_1": [l1.numerator, l1.denominator],
        "lambda_2": [l2.numerator, l2.denominator],
        "N": len(best["coeffs_q"]),
        "M_cert_lower": best["M_cert_lower"],
        "xi_max": best["xi_max"],
        "prec_bits": best["prec_bits"],
        "coeffs_num_den": _fraction_list_to_json(best["coeffs_q"]),
    }
    out_path = _HERE / f"{prefix}_coeffs.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Wrote best G coeffs -> {out_path}")


# ---------------------------------------------------------------------------
# Finer 3-scale sweep around the v4 optimum (d=0.138, 0.055, 0.025; l=0.85, 0.10)
# ---------------------------------------------------------------------------
def _build_fine_sweep():
    """Tight neighborhood around v4's optimum (d2=0.055, d3=0.025, l1=0.85, l2=0.10).

    Cross product:
        d3 in {0.020, 0.022, 0.025, 0.027, 0.030}
        l3 in {0.03, 0.05, 0.07}
        l1 in {0.82, 0.85, 0.88}
    with d2 fixed at the v4 optimum 0.055.  A few extra d2 anchor points are
    swept at the v4 centerpoint (d3=0.025, l1=0.85, l3=0.05) as a sanity check.
    """
    d3_list = [Fraction(20, 1000), Fraction(22, 1000), Fraction(25, 1000),
               Fraction(27, 1000), Fraction(30, 1000)]
    l3_list = [Fraction(3, 100), Fraction(5, 100), Fraction(7, 100)]
    l1_list = [Fraction(82, 100), Fraction(85, 100), Fraction(88, 100)]
    d2_center = Fraction(55, 1000)

    grid = []
    for d3 in d3_list:
        if d3 >= d2_center:
            continue
        for l1 in l1_list:
            for l3 in l3_list:
                l2 = Fraction(1) - l1 - l3
                if l2 <= 0:
                    continue
                grid.append({"d2": d2_center, "d3": d3, "l1": l1, "l2": l2})
    # d2-anchor sanity at the v4 center
    for d2 in (Fraction(50, 1000), Fraction(60, 1000)):
        l1, l3 = Fraction(85, 100), Fraction(5, 100)
        l2 = Fraction(1) - l1 - l3
        grid.append({"d2": d2, "d3": Fraction(25, 1000),
                     "l1": l1, "l2": l2})
    return grid


def _build_optimum_only_point():
    return [{"d2": Fraction(55, 1000), "d3": Fraction(25, 1000),
             "l1": Fraction(85, 100), "l2": Fraction(10, 100)}]


# ---------------------------------------------------------------------------
# Driver mirrors v4.run_threescale but keeps the high-N path explicit
# ---------------------------------------------------------------------------
def run_threescale_N(N: int, sweep_grid, xi_max=10**4,
                     prec_bits=256, arb_grid=ARB_GRID_DEFAULT,
                     verbose: bool = False):
    configure_precision(prec_bits)
    results = []
    best = None
    print(f"\n--- 3-scale N={N} sweep (xi_max={xi_max}, |grid|={len(sweep_grid)}) ---")
    print(f"  {'d2':>6} {'d3':>6} {'l1':>6} {'l2':>6} {'M_cert':>10} "
          f"{'minG':>8} {'S_1':>9} {'K2hi':>8} {'time':>7}")
    for pt in sweep_grid:
        d2, d3, l1, l2 = pt["d2"], pt["d3"], pt["l1"], pt["l2"]
        l3 = Fraction(1) - l1 - l2
        if l3 <= 0 or l1 <= 0 or l2 <= 0:
            continue
        deltas_q = [DELTA1_Q, d2, d3]
        lambdas_q = [l1, l2, l3]
        t0 = time.time()
        try:
            coeffs_q, _ = solve_qp_safe(deltas_q, lambdas_q, N)
            r = certify(deltas_q, lambdas_q, coeffs_q,
                        xi_max=xi_max, prec_bits=prec_bits,
                        arb_grid=arb_grid, verbose=verbose)
        except Exception as exc:
            print(f"  {float(d2):>6.3f} {float(d3):>6.3f} {float(l1):>6.3f} "
                  f"{float(l2):>6.3f}   ERROR: {exc}")
            continue
        el = time.time() - t0
        marker = ""
        if best is None or r["M_cert_lower"] > best["M_cert_lower"]:
            best = dict(r)
            best["coeffs_q"] = coeffs_q
            best["deltas_q"] = deltas_q
            best["lambdas_q"] = lambdas_q
            marker = " *"
        print(f"  {float(d2):>6.3f} {float(d3):>6.3f} {float(l1):>6.3f} "
              f"{float(l2):>6.3f} {r['M_cert_lower']:>10.6f} {r['min_G_lower']:>8.4f} "
              f"{r['S_1_upper']:>9.3f} {r['K_2_upper']:>8.4f} {el:>6.1f}s{marker}")
        results.append(r)
    # High xi_max re-run for the in-sweep best (this is where the rigorous gain comes from)
    if best is not None and "coeffs_q" in best:
        print(f"  [Best 3-scale re-run at xi_max=1e5]")
        r_hi = certify(best["deltas_q"], best["lambdas_q"], best["coeffs_q"],
                       xi_max=10**5, prec_bits=prec_bits, arb_grid=arb_grid,
                       verbose=True)
        results.append(r_hi)
        if r_hi["M_cert_lower"] > best["M_cert_lower"]:
            # carry the coeffs forward, swap stats
            preserved = {k: best[k] for k in ("coeffs_q", "deltas_q", "lambdas_q")}
            best = dict(r_hi)
            best.update(preserved)
    return results, best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="all",
                        choices=["N500", "N1000", "all",
                                 "finepoint", "finepoint-N500",
                                 "finepoint-N1000"])
    parser.add_argument("--xi-max", type=int, default=10**4)
    parser.add_argument("--prec", type=int, default=256)
    args = parser.parse_args()

    print("=" * 78)
    print("Cohn--Elkies 3-scale arcsine certifier v6 (N in {500, 1000})")
    print("=" * 78)
    configure_precision(args.prec)

    all_runs = []
    summary = {}
    bests_full = {}  # carries coeffs_q etc. for exporters

    fine_grid = _build_fine_sweep()
    opt_only = _build_optimum_only_point()

    if args.plan in ("finepoint", "finepoint-N500"):
        runs, best = run_threescale_N(500, opt_only,
                                      xi_max=args.xi_max, prec_bits=args.prec)
        all_runs.extend(runs)
        summary["threescale_N500_finepoint"] = (
            None if best is None
            else {k: v for k, v in best.items()
                  if k not in ("coeffs_q", "deltas_q", "lambdas_q")}
        )
        bests_full["threescale_N500_finepoint"] = best

    if args.plan in ("finepoint", "finepoint-N1000"):
        runs, best = run_threescale_N(1000, opt_only,
                                      xi_max=args.xi_max, prec_bits=args.prec)
        all_runs.extend(runs)
        summary["threescale_N1000_finepoint"] = (
            None if best is None
            else {k: v for k, v in best.items()
                  if k not in ("coeffs_q", "deltas_q", "lambdas_q")}
        )
        bests_full["threescale_N1000_finepoint"] = best

    if args.plan in ("N500", "all"):
        runs, best = run_threescale_N(500, fine_grid,
                                      xi_max=args.xi_max, prec_bits=args.prec)
        all_runs.extend(runs)
        summary["threescale_N500_best"] = (
            None if best is None
            else {k: v for k, v in best.items()
                  if k not in ("coeffs_q", "deltas_q", "lambdas_q")}
        )
        bests_full["threescale_N500_best"] = best

    if args.plan in ("N1000", "all"):
        # 47-pt grid is now tight enough to run at N=1000 in full.
        runs, best = run_threescale_N(1000, fine_grid,
                                      xi_max=args.xi_max, prec_bits=args.prec)
        all_runs.extend(runs)
        summary["threescale_N1000_best"] = (
            None if best is None
            else {k: v for k, v in best.items()
                  if k not in ("coeffs_q", "deltas_q", "lambdas_q")}
        )
        bests_full["threescale_N1000_best"] = best

    # ----------------------- Final summary ---------------------------------
    print()
    print("=" * 78)
    print("FINAL SUMMARY (v6)")
    print("=" * 78)
    for tag, r in summary.items():
        if r is None:
            continue
        ds = ",".join(f"{d:.4f}" for d in r["deltas"])
        ls = ",".join(f"{l:.4f}" for l in r["lambdas"])
        print(f"  {tag:<32}  M_cert >= {r['M_cert_lower']:.8f}  "
              f"d=[{ds}]  l=[{ls}]  N={r['n_modes']}  xi={r['xi_max']}")

    best_all = max(all_runs, key=lambda x: x["M_cert_lower"]) if all_runs else None
    print()
    if best_all is not None:
        print(f"BEST RIGOROUS M_cert (v6): {best_all['M_cert_lower']:.8f}")
        print(f"  deltas = {best_all['deltas']}")
        print(f"  lambdas = {best_all['lambdas']}")
        print(f"  N = {best_all['n_modes']}, xi_max = {best_all['xi_max']}")
    print(f"v4 best for reference: 1.29215650")
    print(f"Target threshold:      1.29300000")
    print("=" * 78)

    # ----------------------- Export best coeffs ---------------------------
    best_overall_full = None
    best_overall_val = -1.0
    for tag, b in bests_full.items():
        if b is None:
            continue
        if b["M_cert_lower"] > best_overall_val:
            best_overall_val = b["M_cert_lower"]
            best_overall_full = (tag, b)
    if best_overall_full is not None:
        tag, b = best_overall_full
        _save_best_coeffs(f"_cohn_elkies_128_v6_{tag}", b)

    # ----------------------- Persist results ------------------------------
    out = {
        "configuration": {
            "delta_1": float(DELTA1_Q),
            "u": float(U_Q),
            "rational_denominator": RATIONAL_DEN,
            "arb_grid_default": ARB_GRID_DEFAULT,
            "prec_bits": args.prec,
        },
        "summary_best_per_plan": summary,
        "all_runs": all_runs,
        "best_overall": best_all,
        "v4_best_reference": 1.29215650,
        "v3_best_reference": 1.28984106,
    }
    json_path = _HERE / "_cohn_elkies_128_v6_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
