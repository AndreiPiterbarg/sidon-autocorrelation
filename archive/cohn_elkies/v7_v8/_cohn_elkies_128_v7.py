"""Cohn--Elkies multi-scale arcsine certifier v7 -- N-scale + N_G sweep.

Extends v5 (N-scale arcsine, rigorous arb pipeline) by allowing the number of
QP cosine modes for G (n_modes / N_G) to vary in {200, 500} and adds 4-scale,
5-scale and 6-scale sweeps requested by the v6 agent task.

Goal: confirm whether 4-scale or 5-scale arcsine kernels beat the 3-scale
ceiling (~1.29216 from v4 at N_G=200) when N_G is large enough.

Rigorous pieces (every endpoint is a flint.arb interval), identical to v5:
  * k_1 = K_hat(1)
  * K_2 = 2 * int_0^XI_MAX K_hat(xi)^2 d xi  +  Bessel-tail correction
         tail bound:  (8/pi^2) * (sum_i lambda_i / delta_i)^2 / XI_MAX
  * S_1 upper:  sum_{j=1}^{N_G} a_j^2 / K_hat(j/u)
  * min_G lower:  dense grid argmin + Lipschitz remainder of G' on [0, 1/4]
  * gain a >= (4/u) * (min_G)^2 / S_1
  * M_* lower via the MV master-inequality bisection

Usage:
    python _cohn_elkies_128_v7.py                # full sweep
    python _cohn_elkies_128_v7.py --quick        # just headline configs
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "delsarte_dual"))

# Pull the rigorous primitives from v5 verbatim (avoid duplication).
import importlib.util
_V5_PATH = _HERE / "_cohn_elkies_128_v5.py"
spec = importlib.util.spec_from_file_location("_cohn_elkies_128_v5", _V5_PATH)
v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v5)

import flint
from flint import acb, arb

# Re-export shared constants/globals
DELTA1_Q = v5.DELTA1_Q
U_Q = v5.U_Q
RATIONAL_DEN = v5.RATIONAL_DEN
PREC_BITS = v5.PREC_BITS
ARB_GRID_DEFAULT = v5.ARB_GRID_DEFAULT

Q_arb = v5.Q_arb
configure_precision = v5.configure_precision
K_hat_arb = v5.K_hat_arb
k1_enclosure = v5.k1_enclosure
K2_rigorous = v5.K2_rigorous
round_to_rationals = v5.round_to_rationals
min_G_lower_bound = v5.min_G_lower_bound
S1_upper_bound = v5.S1_upper_bound
find_M_lower_bisect = v5.find_M_lower_bisect
certify_Nscale = v5.certify_Nscale
_norm_lambdas = v5._norm_lambdas
_Q = v5._Q


# ---------------------------------------------------------------------------
# solve_qp_Nscale_NG -- like v5.solve_qp_Nscale but with configurable n_modes,
# and QP grid scaled with N (n_grid >= 4*N + 1) to match v4's convention.
# ---------------------------------------------------------------------------
def solve_qp_Nscale_NG(deltas_q, lambdas_q, n_modes: int):
    """Solve the QP for G coefficients with n_modes cosine modes."""
    import cvxpy as cp
    from scipy.special import j0
    deltas_f = [float(q) for q in deltas_q]
    lambdas_f = [float(q) for q in lambdas_q]
    u_f = float(U_Q)
    # Adaptive QP grid: at least 4*N+1 points (matching v4)
    n_grid = max(5001, 4 * n_modes + 1)

    w = np.zeros(n_modes)
    for j in range(1, n_modes + 1):
        xi_j = j / u_f
        v_ = 0.0
        for lam, d in zip(lambdas_f, deltas_f):
            v_ += lam * j0(math.pi * d * xi_j) ** 2
        w[j - 1] = v_
    if w.min() <= 0:
        raise RuntimeError(f"w min nonpositive: {w.min()}")
    xs = np.linspace(0.0, 0.25, n_grid)
    B = np.zeros((n_grid, n_modes))
    for j in range(1, n_modes + 1):
        B[:, j - 1] = np.cos(2.0 * math.pi * j * xs / u_f)
    a = cp.Variable(n_modes)
    obj = cp.Minimize(cp.sum(cp.multiply(1.0 / w, cp.square(a))))
    cons = [B @ a >= 1.0]
    prob = cp.Problem(obj, cons)
    for s_name in ("MOSEK", "CLARABEL", "SCS"):
        try:
            prob.solve(solver=s_name, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate") and a.value is not None:
                break
        except Exception:
            continue
    if a.value is None:
        raise RuntimeError(f"QP failed; status={prob.status}")
    return np.asarray(a.value).flatten()


def certify_with_reopt_NG(deltas_q, lambdas_q, xi_max: int, n_modes: int,
                          verbose: bool = False) -> dict:
    """Full pipeline: solve QP at n_modes, rationalize, certify with arb."""
    configure_precision(PREC_BITS)
    a_opt = solve_qp_Nscale_NG(deltas_q, lambdas_q, n_modes=n_modes)
    coeffs_q = round_to_rationals(a_opt)
    return certify_Nscale(deltas_q, lambdas_q, coeffs_q,
                          xi_max=xi_max, verbose=verbose)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def _fmt_tag(scales, deltas, lambdas, n_g):
    ds = ",".join(f"{d:.4g}" for d in deltas)
    ls = ",".join(f"{l:.3g}" for l in lambdas)
    return f"{scales}sc NG={n_g} d=({ds}) l=({ls})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Only run a single 4-scale headline config.")
    parser.add_argument("--xi-max", type=int, default=10**4,
                        help="XI_MAX for sweep certifications (default 1e4).")
    parser.add_argument("--xi-max-best", type=int, default=10**5,
                        help="XI_MAX for the final best recertification.")
    parser.add_argument("--out", type=str,
                        default=str(_HERE / "_cohn_elkies_128_v7_results.json"))
    parser.add_argument("--skip-500", action="store_true",
                        help="Skip the N_G=500 configurations (faster).")
    parser.add_argument("--skip-6scale", action="store_true",
                        help="Skip the 6-scale sweep.")
    args = parser.parse_args()

    configure_precision(PREC_BITS)
    print("=" * 78)
    print("Cohn--Elkies N-scale arcsine certifier v7 (N_G sweep, 4/5/6-scale)")
    print(f"  delta_1 (fixed) = {float(DELTA1_Q):.4f}   u = {float(U_Q):.4f}")
    print(f"  PREC = {PREC_BITS} bits")
    print(f"  XI_MAX (sweep) = {args.xi_max}    XI_MAX (best) = {args.xi_max_best}")
    print("=" * 78)

    all_runs = []

    # ----- 3-scale reference at N_G=200, 500 (calibration baseline) -----
    n_g_list = [200] if args.skip_500 else [200, 500]
    if args.quick:
        n_g_list = [200]

    ref_3sc_q = ([DELTA1_Q, _Q(0.055, 1000), _Q(0.025, 1000)],
                 _norm_lambdas([0.85, 0.10, 0.05]))
    print("\n[3-scale reference] delta=(0.138,0.055,0.025), lambda=(0.85,0.10,0.05)")
    for n_g in n_g_list:
        t0 = time.time()
        try:
            r = certify_with_reopt_NG(*ref_3sc_q, xi_max=args.xi_max,
                                      n_modes=n_g, verbose=False)
        except Exception as exc:
            print(f"  N_G={n_g}: ERROR {exc}")
            continue
        el = time.time() - t0
        tag = _fmt_tag(3, r["deltas"], r["lambdas"], n_g)
        r["tag"] = tag
        r["N_G"] = n_g
        all_runs.append(r)
        print(f"  N_G={n_g:>4}  M_cert = {r['M_cert_lower']:.8f}  "
              f"minG={r['min_G_lower']:.4f}  S1={r['S_1_upper']:.3f}  "
              f"K2hi={r['K_2_upper']:.4f}  ({el:.1f}s)")

    if args.quick:
        # one 4-scale headline
        print("\n[Quick 4-scale] delta=(0.138,0.055,0.025,0.020), N_G=200")
        d_q = [DELTA1_Q, _Q(0.055, 1000), _Q(0.025, 1000), _Q(0.020, 1000)]
        l_q = _norm_lambdas([0.85, 0.08, 0.04, 0.03])
        r = certify_with_reopt_NG(d_q, l_q, xi_max=args.xi_max,
                                  n_modes=200, verbose=True)
        r["tag"] = _fmt_tag(4, r["deltas"], r["lambdas"], 200)
        r["N_G"] = 200
        all_runs.append(r)
        best = max(all_runs, key=lambda x: x["M_cert_lower"])
        _final_summary(all_runs)
        _write(args.out, all_runs, best, n_g_list)
        return

    # ===================== 4-SCALE SWEEP =====================
    print("\n" + "=" * 78)
    print("4-SCALE SWEEP")
    print("=" * 78)
    # Requested deltas + lambda combinations (l1 ~ 0.85)
    four_scale_deltas = [
        (0.138, 0.055, 0.025, 0.020),
        (0.138, 0.055, 0.025, 0.012),
        (0.138, 0.046, 0.025, 0.015),
        # a few neighbors (small perturbations) for robustness
        (0.138, 0.055, 0.030, 0.015),
        (0.138, 0.046, 0.025, 0.012),
        (0.138, 0.055, 0.020, 0.010),
    ]
    four_scale_lambdas = [
        (0.85, 0.08, 0.04, 0.03),
        (0.85, 0.10, 0.03, 0.02),
        (0.86, 0.08, 0.04, 0.02),
        (0.84, 0.10, 0.04, 0.02),
        (0.85, 0.09, 0.04, 0.02),
    ]
    print(f"  {'deltas':<32} {'lambdas':<28} {'N_G':>5} {'M_cert':>10} {'minG':>7} {'S1':>7}")
    best4 = None
    for deltas in four_scale_deltas:
        for lams in four_scale_lambdas:
            d_q = [DELTA1_Q] + [_Q(d, 10**6) for d in deltas[1:]]
            try:
                l_q = _norm_lambdas(list(lams))
            except ValueError:
                continue
            for n_g in n_g_list:
                t0 = time.time()
                try:
                    r = certify_with_reopt_NG(d_q, l_q,
                                              xi_max=args.xi_max,
                                              n_modes=n_g)
                except Exception as exc:
                    print(f"  d={deltas} l={lams} N_G={n_g}: ERROR {exc}")
                    continue
                el = time.time() - t0
                tag = _fmt_tag(4, r["deltas"], r["lambdas"], n_g)
                r["tag"] = tag
                r["N_G"] = n_g
                marker = ""
                if best4 is None or r["M_cert_lower"] > best4["M_cert_lower"]:
                    best4 = r
                    marker = " *"
                d_str = "(" + ",".join(f"{d:.4g}" for d in deltas) + ")"
                l_str = "(" + ",".join(f"{l:.3g}" for l in lams) + ")"
                print(f"  {d_str:<32} {l_str:<28} {n_g:>5} "
                      f"{r['M_cert_lower']:>10.6f} "
                      f"{r['min_G_lower']:>7.4f} "
                      f"{r['S_1_upper']:>7.3f}  ({el:.1f}s){marker}")
                all_runs.append(r)

    # ===================== 5-SCALE SWEEP =====================
    print("\n" + "=" * 78)
    print("5-SCALE SWEEP")
    print("=" * 78)
    five_scale_deltas = [
        (0.138, 0.055, 0.025, 0.015, 0.008),
        # small perturbations
        (0.138, 0.055, 0.025, 0.012, 0.006),
        (0.138, 0.046, 0.025, 0.015, 0.008),
    ]
    five_scale_lambdas = [
        (0.85, 0.07, 0.04, 0.02, 0.02),
        (0.84, 0.08, 0.04, 0.02, 0.02),
        (0.86, 0.07, 0.04, 0.02, 0.01),
    ]
    print(f"  {'deltas':<36} {'lambdas':<32} {'N_G':>5} {'M_cert':>10} {'minG':>7} {'S1':>7}")
    best5 = None
    for deltas in five_scale_deltas:
        for lams in five_scale_lambdas:
            d_q = [DELTA1_Q] + [_Q(d, 10**6) for d in deltas[1:]]
            try:
                l_q = _norm_lambdas(list(lams))
            except ValueError:
                continue
            for n_g in n_g_list:
                t0 = time.time()
                try:
                    r = certify_with_reopt_NG(d_q, l_q,
                                              xi_max=args.xi_max,
                                              n_modes=n_g)
                except Exception as exc:
                    print(f"  d={deltas} l={lams} N_G={n_g}: ERROR {exc}")
                    continue
                el = time.time() - t0
                tag = _fmt_tag(5, r["deltas"], r["lambdas"], n_g)
                r["tag"] = tag
                r["N_G"] = n_g
                marker = ""
                if best5 is None or r["M_cert_lower"] > best5["M_cert_lower"]:
                    best5 = r
                    marker = " *"
                d_str = "(" + ",".join(f"{d:.4g}" for d in deltas) + ")"
                l_str = "(" + ",".join(f"{l:.3g}" for l in lams) + ")"
                print(f"  {d_str:<36} {l_str:<32} {n_g:>5} "
                      f"{r['M_cert_lower']:>10.6f} "
                      f"{r['min_G_lower']:>7.4f} "
                      f"{r['S_1_upper']:>7.3f}  ({el:.1f}s){marker}")
                all_runs.append(r)

    # ===================== 6-SCALE SWEEP =====================
    if not args.skip_6scale:
        print("\n" + "=" * 78)
        print("6-SCALE SWEEP (thoroughness; uniformly spaced)")
        print("=" * 78)
        # delta_1 = 0.138 fixed; remaining 5 deltas uniformly in [0.005, 0.080]
        # (keep small so that the kernel is concentrated)
        rest_low_high_list = [
            (0.005, 0.080),
            (0.008, 0.090),
            (0.010, 0.060),
        ]
        # lambdas: l_1 ~ 0.85, rest distributed roughly geometrically
        six_lambdas = [
            (0.85, 0.06, 0.04, 0.02, 0.02, 0.01),
            (0.84, 0.07, 0.04, 0.02, 0.02, 0.01),
        ]
        best6 = None
        print(f"  {'deltas':<48} {'lambdas':<36} {'N_G':>5} {'M_cert':>10}")
        for (lo_, hi_) in rest_low_high_list:
            rest = list(np.linspace(hi_, lo_, 5))  # descending order
            deltas = (0.138,) + tuple(rest)
            for lams in six_lambdas:
                d_q = [DELTA1_Q] + [_Q(d, 10**6) for d in deltas[1:]]
                try:
                    l_q = _norm_lambdas(list(lams))
                except ValueError:
                    continue
                # only at N_G=200 for thoroughness
                n_g = 200
                t0 = time.time()
                try:
                    r = certify_with_reopt_NG(d_q, l_q,
                                              xi_max=args.xi_max,
                                              n_modes=n_g)
                except Exception as exc:
                    print(f"  d={deltas} l={lams}: ERROR {exc}")
                    continue
                el = time.time() - t0
                tag = _fmt_tag(6, r["deltas"], r["lambdas"], n_g)
                r["tag"] = tag
                r["N_G"] = n_g
                marker = ""
                if best6 is None or r["M_cert_lower"] > best6["M_cert_lower"]:
                    best6 = r
                    marker = " *"
                d_str = "(" + ",".join(f"{d:.3g}" for d in deltas) + ")"
                l_str = "(" + ",".join(f"{l:.2g}" for l in lams) + ")"
                print(f"  {d_str:<48} {l_str:<36} {n_g:>5} "
                      f"{r['M_cert_lower']:>10.6f}  ({el:.1f}s){marker}")
                all_runs.append(r)

    # ===================== RECERTIFY BEST AT HIGH XI_MAX =====================
    print("\n" + "=" * 78)
    best_overall = max(all_runs, key=lambda r: r["M_cert_lower"])
    print(f"Re-certify BEST at XI_MAX={args.xi_max_best}: {best_overall['tag']}")
    print("=" * 78)
    d_best = best_overall["deltas"]
    l_best = best_overall["lambdas"]
    n_g_best = best_overall["N_G"]
    d_q_best = [DELTA1_Q] + [_Q(d, 10**6) for d in d_best[1:]]
    l_q_best = _norm_lambdas(list(l_best))
    r_best_hi = certify_with_reopt_NG(d_q_best, l_q_best,
                                      xi_max=args.xi_max_best,
                                      n_modes=n_g_best, verbose=True)
    r_best_hi["tag"] = best_overall["tag"] + f" @xi={args.xi_max_best}"
    r_best_hi["N_G"] = n_g_best
    all_runs.append(r_best_hi)
    if r_best_hi["M_cert_lower"] > best_overall["M_cert_lower"]:
        best_overall = r_best_hi

    _final_summary(all_runs)
    _write(args.out, all_runs, best_overall, n_g_list)


def _final_summary(all_runs):
    print()
    print("=" * 78)
    print("FINAL SUMMARY (top 20)")
    print("=" * 78)
    print(f"  {'tag':<60}  {'M_cert':>10}  {'K2hi':>7}")
    print(f"  {'-'*60}  {'-'*10}  {'-'*7}")
    sorted_runs = sorted(all_runs, key=lambda r: -r["M_cert_lower"])
    for r in sorted_runs[:20]:
        tag = r.get("tag", "?")[:60]
        print(f"  {tag:<60}  {r['M_cert_lower']:>10.6f}  {r['K_2_upper']:>7.4f}")
    best = sorted_runs[0]
    print()
    print(f"BEST rigorous M_cert: {best['M_cert_lower']:.8f}")
    print(f"  tag: {best.get('tag','?')}")
    print(f"  vs 3-scale ref (1.29216): {best['M_cert_lower'] - 1.29216:+.6f}")
    print(f"  vs v5 best (1.29136):     {best['M_cert_lower'] - 1.29136:+.6f}")
    print(f"  vs CS17 (1.28020):        {best['M_cert_lower'] - 1.28020:+.6f}")
    print(f"  empirical MV-arcsine ceiling ~1.2924")
    print("=" * 78)


def _write(path, all_runs, best_overall, n_g_list):
    out = {
        "configuration": {
            "delta_1": float(DELTA1_Q),
            "u": float(U_Q),
            "prec_bits": PREC_BITS,
            "N_G_values": n_g_list,
        },
        "runs": all_runs,
        "best_overall": best_overall,
        "baselines": {
            "v4_3sc_N200": 1.29216,
            "v5_4sc_N119": 1.29136,
            "cs17_paper": 1.28020,
            "MV_numerical": 1.27428,
            "MV_arcsine_empirical_ceiling": 1.2924,
        },
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
