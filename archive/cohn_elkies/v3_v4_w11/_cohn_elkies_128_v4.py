"""Cohn--Elkies multi-scale arcsine certifier v4 -- larger N + 3-scale arcsine.

Builds directly on `_cohn_elkies_128_v3.py` but adds:

  (1) Variable N (number of QP cosine modes for G).  At N=119 v3 had
      M_cert >= 1.289841.  Larger N reduces S_1; we try N in {200, 500, 1000}.
      QP grid is scaled so n_grid >= 4*N + 1.

  (2) 3-scale arcsine kernel K_hat(xi) = sum_i lambda_i * J_0(pi * delta_i * xi)^2,
      with i in {1,2,3}.  Three CROSS integrals I(delta_i, delta_j) are evaluated
      rigorously via acb.integral + Bessel tail.

Every output is a rigorous arb (or float upper/lower bound from arb).  G coeffs
are rationalized to Fraction with denom 10^12, then verified in arb.

Usage:
    python _cohn_elkies_128_v4.py --plan twoscale-N200
    python _cohn_elkies_128_v4.py --plan twoscale-N500
    python _cohn_elkies_128_v4.py --plan twoscale-N1000
    python _cohn_elkies_128_v4.py --plan threescale-N200 --sweep-3s
    python _cohn_elkies_128_v4.py --plan all
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

import flint
from flint import acb, arb


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DELTA1_Q = Fraction(138, 1000)              # 0.138 (MV's delta_1)
U_Q = Fraction(1, 2) + DELTA1_Q              # 0.638
RATIONAL_DEN = 10 ** 12
ARB_GRID_DEFAULT = 200001

# Globals filled in by configure_precision
DELTA1 = U = PI = TWO_OVER_U = None


def Q_arb(q: Fraction) -> arb:
    return arb(q.numerator) / arb(q.denominator)


def configure_precision(prec_bits: int) -> None:
    global DELTA1, U, PI, TWO_OVER_U
    flint.ctx.prec = prec_bits
    DELTA1 = Q_arb(DELTA1_Q)
    U = Q_arb(U_Q)
    PI = arb.pi()
    TWO_OVER_U = arb(2) / U


# ---------------------------------------------------------------------------
# arb K_hat (multi-scale, generic in deltas/lambdas)
# ---------------------------------------------------------------------------
def K_hat_arb(xi: arb, deltas_arb, lambdas_arb) -> arb:
    out = arb(0)
    for lam, d in zip(lambdas_arb, deltas_arb):
        j0_val = (PI * d * xi).bessel_j(0)
        out = out + lam * j0_val * j0_val
    return out


def k1_enclosure(deltas_arb, lambdas_arb) -> arb:
    return K_hat_arb(arb(1), deltas_arb, lambdas_arb)


# ---------------------------------------------------------------------------
# K_2 rigorous: 2 * int_0^XI_MAX K_hat^2 d xi  +  Bessel tail bound
# ---------------------------------------------------------------------------
def K2_rigorous(deltas_arb, lambdas_arb, xi_max: int,
                eval_limit: int = 10**7):
    """Return (K_2_arb, bulk_arb, tail_upper_arb).

    K_2 = 2*int_0^infty K_hat^2 = 2*I_bulk + R_tail,
       0 <= R_tail <= (8/pi^2) * C^2 / XI_MAX,   C = sum_i lam_i / delta_i.

    Works for ANY number of scales (Bessel tail bound holds termwise).
    """
    pi_d = [PI * d for d in deltas_arb]

    def _integrand(x, analytic):
        Kh = arb(0)
        for lam, pd in zip(lambdas_arb, pi_d):
            j0_val = (pd * x).bessel_j(0)
            Kh = Kh + lam * j0_val * j0_val
        return Kh * Kh

    I_bulk_acb = acb.integral(_integrand, 0, xi_max, eval_limit=eval_limit)
    I_bulk = I_bulk_acb.real
    bulk = arb(2) * I_bulk

    C = arb(0)
    for lam, d in zip(lambdas_arb, deltas_arb):
        C = C + lam / d
    tail_upper = (arb(8) / (PI * PI)) * (C * C) / arb(xi_max)

    lo = arb(float(bulk.lower()))
    hi = bulk + tail_upper
    mid_str = repr(0.5 * (float(lo.lower()) + float(hi.upper())))
    rad_str = repr(0.5 * (float(hi.upper()) - float(lo.lower())) + 1e-30)
    K_2_arb = arb(mid_str, rad_str)
    return K_2_arb, bulk, tail_upper


# ---------------------------------------------------------------------------
# G handling
# ---------------------------------------------------------------------------
def round_to_rationals(a_opt, denom=RATIONAL_DEN):
    return [Fraction(int(round(a * denom)), denom) for a in a_opt]


def solve_qp(deltas_q, lambdas_q, n_modes: int, n_grid: int = None,
             verbose: bool = False):
    """QP for G(x) = sum a_j cos(2 pi j x / u)  s.t. G(x) >= 1 on [0, 1/4],
    minimizing S_1 = sum a_j^2 / K_hat(j/u).

    Supports any number of scales via deltas_q, lambdas_q (Fractions).
    """
    import cvxpy as cp
    from scipy.special import j0
    deltas_f = [float(d) for d in deltas_q]
    lambdas_f = [float(l) for l in lambdas_q]
    u_f = float(U_Q)

    if n_grid is None:
        n_grid = max(5001, 4 * n_modes + 1)

    w = np.zeros(n_modes)
    for j in range(1, n_modes + 1):
        xi_j = j / u_f
        v = 0.0
        for lam, d in zip(lambdas_f, deltas_f):
            v += lam * j0(math.pi * d * xi_j) ** 2
        w[j - 1] = v
    if w.min() <= 0:
        raise RuntimeError(f"w has nonpositive entry: min={w.min()} at j={int(w.argmin())+1}")

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
                if verbose:
                    print(f"   QP solved with {s_name}; status={prob.status}")
                break
        except Exception as exc:
            if verbose:
                print(f"   QP solver {s_name} failed: {exc}")
            continue
    if a.value is None:
        raise RuntimeError(f"QP failed; status={prob.status}")
    return np.asarray(a.value).flatten()


# ---------------------------------------------------------------------------
# Rigorous min G on [0, 1/4] via Lipschitz + dense grid
# ---------------------------------------------------------------------------
def lipschitz_upper(coeffs_arb) -> arb:
    two_pi_over_u = (arb(2) * PI) / U
    total = arb(0)
    for j, a in enumerate(coeffs_arb, start=1):
        total = total + arb(j) * abs(a)
    return two_pi_over_u * total


def min_G_lower_bound(coeffs_arb, coeffs_float, n_grid: int = ARB_GRID_DEFAULT):
    h = arb(1) / (arb(4) * arb(n_grid))
    L = lipschitz_upper(coeffs_arb)

    xs = np.linspace(0.0, 0.25, n_grid + 1)
    vals = np.zeros_like(xs)
    u_f = float(U.mid())
    for j, aj in enumerate(coeffs_float, start=1):
        vals += aj * np.cos(2 * np.pi * j * xs / u_f)
    idx_min = int(vals.argmin())
    x_min_q = Fraction(idx_min, 4 * n_grid)
    x_min = arb(x_min_q.numerator) / arb(x_min_q.denominator)

    G_at_min = arb(0)
    two_pi_over_u = (arb(2) * PI) / U
    for j, aj in enumerate(coeffs_arb, start=1):
        G_at_min = G_at_min + aj * (two_pi_over_u * arb(j) * x_min).cos()

    min_grid_lo = G_at_min - arb("1e-30")
    bound = min_grid_lo - L * h
    return bound, L * h


# ---------------------------------------------------------------------------
# Rigorous S_1 upper bound
# ---------------------------------------------------------------------------
def S1_upper_bound(coeffs_arb, deltas_arb, lambdas_arb) -> arb:
    inv_u = arb(1) / U
    total = arb(0)
    for j, a in enumerate(coeffs_arb, start=1):
        xi = arb(j) * inv_u
        kh = K_hat_arb(xi, deltas_arb, lambdas_arb)
        term = (a * a) / kh
        total = total + term
    return total


# ---------------------------------------------------------------------------
# Master inequality: rigorous lower bound on M_*
# ---------------------------------------------------------------------------
def sup_R_upper(M_arb: arb, k_1: arb, K_2_up: arb) -> arb:
    one = arb(1)
    two = arb(2)
    if M_arb <= one:
        return arb("inf")
    mu_arb = M_arb * (PI / M_arb).sin() / PI
    two_mu_sq = two * mu_arb * mu_arb
    rad1 = M_arb - one - two_mu_sq
    rad2 = K_2_up - one - two * k_1 * k_1
    no_z1 = M_arb + one + ((M_arb - one) * (K_2_up - one)).sqrt()
    if float(rad1.lower()) <= 0.0:
        return no_z1
    refined = M_arb + one + two * mu_arb * k_1 + (rad1 * rad2).sqrt()
    return arb.max(refined, no_z1)


def certify_M_lower(M_candidate_arb, k_1, K_2_up, a_lower):
    sup_R = sup_R_upper(M_candidate_arb, k_1, K_2_up)
    target = TWO_OVER_U + a_lower
    ok = float(sup_R.upper()) < float(target.lower())
    return ok, sup_R, target


def find_M_lower_bisect(k_1, K_2_up, a_lower,
                        M_lo_init=1.001, M_hi_init=1.30, n_iter=80) -> arb:
    lo, hi = M_lo_init, M_hi_init
    ok_lo, _, _ = certify_M_lower(arb(repr(lo)), k_1, K_2_up, a_lower)
    if not ok_lo:
        return arb(1)
    ok_hi, _, _ = certify_M_lower(arb(repr(hi)), k_1, K_2_up, a_lower)
    if ok_hi:
        hi = 1.5
        ok_hi2, _, _ = certify_M_lower(arb(repr(hi)), k_1, K_2_up, a_lower)
        if ok_hi2:
            return arb(repr(hi))
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        ok, _, _ = certify_M_lower(arb(repr(mid)), k_1, K_2_up, a_lower)
        if ok:
            lo = mid
        else:
            hi = mid
    return arb(repr(lo))


# ---------------------------------------------------------------------------
# One full rigorous certification.
# ---------------------------------------------------------------------------
def certify(deltas_q, lambdas_q, coeffs_q,
            xi_max: int, prec_bits: int,
            arb_grid: int = ARB_GRID_DEFAULT,
            verbose: bool = True) -> dict:
    configure_precision(prec_bits)

    deltas_arb = [Q_arb(d) for d in deltas_q]
    lambdas_arb = [Q_arb(l) for l in lambdas_q]
    coeffs_arb = [Q_arb(q) for q in coeffs_q]
    coeffs_float = [float(q) for q in coeffs_q]
    n_modes = len(coeffs_arb)

    if verbose:
        ds = "[" + ", ".join(f"{float(d):.4f}" for d in deltas_q) + "]"
        ls = "[" + ", ".join(f"{float(l):.4f}" for l in lambdas_q) + "]"
        print(f"  deltas = {ds}  lambdas = {ls}")
        print(f"  n_modes = {n_modes}  XI_MAX = {xi_max}  prec = {prec_bits} bits  arb_grid = {arb_grid}")

    k_1 = k1_enclosure(deltas_arb, lambdas_arb)
    if verbose:
        print(f"  [arb] k_1 mid={float(k_1.mid()):.10f}  rad<={float(k_1.rad()):.2e}")

    t0 = time.time()
    min_G_lo, lip_rem = min_G_lower_bound(coeffs_arb, coeffs_float, n_grid=arb_grid)
    cond_G2 = float(min_G_lo.lower()) > 0
    t_minG = time.time() - t0
    if verbose:
        print(f"  [arb] min_G >= {float(min_G_lo.lower()):.8f} "
              f"(lip rem <= {float(lip_rem.upper()):.2e}) "
              f"G2 = {'OK' if cond_G2 else 'FAIL'}  ({t_minG:.1f}s)")

    S1_hi = S1_upper_bound(coeffs_arb, deltas_arb, lambdas_arb)
    if verbose:
        print(f"  [arb] S_1 <= {float(S1_hi.upper()):.6f}")

    t0 = time.time()
    K_2_arb, bulk_arb, tail_up = K2_rigorous(deltas_arb, lambdas_arb, xi_max=xi_max)
    t_K2 = time.time() - t0
    if verbose:
        print(f"  [arb] K_2 in [{float(K_2_arb.lower()):.8f}, {float(K_2_arb.upper()):.8f}]  "
              f"tail <= {float(tail_up.upper()):.2e}  ({t_K2:.1f}s)")

    min_G_lo_pos = arb.max(min_G_lo, arb(0))
    a_gain_lo = (arb(4) / U) * (min_G_lo_pos * min_G_lo_pos) / S1_hi
    if verbose:
        print(f"  [arb] gain a >= {float(a_gain_lo.lower()):.6f}")

    K_2_upper = arb(float(K_2_arb.upper()))
    M_lo = find_M_lower_bisect(k_1, K_2_upper, a_gain_lo)
    M_lo_f = float(M_lo.lower())
    if verbose:
        print(f"  [arb] certified M_* >= {M_lo_f:.8f}")

    return {
        "deltas": [float(d) for d in deltas_q],
        "lambdas": [float(l) for l in lambdas_q],
        "n_scales": len(deltas_q),
        "n_modes": n_modes,
        "xi_max": xi_max,
        "prec_bits": prec_bits,
        "arb_grid": arb_grid,
        "k_1_mid": float(k_1.mid()),
        "k_1_rad": float(k_1.rad()),
        "min_G_lower": float(min_G_lo.lower()),
        "lip_remainder_upper": float(lip_rem.upper()),
        "G2_OK": bool(cond_G2),
        "S_1_upper": float(S1_hi.upper()),
        "K_2_lower": float(K_2_arb.lower()),
        "K_2_upper": float(K_2_arb.upper()),
        "K_2_bulk_lower": float(bulk_arb.lower()),
        "K_2_bulk_upper": float(bulk_arb.upper()),
        "K_2_tail_upper": float(tail_up.upper()),
        "a_gain_lower": float(a_gain_lo.lower()),
        "M_cert_lower": M_lo_f,
        "time_K2_seconds": t_K2,
        "time_minG_seconds": t_minG,
    }


# ---------------------------------------------------------------------------
# G optimization with optional inflation if min_G < 1 (sanity safety net).
# ---------------------------------------------------------------------------
def solve_qp_safe(deltas_q, lambdas_q, n_modes, scale_factor=1.0):
    a = solve_qp(deltas_q, lambdas_q, n_modes=n_modes)
    a = a * scale_factor
    coeffs_q = round_to_rationals(a, RATIONAL_DEN)
    return coeffs_q, a


# ---------------------------------------------------------------------------
# Main: drive the sweeps
# ---------------------------------------------------------------------------
def run_twoscale(N: int, delta_2_list, lambda_1_list, xi_max=10**4,
                 prec_bits=256, arb_grid=ARB_GRID_DEFAULT):
    """Two-scale (delta_1 fixed at 0.138) sweep at the given N."""
    configure_precision(prec_bits)
    results = []
    best = None
    cache_coeffs = {}
    print(f"\n--- 2-scale N={N} sweep (xi_max={xi_max}) ---")
    print(f"  {'d_2':>7} {'lam_1':>8} {'M_cert':>10} {'minG':>8} {'S_1':>9} {'K2hi':>8} {'time':>7}")
    for d2 in delta_2_list:
        for l1 in lambda_1_list:
            deltas_q = [DELTA1_Q, d2]
            lambdas_q = [l1, Fraction(1) - l1]
            t0 = time.time()
            try:
                coeffs_q, _ = solve_qp_safe(deltas_q, lambdas_q, N)
                r = certify(deltas_q, lambdas_q, coeffs_q,
                            xi_max=xi_max, prec_bits=prec_bits,
                            arb_grid=arb_grid, verbose=False)
            except Exception as exc:
                print(f"  {float(d2):>7.4f} {float(l1):>8.4f}   ERROR: {exc}")
                continue
            el = time.time() - t0
            marker = ""
            if best is None or r["M_cert_lower"] > best["M_cert_lower"]:
                best = r
                best_key = (d2, l1, tuple(coeffs_q))
                marker = " *"
            print(f"  {float(d2):>7.4f} {float(l1):>8.4f} "
                  f"{r['M_cert_lower']:>10.6f} {r['min_G_lower']:>8.4f} "
                  f"{r['S_1_upper']:>9.3f} {r['K_2_upper']:>8.4f} {el:>6.1f}s{marker}")
            results.append(r)
            cache_coeffs[(float(d2), float(l1))] = coeffs_q
    # high xi_max re-run for best
    if best is not None and xi_max < 10**5:
        d2_best = best["deltas"][1]
        l1_best = best["lambdas"][0]
        key = None
        for k in cache_coeffs:
            if abs(k[0] - d2_best) < 1e-9 and abs(k[1] - l1_best) < 1e-9:
                key = k
                break
        if key is not None:
            d2_q = Fraction(int(round(d2_best * 1000)), 1000)
            l1_q = Fraction(int(round(l1_best * 1000)), 1000)
            # find matching Fraction objects
            for q in delta_2_list:
                if abs(float(q) - d2_best) < 1e-9:
                    d2_q = q
                    break
            for q in lambda_1_list:
                if abs(float(q) - l1_best) < 1e-9:
                    l1_q = q
                    break
            print(f"  [Best 2-scale re-run at xi_max=1e5]")
            r_hi = certify([DELTA1_Q, d2_q], [l1_q, Fraction(1) - l1_q],
                           cache_coeffs[key], xi_max=10**5, prec_bits=prec_bits,
                           arb_grid=arb_grid, verbose=True)
            results.append(r_hi)
            if r_hi["M_cert_lower"] > best["M_cert_lower"]:
                best = r_hi
    return results, best


def run_threescale(N: int, sweep_grid, xi_max=10**4,
                   prec_bits=256, arb_grid=ARB_GRID_DEFAULT):
    """3-scale sweep: (delta_2, delta_3, lambda_1, lambda_2) with delta_1 = 0.138.

    sweep_grid: list of dict-like points {"d2","d3","l1","l2"} with Fractions.
    """
    configure_precision(prec_bits)
    results = []
    best = None
    print(f"\n--- 3-scale N={N} sweep (xi_max={xi_max}) ---")
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
                        arb_grid=arb_grid, verbose=False)
        except Exception as exc:
            print(f"  {float(d2):>6.3f} {float(d3):>6.3f} {float(l1):>6.3f} "
                  f"{float(l2):>6.3f}   ERROR: {exc}")
            continue
        el = time.time() - t0
        marker = ""
        if best is None or r["M_cert_lower"] > best["M_cert_lower"]:
            best = r
            best["coeffs_q"] = coeffs_q
            best["deltas_q"] = deltas_q
            best["lambdas_q"] = lambdas_q
            marker = " *"
        print(f"  {float(d2):>6.3f} {float(d3):>6.3f} {float(l1):>6.3f} "
              f"{float(l2):>6.3f} {r['M_cert_lower']:>10.6f} {r['min_G_lower']:>8.4f} "
              f"{r['S_1_upper']:>9.3f} {r['K_2_upper']:>8.4f} {el:>6.1f}s{marker}")
        results.append(r)
    # high xi_max re-run for best
    if best is not None and "coeffs_q" in best:
        print(f"  [Best 3-scale re-run at xi_max=1e5]")
        r_hi = certify(best["deltas_q"], best["lambdas_q"], best["coeffs_q"],
                       xi_max=10**5, prec_bits=prec_bits, arb_grid=arb_grid,
                       verbose=True)
        results.append(r_hi)
        # strip non-JSON-safe fields
        best_clean = {k: v for k, v in best.items()
                      if k not in ("coeffs_q", "deltas_q", "lambdas_q")}
        if r_hi["M_cert_lower"] > best_clean["M_cert_lower"]:
            best = r_hi
        else:
            best = best_clean
    return results, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="all",
                        choices=["twoscale-N200", "twoscale-N500",
                                 "twoscale-N1000", "threescale-N200", "all"])
    parser.add_argument("--xi-max", type=int, default=10**4)
    parser.add_argument("--prec", type=int, default=256)
    args = parser.parse_args()

    print("=" * 78)
    print("Cohn--Elkies multi-scale arcsine certifier v4 (large-N + 3-scale)")
    print("=" * 78)
    configure_precision(args.prec)

    all_runs = []
    summary = {}

    # focused 2-scale sweep -- around v3 best (d2=0.046, l1=0.85)
    d2_list = [Fraction(40, 1000), Fraction(43, 1000), Fraction(46, 1000),
               Fraction(50, 1000)]
    l1_list = [Fraction(82, 100), Fraction(85, 100), Fraction(88, 100),
               Fraction(90, 100)]

    if args.plan in ("twoscale-N200", "all"):
        runs, best = run_twoscale(N=200, delta_2_list=d2_list,
                                  lambda_1_list=l1_list,
                                  xi_max=args.xi_max, prec_bits=args.prec)
        all_runs.extend(runs)
        summary["twoscale_N200_best"] = best

    if args.plan in ("twoscale-N500", "all"):
        runs, best = run_twoscale(N=500, delta_2_list=d2_list,
                                  lambda_1_list=l1_list,
                                  xi_max=args.xi_max, prec_bits=args.prec)
        all_runs.extend(runs)
        summary["twoscale_N500_best"] = best

    if args.plan in ("twoscale-N1000", "all"):
        runs, best = run_twoscale(N=1000, delta_2_list=d2_list,
                                  lambda_1_list=l1_list,
                                  xi_max=args.xi_max, prec_bits=args.prec)
        all_runs.extend(runs)
        summary["twoscale_N1000_best"] = best

    if args.plan in ("threescale-N200", "all"):
        # focused 3-scale grid: d2 around 0.046, d3 around 0.020 (smaller scale)
        sweep_3s = []
        for d2 in [Fraction(40, 1000), Fraction(46, 1000), Fraction(55, 1000)]:
            for d3 in [Fraction(15, 1000), Fraction(20, 1000), Fraction(25, 1000)]:
                for l1 in [Fraction(80, 100), Fraction(85, 100), Fraction(88, 100)]:
                    for l2 in [Fraction(8, 100), Fraction(10, 100), Fraction(12, 100)]:
                        if l1 + l2 < Fraction(99, 100):
                            sweep_3s.append({"d2": d2, "d3": d3, "l1": l1, "l2": l2})
        runs, best = run_threescale(N=200, sweep_grid=sweep_3s,
                                    xi_max=args.xi_max, prec_bits=args.prec)
        all_runs.extend(runs)
        summary["threescale_N200_best"] = best

    # final summary
    print()
    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    for tag, r in summary.items():
        if r is None:
            continue
        ds = ",".join(f"{d:.4f}" for d in r["deltas"])
        ls = ",".join(f"{l:.4f}" for l in r["lambdas"])
        print(f"  {tag:<28}  M_cert >= {r['M_cert_lower']:.8f}  "
              f"d=[{ds}]  l=[{ls}]  N={r['n_modes']}  xi={r['xi_max']}")

    best_all = max(all_runs, key=lambda x: x["M_cert_lower"]) if all_runs else None
    print()
    if best_all is not None:
        print(f"BEST RIGOROUS M_cert (v4): {best_all['M_cert_lower']:.8f}")
        print(f"  deltas = {best_all['deltas']}")
        print(f"  lambdas = {best_all['lambdas']}")
        print(f"  N = {best_all['n_modes']}, xi_max = {best_all['xi_max']}")
    print(f"v3 best for reference: 1.28984106")
    print(f"Target threshold:      1.29000000")
    print("=" * 78)

    out = {
        "configuration": {
            "delta_1": float(DELTA1_Q),
            "u": float(U_Q),
        },
        "summary_best_per_plan": summary,
        "all_runs": all_runs,
        "best_overall": best_all,
        "v3_best_reference": 1.28984106,
    }
    json_path = _HERE / "_cohn_elkies_128_v4_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
