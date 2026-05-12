"""Cohn--Elkies multi-scale arcsine certifier v5 -- N-scale generalization.

Extends v3 (2-scale + rigorous arb K_2 + G re-optimization) to arbitrary
N >= 1 scales.  K_hat(xi) = sum_i lambda_i J_0(pi delta_i xi)^2 with
sum_i lambda_i = 1, lambda_i >= 0, delta_i in (0, 1/2].

Rigorous pieces (every endpoint is a flint.arb interval):
  * k_1 = K_hat(1)
  * K_2 = 2 * int_0^XI_MAX K_hat(xi)^2 d xi  +  Bessel-tail correction
         tail bound:  (8/pi^2) * (sum_i lambda_i / delta_i)^2 / XI_MAX
  * S_1 upper:  sum_{j=1}^{N_QP} a_j^2 / K_hat(j/u)
  * min_G lower:  dense grid argmin + Lipschitz remainder of G' on [0, 1/4]
  * gain a >= (4/u) * (min_G)^2 / S_1
  * M_* lower via the MV master-inequality bisection

The G in front is solved as a QP (n_modes = 119 cosines) for the *actual*
N-scale K_hat using cvxpy (MOSEK / Clarabel / SCS).

Goal: push rigorous M_cert above v3's 1.28984.

Usage:
    python _cohn_elkies_128_v5.py                  # default 3-scale + 4-scale + frontier
    python _cohn_elkies_128_v5.py --quick          # just the headline config
    python _cohn_elkies_128_v5.py --xi-max 100000  # tighter tail for the best point
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
DELTA1_Q = Fraction(138, 1000)
U_Q = Fraction(1, 2) + DELTA1_Q
N_QP = 119
RATIONAL_DEN = 10 ** 12
QP_GRID = 5001
ARB_GRID_DEFAULT = 200001
PREC_BITS = 256

DELTA1 = U = PI = TWO_OVER_U = None


def Q_arb(q: Fraction) -> arb:
    return arb(q.numerator) / arb(q.denominator)


def configure_precision(prec_bits: int = PREC_BITS) -> None:
    global DELTA1, U, PI, TWO_OVER_U
    flint.ctx.prec = prec_bits
    DELTA1 = Q_arb(DELTA1_Q)
    U = Q_arb(U_Q)
    PI = arb.pi()
    TWO_OVER_U = arb(2) / U


# ---------------------------------------------------------------------------
# K_hat (generic N-scale)
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
# K_2 (rigorous, generic N-scale).  K_hat^2 already expands into the full
# Sum_{i,j} lambda_i lambda_j J_0(pi d_i xi)^2 J_0(pi d_j xi)^2 inside the
# integrand, so we just integrate K_hat^2 directly.
# ---------------------------------------------------------------------------
def K2_rigorous(deltas_arb, lambdas_arb, xi_max: int,
                eval_limit: int = 10**7):
    pi_d = [PI * d for d in deltas_arb]

    def _integrand(x, analytic):
        Kh = arb(0)
        for lam, pd in zip(lambdas_arb, pi_d):
            j = (pd * x).bessel_j(0)
            Kh = Kh + lam * j * j
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
    mid = 0.5 * (float(lo.lower()) + float(hi.upper()))
    rad = 0.5 * (float(hi.upper()) - float(lo.lower())) + 1e-30
    K_2_arb = arb(repr(mid), repr(rad))
    return K_2_arb, bulk, tail_upper


# ---------------------------------------------------------------------------
# QP for G coefficients given an N-scale K_hat
# ---------------------------------------------------------------------------
def solve_qp_Nscale(deltas_q, lambdas_q, n_grid=QP_GRID, n_modes=N_QP):
    import cvxpy as cp
    from scipy.special import j0
    deltas_f = [float(q) for q in deltas_q]
    lambdas_f = [float(q) for q in lambdas_q]
    u_f = float(U_Q)

    w = np.zeros(n_modes)
    for j in range(1, n_modes + 1):
        xi_j = j / u_f
        v = 0.0
        for lam, d in zip(lambdas_f, deltas_f):
            v += lam * j0(math.pi * d * xi_j) ** 2
        w[j - 1] = v
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


def round_to_rationals(a_opt, denom=RATIONAL_DEN):
    return [Fraction(int(round(a * denom)), denom) for a in a_opt]


# ---------------------------------------------------------------------------
# Rigorous min G over [0, 1/4]
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
# Master inequality
# ---------------------------------------------------------------------------
def sup_R_upper(M_arb, k_1, K_2_up):
    one = arb(1)
    two = arb(2)
    if M_arb <= one:
        return arb("inf")
    mu = M_arb * (PI / M_arb).sin() / PI
    two_mu_sq = two * mu * mu
    rad1 = M_arb - one - two_mu_sq
    rad2 = K_2_up - one - two * k_1 * k_1
    no_z1 = M_arb + one + ((M_arb - one) * (K_2_up - one)).sqrt()
    if float(rad1.lower()) <= 0.0:
        return no_z1
    refined = M_arb + one + two * mu * k_1 + (rad1 * rad2).sqrt()
    return arb.max(refined, no_z1)


def certify_M_lower(M_arb, k_1, K_2_up, a_lo):
    sup_R = sup_R_upper(M_arb, k_1, K_2_up)
    target = TWO_OVER_U + a_lo
    return float(sup_R.upper()) < float(target.lower()), sup_R, target


def find_M_lower_bisect(k_1, K_2_up, a_lo,
                        M_lo_init=1.001, M_hi_init=1.30, n_iter=80):
    lo, hi = M_lo_init, M_hi_init
    ok, _, _ = certify_M_lower(arb(repr(lo)), k_1, K_2_up, a_lo)
    if not ok:
        return arb(1)
    ok_hi, _, _ = certify_M_lower(arb(repr(hi)), k_1, K_2_up, a_lo)
    if ok_hi:
        hi = 1.5
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        ok, _, _ = certify_M_lower(arb(repr(mid)), k_1, K_2_up, a_lo)
        if ok:
            lo = mid
        else:
            hi = mid
    return arb(repr(lo))


# ---------------------------------------------------------------------------
# Full rigorous certification at a generic (deltas_q, lambdas_q, coeffs_q)
# ---------------------------------------------------------------------------
def certify_Nscale(deltas_q, lambdas_q, coeffs_q,
                   xi_max: int, prec_bits: int = PREC_BITS,
                   arb_grid: int = ARB_GRID_DEFAULT,
                   verbose: bool = False) -> dict:
    configure_precision(prec_bits)
    assert sum(lambdas_q) == Fraction(1), f"lambdas don't sum to 1: {sum(lambdas_q)}"

    deltas_arb = [Q_arb(q) for q in deltas_q]
    lambdas_arb = [Q_arb(q) for q in lambdas_q]
    coeffs_arb = [Q_arb(q) for q in coeffs_q]
    coeffs_float = [float(q) for q in coeffs_q]

    if verbose:
        print(f"  deltas  = {[float(q) for q in deltas_q]}")
        print(f"  lambdas = {[float(q) for q in lambdas_q]}")
        print(f"  XI_MAX  = {xi_max}  prec = {prec_bits}")

    k_1 = k1_enclosure(deltas_arb, lambdas_arb)
    if verbose:
        print(f"  k_1 ~ {float(k_1.mid()):.10f}")

    t0 = time.time()
    min_G_lo, lip_rem = min_G_lower_bound(coeffs_arb, coeffs_float, n_grid=arb_grid)
    cond_G2 = float(min_G_lo.lower()) > 0
    t_minG = time.time() - t0
    if verbose:
        print(f"  min_G >= {float(min_G_lo.lower()):.6f}  G2={cond_G2}  ({t_minG:.1f}s)")

    S1_hi = S1_upper_bound(coeffs_arb, deltas_arb, lambdas_arb)
    if verbose:
        print(f"  S_1 <= {float(S1_hi.upper()):.4f}")

    t0 = time.time()
    K_2_arb, bulk_arb, tail_up = K2_rigorous(deltas_arb, lambdas_arb, xi_max=xi_max)
    t_K2 = time.time() - t0
    if verbose:
        print(f"  K_2 in [{float(K_2_arb.lower()):.6f}, "
              f"{float(K_2_arb.upper()):.6f}]  tail<={float(tail_up.upper()):.2e}  "
              f"({t_K2:.1f}s)")

    min_G_lo_pos = arb.max(min_G_lo, arb(0))
    a_gain_lo = (arb(4) / U) * (min_G_lo_pos * min_G_lo_pos) / S1_hi

    K_2_upper = arb(float(K_2_arb.upper()))
    M_lo = find_M_lower_bisect(k_1, K_2_upper, a_gain_lo)
    M_lo_f = float(M_lo.lower())
    if verbose:
        print(f"  M_cert >= {M_lo_f:.8f}")

    return {
        "deltas": [float(q) for q in deltas_q],
        "lambdas": [float(q) for q in lambdas_q],
        "n_scales": len(deltas_q),
        "n_modes": len(coeffs_arb),
        "xi_max": xi_max,
        "prec_bits": prec_bits,
        "arb_grid": arb_grid,
        "k_1_mid": float(k_1.mid()),
        "min_G_lower": float(min_G_lo.lower()),
        "G2_OK": bool(cond_G2),
        "S_1_upper": float(S1_hi.upper()),
        "K_2_lower": float(K_2_arb.lower()),
        "K_2_upper": float(K_2_arb.upper()),
        "K_2_tail_upper": float(tail_up.upper()),
        "a_gain_lower": float(a_gain_lo.lower()),
        "M_cert_lower": M_lo_f,
        "time_K2_seconds": t_K2,
        "time_minG_seconds": t_minG,
    }


# ---------------------------------------------------------------------------
# Convenience: solve_qp and certify combined
# ---------------------------------------------------------------------------
def certify_with_reopt(deltas_q, lambdas_q, xi_max, verbose=False):
    configure_precision(PREC_BITS)
    a_opt = solve_qp_Nscale(deltas_q, lambdas_q)
    coeffs_q = round_to_rationals(a_opt)
    return certify_Nscale(deltas_q, lambdas_q, coeffs_q, xi_max=xi_max, verbose=verbose)


# ---------------------------------------------------------------------------
# Main: sweep over 2-, 3-, 4-scale configs around the known best.
# ---------------------------------------------------------------------------
def _norm_lambdas(lams_f):
    """Normalize a float lambda vector to sum=1 as exact Fractions."""
    den = 10 ** 6
    nums = [int(round(l * den)) for l in lams_f]
    s = sum(nums)
    if s == 0:
        raise ValueError("zero lambdas")
    fracs = [Fraction(n, s) for n in nums]
    # ensure exact sum to 1
    drift = Fraction(1) - sum(fracs)
    fracs[-1] = fracs[-1] + drift
    return fracs


def _Q(x, den=10**6):
    return Fraction(int(round(x * den)), den)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run only the headline 3-scale config.")
    parser.add_argument("--xi-max", type=int, default=10**4,
                        help="XI_MAX for sweeps (default 1e4).")
    parser.add_argument("--xi-max-best", type=int, default=10**5,
                        help="XI_MAX to re-certify the best point.")
    parser.add_argument("--out", type=str,
                        default=str(_HERE / "_cohn_elkies_128_v5_results.json"))
    args = parser.parse_args()

    configure_precision(PREC_BITS)
    print("=" * 76)
    print("Cohn--Elkies N-scale arcsine certifier v5")
    print(f"  delta_1 (fixed) = {float(DELTA1_Q):.4f}   u = {float(U_Q):.4f}")
    print(f"  n_modes (G QP) = {N_QP}     PREC = {PREC_BITS} bits")
    print(f"  XI_MAX (sweep) = {args.xi_max}    XI_MAX (best) = {args.xi_max_best}")
    print("=" * 76)

    # v3 reference (2-scale, delta_2=0.046, lambda_1=0.85, G reoptimized)
    print("\n[Reference] 2-scale v3 best: (d_2=0.046, l_1=0.85)")
    deltas_ref = [DELTA1_Q, Fraction(46, 1000)]
    lambdas_ref = [Fraction(85, 100), Fraction(15, 100)]
    r_ref = certify_with_reopt(deltas_ref, lambdas_ref,
                               xi_max=args.xi_max, verbose=True)
    print(f"  -> rigorous M_cert = {r_ref['M_cert_lower']:.6f}\n")

    all_runs = [{"tag": "v3-ref 2sc (0.138,0.046; 0.85)", **r_ref}]

    if args.quick:
        best_overall = max(all_runs, key=lambda r: r["M_cert_lower"])
        _write(args.out, all_runs, best_overall, r_ref)
        _final_summary(all_runs, r_ref)
        return

    # ---------------- 3-scale grid ----------------
    print("=" * 76)
    print("3-SCALE SWEEP")
    print("=" * 76)
    # Anchor (delta_1=0.138, delta_2~0.046) and add a third atom in (0.020, 0.10)
    # with lambda_3 small (5-15%) drawn from lambda_2.
    three_scale_configs = []
    # Around v3 best: delta_2 in {0.040, 0.046, 0.055}, delta_3 in {0.025, 0.030, 0.070}
    for d2 in [0.040, 0.046, 0.055]:
        for d3 in [0.025, 0.030, 0.070, 0.090]:
            if d3 == d2 or d3 >= 0.10:
                continue
            for split in [
                (0.85, 0.10, 0.05),
                (0.80, 0.10, 0.10),
                (0.85, 0.07, 0.08),
                (0.78, 0.15, 0.07),
                (0.82, 0.12, 0.06),
            ]:
                three_scale_configs.append((d2, d3, split))

    print(f"  Testing {len(three_scale_configs)} 3-scale configurations...")
    print(f"  {'d2':>6} {'d3':>6} {'l1':>5} {'l2':>5} {'l3':>5} "
          f"{'M_cert':>10} {'minG':>7} {'S1':>7} {'K2hi':>7}")
    best3 = None
    for (d2, d3, (l1, l2, l3)) in three_scale_configs:
        deltas_q = [DELTA1_Q, _Q(d2, 1000), _Q(d3, 1000)]
        lambdas_q = _norm_lambdas([l1, l2, l3])
        t0 = time.time()
        try:
            r = certify_with_reopt(deltas_q, lambdas_q, xi_max=args.xi_max)
        except Exception as exc:
            print(f"  {d2:>6.3f} {d3:>6.3f} {l1:>5.2f} {l2:>5.2f} {l3:>5.2f}   ERROR: {exc}")
            continue
        el = time.time() - t0
        marker = ""
        if best3 is None or r["M_cert_lower"] > best3["M_cert_lower"]:
            best3 = r
            best3["tag"] = f"3sc d=({d2},{d3}) l=({l1},{l2},{l3})"
            marker = " *"
        print(f"  {d2:>6.3f} {d3:>6.3f} {l1:>5.2f} {l2:>5.2f} {l3:>5.2f} "
              f"{r['M_cert_lower']:>10.6f} {r['min_G_lower']:>7.4f} "
              f"{r['S_1_upper']:>7.3f} {r['K_2_upper']:>7.4f}  "
              f"({el:.1f}s){marker}")
        r_tag = {"tag": f"3sc d=({d2},{d3}) l=({l1:.2f},{l2:.2f},{l3:.2f})", **r}
        all_runs.append(r_tag)
    print()

    # ---------------- 4-scale ----------------
    print("=" * 76)
    print("4-SCALE SWEEP")
    print("=" * 76)
    four_scale_configs = [
        # (delta_2, delta_3, delta_4, (lambdas))
        (0.046, 0.025, 0.080, (0.82, 0.10, 0.04, 0.04)),
        (0.046, 0.025, 0.090, (0.80, 0.10, 0.05, 0.05)),
        (0.046, 0.020, 0.070, (0.83, 0.10, 0.04, 0.03)),
        (0.046, 0.030, 0.075, (0.80, 0.10, 0.05, 0.05)),
        (0.050, 0.025, 0.080, (0.80, 0.10, 0.05, 0.05)),
        (0.040, 0.025, 0.070, (0.82, 0.08, 0.05, 0.05)),
    ]
    print(f"  {'d2':>6} {'d3':>6} {'d4':>6} {'lambdas':>22} "
          f"{'M_cert':>10} {'minG':>7} {'S1':>7} {'K2hi':>7}")
    best4 = None
    for (d2, d3, d4, lams) in four_scale_configs:
        deltas_q = [DELTA1_Q, _Q(d2, 1000), _Q(d3, 1000), _Q(d4, 1000)]
        lambdas_q = _norm_lambdas(list(lams))
        t0 = time.time()
        try:
            r = certify_with_reopt(deltas_q, lambdas_q, xi_max=args.xi_max)
        except Exception as exc:
            print(f"  {d2:>6.3f} {d3:>6.3f} {d4:>6.3f}   ERROR: {exc}")
            continue
        el = time.time() - t0
        marker = ""
        if best4 is None or r["M_cert_lower"] > best4["M_cert_lower"]:
            best4 = r
            marker = " *"
        lam_str = "(" + ",".join(f"{l:.2f}" for l in lams) + ")"
        print(f"  {d2:>6.3f} {d3:>6.3f} {d4:>6.3f} {lam_str:>22} "
              f"{r['M_cert_lower']:>10.6f} {r['min_G_lower']:>7.4f} "
              f"{r['S_1_upper']:>7.3f} {r['K_2_upper']:>7.4f}  "
              f"({el:.1f}s){marker}")
        all_runs.append({"tag": f"4sc d=({d2},{d3},{d4}) l={lam_str}", **r})
    print()

    # ---------------- Best re-certify at XI_MAX = 1e5 ----------------
    best_overall = max(all_runs, key=lambda r: r["M_cert_lower"])
    print("=" * 76)
    print(f"Re-certify BEST at XI_MAX={args.xi_max_best}: {best_overall['tag']}")
    print("=" * 76)
    deltas_q_best = [_Q(d, 10**6) if i > 0 else DELTA1_Q
                     for i, d in enumerate(best_overall["deltas"])]
    lambdas_q_best = _norm_lambdas(best_overall["lambdas"])
    r_best_hi = certify_with_reopt(deltas_q_best, lambdas_q_best,
                                   xi_max=args.xi_max_best, verbose=True)
    r_best_hi_tag = {"tag": best_overall["tag"] + f" @xi={args.xi_max_best}",
                     **r_best_hi}
    all_runs.append(r_best_hi_tag)
    if r_best_hi["M_cert_lower"] > best_overall["M_cert_lower"]:
        best_overall = r_best_hi_tag

    _final_summary(all_runs, r_ref)
    _write(args.out, all_runs, best_overall, r_ref)


def _final_summary(all_runs, r_ref):
    print()
    print("=" * 76)
    print("FINAL SUMMARY")
    print("=" * 76)
    print(f"  {'tag':<48}  {'M_cert':>10}  {'K2hi':>7}")
    print(f"  {'-'*48}  {'-'*10}  {'-'*7}")
    sorted_runs = sorted(all_runs, key=lambda r: -r["M_cert_lower"])
    for r in sorted_runs[:15]:
        tag = r.get("tag", "?")[:48]
        print(f"  {tag:<48}  {r['M_cert_lower']:>10.6f}  {r['K_2_upper']:>7.4f}")
    best = sorted_runs[0]
    print()
    print(f"BEST rigorous M_cert: {best['M_cert_lower']:.8f}")
    print(f"  tag: {best.get('tag','?')}")
    print(f"  vs v3 (1.28984): {best['M_cert_lower'] - 1.28984:+.6f}")
    print(f"  vs v3 ref-here:  {best['M_cert_lower'] - r_ref['M_cert_lower']:+.6f}")
    print("=" * 76)


def _write(path, all_runs, best_overall, r_ref):
    out = {
        "configuration": {
            "delta_1": float(DELTA1_Q),
            "u": float(U_Q),
            "n_modes": N_QP,
            "prec_bits": PREC_BITS,
        },
        "v3_reference_2sc": r_ref,
        "runs": all_runs,
        "best_overall": best_overall,
        "baselines": {
            "v3_paper": 1.28984,
            "MV_numerical": 1.27428,
        },
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
