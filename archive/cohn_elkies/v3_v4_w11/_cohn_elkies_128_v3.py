"""Cohn--Elkies multi-scale arcsine certifier v3 -- *combined* rigorous K_2 + re-optimized G.

This script merges:

  (A) the rigorous K_2 = 2 * int_0^inf K_hat^2 d xi pipeline of v2
      (`_cohn_elkies_128_v2.py`):  acb.integral on [0, XI_MAX] + Bessel tail bound;
      both endpoints are tracked as flint.arb intervals.

  (B) the G re-optimized coefficients produced by `_cohn_elkies_128_greopt.py`
      (saved in `_cohn_elkies_128_greopt_coeffs.json`):  a fresh QP solution
      that drops S_1 from 87.35 (MV's stock G) to 49.82, *for the K26 best
      multi-scale K_hat (delta_1 = 0.138, delta_2 = 0.055, lambda_1 = 0.9312).*

In v2 the K_2 was rigorous but G was MV's stock 119-coefficient cosine, giving
M_cert >= 1.27974 at XI_MAX = 1e5.  In greopt the G was re-optimized (S_1
nearly halved), but the rigorous K_2 surrogate used was the loose Minkowski
upper bound 4.5059, dragging the rigorous M_cert down to 1.27624.  Plugging
the *direct numerical* K_2 ~ 4.3587 into the greopt master inequality predicted
M_cert ~ 1.2840 -- but that was NOT rigorous.

v3 closes the loop:  every input (k_1, K_2, S_1, min_G) is a flint.arb
interval.  We then run the MV master inequality bisection with the K_2 upper
endpoint from acb.integral + Bessel tail.

We *also* support, optionally, on-the-fly re-optimization of G at each (delta_2,
lambda_1) point in a small sweep around the K26 best, so that for the final
sweep step the G is matched to the K_hat (otherwise we are stuck with the
single fixed K26 G).

Usage:
    python _cohn_elkies_128_v3.py             # default: run K26-point at two XI_MAX, then sweep
    python _cohn_elkies_128_v3.py --no-sweep  # skip the sweep (faster)
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
DELTA1_Q = Fraction(138, 1000)            # 0.138 (MV's delta_1)
U_Q = Fraction(1, 2) + DELTA1_Q            # 0.638
N_QP = 119
RATIONAL_DEN = 10 ** 12
QP_GRID = 5001
ARB_GRID_DEFAULT = 200001                  # grid for rigorous min_G verification
COEFFS_JSON = _HERE / "_cohn_elkies_128_greopt_coeffs.json"

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
# arb plumbing for K_hat (multi-scale, generic in deltas/lambdas)
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
                eval_limit: int = 10**7) -> tuple[arb, arb, arb]:
    """Return (K_2_arb, bulk_arb, tail_upper_arb).

    K_2 = 2 * int_0^infty K_hat^2 = 2 * I_bulk + R_tail,
       0 <= R_tail <= (8/pi^2) * C^2 / XI_MAX,
       C = sum_i lambda_i / delta_i.

    K_2_arb is a rigorous arb interval containing the true K_2.
    """
    pi_d = [PI * d for d in deltas_arb]

    def _integrand(x, analytic):
        j0s = [(pd * x).bessel_j(0) for pd in pi_d]
        Kh = arb(0)
        for lam, j in zip(lambdas_arb, j0s):
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
    mid_str = repr(0.5 * (float(lo.lower()) + float(hi.upper())))
    rad_str = repr(0.5 * (float(hi.upper()) - float(lo.lower())) + 1e-30)
    K_2_arb = arb(mid_str, rad_str)
    return K_2_arb, bulk, tail_upper


# ---------------------------------------------------------------------------
# G handling: load coeffs from JSON (rational) or re-optimize via cvxpy.
# ---------------------------------------------------------------------------
def load_coeffs_from_json(path: Path):
    with open(path) as f:
        d = json.load(f)
    coeffs_q = [Fraction(int(n), int(dd)) for n, dd in d["coeffs_num_den"]]
    delta_1_q = Fraction(int(d["delta_1"][0]), int(d["delta_1"][1]))
    delta_2_q = Fraction(int(d["delta_2"][0]), int(d["delta_2"][1]))
    lambda_1_q = Fraction(int(d["lambda_1"][0]), int(d["lambda_1"][1]))
    return coeffs_q, delta_1_q, delta_2_q, lambda_1_q


def round_to_rationals(a_opt, denom=RATIONAL_DEN):
    return [Fraction(int(round(a * denom)), denom) for a in a_opt]


def solve_qp(delta_2_q, lambda_1_q, n_grid=QP_GRID, n_modes=N_QP, verbose=False):
    import cvxpy as cp
    from scipy.special import j0
    deltas_f = [float(DELTA1_Q), float(delta_2_q)]
    lambdas_f = [float(lambda_1_q), 1.0 - float(lambda_1_q)]
    u_f = float(U_Q)

    w = np.zeros(n_modes)
    for j in range(1, n_modes + 1):
        xi_j = j / u_f
        v = 0.0
        for lam, d in zip(lambdas_f, deltas_f):
            v += lam * j0(math.pi * d * xi_j) ** 2
        w[j - 1] = v
    if w.min() <= 0:
        raise RuntimeError(f"w has nonpositive entry: min={w.min()}")
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
    a_opt = np.asarray(a.value).flatten()
    return a_opt


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
    """Rigorous LOWER bound on min_{x in [0, 1/4]} G(x).

    Strategy: dense float grid argmin -> rigorous arb re-evaluation at
    argmin -> Lipschitz_upper(G') * h subtraction.  Identical to the v2 /
    greopt routines (now wrapped to accept generic coeffs).
    """
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
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        ok, _, _ = certify_M_lower(arb(repr(mid)), k_1, K_2_up, a_lower)
        if ok:
            lo = mid
        else:
            hi = mid
    return arb(repr(lo))


# ---------------------------------------------------------------------------
# One full rigorous certification at (delta_2, lambda_1, coeffs).
# ---------------------------------------------------------------------------
def certify_combined(delta_2_q: Fraction, lambda_1_q: Fraction,
                     coeffs_q,
                     xi_max: int, prec_bits: int,
                     arb_grid: int = ARB_GRID_DEFAULT,
                     verbose: bool = True) -> dict:
    configure_precision(prec_bits)

    deltas_arb = [Q_arb(DELTA1_Q), Q_arb(delta_2_q)]
    lambdas_arb = [Q_arb(lambda_1_q), Q_arb(Fraction(1) - lambda_1_q)]
    coeffs_arb = [Q_arb(q) for q in coeffs_q]
    coeffs_float = [float(q) for q in coeffs_q]

    if verbose:
        print(f"  (delta_1, delta_2) = ({float(DELTA1_Q):.4f}, {float(delta_2_q):.4f})")
        print(f"  (lambda_1, lambda_2) = ({float(lambda_1_q):.4f}, "
              f"{1.0 - float(lambda_1_q):.4f})")
        print(f"  n_modes = {len(coeffs_arb)}  XI_MAX = {xi_max}  prec = {prec_bits} bits")

    # k_1
    k_1 = k1_enclosure(deltas_arb, lambdas_arb)
    if verbose:
        print(f"  [arb] k_1 enclosure mid={float(k_1.mid()):.10f}  "
              f"rad<={float(k_1.rad()):.2e}")

    # min G (rigorous)
    t0 = time.time()
    min_G_lo, lip_rem = min_G_lower_bound(coeffs_arb, coeffs_float, n_grid=arb_grid)
    cond_G2 = float(min_G_lo.lower()) > 0
    t_minG = time.time() - t0
    if verbose:
        print(f"  [arb] min_G >= {float(min_G_lo.lower()):.8f}  "
              f"(Lipschitz rem <= {float(lip_rem.upper()):.2e})  "
              f"G2 = {'OK' if cond_G2 else 'FAIL'}  ({t_minG:.1f}s)")

    # S_1 (rigorous upper)
    S1_hi = S1_upper_bound(coeffs_arb, deltas_arb, lambdas_arb)
    if verbose:
        print(f"  [arb] S_1 <= {float(S1_hi.upper()):.6f}")

    # K_2 (rigorous direct quad + Bessel tail)
    t0 = time.time()
    K_2_arb, bulk_arb, tail_up = K2_rigorous(deltas_arb, lambdas_arb, xi_max=xi_max)
    t_K2 = time.time() - t0
    if verbose:
        print(f"  [arb] K_2 in [{float(K_2_arb.lower()):.8f}, "
              f"{float(K_2_arb.upper()):.8f}]  tail <= {float(tail_up.upper()):.2e}  "
              f"({t_K2:.1f}s)")

    # gain a
    min_G_lo_pos = arb.max(min_G_lo, arb(0))
    a_gain_lo = (arb(4) / U) * (min_G_lo_pos * min_G_lo_pos) / S1_hi
    if verbose:
        print(f"  [arb] gain a >= {float(a_gain_lo.lower()):.6f}")

    # Rigorous M_* lower
    K_2_upper = arb(float(K_2_arb.upper()))
    M_lo = find_M_lower_bisect(k_1, K_2_upper, a_gain_lo)
    M_lo_f = float(M_lo.lower())
    if verbose:
        print(f"  [arb] certified M_* >= {M_lo_f:.8f}")

    return {
        "delta_1": float(DELTA1_Q),
        "delta_2": float(delta_2_q),
        "lambda_1": float(lambda_1_q),
        "lambda_2": 1.0 - float(lambda_1_q),
        "n_modes": len(coeffs_arb),
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
# Sweep helpers
# ---------------------------------------------------------------------------
def cached_solve_qp(delta_2_q, lambda_1_q, cache: dict):
    key = (delta_2_q, lambda_1_q)
    if key in cache:
        return cache[key]
    a_opt = solve_qp(delta_2_q, lambda_1_q)
    coeffs_q = round_to_rationals(a_opt, RATIONAL_DEN)
    cache[key] = coeffs_q
    return coeffs_q


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-sweep", action="store_true",
                        help="Skip the (delta_2, lambda_1) sweep.")
    parser.add_argument("--sweep-xi-max", type=int, default=10**4,
                        help="XI_MAX for sweep points (default 1e4).")
    args = parser.parse_args()

    print("=" * 76)
    print("Cohn--Elkies multi-scale arcsine certifier v3")
    print("Combined: rigorous arb K_2  +  re-optimized G")
    print("=" * 76)
    print(f"  delta_1 = {float(DELTA1_Q):.4f}    u = {float(U_Q):.4f}")
    print(f"  n_modes = {N_QP}")
    print()

    # --------------- 1) K26 best point with greopt G ---------------
    coeffs_q, d1_loaded, d2_loaded, l1_loaded = load_coeffs_from_json(COEFFS_JSON)
    assert d1_loaded == DELTA1_Q, f"delta_1 mismatch: {d1_loaded} vs {DELTA1_Q}"
    print(f"Loaded G coeffs from {COEFFS_JSON.name}:")
    print(f"  delta_2 = {float(d2_loaded):.4f}  lambda_1 = {float(l1_loaded):.4f}  "
          f"n_modes = {len(coeffs_q)}")
    print()

    print("-" * 76)
    print("Run 1:  K26 best, XI_MAX = 1e4, prec = 256")
    print("-" * 76)
    r1 = certify_combined(d2_loaded, l1_loaded, coeffs_q,
                          xi_max=10**4, prec_bits=256, verbose=True)
    print()

    print("-" * 76)
    print("Run 2:  K26 best, XI_MAX = 1e5, prec = 256")
    print("-" * 76)
    r2 = certify_combined(d2_loaded, l1_loaded, coeffs_q,
                          xi_max=10**5, prec_bits=256, verbose=True)
    print()

    runs = [r1, r2]

    # --------------- 2) Sweep (delta_2, lambda_1) with on-the-fly G re-opt ---
    sweep_runs = []
    best_sweep = None
    if not args.no_sweep:
        print("=" * 76)
        print("Sweep (delta_2, lambda_1) with G re-optimized at each point")
        print(f"  (XI_MAX = {args.sweep_xi_max} for speed)")
        print("=" * 76)

        delta_2_list = [Fraction(30, 1000), Fraction(35, 1000), Fraction(40, 1000),
                        Fraction(46, 1000), Fraction(50, 1000), Fraction(55, 1000),
                        Fraction(60, 1000)]
        lambda_1_list = [Fraction(75, 100), Fraction(80, 100), Fraction(85, 100),
                         Fraction(88, 100), Fraction(90, 100), Fraction(92, 100),
                         Fraction(94, 100)]
        print(f"  {'d_2':>7} {'lam_1':>8} {'M_cert':>10} {'minG':>8} {'S_1':>8} {'K2hi':>8}")
        cache: dict = {}
        for d2 in delta_2_list:
            for l1 in lambda_1_list:
                t0 = time.time()
                try:
                    coeffs_q_pt = cached_solve_qp(d2, l1, cache)
                    r = certify_combined(d2, l1, coeffs_q_pt,
                                         xi_max=args.sweep_xi_max,
                                         prec_bits=256,
                                         verbose=False)
                except Exception as exc:
                    print(f"  {float(d2):>7.4f} {float(l1):>8.4f}   ERROR: {exc}")
                    continue
                el = time.time() - t0
                marker = ""
                if best_sweep is None or r["M_cert_lower"] > best_sweep["M_cert_lower"]:
                    best_sweep = r
                    marker = " *"
                print(f"  {float(d2):>7.4f} {float(l1):>8.4f} "
                      f"{r['M_cert_lower']:>10.6f} {r['min_G_lower']:>8.4f} "
                      f"{r['S_1_upper']:>8.3f} {r['K_2_upper']:>8.4f}  "
                      f"({el:.1f}s){marker}")
                sweep_runs.append(r)
        print()
        # Best sweep point: also re-run at higher XI_MAX for a tightened certificate
        if best_sweep is not None:
            print("-" * 76)
            print(f"Best sweep point re-run at XI_MAX = 1e5: "
                  f"(d_2={best_sweep['delta_2']:.4f}, lam_1={best_sweep['lambda_1']:.4f})")
            print("-" * 76)
            d2 = Fraction(int(round(best_sweep["delta_2"] * 1000)), 1000)
            l1_val = best_sweep["lambda_1"]
            # locate the cached Fraction
            l1_key = None
            for (kd, kl) in cache:
                if abs(float(kd) - best_sweep["delta_2"]) < 1e-9 and \
                   abs(float(kl) - l1_val) < 1e-9:
                    l1_key = kl
                    d2 = kd
                    break
            best_high = certify_combined(d2, l1_key, cache[(d2, l1_key)],
                                         xi_max=10**5, prec_bits=256, verbose=True)
            sweep_runs.append(best_high)
            if best_high["M_cert_lower"] > best_sweep["M_cert_lower"]:
                best_sweep = best_high
            print()

    # --------------- Final summary ----------------------------------
    print("=" * 76)
    print("FINAL SUMMARY")
    print("=" * 76)
    all_runs = runs + sweep_runs
    print(f"  {'tag':<32}  {'M_cert':>10}  {'K2hi':>8}  {'minG':>7}  {'S1':>7}")
    print(f"  {'-'*32}  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*7}")
    print(f"  {'K26 loaded, xi=1e4':<32}  {r1['M_cert_lower']:>10.6f}  "
          f"{r1['K_2_upper']:>8.4f}  {r1['min_G_lower']:>7.4f}  {r1['S_1_upper']:>7.3f}")
    print(f"  {'K26 loaded, xi=1e5':<32}  {r2['M_cert_lower']:>10.6f}  "
          f"{r2['K_2_upper']:>8.4f}  {r2['min_G_lower']:>7.4f}  {r2['S_1_upper']:>7.3f}")
    if best_sweep is not None:
        tag = (f"sweep best d2={best_sweep['delta_2']:.3f},l1={best_sweep['lambda_1']:.3f}")
        print(f"  {tag:<32}  {best_sweep['M_cert_lower']:>10.6f}  "
              f"{best_sweep['K_2_upper']:>8.4f}  {best_sweep['min_G_lower']:>7.4f}  "
              f"{best_sweep['S_1_upper']:>7.3f}")
    print()
    best_all = max(all_runs, key=lambda r: r["M_cert_lower"])
    print(f"BEST RIGOROUS M_cert:  {best_all['M_cert_lower']:.8f}")
    print(f"  at  (delta_2, lambda_1) = ({best_all['delta_2']:.4f}, "
          f"{best_all['lambda_1']:.4f}),  XI_MAX = {best_all['xi_max']}")
    print()
    print(f"Baselines:")
    print(f"  v2 (MV stock G + rigorous K_2) :  1.27974")
    print(f"  greopt (greopt G + Minkowski K_2):  1.27624")
    print(f"  MV's reported numerical (1-arcs):  1.27428")
    print(f"  CS17 1.2802                       :  INVALID (matlab mass/height bug)")
    print(f"  greopt prediction with direct K_2:  1.283886")
    print()
    print(f"Improvement over v2 (1.27974):  {best_all['M_cert_lower'] - 1.27974:+.6f}")
    print(f"Improvement over greopt (1.27624):  {best_all['M_cert_lower'] - 1.27624:+.6f}")
    print(f"Improvement over MV (1.27428):  {best_all['M_cert_lower'] - 1.27428:+.6f}")
    print("=" * 76)

    out = {
        "configuration": {
            "delta_1": float(DELTA1_Q),
            "u": float(U_Q),
            "n_modes": N_QP,
        },
        "K26_runs_with_loaded_G": [r1, r2],
        "sweep_runs": sweep_runs,
        "sweep_best": best_sweep,
        "best_overall": best_all,
        "baselines": {
            "v2_rigorous_M_cert": 1.27974,
            "greopt_rigorous_M_cert": 1.27624,
            "MV_numerical": 1.27428,
            "CS17_claim_unsound": True,
            "greopt_direct_K2_prediction": 1.283886,
        },
    }
    json_path = _HERE / "_cohn_elkies_128_v3_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {json_path}")
    return out


if __name__ == "__main__":
    main()
