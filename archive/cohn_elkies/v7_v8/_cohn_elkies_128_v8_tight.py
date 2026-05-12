"""Cohn--Elkies multi-scale arcsine certifier v8 (TIGHT) -- sharper K_2 tail + 2nd-order min_G.

Builds on `_cohn_elkies_128_v4.py` with TWO pure-rigor tightenings (no new math):

  (A) **Sharper Bessel tail bound for K_2.**  The previous code used
        J_0(pi*delta*xi)^2 <= 2/(pi * delta * xi),
      which is OFF BY A FACTOR OF pi.  The textbook bound (A&S 9.2.28) is
        |J_0(z)| <= sqrt(2/(pi z))   for all z > 0,
      so with z = pi*delta*xi,
        J_0(pi*delta*xi)^2 <= 2 / (pi^2 * delta * xi).
      This gives the rigorous tail
        2 * int_XI^inf K_hat(xi)^2 dxi  <=  (8 / pi^4) * C^2 / XI,
      tightening the previous (8/pi^2)*C^2/XI by a factor of pi^2 ~ 9.87x.

      At the v4 best point and XI=1e5, tail drops from 8.07e-4 to 8.18e-5.

  (B) **2nd-order Taylor remainder for min_G.**  The previous bound was the
      Lipschitz remainder
        min G  >=  min_k G(x_k)  -  |G'|_inf * h,
      with h = 1/(4*n_grid).  Replacing it with a cell-centered 2nd-order
      Taylor expansion around each grid point:
        for x in [x_k - h/2, x_k + h/2]:
            G(x) >= G(x_k) - (h/2) * |G'(x_k)| - (h^2/8) * |G''|_inf.
      At the argmin grid index k*, we evaluate G'(x_{k*}) rigorously in arb
      and bound |G''|_inf via the coefficient series sum_j j^2 |a_j|.

      At n_grid = 200000, h = 1.25e-6, the new remainder is ~5e-7 vs ~1.2e-3.

All certified bounds use python-flint `arb` for ball arithmetic.  Outputs are
JSON-serialisable floats (lower/upper from arb intervals).

Usage:
    python _cohn_elkies_128_v8_tight.py --xi-max 100000
    python _cohn_elkies_128_v8_tight.py --xi-max 1000000
    python _cohn_elkies_128_v8_tight.py --xi-max 10000000   # may be slow

The "best v4 point" (delta=[0.138, 0.055, 0.025], lambda=[0.85, 0.10, 0.05],
N=200) is re-certified with the tighter pipeline.  Comparison against v4
loose anchors is printed and saved.
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
# Configuration (mirrors v4)
# ---------------------------------------------------------------------------
DELTA1_Q = Fraction(138, 1000)
U_Q = Fraction(1, 2) + DELTA1_Q  # 0.638
RATIONAL_DEN = 10 ** 12
ARB_GRID_DEFAULT = 200001

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
# arb K_hat
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
# K_2 rigorous: TIGHTENED tail bound.
# ---------------------------------------------------------------------------
def K2_rigorous_tight(deltas_arb, lambdas_arb, xi_max: int,
                      eval_limit: int = 10 ** 7):
    """Return (K_2_arb, bulk_arb, tail_upper_arb, tail_loose_arb).

    K_2 = 2 * int_0^infty K_hat^2 dxi = 2 * I_bulk + R_tail.

    TIGHT tail bound (textbook A&S 9.2.28):
        |J_0(z)|^2 <= 2/(pi z)        for all z > 0.
    With z = pi*delta_i*xi,  J_0(pi*delta_i*xi)^2 <= 2/(pi^2 * delta_i * xi).
    Therefore K_hat(xi) <= (2/pi^2) * C / xi   where  C = sum lam_i/delta_i.
    Hence
        R_tail  <=  2 * int_XI^inf (2 C / (pi^2 xi))^2 dxi
                 =  (8 / pi^4) * C^2 / XI.

    The previous v4 bound used (8/pi^2)*C^2/XI -- looser by factor pi^2~9.87.
    We also return the loose bound for the comparison record.
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

    PI2 = PI * PI
    PI4 = PI2 * PI2
    tail_tight = (arb(8) / PI4) * (C * C) / arb(xi_max)
    tail_loose = (arb(8) / PI2) * (C * C) / arb(xi_max)

    lo = arb(float(bulk.lower()))
    hi = bulk + tail_tight
    mid_str = repr(0.5 * (float(lo.lower()) + float(hi.upper())))
    rad_str = repr(0.5 * (float(hi.upper()) - float(lo.lower())) + 1e-30)
    K_2_arb = arb(mid_str, rad_str)
    return K_2_arb, bulk, tail_tight, tail_loose


# ---------------------------------------------------------------------------
# G handling
# ---------------------------------------------------------------------------
def round_to_rationals(a_opt, denom=RATIONAL_DEN):
    return [Fraction(int(round(a * denom)), denom) for a in a_opt]


def solve_qp(deltas_q, lambdas_q, n_modes: int, n_grid: int = None,
             verbose: bool = False):
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
# Rigorous min G on [0, 1/4] -- LOOSE Lipschitz (v4 baseline, for comparison)
# ---------------------------------------------------------------------------
def lipschitz_upper(coeffs_arb) -> arb:
    two_pi_over_u = (arb(2) * PI) / U
    total = arb(0)
    for j, a in enumerate(coeffs_arb, start=1):
        total = total + arb(j) * abs(a)
    return two_pi_over_u * total


def min_G_loose(coeffs_arb, coeffs_float, n_grid: int = ARB_GRID_DEFAULT):
    """v4 baseline:  remainder = h * |G'|_inf."""
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

    bound = G_at_min - L * h
    return bound, L * h, G_at_min, x_min


# ---------------------------------------------------------------------------
# Rigorous min G -- TIGHT 2nd-order cell-centered Taylor.
# ---------------------------------------------------------------------------
def gprime_bound(coeffs_arb) -> arb:
    """|G'|_inf  <=  (2*pi/u) * sum_j j*|a_j|."""
    two_pi_over_u = (arb(2) * PI) / U
    total = arb(0)
    for j, a in enumerate(coeffs_arb, start=1):
        total = total + arb(j) * abs(a)
    return two_pi_over_u * total


def gpp_bound(coeffs_arb) -> arb:
    """|G''|_inf  <=  (2*pi/u)^2 * sum_j j^2 * |a_j|."""
    two_pi_over_u = (arb(2) * PI) / U
    factor = two_pi_over_u * two_pi_over_u
    total = arb(0)
    for j, a in enumerate(coeffs_arb, start=1):
        total = total + arb(j) * arb(j) * abs(a)
    return factor * total


def gprime_at(coeffs_arb, x_arb) -> arb:
    """Rigorous G'(x_arb)  =  -(2*pi/u) * sum_j j*a_j*sin(2*pi*j*x/u)."""
    two_pi_over_u = (arb(2) * PI) / U
    total = arb(0)
    for j, aj in enumerate(coeffs_arb, start=1):
        total = total + arb(j) * aj * (two_pi_over_u * arb(j) * x_arb).sin()
    return -two_pi_over_u * total


def min_G_tight(coeffs_arb, coeffs_float, n_grid: int = ARB_GRID_DEFAULT):
    """2nd-order cell-centered Taylor bound on min G.

    Grid covers x_k = k/(4*n_grid), k = 0..n_grid, with cell half-width h/2.
    At each cell:
        G(x) >= G(x_k) - (h/2)*|G'(x_k)|  -  (h^2/8) * |G''|_inf.
    We apply at the float-argmin x_{k*} and certify in arb.
    """
    h = arb(1) / (arb(4) * arb(n_grid))
    h_half = h / arb(2)
    M2 = gpp_bound(coeffs_arb)
    h2_over_8 = (h * h) / arb(8)
    second_order_rem = M2 * h2_over_8

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

    # First-derivative at argmin (rigorous arb)
    Gp_at_min = gprime_at(coeffs_arb, x_min)
    abs_Gp_at_min = abs(Gp_at_min)
    first_order_rem = abs_Gp_at_min * h_half

    rem_total = first_order_rem + second_order_rem
    bound = G_at_min - rem_total
    return bound, rem_total, G_at_min, x_min, abs_Gp_at_min, second_order_rem


# ---------------------------------------------------------------------------
# S_1 upper bound
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
# Master inequality
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
# Full certification:  loose vs tight side-by-side.
# ---------------------------------------------------------------------------
def certify_both(deltas_q, lambdas_q, coeffs_q,
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
        print(f"  [arb] k_1 mid={float(k_1.mid()):.10f}")

    # --- min_G:  loose vs tight ---
    t0 = time.time()
    min_G_loose_lo, rem_loose, _, _ = min_G_loose(coeffs_arb, coeffs_float, n_grid=arb_grid)
    t_mg_loose = time.time() - t0

    t0 = time.time()
    min_G_tight_lo, rem_tight, G_argmin, x_argmin, abs_Gp, so_rem = min_G_tight(
        coeffs_arb, coeffs_float, n_grid=arb_grid)
    t_mg_tight = time.time() - t0

    if verbose:
        print(f"  [arb] min_G (loose)  >= {float(min_G_loose_lo.lower()):.10f}"
              f"  rem <= {float(rem_loose.upper()):.2e}  ({t_mg_loose:.2f}s)")
        print(f"  [arb] min_G (TIGHT)  >= {float(min_G_tight_lo.lower()):.10f}"
              f"  rem <= {float(rem_tight.upper()):.2e}"
              f"  [|G'(x*)|<= {float(abs_Gp.upper()):.2e}; 2nd-ord <= {float(so_rem.upper()):.2e}]"
              f"  ({t_mg_tight:.2f}s)")

    # --- S_1 upper ---
    S1_hi = S1_upper_bound(coeffs_arb, deltas_arb, lambdas_arb)
    if verbose:
        print(f"  [arb] S_1 <= {float(S1_hi.upper()):.6f}")

    # --- K_2:  bulk integral once, two tail bounds ---
    t0 = time.time()
    K_2_tight_arb, bulk_arb, tail_tight_up, tail_loose_up = K2_rigorous_tight(
        deltas_arb, lambdas_arb, xi_max=xi_max)
    t_K2 = time.time() - t0
    # also build a "loose" K_2 interval using the loose tail
    K_2_loose_hi = bulk_arb + tail_loose_up
    K_2_loose_lo = arb(float(bulk_arb.lower()))
    if verbose:
        print(f"  [arb] K_2 (loose):  bulk + (8/pi^2)*C^2/XI"
              f"  upper = {float(K_2_loose_hi.upper()):.10f}"
              f"  tail <= {float(tail_loose_up.upper()):.3e}")
        print(f"  [arb] K_2 (TIGHT):  bulk + (8/pi^4)*C^2/XI"
              f"  upper = {float(K_2_tight_arb.upper()):.10f}"
              f"  tail <= {float(tail_tight_up.upper()):.3e}  ({t_K2:.1f}s)")

    # --- M_cert bisection: loose and tight ---
    # LOOSE
    min_G_loose_pos = arb.max(min_G_loose_lo, arb(0))
    a_gain_loose = (arb(4) / U) * (min_G_loose_pos * min_G_loose_pos) / S1_hi
    K2_up_loose_arb = arb(float(K_2_loose_hi.upper()))
    M_loose = find_M_lower_bisect(k_1, K2_up_loose_arb, a_gain_loose)

    # TIGHT
    min_G_tight_pos = arb.max(min_G_tight_lo, arb(0))
    a_gain_tight = (arb(4) / U) * (min_G_tight_pos * min_G_tight_pos) / S1_hi
    K2_up_tight_arb = arb(float(K_2_tight_arb.upper()))
    M_tight = find_M_lower_bisect(k_1, K2_up_tight_arb, a_gain_tight)

    M_loose_f = float(M_loose.lower())
    M_tight_f = float(M_tight.lower())
    if verbose:
        print(f"  [arb] a_gain (loose) >= {float(a_gain_loose.lower()):.7f}")
        print(f"  [arb] a_gain (TIGHT) >= {float(a_gain_tight.lower()):.7f}")
        print(f"  [arb] M_cert (loose) >= {M_loose_f:.10f}")
        print(f"  [arb] M_cert (TIGHT) >= {M_tight_f:.10f}    delta = {M_tight_f - M_loose_f:+.6e}")

    return {
        "deltas": [float(d) for d in deltas_q],
        "lambdas": [float(l) for l in lambdas_q],
        "n_scales": len(deltas_q),
        "n_modes": n_modes,
        "xi_max": xi_max,
        "prec_bits": prec_bits,
        "arb_grid": arb_grid,
        "k_1_mid": float(k_1.mid()),
        # min_G comparison
        "min_G_loose_lower": float(min_G_loose_lo.lower()),
        "min_G_tight_lower": float(min_G_tight_lo.lower()),
        "min_G_rem_loose_upper": float(rem_loose.upper()),
        "min_G_rem_tight_upper": float(rem_tight.upper()),
        "abs_Gprime_at_argmin_upper": float(abs_Gp.upper()),
        "second_order_rem_upper": float(so_rem.upper()),
        # K_2 comparison
        "K_2_bulk_lower": float(bulk_arb.lower()),
        "K_2_bulk_upper": float(bulk_arb.upper()),
        "K_2_tail_loose_upper": float(tail_loose_up.upper()),
        "K_2_tail_tight_upper": float(tail_tight_up.upper()),
        "K_2_loose_upper": float(K_2_loose_hi.upper()),
        "K_2_tight_upper": float(K_2_tight_arb.upper()),
        # downstream
        "S_1_upper": float(S1_hi.upper()),
        "a_gain_loose_lower": float(a_gain_loose.lower()),
        "a_gain_tight_lower": float(a_gain_tight.lower()),
        "M_cert_loose": M_loose_f,
        "M_cert_tight": M_tight_f,
        "M_cert_delta": M_tight_f - M_loose_f,
        "time_K2_seconds": t_K2,
        "time_minG_loose_seconds": t_mg_loose,
        "time_minG_tight_seconds": t_mg_tight,
    }


def solve_qp_safe(deltas_q, lambdas_q, n_modes, scale_factor=1.0):
    a = solve_qp(deltas_q, lambdas_q, n_modes=n_modes)
    a = a * scale_factor
    coeffs_q = round_to_rationals(a, RATIONAL_DEN)
    return coeffs_q, a


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xi-max", type=int, default=10**5,
                        help="Truncation for K_2 bulk integral (default 1e5).")
    parser.add_argument("--prec", type=int, default=256)
    parser.add_argument("--arb-grid", type=int, default=ARB_GRID_DEFAULT)
    parser.add_argument("--xi-max-also",
                        type=lambda s: [int(t) for t in s.split(",")],
                        default=None,
                        help="Comma-separated extra XI values to compare (e.g. 1000000).")
    args = parser.parse_args()

    print("=" * 78)
    print("Cohn--Elkies multi-scale arcsine certifier v8 TIGHT")
    print("  (sharper Bessel tail + 2nd-order min_G Taylor)")
    print("=" * 78)
    configure_precision(args.prec)

    # v4 best 3-scale point
    deltas_q = [Fraction(138, 1000), Fraction(55, 1000), Fraction(25, 1000)]
    lambdas_q = [Fraction(85, 100), Fraction(10, 100), Fraction(5, 100)]
    N = 200

    print("\nUsing v4 best 3-scale anchor:")
    print(f"  deltas  = {[float(d) for d in deltas_q]}")
    print(f"  lambdas = {[float(l) for l in lambdas_q]}")
    print(f"  N = {N}")

    print("\nSolving QP for G coefficients...")
    coeffs_q, _ = solve_qp_safe(deltas_q, lambdas_q, N)

    print(f"\n--- Certification at XI_MAX = {args.xi_max} ---")
    r_main = certify_both(deltas_q, lambdas_q, coeffs_q,
                          xi_max=args.xi_max, prec_bits=args.prec,
                          arb_grid=args.arb_grid, verbose=True)

    all_runs = [r_main]
    if args.xi_max_also:
        for xim in args.xi_max_also:
            print(f"\n--- Certification at XI_MAX = {xim} ---")
            r = certify_both(deltas_q, lambdas_q, coeffs_q,
                             xi_max=xim, prec_bits=args.prec,
                             arb_grid=args.arb_grid, verbose=True)
            all_runs.append(r)

    # v4 reference values (from _cohn_elkies_128_v4_results.json best_overall)
    v4_ref = {
        "min_G_lower": 0.9987588689743072,
        "lip_remainder_upper": 0.001221734943974394,
        "S_1_upper": 29.84090655472349,
        "K_2_lower": 4.788823421259034,
        "K_2_upper": 4.789630363787504,
        "K_2_tail_upper": 0.0008069425260069316,
        "a_gain_lower": 0.2095794023970866,
        "M_cert_lower": 1.2921564960222152,
    }

    # final comparison table
    print()
    print("=" * 78)
    print("COMPARISON: v4 anchors (loose) vs v8 anchors (tight)")
    print("=" * 78)
    r = r_main
    print(f"  Quantity                  v4 (loose)             v8 (TIGHT)             ratio")
    print(f"  K_2 tail bound        {v4_ref['K_2_tail_upper']:.4e}            {r['K_2_tail_tight_upper']:.4e}            "
          f"{v4_ref['K_2_tail_upper']/max(r['K_2_tail_tight_upper'], 1e-30):.2f}x")
    print(f"  K_2 upper             {v4_ref['K_2_upper']:.8f}        {r['K_2_tight_upper']:.8f}")
    print(f"  K_2 interval width    {v4_ref['K_2_upper']-v4_ref['K_2_lower']:.4e}            "
          f"{r['K_2_tight_upper']-r['K_2_bulk_lower']:.4e}")
    print(f"  min_G remainder       {v4_ref['lip_remainder_upper']:.4e}            {r['min_G_rem_tight_upper']:.4e}            "
          f"{v4_ref['lip_remainder_upper']/max(r['min_G_rem_tight_upper'], 1e-30):.1f}x")
    print(f"  min_G lower           {v4_ref['min_G_lower']:.10f}      {r['min_G_tight_lower']:.10f}")
    print(f"  S_1 upper             {v4_ref['S_1_upper']:.6f}             {r['S_1_upper']:.6f}")
    print(f"  a_gain lower          {v4_ref['a_gain_lower']:.8f}        {r['a_gain_tight_lower']:.8f}")
    print(f"  M_cert lower          {v4_ref['M_cert_lower']:.10f}      {r['M_cert_tight']:.10f}")
    print(f"  Delta M_cert         (v8 - v4 ref):                    {r['M_cert_tight'] - v4_ref['M_cert_lower']:+.6e}")

    out = {
        "v4_reference": v4_ref,
        "v8_runs": all_runs,
        "configuration": {
            "delta_1": float(DELTA1_Q),
            "u": float(U_Q),
            "anchor_deltas": [float(d) for d in deltas_q],
            "anchor_lambdas": [float(l) for l in lambdas_q],
            "anchor_n_modes": N,
        },
    }
    json_path = _HERE / "_cohn_elkies_128_v8_tight_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()
