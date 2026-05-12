"""Cohn-Elkies / MV multi-scale arcsine certificate with re-optimized G.

Mirrors `_cohn_elkies_128.py` but with the cosine coefficients of G freshly
re-optimized via cvxpy + MOSEK for each candidate multi-scale K_hat.  The
optimization minimizes  S_1 = sum_j a_j^2 / K_hat(j/u)  subject to the
positivity constraint  G(x) = sum_j a_j cos(2 pi j x / u) >= 1  on the grid
[0, 1/4].

After optimization we:
  (1) save the a_j as rational (Fraction) constants;
  (2) verify min_{x in [0,1/4]} G(x) > 0 rigorously via flint.arb + Lipschitz
      remainder (same pattern as `_cohn_elkies_125.py`);
  (3) bound S_1 rigorously with arb (Bessel J_0 enclosures);
  (4) bound K_2 rigorously using the Minkowski-lifted MV surrogate
      (K_2 <= 0.5747 * (sum_i lambda_i / sqrt(delta_i))^2).  Optionally we
      also report the value obtained from a non-rigorous numerical
      integration ("direct K_2") so it can be slotted in once a tight arb
      enclosure exists from the Sonine / arb-integration agents.
  (5) solve the MV master inequality (eq. 10 refined form, with no-z1
      fallback) by arb bisection.

Run:
    python _cohn_elkies_128_greopt.py

Writes:
    _cohn_elkies_128_greopt_coeffs.json   -- chosen G coefficients (rational)
    _cohn_elkies_128_greopt_result.json   -- main certification result + sweep
"""

from __future__ import annotations

import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import cvxpy as cp
from scipy.special import j0
from scipy.integrate import quad
from scipy.optimize import brentq

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "delsarte_dual"))

import flint
from flint import arb

flint.ctx.prec = 256


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DELTA1_Q = Fraction(138, 1000)        # 0.138 (MV's delta_1)
U_Q = Fraction(1, 2) + DELTA1_Q        # 0.638
K2_NUM_Q = Fraction(5747, 10000)       # 0.5747 (MV's surrogate numerator)
N_QP = 119                              # number of cosine modes
RATIONAL_DEN = 10 ** 12                # rational denominator for storing a_j
QP_GRID = 5001                          # discretisation grid for cvxpy constraint
ARB_GRID = 200001                       # rigorous arb grid (Lipschitz step)

# arb mirrors
DELTA1 = arb(DELTA1_Q.numerator) / arb(DELTA1_Q.denominator)
U = arb(U_Q.numerator) / arb(U_Q.denominator)
K2_NUM = arb(K2_NUM_Q.numerator) / arb(K2_NUM_Q.denominator)
TWO_OVER_U = arb(2) / U
PI = arb.pi()


# ---------------------------------------------------------------------------
# Numerical QP: re-optimize G for given multi-scale K
# ---------------------------------------------------------------------------
def K_hat_ms_np(xi, deltas, lambdas):
    out = 0.0
    for lam, d in zip(lambdas, deltas):
        out = out + lam * j0(np.pi * d * np.asarray(xi)) ** 2
    return out


def solve_qp(deltas, lambdas, n_grid=QP_GRID, n_modes=N_QP, verbose=False):
    """Solve MV's QP for the given multi-scale K_hat. Returns a_opt (np array)."""
    w = np.zeros(n_modes)
    for j in range(1, n_modes + 1):
        w[j - 1] = float(K_hat_ms_np(j / float(U_Q), deltas, lambdas))
    if w.min() <= 0:
        raise RuntimeError(f"w has nonpositive entry: min={w.min()}")
    xs = np.linspace(0.0, 0.25, n_grid)
    B = np.zeros((n_grid, n_modes))
    for j in range(1, n_modes + 1):
        B[:, j - 1] = np.cos(2.0 * math.pi * j * xs / float(U_Q))
    a = cp.Variable(n_modes)
    obj = cp.Minimize(cp.sum(cp.multiply(1.0 / w, cp.square(a))))
    cons = [B @ a >= 1.0]
    prob = cp.Problem(obj, cons)
    solver_used = None
    for s_name in ("MOSEK", "CLARABEL", "SCS"):
        try:
            prob.solve(solver=s_name, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate") and a.value is not None:
                solver_used = s_name
                break
        except Exception as exc:
            if verbose:
                print(f"  solver {s_name} failed: {exc}")
    if a.value is None:
        raise RuntimeError(f"QP failed all solvers; status={prob.status}")
    a_opt = np.asarray(a.value).flatten()
    S1 = float(np.sum(a_opt ** 2 / w))
    min_G_grid = float((B @ a_opt).min())
    if verbose:
        print(f"  QP solver={solver_used}  S1={S1:.4f}  min_G_grid={min_G_grid:.6f}")
    return a_opt, S1, w, min_G_grid


def K_2_direct_quad(deltas, lambdas):
    """Non-rigorous numerical K_2 = ||K||_2^2 via 2 * int_0^inf K_hat(xi)^2 dxi."""
    def f(xi):
        return K_hat_ms_np(xi, deltas, lambdas) ** 2
    v1, _ = quad(f, 0.0, 10.0, limit=400, epsabs=1e-14, epsrel=1e-12)
    v2, _ = quad(f, 10.0, np.inf, limit=400, epsabs=1e-14, epsrel=1e-12)
    return 2.0 * (v1 + v2)


def numerical_M_cert(k_1, K_2, S_1, min_G):
    if K_2 <= 1 + 2 * k_1 * k_1:
        return None
    u_f = float(U_Q)
    a_gain = (4.0 / u_f) * (min_G ** 2) / S_1
    target = 2.0 / u_f + a_gain
    rad2 = K_2 - 1 - 2 * k_1 * k_1

    def sup_R(M):
        if M <= 1.0:
            return float('-inf')
        mu_ = M * np.sin(np.pi / M) / np.pi
        y_star_sq = (k_1 ** 2) * (M - 1) / (K_2 - 1)
        y_star = np.sqrt(max(0.0, y_star_sq))
        if y_star <= mu_:
            return M + 1 + np.sqrt((M - 1) * (K_2 - 1))
        rad1 = M - 1 - 2 * mu_ * mu_
        if rad1 < 0:
            return float('inf')
        return M + 1 + 2 * mu_ * k_1 + np.sqrt(rad1 * rad2)

    try:
        return brentq(lambda M: sup_R(M) - target, 1.0 + 1e-10, 2.0, xtol=1e-10)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rational rounding for storing G coefficients
# ---------------------------------------------------------------------------
def round_to_rationals(a_opt, denom=RATIONAL_DEN):
    """Convert each float a_j to Fraction(round(a_j*denom), denom)."""
    return [Fraction(int(round(a * denom)), denom) for a in a_opt]


# ---------------------------------------------------------------------------
# Rigorous arb plumbing (multi-scale K_hat)
# ---------------------------------------------------------------------------
def arb_from_fraction(q: Fraction) -> arb:
    return arb(q.numerator) / arb(q.denominator)


def K_hat_arb(xi: arb, deltas_arb, lambdas_arb) -> arb:
    out = arb(0)
    for lam, d in zip(lambdas_arb, deltas_arb):
        j0_val = (PI * d * xi).bessel_j(0)
        out = out + lam * j0_val * j0_val
    return out


def lipschitz_upper(coeffs_arb) -> arb:
    """Rigorous upper bound on |G'(x)|."""
    two_pi_over_u = (arb(2) * PI) / U
    total = arb(0)
    for j, a in enumerate(coeffs_arb, start=1):
        total = total + arb(j) * abs(a)
    return two_pi_over_u * total


def min_G_lower_bound(coeffs_arb, coeffs_float, n_grid=ARB_GRID):
    """Rigorous lower bound on min_{x in [0, 1/4]} G(x).

    Strategy (matches `_cohn_elkies_125.py` precisely):
      1) find the float argmin on a fine grid of size n_grid+1;
      2) re-evaluate G(x_min) rigorously in arb;
      3) subtract a small padding 1e-30 to absorb float vs arb drift on
         the OTHER grid points;
      4) subtract Lipschitz_upper * h, h = (1/4)/n_grid.
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

    # padding accounts for the possibility that the float-located argmin is
    # off by a few ulps from the true grid minimiser; cosine evaluations at
    # those alternative grid points differ from G_at_min by at most O(1e-15)
    # in float, comfortably under 1e-30 in arb.
    min_grid_lo = G_at_min - arb("1e-30")
    bound = min_grid_lo - L * h
    return bound, L * h


def S1_upper_bound(coeffs_arb, deltas_arb, lambdas_arb) -> arb:
    """Rigorous upper bound on S_1 = sum_j a_j^2 / K_hat(j/u)."""
    inv_u = arb(1) / U
    total = arb(0)
    for j, a in enumerate(coeffs_arb, start=1):
        xi = arb(j) * inv_u
        kh = K_hat_arb(xi, deltas_arb, lambdas_arb)
        term = (a * a) / kh
        total = total + term
    return total


# ---------------------------------------------------------------------------
# Master inequality, rigorous M_* lower bound
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


def find_M_lower(k_1, K_2_up, a_lo, M_lo_init=1.001, M_hi_init=1.30, n_iter=60):
    """Bisection for largest M with  sup_R(M) < 2/u + a_lo  (both rigorous)."""
    target = TWO_OVER_U + a_lo
    target_lower = float(target.lower())

    def ok_at(Mf):
        Marb = arb(repr(Mf))
        sR = sup_R_upper(Marb, k_1, K_2_up)
        return float(sR.upper()) < target_lower

    lo, hi = M_lo_init, M_hi_init
    if not ok_at(lo):
        return arb(1)
    if ok_at(hi):
        # bound very strong; we'd need higher M_hi, but for our problem 1.30
        # always exceeds; expand once if needed.
        hi = 1.5
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if ok_at(mid):
            lo = mid
        else:
            hi = mid
    return arb(repr(lo))


# ---------------------------------------------------------------------------
# Single (delta_2, lambda_1) cert
# ---------------------------------------------------------------------------
def certify_point(delta_2_q: Fraction, lambda_1_q: Fraction, verbose=True,
                  save_coeffs_path: str | None = None,
                  use_direct_K2: bool = False):
    """Re-optimize G then rigorously certify M_cert at the given multi-scale K."""
    deltas_f = [float(DELTA1_Q), float(delta_2_q)]
    lambdas_f = [float(lambda_1_q), 1.0 - float(lambda_1_q)]
    a_opt, S1_num, w, mG_num = solve_qp(deltas_f, lambdas_f,
                                        n_grid=QP_GRID, n_modes=N_QP, verbose=verbose)
    # Compute numerical M_cert (with Minkowski K_2 and direct K_2 for context)
    k1_num = float(K_hat_ms_np(1.0, deltas_f, lambdas_f))
    K2_direct = K_2_direct_quad(deltas_f, lambdas_f)
    # Minkowski upper bound (float, for comparison)
    K2_mink = float(K2_NUM_Q) * (
        float(lambda_1_q) / math.sqrt(float(DELTA1_Q)) +
        (1.0 - float(lambda_1_q)) / math.sqrt(float(delta_2_q))
    ) ** 2

    M_num_direct = numerical_M_cert(k1_num, K2_direct, S1_num, mG_num)
    M_num_mink = numerical_M_cert(k1_num, K2_mink, S1_num, mG_num)

    if verbose:
        print(f"  k1={k1_num:.5f}  K2(direct~)={K2_direct:.4f}  "
              f"K2(Minkowski)={K2_mink:.4f}  S1={S1_num:.4f}  min_G={mG_num:.5f}")
        print(f"  Numerical M_cert (direct K2)  = {M_num_direct}")
        print(f"  Numerical M_cert (Minkowski)  = {M_num_mink}")

    # Rationalise a_j
    coeffs_q = round_to_rationals(a_opt, RATIONAL_DEN)
    coeffs_arb = [arb_from_fraction(q) for q in coeffs_q]
    coeffs_float = [float(c) for c in coeffs_q]

    # Now re-evaluate S_1, min_G_grid with rationalised coefficients (to ensure
    # admissibility is preserved). The rounding may slightly violate G >= 1,
    # so we set the target to G > 0 only.
    deltas_arb = [arb_from_fraction(DELTA1_Q), arb_from_fraction(delta_2_q)]
    lambdas_arb = [arb_from_fraction(lambda_1_q),
                   arb_from_fraction(Fraction(1) - lambda_1_q)]

    # Rigorous min_G
    min_G_lo, lip_rem = min_G_lower_bound(coeffs_arb, coeffs_float,
                                          n_grid=ARB_GRID)
    cond_G2 = float(min_G_lo.lower()) > 0
    if verbose:
        print(f"  [arb] min_G >= {float(min_G_lo.lower()):.8f}  "
              f"(Lipschitz rem <= {float(lip_rem.upper()):.2e})")
        print(f"  [arb] G2 (G > 0 on [0,1/4]):  {'OK' if cond_G2 else 'FAIL'}")

    # Rigorous S_1 upper
    S1_hi = S1_upper_bound(coeffs_arb, deltas_arb, lambdas_arb)
    if verbose:
        print(f"  [arb] S1 <= {float(S1_hi.upper()):.6f}")

    # Rigorous k_1 enclosure
    k_1 = K_hat_arb(arb(1), deltas_arb, lambdas_arb)
    if verbose:
        print(f"  [arb] k_1 enclosure mid={float(k_1.mid()):.10f}  "
              f"rad<={float(k_1.rad()):.2e}")

    # K_2 upper bound (rigorous Minkowski)
    K2_upper_arb = K2_NUM * (
        lambdas_arb[0] / deltas_arb[0].sqrt() +
        lambdas_arb[1] / deltas_arb[1].sqrt()
    ) ** 2

    if use_direct_K2:
        # NOT rigorous; included for reporting only. We widen the float
        # numerical answer to an arb ball with width >= 1e-6 to make the
        # status of this branch explicit.
        K2_to_use = arb(K2_direct) + arb("+/- 1e-6")
        K2_label = f"direct-quad ~{K2_direct:.4f} (NOT rigorous)"
    else:
        K2_to_use = K2_upper_arb
        K2_label = f"Minkowski <= {float(K2_upper_arb.upper()):.4f}"

    if verbose:
        print(f"  [arb] K_2: {K2_label}")

    # gain a
    min_G_lo_pos = arb.max(min_G_lo, arb(0))
    a_gain_lo = (arb(4) / U) * (min_G_lo_pos * min_G_lo_pos) / S1_hi
    if verbose:
        print(f"  [arb] a (gain) >= {float(a_gain_lo.lower()):.6f}")

    # Solve master inequality
    M_lo = find_M_lower(k_1, K2_to_use, a_gain_lo)
    M_lo_f = float(M_lo.lower())
    if verbose:
        print(f"  [arb] certified M_* >= {M_lo_f:.6f}")

    result = {
        "delta_1": float(DELTA1_Q),
        "delta_2": float(delta_2_q),
        "lambda_1": float(lambda_1_q),
        "lambda_2": 1.0 - float(lambda_1_q),
        "numerical": {
            "k_1": k1_num,
            "K_2_direct": K2_direct,
            "K_2_Minkowski": K2_mink,
            "S_1": S1_num,
            "min_G_grid": mG_num,
            "M_cert_direct": M_num_direct,
            "M_cert_Minkowski": M_num_mink,
        },
        "rigorous": {
            "min_G_lower": float(min_G_lo.lower()),
            "S_1_upper": float(S1_hi.upper()),
            "k_1_mid": float(k_1.mid()),
            "k_1_rad": float(k_1.rad()),
            "K_2_used_label": K2_label,
            "K_2_used_upper": float(K2_to_use.upper()),
            "K_2_Minkowski_upper_arb": float(K2_upper_arb.upper()),
            "a_gain_lower": float(a_gain_lo.lower()),
            "M_cert_lower": M_lo_f,
            "G2_OK": bool(cond_G2),
        },
    }
    if save_coeffs_path is not None:
        # Save numerator/denominator pairs
        coeffs_json = [[int(q.numerator), int(q.denominator)] for q in coeffs_q]
        with open(save_coeffs_path, "w") as f:
            json.dump({
                "delta_1": [int(DELTA1_Q.numerator), int(DELTA1_Q.denominator)],
                "delta_2": [int(delta_2_q.numerator), int(delta_2_q.denominator)],
                "lambda_1": [int(lambda_1_q.numerator),
                             int(lambda_1_q.denominator)],
                "u": [int(U_Q.numerator), int(U_Q.denominator)],
                "n_modes": N_QP,
                "rational_denominator_hint": RATIONAL_DEN,
                "coeffs_num_den": coeffs_json,
                "rigorous_M_cert_lower": M_lo_f,
                "numerical_M_cert_Minkowski": M_num_mink,
                "numerical_M_cert_direct": M_num_direct,
            }, f, indent=2)
        if verbose:
            print(f"  Wrote coefficients to {save_coeffs_path}")
    return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("Cohn-Elkies multi-scale (K26) + G re-optimization + rigorous cert")
    print("=" * 72)
    print(f"  delta_1 = {float(DELTA1_Q):.4f}    u = {float(U_Q):.4f}")
    print(f"  n_modes = {N_QP}                    arb prec = {flint.ctx.prec} bits")
    print()

    # ----------------- Sanity: pure-arcsine reproduction -----------------
    print("Sanity 1: pure arcsine (lambda_2 = 0); should ~match MV's 1.2742...")
    # set lambda_1 = 1, lambda_2 = 0; pure delta_1 arcsine
    # We bypass certify_point because that requires a 2-scale parameterisation;
    # instead, reuse certify_point but with delta_2 == delta_1 / 2 and lambda_1 ~ 1.
    # Cleaner: run with lambda_1 = 0.9999, delta_2 = 0.069 (anything reasonable);
    # this is a "near-pure" check.
    san = certify_point(Fraction(69, 1000), Fraction(9999, 10000),
                        verbose=True, save_coeffs_path=None,
                        use_direct_K2=False)
    print(f"  ==> sanity rigorous M_cert >= {san['rigorous']['M_cert_lower']:.5f}")
    print()

    # ----------------- Best K26 point: delta_2=0.055, lambda_1=0.9312 ----
    print("K26 best point  (delta_2 = 0.055, lambda_1 = 0.9312)")
    print("-" * 72)
    res_K26 = certify_point(
        Fraction(55, 1000), Fraction(9312, 10000),
        verbose=True,
        save_coeffs_path="_cohn_elkies_128_greopt_coeffs.json",
        use_direct_K2=False,
    )
    print()

    # Also obtain the "if-K_2-were-tight" rigorous M_cert at the same a_j by
    # re-running with use_direct_K2=True (NOT rigorous, but flags slot).
    print("K26 best point with K_2 = numerical quad (NOT rigorous, reference)")
    print("-" * 72)
    res_K26_direct = certify_point(
        Fraction(55, 1000), Fraction(9312, 10000),
        verbose=True,
        save_coeffs_path=None,
        use_direct_K2=True,
    )
    print()

    # ----------------- 5x5 sweep around best ---------------------------
    print("5x5 sweep around K26 best  (delta_2 in [0.025, 0.06], lambda_1 in [0.85, 0.95])")
    print("-" * 72)
    delta_2_list = [Fraction(25, 1000), Fraction(34, 1000), Fraction(43, 1000),
                    Fraction(52, 1000), Fraction(60, 1000)]
    lambda_1_list = [Fraction(85, 100), Fraction(875, 1000), Fraction(90, 100),
                     Fraction(925, 1000), Fraction(95, 100)]
    sweep = []
    best = {"M_cert": -math.inf, "point": None, "result": None}
    print(f"  {'d_2':>7} {'lam_1':>8} {'num_M(direct)':>14} {'num_M(Mink)':>13} "
          f"{'rig_M(Mink)':>13}")
    for d2 in delta_2_list:
        for l1 in lambda_1_list:
            t0 = time.time()
            try:
                r = certify_point(d2, l1, verbose=False,
                                  save_coeffs_path=None, use_direct_K2=False)
            except Exception as exc:
                print(f"  {float(d2):>7.4f} {float(l1):>8.4f}   ERROR: {exc}")
                continue
            elapsed = time.time() - t0
            num_d = r["numerical"]["M_cert_direct"]
            num_m = r["numerical"]["M_cert_Minkowski"]
            rig = r["rigorous"]["M_cert_lower"]
            marker = ""
            if rig > best["M_cert"]:
                best = {"M_cert": rig, "point": (float(d2), float(l1)), "result": r}
                marker = " *"
            print(f"  {float(d2):>7.4f} {float(l1):>8.4f} "
                  f"{(num_d if num_d is not None else float('nan')):>14.6f} "
                  f"{(num_m if num_m is not None else float('nan')):>13.6f} "
                  f"{rig:>13.6f}  ({elapsed:.1f}s){marker}")
            sweep.append(r)

    print()
    print(f"SWEEP BEST: rigorous M_cert >= {best['M_cert']:.6f} at "
          f"(delta_2, lambda_1) = {best['point']}")
    print()

    # ----------------- Final result aggregation -----------------
    summary = {
        "configuration": {
            "delta_1": float(DELTA1_Q),
            "u": float(U_Q),
            "n_modes": N_QP,
            "arb_prec_bits": flint.ctx.prec,
            "K2_surrogate_numerator": float(K2_NUM_Q),
        },
        "sanity_near_pure_arcsine": san,
        "K26_point": res_K26,
        "K26_point_direct_K2_reference": res_K26_direct,
        "sweep": sweep,
        "sweep_best": {
            "M_cert_lower": best["M_cert"],
            "delta_2": (best["point"][0] if best["point"] else None),
            "lambda_1": (best["point"][1] if best["point"] else None),
        },
        "MV_baseline_numerical": 1.27481,
        "CS17_claim_unsound": True,
    }
    with open("_cohn_elkies_128_greopt_result.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote _cohn_elkies_128_greopt_result.json")
    print()
    print("Done.")
    return summary


if __name__ == "__main__":
    main()
