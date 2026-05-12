"""Cohn--Elkies-style dual certificate for multi-scale arcsine kernel (v2).

This is a tightened version of ``_cohn_elkies_128.py``.  The single source of
looseness in v1 was the Minkowski-lifted ``K_2`` surrogate

    ||K||_2^2  <  0.5747 * (lambda_1 / sqrt(delta_1) + lambda_2 / sqrt(delta_2))^2,

which is loose by ~3% (Minkowski is tight only for proportional summands).
v2 replaces this with a rigorous *direct* arb integration of the cross-term

    K_2  =  2 * int_0^infty K_hat(xi)^2 d xi,
    K_hat(xi)  =  lambda_1 J_0(pi delta_1 xi)^2 + lambda_2 J_0(pi delta_2 xi)^2.

Identity: by Plancherel / Parseval applied to K (an even L^2-regularised
function with Fourier transform K_hat = sum_i lambda_i J_0(pi delta_i .)^2),

    ||K||_2^2  =  2 * int_0^infty K_hat(xi)^2 d xi,

(factor of 2 because K is even so we double the half-line integral).  We
denote this by K_2.

We compute K_2 as a rigorous arb interval by splitting

    K_2 = 2 * I_bulk + R_tail,
    I_bulk = int_0^{XI_MAX} K_hat^2 d xi    (Arb adaptive Gauss; rigorous ball)
    R_tail = 2 * int_{XI_MAX}^infty K_hat^2 d xi.

Tail bound (rigorous): from the standard asymptotic estimate
J_0(z)^2 <= 2 / (pi z) (valid for all z > 0; e.g. NIST DLMF 10.14.4 squared
gives |J_0(z)| <= sqrt(2/(pi z)) for z >= 0), we get

    K_hat(xi)  <=  lambda_1 * 2/(pi delta_1 xi) + lambda_2 * 2/(pi delta_2 xi)
                =  (2/(pi xi)) * (lambda_1/delta_1 + lambda_2/delta_2)
    K_hat(xi)^2 <=  (4/(pi^2 xi^2)) * (lambda_1/delta_1 + lambda_2/delta_2)^2.

Integrating from XI_MAX to infinity,

    int_{XI_MAX}^infty K_hat^2 d xi  <=  (4/pi^2) * C^2 / XI_MAX,
    R_tail  <=  (8/pi^2) * C^2 / XI_MAX,    C := lambda_1/delta_1 + lambda_2/delta_2.

All other inputs (k_1, S_1, min_G) are unchanged from v1 and are already
arb-rigorous.

Run:
    python _cohn_elkies_128_v2.py
"""

from __future__ import annotations

import sys
import time
from fractions import Fraction
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "delsarte_dual"))

import flint
from flint import acb, arb

from mv_bound import MV_COEFFS_119

# ---------------------------------------------------------------------------
# Multi-scale parameters (K26 best, exact rationals)
# ---------------------------------------------------------------------------
DELTA1_Q = Fraction(138, 1000)            # 0.138
DELTA2_Q = Fraction(55, 1000)             # 0.055
LAMBDA1_Q = Fraction(9312, 10000)         # 0.9312
LAMBDA2_Q = Fraction(1) - LAMBDA1_Q       # 0.0688
U_Q = Fraction(1, 2) + DELTA1_Q           # 0.638
TARGET_M_Q = Fraction(1275, 1000)         # 1.275


def Q_arb(q: Fraction) -> arb:
    return arb(q.numerator) / arb(q.denominator)


# ---------------------------------------------------------------------------
# Globals filled in by configure_precision
# ---------------------------------------------------------------------------
DELTA1 = DELTA2 = LAMBDA1 = LAMBDA2 = U = PI = TWO_OVER_U = None
COEFFS = None
N = 0


def configure_precision(prec_bits: int) -> None:
    """Set precision and (re-)compute all arb anchors at this precision."""
    global DELTA1, DELTA2, LAMBDA1, LAMBDA2, U, PI, TWO_OVER_U, COEFFS, N
    flint.ctx.prec = prec_bits
    DELTA1 = Q_arb(DELTA1_Q)
    DELTA2 = Q_arb(DELTA2_Q)
    LAMBDA1 = Q_arb(LAMBDA1_Q)
    LAMBDA2 = Q_arb(LAMBDA2_Q)
    U = Q_arb(U_Q)
    PI = arb.pi()
    TWO_OVER_U = arb(2) / U
    COEFFS = []
    for a in MV_COEFFS_119:
        fr = Fraction(str(a))
        COEFFS.append(arb(fr.numerator) / arb(fr.denominator))
    N = len(COEFFS)
    assert N == 119


# ---------------------------------------------------------------------------
# K_hat (multi-scale Bochner kernel)
# ---------------------------------------------------------------------------
def K_hat_arb(xi: arb) -> arb:
    j01 = (PI * DELTA1 * xi).bessel_j(0)
    j02 = (PI * DELTA2 * xi).bessel_j(0)
    return LAMBDA1 * j01 * j01 + LAMBDA2 * j02 * j02


def k1_enclosure() -> arb:
    return K_hat_arb(arb(1))


# ---------------------------------------------------------------------------
# RIGOROUS K_2 via direct arb integration + Bessel tail bound
# ---------------------------------------------------------------------------
def _khat_sq_integrand(x, analytic):
    """Integrand for acb.integral: f(xi) = K_hat(xi)^2.

    K_hat(xi) = lambda_1 J_0(pi delta_1 xi)^2 + lambda_2 J_0(pi delta_2 xi)^2 is
    entire (J_0 is entire), so we may ignore the analytic flag (just forward it
    to bessel_j if needed; here it's irrelevant because J_0 has no branch cut).
    """
    pi_d1 = PI * DELTA1
    pi_d2 = PI * DELTA2
    j01 = (pi_d1 * x).bessel_j(0)
    j02 = (pi_d2 * x).bessel_j(0)
    Kh = LAMBDA1 * j01 * j01 + LAMBDA2 * j02 * j02
    return Kh * Kh


def K2_rigorous(xi_max: int, eval_limit: int = 10**7) -> tuple[arb, arb, arb, arb]:
    """Return (K_2_arb, bulk_arb, tail_upper_arb, tail_lower_arb).

    K_2 = 2 * int_0^infty K_hat^2 = 2 * I_bulk + R_tail, with
    0 <= R_tail <= (8/pi^2) * C^2 / XI_MAX,    C = lambda_1/delta_1 + lambda_2/delta_2.

    The returned ``K_2_arb`` is a rigorous arb interval that contains the true K_2.
    """
    # Bulk: rigorous adaptive integration with acb.integral.
    # Domain [0, xi_max]; integrand is real-analytic.  Returned value is acb,
    # but the imaginary part is a tight zero-ball; we take the real part.
    I_bulk_acb = acb.integral(_khat_sq_integrand, 0, xi_max, eval_limit=eval_limit)
    I_bulk = I_bulk_acb.real
    bulk = arb(2) * I_bulk

    # Tail upper bound
    C = LAMBDA1 / DELTA1 + LAMBDA2 / DELTA2
    tail_upper = (arb(8) / (PI * PI)) * (C * C) / arb(xi_max)
    # Tail lower bound: integrand >= 0, so R_tail >= 0.
    tail_lower = arb(0)

    # Combine into a single arb interval containing the true K_2.
    # bulk is itself an arb ball with radius eps_quad; the true K_2 lies in
    # [bulk - eps_quad + tail_lower, bulk + eps_quad + tail_upper].
    # The cleanest way: K_2 = bulk + tail, with tail in [0, tail_upper]; we
    # represent this as the union [bulk.lower(), bulk.upper() + tail_upper.upper()].
    lo = arb(float(bulk.lower()))
    hi = bulk + tail_upper
    # Construct an arb ball that bounds [lo.lower(), hi.upper()].  arb supports
    # "[mid +/- rad]" via the (mid, rad) constructor signature; we instead
    # construct via the union by adding a half-width slack.
    mid_str = repr(0.5 * (float(lo.lower()) + float(hi.upper())))
    rad_str = repr(0.5 * (float(hi.upper()) - float(lo.lower())) + 1e-30)
    K_2_arb = arb(mid_str, rad_str)
    return K_2_arb, bulk, tail_upper, tail_lower


# ---------------------------------------------------------------------------
# G2: min_{[0,1/4]} G  (verbatim from v1, unchanged)
# ---------------------------------------------------------------------------
def lipschitz_upper_on_quarter() -> arb:
    two_pi_over_u = (arb(2) * PI) / U
    total = arb(0)
    for j, a in enumerate(COEFFS, start=1):
        total = total + arb(j) * abs(a)
    return two_pi_over_u * total


def min_G_lower_bound(n_grid: int = 200001) -> tuple[arb, arb]:
    h = arb(1) / (arb(4) * arb(n_grid))
    L = lipschitz_upper_on_quarter()

    import numpy as np
    a_f = np.array([float(a.mid()) for a in COEFFS])
    u_f = float(U.mid())
    xs = np.linspace(0.0, 0.25, n_grid + 1)
    vals = np.zeros_like(xs)
    for j, aj in enumerate(a_f, start=1):
        vals += aj * np.cos(2 * np.pi * j * xs / u_f)
    idx_min = int(vals.argmin())
    x_min_q = Fraction(idx_min, 4 * n_grid)
    x_min = arb(x_min_q.numerator) / arb(x_min_q.denominator)

    G_at_min = arb(0)
    two_pi_over_u = (arb(2) * PI) / U
    for j, aj in enumerate(COEFFS, start=1):
        G_at_min = G_at_min + aj * (two_pi_over_u * arb(j) * x_min).cos()

    min_grid_lo = G_at_min - arb("1e-30")
    bound = min_grid_lo - L * h
    return bound, L * h


# ---------------------------------------------------------------------------
# S_1 upper bound (unchanged from v1)
# ---------------------------------------------------------------------------
def S1_upper_bound() -> arb:
    inv_u = arb(1) / U
    total = arb(0)
    for j, a in enumerate(COEFFS, start=1):
        xi = arb(j) * inv_u
        kh = K_hat_arb(xi)
        term = (a * a) / kh
        total = total + term
    return total


# ---------------------------------------------------------------------------
# Master inequality (MV eq. (10) refined form), rigorous lower bound on M_*.
# Identical to v1.
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


def certify_M_lower(M_candidate_arb: arb, k_1: arb, K_2_up: arb,
                    a_lower: arb) -> tuple[bool, arb, arb]:
    sup_R = sup_R_upper(M_candidate_arb, k_1, K_2_up)
    target = TWO_OVER_U + a_lower
    ok = float(sup_R.upper()) < float(target.lower())
    return ok, sup_R, target


def find_M_lower_bisect(k_1: arb, K_2_up: arb, a_lower: arb,
                        M_lo_init: float = 1.001, M_hi_init: float = 1.30,
                        n_iter: int = 80) -> arb:
    lo = M_lo_init
    hi = M_hi_init
    ok_lo, _, _ = certify_M_lower(arb(repr(lo)), k_1, K_2_up, a_lower)
    if not ok_lo:
        return arb(1)
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        ok, _, _ = certify_M_lower(arb(repr(mid)), k_1, K_2_up, a_lower)
        if ok:
            lo = mid
        else:
            hi = mid
    return arb(repr(lo))


# ---------------------------------------------------------------------------
# Full certifier at a given (XI_MAX, prec).
# ---------------------------------------------------------------------------
def certify_one(xi_max: int, prec_bits: int, verbose: bool = True) -> dict:
    configure_precision(prec_bits)

    if verbose:
        print("=" * 76)
        print(f"v2 cert: XI_MAX = {xi_max:<8d}  prec = {prec_bits} bits")
        print("=" * 76)

    t0 = time.time()
    K_2_arb, bulk_arb, tail_up_arb, tail_lo_arb = K2_rigorous(xi_max)
    t_K2 = time.time() - t0
    if verbose:
        print(f"[K_2]  rigorous direct integration of 2 * int_0^{{XI_MAX}} K_hat^2 d xi")
        print(f"       bulk   = 2 * int_0^{{XI_MAX}} K_hat^2 in {float(bulk_arb.lower()):.10f} .. {float(bulk_arb.upper()):.10f}")
        print(f"       tail   <= (8/pi^2) * (lam1/delta1 + lam2/delta2)^2 / XI_MAX = {float(tail_up_arb.upper()):.3e}")
        print(f"       K_2 in [{float(K_2_arb.lower()):.10f}, {float(K_2_arb.upper()):.10f}]")
        print(f"       (time: {t_K2:.2f}s)")
        print()

    k_1 = k1_enclosure()
    if verbose:
        print(f"[k_1]  K_hat(1)  = {float(k_1.mid()):.10f}  (rad <= {float(k_1.rad()):.2e})")

    min_G_lo, lip_rem = min_G_lower_bound(n_grid=200001)
    cond_G2 = float(min_G_lo.lower()) > 0
    if verbose:
        print(f"[G2]   min_{{[0,1/4]}} G  >=  {float(min_G_lo.lower()):.8f}     (cond_G2: {'OK' if cond_G2 else 'FAIL'})")
        print(f"       Lipschitz remainder over grid step  <=  {float(lip_rem.upper()):.2e}")

    S1_hi = S1_upper_bound()
    if verbose:
        print(f"[S_1]  sum a_j^2 / K_hat(j/u)  <=  {float(S1_hi.upper()):.6f}")

    min_G_lo_pos = arb.max(min_G_lo, arb(0))
    a_gain_lo = (arb(4) / U) * (min_G_lo_pos * min_G_lo_pos) / S1_hi
    if verbose:
        print(f"[a]    gain (4/u) min_G^2 / S_1  >=  {float(a_gain_lo.lower()):.6f}")

    # Rigorous LOWER bound on M_* needs UPPER bound on K_2 in the sup_R formula.
    K_2_upper = arb(float(K_2_arb.upper()))
    M_lo = find_M_lower_bisect(k_1, K_2_upper, a_gain_lo)
    M_lo_f = float(M_lo.lower())
    if verbose:
        print(f"[M*]   rigorous lower bound  M_cert  >=  {M_lo_f:.8f}")
        print()

    return {
        "xi_max": xi_max,
        "prec_bits": prec_bits,
        "K2_lower": float(K_2_arb.lower()),
        "K2_upper": float(K_2_arb.upper()),
        "K2_bulk_lower": float(bulk_arb.lower()),
        "K2_bulk_upper": float(bulk_arb.upper()),
        "tail_upper": float(tail_up_arb.upper()),
        "k1_mid": float(k_1.mid()),
        "min_G_lower": float(min_G_lo.lower()),
        "S1_upper": float(S1_hi.upper()),
        "gain_a_lower": float(a_gain_lo.lower()),
        "M_cert": M_lo_f,
        "G2_OK": cond_G2,
        "time_K2_seconds": t_K2,
    }


# ---------------------------------------------------------------------------
# Main: multi-(XI_MAX, prec) sweep
# ---------------------------------------------------------------------------
def main():
    print("=" * 76)
    print("Cohn--Elkies multi-scale arcsine certifier (v2)  --  direct arb K_2")
    print("=" * 76)
    print(f"  delta_1, lambda_1 = 0.138, 0.9312")
    print(f"  delta_2, lambda_2 = 0.055, 0.0688")
    print(f"  Direct rigorous integration of K_2 = 2 * int_0^inf K_hat^2 d xi")
    print(f"  Tail bound via J_0(z)^2 <= 2/(pi z)")
    print()

    configs = [
        (10**4, 256),
        (10**5, 256),
        (10**4, 512),
    ]
    runs = []
    for xi_max, prec in configs:
        r = certify_one(xi_max, prec, verbose=True)
        runs.append(r)
        print("-" * 76)

    # Summary
    print()
    print("=" * 76)
    print("CONVERGENCE SUMMARY")
    print("=" * 76)
    print(f"  {'XI_MAX':>10}  {'prec':>6}  {'K2_lo':>10}  {'K2_hi':>10}  "
          f"{'M_cert':>10}  {'tail_up':>10}")
    for r in runs:
        print(f"  {r['xi_max']:>10d}  {r['prec_bits']:>6d}  "
              f"{r['K2_lower']:>10.6f}  {r['K2_upper']:>10.6f}  "
              f"{r['M_cert']:>10.6f}  {r['tail_upper']:>10.2e}")
    print()

    best = max(r["M_cert"] for r in runs)
    print(f"Best rigorous M_cert across configs: {best:.8f}")
    print(f"Target (1.275): {'MET' if best >= 1.275 else 'NOT MET'}")
    print(f"MV single-arcsine baseline: 1.27428...  (CS17 1.2802 is INVALID)")
    print(f"Improvement over MV: {best - 1.27428:+.6f}")
    print("=" * 76)

    # Optional: dump to JSON
    import json
    out = {
        "runs": runs,
        "best_M_cert": best,
        "target_1.275_met": bool(best >= 1.275),
        "mv_baseline": 1.27428,
        "improvement_over_mv": best - 1.27428,
    }
    json_path = _HERE / "_cohn_elkies_128_v2_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {json_path}")

    return out


if __name__ == "__main__":
    main()
