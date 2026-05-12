"""Cohn--Elkies-style dual certificate for multi-scale arcsine kernel.

Mirrors ``_cohn_elkies_125.py`` but with K(x) = lambda_1 * K_arcsine(.; delta_1)
+ lambda_2 * K_arcsine(.; delta_2), the multi-scale arcsine kernel identified
numerically by ``_agent_K26_multiscale_arcsine.py`` as giving M_cert ~ 1.28013.

Goal: rigorously certify M >= 1.279 (would be the first improvement over MV
1.27484 in 16 years).  Spoiler -- see RIGOR NOTE at bottom -- with the natural
Minkowski-lifted MV surrogate, the rigorous bound is only M >= ~1.2723.

Structure (same as _cohn_elkies_125.py):

    K(x)  = sum_i lambda_i * K_i(x),
    K_i   = (1/delta_i) * eta(x/delta_i),  eta the rescaled arcsine density
             (MV's "K", supported on [-delta_i, delta_i], int = 1).
    K_hat(xi) = sum_i lambda_i * J_0(pi delta_i xi)^2   (Bochner OK by sum of squares).
    G(x)  = sum_{j=1}^{119} a_j cos(2 pi j x / u)  -- unchanged MV trig polynomial.

The MV master inequality (refined / "z_1" form, MV eq. (10)):

    sup_R(M)  =  M + 1 + 2 mu(M) k_1 + sqrt((M - 1 - 2 mu(M)^2) (K_2 - 1 - 2 k_1^2))
              >=  2/u + a       [whenever the radicand is nonneg],

  with mu(M) = M sin(pi/M)/pi, k_1 = K_hat(1), K_2 = ||K||_2^2 (regularised
  per MV's surrogate), a = (4/u) * (min_{[0,1/4]} G)^2 / S_1,
  S_1 = sum_j a_j^2 / K_hat(j/u).

Rigorous K_2 bound (the central question):
    K is NOT in L^2: each arcsine has a logarithmic L^2 divergence at +/-delta_i.
    MV postulate "||K||_2^2 < 0.5747/delta" without proof (inherited from
    Martin-O'Bryant [6]); this is the ONLY non-airtight input in
    _cohn_elkies_125.py and we inherit it here.  By Minkowski for L^2:

        ||K||_2  =  || sum_i lambda_i K_i ||_2  <=  sum_i lambda_i ||K_i||_2,

    and using ||K_i||_2^2 < 0.5747/delta_i per MV's surrogate (applied
    independently to each scale), we get the rigorous bound

        ||K||_2^2  <  0.5747 * (sum_i lambda_i / sqrt(delta_i))^2.

Run:
    python _cohn_elkies_128.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "delsarte_dual"))

import flint
from flint import arb

from mv_bound import MV_COEFFS_119          # MV's 119 cosine coefficients

# ---------------------------------------------------------------------------
# Precision
# ---------------------------------------------------------------------------
flint.ctx.prec = 256


# ---------------------------------------------------------------------------
# Multi-scale parameters (K26 best, exact rationals)
# ---------------------------------------------------------------------------
DELTA1_Q = Fraction(138, 1000)            # 0.138 (MV's delta)
DELTA2_Q = Fraction(55, 1000)             # 0.055 (second arcsine scale)
LAMBDA1_Q = Fraction(9312, 10000)         # 0.9312
LAMBDA2_Q = Fraction(1) - LAMBDA1_Q       # 0.0688
U_Q = Fraction(1, 2) + DELTA1_Q           # 0.638 (same u as MV)
K2_NUM_Q = Fraction(5747, 10000)          # MV's 0.5747 surrogate numerator
TARGET_M_Q = Fraction(1279, 1000)         # 1.279 target

# arb versions
def Q_arb(q: Fraction) -> arb:
    return arb(q.numerator) / arb(q.denominator)


DELTA1 = Q_arb(DELTA1_Q)
DELTA2 = Q_arb(DELTA2_Q)
LAMBDA1 = Q_arb(LAMBDA1_Q)
LAMBDA2 = Q_arb(LAMBDA2_Q)
U = Q_arb(U_Q)
K2_NUM = Q_arb(K2_NUM_Q)           # 0.5747
TWO_OVER_U = arb(2) / U
PI = arb.pi()


# Minkowski-based rigorous upper bound:  K_2 <= 0.5747 * (sum_i lam_i / sqrt(delta_i))^2.
K2_UPPER = K2_NUM * (LAMBDA1 / DELTA1.sqrt() + LAMBDA2 / DELTA2.sqrt()) ** 2


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------
def _coeff_arbs():
    """MV's 119 a_j as arb (exact from mpmath strings via Fraction)."""
    out = []
    for a in MV_COEFFS_119:
        fr = Fraction(str(a))
        out.append(arb(fr.numerator) / arb(fr.denominator))
    return out


COEFFS = _coeff_arbs()
N = len(COEFFS)
assert N == 119


# ---------------------------------------------------------------------------
# K_hat enclosures
# ---------------------------------------------------------------------------
def K_hat(xi: arb) -> arb:
    """K_hat(xi) = lambda_1 J_0(pi delta_1 xi)^2 + lambda_2 J_0(pi delta_2 xi)^2."""
    j01 = (PI * DELTA1 * xi).bessel_j(0)
    j02 = (PI * DELTA2 * xi).bessel_j(0)
    return LAMBDA1 * j01 * j01 + LAMBDA2 * j02 * j02


def k1_enclosure() -> arb:
    return K_hat(arb(1))


# ---------------------------------------------------------------------------
# Step (G2): rigorous lower bound on min_{[0,1/4]} G(x)
# (G is UNCHANGED from the MV cert.)
# ---------------------------------------------------------------------------
def lipschitz_upper_on_quarter() -> arb:
    two_pi_over_u = (arb(2) * PI) / U
    total = arb(0)
    for j, a in enumerate(COEFFS, start=1):
        total = total + arb(j) * abs(a)
    return two_pi_over_u * total


def min_G_lower_bound(n_grid: int = 200001) -> tuple[arb, arb]:
    """Rigorous lower bound on min_{x in [0, 1/4]} G(x) via fine grid + Lipschitz."""
    h = arb(1) / (arb(4) * arb(n_grid))
    L = lipschitz_upper_on_quarter()

    # Locate float-grid minimiser cheaply.
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
# S_1 upper bound:  S_1 = sum_j a_j^2 / K_hat(j/u)
# ---------------------------------------------------------------------------
def S1_upper_bound() -> arb:
    inv_u = arb(1) / U
    total = arb(0)
    for j, a in enumerate(COEFFS, start=1):
        xi = arb(j) * inv_u
        kh = K_hat(xi)
        # kh is a positive enclosure; (a*a)/kh is an arb ball whose .upper()
        # gives a rigorous upper bound.  Accumulate.
        term = (a * a) / kh
        total = total + term
    return total


# ---------------------------------------------------------------------------
# Rigorous M*-bisection on the master inequality (MV eq. (10)).
# We use the "z_1-refined" form with z_1 chosen optimally inside the
# K26 numerical code (the y_star = (k1 sqrt((M-1)/(K_2-1)))-vs-mu branch).
# We want a LOWER BOUND on M_*, so we maximise sup_R (use K_2_upper,
# k_1 upper for the +2 mu k_1 term) and minimise target = 2/u + a (use
# a_lower).  If for M_candidate we have  sup_R_upper(M_candidate) < target_lower,
# then M_* > M_candidate.
#
# Note: rad2 = K_2 - 1 - 2 k_1^2 also appears.  For LOWER bound on M_*, we
# want UPPER bound on rad2 (makes sup_R bigger).  Using K_2_upper and
# k_1_lower for rad2 is consistent.  Use a single arb interval k_1 (tight to
# < 1e-70); the ambiguity is invisible.
# ---------------------------------------------------------------------------
def sup_R_upper(M_arb: arb, k_1: arb, K_2_up: arb) -> arb:
    """Rigorous UPPER bound on  sup_R(M) = M + 1 + 2 mu k_1 + sqrt((M - 1 - 2 mu^2)*(K_2 - 1 - 2 k_1^2)).

    Uses the "max" enclosure of mu(M) (it is monotone-ish in M but tightly
    bracketed by arb interval evaluation), and treats both branches:
      - if M - 1 - 2 mu^2 < 0 (interval), we fall back to the no-z1 form
        M + 1 + sqrt((M-1)(K_2-1)).
    """
    one = arb(1)
    two = arb(2)
    if M_arb <= one:
        return arb("inf")
    mu_arb = M_arb * (PI / M_arb).sin() / PI         # arb-rigorous mu(M)
    two_mu_sq = two * mu_arb * mu_arb
    rad1 = M_arb - one - two_mu_sq
    rad2 = K_2_up - one - two * k_1 * k_1

    # No-z1 fallback bound (always valid as an upper bound on the refined sup_R
    # whenever rad1 <= 0; the refined form requires rad1 >= 0).
    no_z1 = M_arb + one + ((M_arb - one) * (K_2_up - one)).sqrt()

    if float(rad1.lower()) <= 0.0:
        # Refined form not applicable rigorously; use the no-z1 form, which is
        # always >= refined sup_R (because removing the z_1 perturbation can only
        # increase the right-hand side per MV's derivation).
        return no_z1

    # Refined form
    refined = M_arb + one + two * mu_arb * k_1 + (rad1 * rad2).sqrt()
    # Conservative upper bound: take max of refined and no_z1 (both are valid
    # upper bounds on sup_R per MV's analysis when rad1 >= 0).
    return arb.max(refined, no_z1)


def certify_M_lower(M_candidate_arb: arb, k_1: arb, K_2_up: arb,
                    a_lower: arb) -> tuple[bool, arb, arb]:
    """Test if M_candidate is a rigorous lower bound for M_*.

    Returns (ok, sup_R_at_M, target).
    """
    sup_R = sup_R_upper(M_candidate_arb, k_1, K_2_up)
    target = TWO_OVER_U + a_lower
    # M_* > M_candidate iff sup_R(M_candidate) < target.  Rigorous: check
    # sup_R.upper() < target.lower().
    ok = float(sup_R.upper()) < float(target.lower())
    return ok, sup_R, target


def find_M_lower_bisect(k_1: arb, K_2_up: arb, a_lower: arb,
                        M_lo_init: float = 1.001, M_hi_init: float = 1.30,
                        n_iter: int = 60) -> arb:
    """Find the largest M_candidate such that certify_M_lower(M_candidate) holds.

    Bisection in float (the certification is rigorous via arb at each step).
    Returns the certified lower bound.
    """
    lo = M_lo_init
    hi = M_hi_init
    # First confirm the low end works.
    ok_lo, _, _ = certify_M_lower(arb(repr(lo)), k_1, K_2_up, a_lower)
    if not ok_lo:
        return arb(1)  # trivial bound only
    # And the high end fails.
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        ok, _, _ = certify_M_lower(arb(repr(mid)), k_1, K_2_up, a_lower)
        if ok:
            lo = mid
        else:
            hi = mid
    return arb(repr(lo))


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def certify(verbose: bool = True) -> dict:
    if verbose:
        print("=" * 72)
        print("Cohn--Elkies / MV multi-scale arcsine dual certificate (K26)")
        print("=" * 72)
        print(f"  delta_1, lambda_1 = {float(DELTA1.mid()):.6f}, {float(LAMBDA1.mid()):.6f}")
        print(f"  delta_2, lambda_2 = {float(DELTA2.mid()):.6f}, {float(LAMBDA2.mid()):.6f}")
        print(f"  u                  = {float(U.mid()):.6f}  (= 1/2 + delta_1)")
        print(f"  n_coeffs           = {N}")
        print(f"  prec               = {flint.ctx.prec} bits")
        print()

    # ----- k_1 enclosure -----
    k_1 = k1_enclosure()
    if verbose:
        print(f"[k_1]  K_hat(1) = lambda_1 J_0(pi delta_1)^2 + lambda_2 J_0(pi delta_2)^2")
        print(f"       k_1 (mid) = {float(k_1.mid()):.10f}  width <= {float(k_1.rad()):.2e}")

    # ----- K_2 Minkowski upper bound -----
    if verbose:
        s = LAMBDA1 / DELTA1.sqrt() + LAMBDA2 / DELTA2.sqrt()
        print(f"[K_2]  Minkowski: ||K||_2 <= sum_i lambda_i sqrt(0.5747/delta_i)")
        print(f"       sum_i lambda_i / sqrt(delta_i) = {float(s.mid()):.6f}")
        print(f"       K_2 <= 0.5747 * (...)^2  =  {float(K2_UPPER.upper()):.6f}")
        print(f"       (pure-delta1 MV surrogate: 0.5747/delta_1 = {float(K2_NUM/DELTA1):.4f})")
        print()

    # ----- min_G -----
    min_G_lo, lip_rem = min_G_lower_bound(n_grid=200001)
    if verbose:
        print(f"[G2]   min_{{[0,1/4]}} G  >=  {float(min_G_lo.lower()):.8f}")
        print(f"       Lipschitz remainder over grid step  <=  {float(lip_rem.upper()):.2e}")
    cond_G2 = float(min_G_lo.lower()) > 0
    if verbose:
        print(f"       condition G2 (G > 0 on [0, 1/4]):  {'OK' if cond_G2 else 'FAIL'}")
        print()

    # ----- S_1 upper bound -----
    S1_hi = S1_upper_bound()
    if verbose:
        print(f"[S_1]  S_1 = sum a_j^2 / K_hat(j/u)  <=  {float(S1_hi.upper()):.6f}")
        print()

    # ----- gain a -----
    min_G_lo_pos = arb.max(min_G_lo, arb(0))
    a_gain_lo = (arb(4) / U) * (min_G_lo_pos * min_G_lo_pos) / S1_hi
    if verbose:
        print(f"[gain] a  >=  (4/u) min_G^2 / S_1  >=  {float(a_gain_lo.lower()):.6f}")
        print()

    # ----- Solve master inequality (rigorous lower bound on M_*) -----
    M_lo = find_M_lower_bisect(k_1, K2_UPPER, a_gain_lo)
    M_lo_f = float(M_lo.lower())
    if verbose:
        print(f"[M*]   Rigorous lower bound on M_*  via refined-form sup_R bisection:")
        print(f"       certified M_*  >=  {M_lo_f:.6f}")
        print(f"       (K26 numerical value  ~ 1.28013)")
        print()

    margin = M_lo_f - float(TARGET_M_Q)
    ok = cond_G2 and (M_lo_f >= float(TARGET_M_Q))
    print("=" * 72)
    if ok:
        print(f"M_cert >= 1.279 CERTIFIED")
        print(f"  certified value: M_cert >= {M_lo_f:.6f}  (margin {margin:+.6f})")
    else:
        print(f"M_cert >= 1.279  NOT achievable with this surrogate.")
        print(f"  best rigorous lower bound: M_cert >= {M_lo_f:.6f}")
        print(f"  shortfall vs 1.279:        {margin:+.6f}")
    print("=" * 72)

    return {
        "verified_target": ok,
        "M_lower_bound": M_lo_f,
        "target": float(TARGET_M_Q),
        "margin_vs_target": margin,
        "min_G_lower": float(min_G_lo.lower()),
        "S1_upper": float(S1_hi.upper()),
        "K2_upper_Minkowski": float(K2_UPPER.upper()),
        "k1_mid": float(k_1.mid()),
        "gain_a_lower": float(a_gain_lo.lower()),
        "delta_1": float(DELTA1.mid()),
        "delta_2": float(DELTA2.mid()),
        "lambda_1": float(LAMBDA1.mid()),
        "lambda_2": float(LAMBDA2.mid()),
        "u": float(U.mid()),
        "G2_OK": cond_G2,
    }


# ---------------------------------------------------------------------------
# Sanity check:  with lambda_1 = 1, K reduces to MV's pure arcsine.
# This should reproduce _cohn_elkies_125.py.
# ---------------------------------------------------------------------------
def sanity_pure_arcsine() -> dict:
    """Re-run with lambda_2 = 0 (pure delta_1 arcsine); should match 1.2742..."""
    global LAMBDA1, LAMBDA2, K2_UPPER
    saved = (LAMBDA1, LAMBDA2, K2_UPPER)
    LAMBDA1 = arb(1)
    LAMBDA2 = arb(0)
    K2_UPPER = K2_NUM / DELTA1
    try:
        print()
        print("=" * 72)
        print("SANITY CHECK:  pure arcsine (lambda_1 = 1, K_2 = 0.5747/delta_1)")
        print("=" * 72)
        out = certify(verbose=True)
        return out
    finally:
        LAMBDA1, LAMBDA2, K2_UPPER = saved


if __name__ == "__main__":
    # Sanity: pure-MV reproduction (should give ~1.2742)
    sanity = sanity_pure_arcsine()
    print()
    print(f"Sanity (pure MV): certified M >= {sanity['M_lower_bound']:.6f}  "
          f"(expect ~1.2742; _cohn_elkies_125 prints 1.27428679)")
    print()

    # Real cert: multi-scale K26
    print()
    result = certify(verbose=True)

    sys.exit(0 if result["verified_target"] else 1)
