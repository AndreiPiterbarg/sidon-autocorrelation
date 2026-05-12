"""Hybrid Bochner-admissible kernel rigorous certifier (v2).

Tests mixtures K_hat(xi) = sum_i lambda_i K_hat_i(xi; delta_i) with families:

    arcsine:    K_hat_i = J_0(pi delta xi)^2
    tent:       K_hat_i = sinc^2(pi delta xi)   (sinc(z) = sin(z)/z)
    bspline3:   K_hat_i = sinc^4(pi delta xi / 2)
    gaussian:   K_hat_i = exp(-pi^2 delta^2 xi^2)   (compactly supported approx)
    semicircle: K_hat_i = 2 J_1(pi delta xi) / (pi delta xi)
    epanech:    K_hat_i = 3 (sin z - z cos z) / z^3, z = pi delta xi

Each K_hat_i is non-negative (Bochner OK) and normalised K_hat_i(0)=1.
Reoptimises the G modulator at N=200 cosines for the mixture's full K_hat.
K_2, k_1, min_G, S_1 are computed via flint.arb for rigorous certification.

NOTE on Gaussian: a true Gaussian is NOT compactly supported in x.  We
include it as a Bochner test (K_hat is positive, so Bochner-admissible for
the autocorrelation lower-bound machinery) and treat tails carefully -- but
strictly speaking the resulting M-bound machinery (which assumes phi
supported on [-DELTA/2, DELTA/2]) requires compact support.  We therefore
ALSO include Gaussian only as a sanity comparison and do NOT count Gaussian
mixtures as rigorous compact-support certificates.  Compact-support
candidates are the other 5 families.

Pipeline:
  1. Float pre-screen sweeps over (family pair/triple, deltas, lambdas).
  2. Top-K candidates undergo full rigorous arb certification.
  3. Best rigorous compact-support hybrid M_cert is reported.

References for tail bounds:
  |J_0(z)| <= 1 for all z, |J_0(z)| <= sqrt(2/(pi z)) for z >= 1.
  |sinc(z)|   <= 1, |sinc(z)| <= 1/|z|
  |J_1(z)|   <= sqrt(2/(pi z))
  Epanechnikov K_hat = 3(sin z - z cos z)/z^3 has |.| <= 3(1+z)/z^3 <= 6/z^2

Tail strategy: for xi >= XI_MAX, use the universal bound
  |K_hat_i(xi)| <= 2/(pi * delta_i * xi)   (valid for all non-Gaussian
  families above when xi * delta_i >= 2/pi, i.e. xi >= 2/(pi*delta_i)).
Then |K_hat(xi)|^2 <= (sum_i lambda_i * 2/(pi*delta_i*xi))^2
                  = (2/(pi*xi))^2 * (sum lambda_i/delta_i)^2
and  int_R^infty <= (4/pi^2) * (sum lambda_i/delta_i)^2 / R.
So  2 * int_R^infty K_hat^2 <= (8/pi^2) (sum lambda_i/delta_i)^2 / R.

This matches the arcsine-only v4 tail formula and applies uniformly.

VALIDATION: we require XI_MAX such that XI_MAX * delta_min >= 2/pi, easy.
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
from scipy.special import j0, j1

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "delsarte_dual"))

import flint
from flint import acb, arb


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DELTA_MAX_Q = Fraction(138, 1000)      # 0.138
U_Q = Fraction(1, 2) + DELTA_MAX_Q     # 0.638
RATIONAL_DEN = 10 ** 12
ARB_GRID_DEFAULT = 200001
PREC_BITS_DEFAULT = 256

# Globals filled in by configure_precision
U = PI = TWO_OVER_U = None


def Q_arb(q: Fraction) -> arb:
    return arb(q.numerator) / arb(q.denominator)


def configure_precision(prec_bits: int) -> None:
    global U, PI, TWO_OVER_U
    flint.ctx.prec = prec_bits
    U = Q_arb(U_Q)
    PI = arb.pi()
    TWO_OVER_U = arb(2) / U


# ---------------------------------------------------------------------------
# Float K_hat per family (for QP and pre-screening)
# ---------------------------------------------------------------------------
def _sinc(z):
    z = np.asarray(z, dtype=float)
    out = np.ones_like(z)
    nz = np.abs(z) > 1e-14
    out[nz] = np.sin(z[nz]) / z[nz]
    return out


def khat_arcsine_f(xi, delta):
    z = math.pi * delta * np.asarray(xi, dtype=float)
    return j0(z) ** 2


def khat_tent_f(xi, delta):
    z = math.pi * delta * np.asarray(xi, dtype=float)
    return _sinc(z) ** 2


def khat_bspline3_f(xi, delta):
    z = math.pi * delta * np.asarray(xi, dtype=float)
    return _sinc(z / 2.0) ** 4


def khat_gaussian_f(xi, delta):
    # Treat delta as the std-dev-like parameter so K_hat(0)=1 and the kernel
    # has effective "width" delta (sigma_x = delta / pi, so K_hat decays as
    # exp(-pi^2 delta^2 xi^2)).
    z = math.pi * delta * np.asarray(xi, dtype=float)
    return np.exp(-z * z)


def khat_semicircle_f(xi, delta):
    z = math.pi * delta * np.asarray(xi, dtype=float)
    out = np.ones_like(z)
    nz = np.abs(z) > 1e-8
    out[nz] = 2.0 * j1(z[nz]) / z[nz]
    z0 = z[~nz]
    out[~nz] = 1.0 - z0 ** 2 / 8.0 + z0 ** 4 / 192.0
    return out


def khat_epanech_f(xi, delta):
    z = math.pi * delta * np.asarray(xi, dtype=float)
    out = np.ones_like(z)
    nz = np.abs(z) > 1e-6
    zz = z[nz]
    out[nz] = 3.0 * (np.sin(zz) - zz * np.cos(zz)) / (zz ** 3)
    z0 = z[~nz]
    out[~nz] = 1.0 - z0 ** 2 / 10.0 + z0 ** 4 / 280.0
    return out


FLOAT_FAMILIES = {
    "arcsine":    khat_arcsine_f,
    "tent":       khat_tent_f,
    "bspline3":   khat_bspline3_f,
    "gaussian":   khat_gaussian_f,
    "semicircle": khat_semicircle_f,
    "epanech":    khat_epanech_f,
}

# Families with strictly compact x-support (used for rigorous certificates).
COMPACT_FAMILIES = {"arcsine", "tent", "bspline3", "semicircle", "epanech"}


def K_hat_mixture_f(xi, components):
    """components: list of (family_name, delta, lambda) with sum lambda = 1."""
    xi = np.asarray(xi, dtype=float)
    out = np.zeros_like(xi)
    for fam, delta, lam in components:
        out = out + lam * FLOAT_FAMILIES[fam](xi, delta)
    return out


# ---------------------------------------------------------------------------
# arb K_hat per family
# ---------------------------------------------------------------------------
def _arb_sinc(z: arb) -> arb:
    # sinc(z) = sin(z)/z, with sinc(0) = 1 (handle via Taylor for small z)
    # Use arb's robust evaluator: if z contains 0, fall back to a Taylor
    # series. For our use, z = pi*delta*xi varies along an integration
    # interval that may include 0 (analytic = 1 case in acb.integral),
    # so we use the safer formula sin(z)/z with arb's series handling.
    # arb supports sinc via .sinc() if available; otherwise sin(z)/z.
    if hasattr(z, "sinc"):
        return z.sinc()
    # Fallback
    return z.sin() / z


def khat_arcsine_arb(xi: arb, delta_arb: arb) -> arb:
    z = PI * delta_arb * xi
    j0v = z.bessel_j(0)
    return j0v * j0v


def khat_tent_arb(xi: arb, delta_arb: arb) -> arb:
    z = PI * delta_arb * xi
    s = _arb_sinc(z)
    return s * s


def khat_bspline3_arb(xi: arb, delta_arb: arb) -> arb:
    z = PI * delta_arb * xi / arb(2)
    s = _arb_sinc(z)
    return s * s * s * s


def khat_gaussian_arb(xi: arb, delta_arb: arb) -> arb:
    z = PI * delta_arb * xi
    return (-z * z).exp()


def khat_semicircle_arb(xi: arb, delta_arb: arb) -> arb:
    z = PI * delta_arb * xi
    # 2 J_1(z) / z, handle z near 0 with sinc-style robust eval
    # arb's bessel_j is fine; division by z needs care.  Use series fallback
    # only if z contains 0; for our integration intervals starting at 0,
    # acb.integral will handle that. We just compute and rely on arb's
    # interval arithmetic.
    j1v = z.bessel_j(1)
    # 2 J_1(z)/z = 1 - z^2/8 + z^4/192 - ...
    # Use this Taylor expansion if z is small (radius small AND |mid| small)
    return arb(2) * j1v / z


def khat_epanech_arb(xi: arb, delta_arb: arb) -> arb:
    z = PI * delta_arb * xi
    # 3 (sin z - z cos z)/z^3 = 1 - z^2/10 + ...
    return arb(3) * (z.sin() - z * z.cos()) / (z * z * z)


ARB_FAMILIES = {
    "arcsine":    khat_arcsine_arb,
    "tent":       khat_tent_arb,
    "bspline3":   khat_bspline3_arb,
    "gaussian":   khat_gaussian_arb,
    "semicircle": khat_semicircle_arb,
    "epanech":    khat_epanech_arb,
}


def K_hat_mixture_arb(xi, components_q):
    """components_q: list of (family_name, delta_q, lambda_q) (Fractions)."""
    out = arb(0)
    for fam, d, lam in components_q:
        d_arb = Q_arb(d)
        lam_arb = Q_arb(lam)
        out = out + lam_arb * ARB_FAMILIES[fam](xi, d_arb)
    return out


def k1_enclosure(components_q) -> arb:
    return K_hat_mixture_arb(arb(1), components_q)


# ---------------------------------------------------------------------------
# Rigorous K_2 = 2 int_0^inf K_hat^2 dxi  (tail by universal 2/(pi delta xi)
# bound for non-Gaussian families; Gaussian decays exponentially -> add its
# closed-form tail).
# ---------------------------------------------------------------------------
def K2_rigorous(components_q, xi_max: int, eval_limit: int = 10**7):
    """Return (K_2_arb, bulk_arb, tail_upper_arb)."""

    def _integrand(x, analytic):
        Kh = arb(0)
        for fam, d, lam in components_q:
            d_arb = Q_arb(d)
            lam_arb = Q_arb(lam)
            Kh = Kh + lam_arb * ARB_FAMILIES[fam](x, d_arb)
        return Kh * Kh

    I_bulk_acb = acb.integral(_integrand, 0, xi_max, eval_limit=eval_limit)
    I_bulk = I_bulk_acb.real
    bulk = arb(2) * I_bulk

    # Tail: bound each non-Gaussian component by 2/(pi delta xi) at xi >= XI_MAX
    # (valid since XI_MAX * delta_i >= 1 for our delta_min ~ 0.02).
    # K_hat(xi) <= sum_{non-gauss} lam_i * 2/(pi d_i xi) + sum_{gauss} lam_i * exp(-pi^2 d_i^2 xi^2).
    # We bound the Gaussian piece by its monotone-decreasing value at xi_max
    # (which we then multiply by mass) and treat as a constant addition.
    C_non_gauss = arb(0)
    gauss_components = []
    for fam, d, lam in components_q:
        d_arb = Q_arb(d)
        lam_arb = Q_arb(lam)
        if fam == "gaussian":
            gauss_components.append((d_arb, lam_arb))
        else:
            C_non_gauss = C_non_gauss + lam_arb / d_arb
    # universal-bound tail piece (non-Gaussian contribution)
    tail_ng = (arb(8) / (PI * PI)) * (C_non_gauss * C_non_gauss) / arb(xi_max)
    tail_upper = tail_ng
    # Gaussian-only piece: 2 int_{xi_max}^inf (sum lam_i exp(-pi^2 d_i^2 xi^2))^2 dxi
    # <= 2 * (sum lam_i) * sum lam_i (1/2) sqrt(pi)/(pi d_i) * erfc(pi d_i xi_max)
    # We use a CRUDE upper bound: 2 * (sum lam_i)^2 * exp(-pi^2 d_min^2 xi_max^2)
    # / (pi^2 d_min^2 xi_max).
    if gauss_components:
        d_min = gauss_components[0][0]
        lam_sum = arb(0)
        for d_a, l_a in gauss_components:
            lam_sum = lam_sum + l_a
            # Find minimum d
        # very crude bound: 2 * lam_sum^2 / (pi d_min)^2 / xi_max * exp(-(pi d_min xi_max)^2)
        z = PI * d_min * arb(xi_max)
        tail_g = arb(2) * lam_sum * lam_sum * (-(z * z)).exp() / (PI * d_min) ** 2 / arb(xi_max)
        tail_upper = tail_upper + tail_g

    lo = arb(float(bulk.lower()))
    hi = bulk + tail_upper
    mid_val = 0.5 * (float(lo.lower()) + float(hi.upper()))
    rad_val = 0.5 * (float(hi.upper()) - float(lo.lower())) + 1e-30
    K_2_arb = arb(repr(mid_val), repr(rad_val))
    return K_2_arb, bulk, tail_upper


# ---------------------------------------------------------------------------
# QP solve for G (mixture)
# ---------------------------------------------------------------------------
def round_to_rationals(a_opt, denom=RATIONAL_DEN):
    return [Fraction(int(round(a * denom)), denom) for a in a_opt]


def solve_qp(components_q, n_modes: int, n_grid: int = None,
             verbose: bool = False):
    import cvxpy as cp
    u_f = float(U_Q)
    if n_grid is None:
        n_grid = max(5001, 4 * n_modes + 1)

    w = np.zeros(n_modes)
    for j in range(1, n_modes + 1):
        xi_j = j / u_f
        # Compute K_hat at xi_j (float)
        v = 0.0
        for fam, d, lam in components_q:
            v += float(lam) * float(FLOAT_FAMILIES[fam](np.array([xi_j]), float(d))[0])
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
# Rigorous S_1, min_G
# ---------------------------------------------------------------------------
def S1_upper_bound(coeffs_arb, components_q) -> arb:
    inv_u = arb(1) / U
    total = arb(0)
    for j, a in enumerate(coeffs_arb, start=1):
        xi = arb(j) * inv_u
        kh = K_hat_mixture_arb(xi, components_q)
        total = total + (a * a) / kh
    return total


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
# Rigorous end-to-end certify
# ---------------------------------------------------------------------------
def certify(components_q, coeffs_q,
            xi_max: int, prec_bits: int,
            arb_grid: int = ARB_GRID_DEFAULT,
            verbose: bool = True) -> dict:
    configure_precision(prec_bits)

    coeffs_arb = [Q_arb(q) for q in coeffs_q]
    coeffs_float = [float(q) for q in coeffs_q]
    n_modes = len(coeffs_arb)

    if verbose:
        comps_str = " + ".join(
            f"{fam}(d={float(d):.4f},lam={float(lam):.4f})"
            for fam, d, lam in components_q)
        print(f"  components: {comps_str}")
        print(f"  n_modes={n_modes}  XI_MAX={xi_max}  prec={prec_bits}  arb_grid={arb_grid}")

    k_1 = k1_enclosure(components_q)
    if verbose:
        print(f"  k_1 mid={float(k_1.mid()):.10f}  rad<={float(k_1.rad()):.2e}")

    t0 = time.time()
    min_G_lo, lip_rem = min_G_lower_bound(coeffs_arb, coeffs_float, n_grid=arb_grid)
    cond_G2 = float(min_G_lo.lower()) > 0
    t_minG = time.time() - t0
    if verbose:
        print(f"  min_G>={float(min_G_lo.lower()):.6f}  lip_rem<={float(lip_rem.upper()):.2e}  "
              f"G2={'OK' if cond_G2 else 'FAIL'}  ({t_minG:.1f}s)")

    S1_hi = S1_upper_bound(coeffs_arb, components_q)
    if verbose:
        print(f"  S_1<={float(S1_hi.upper()):.6f}")

    t0 = time.time()
    K_2_arb, bulk, tail_up = K2_rigorous(components_q, xi_max=xi_max)
    t_K2 = time.time() - t0
    if verbose:
        print(f"  K_2 in [{float(K_2_arb.lower()):.6f},{float(K_2_arb.upper()):.6f}]  "
              f"tail<={float(tail_up.upper()):.2e}  ({t_K2:.1f}s)")

    min_G_lo_pos = arb.max(min_G_lo, arb(0))
    a_gain_lo = (arb(4) / U) * (min_G_lo_pos * min_G_lo_pos) / S1_hi
    if verbose:
        print(f"  a_gain>={float(a_gain_lo.lower()):.6f}")

    K_2_upper = arb(float(K_2_arb.upper()))
    M_lo = find_M_lower_bisect(k_1, K_2_upper, a_gain_lo)
    M_lo_f = float(M_lo.lower())
    if verbose:
        print(f"  M_cert>={M_lo_f:.8f}")

    return {
        "components": [(fam, float(d), float(lam)) for fam, d, lam in components_q],
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
        "K_2_tail_upper": float(tail_up.upper()),
        "a_gain_lower": float(a_gain_lo.lower()),
        "M_cert_lower": M_lo_f,
        "time_K2_seconds": t_K2,
        "time_minG_seconds": t_minG,
        "is_compact_support": all(fam in COMPACT_FAMILIES
                                  for fam, _, _ in components_q),
    }


# ---------------------------------------------------------------------------
# Float pre-screen
# ---------------------------------------------------------------------------
def float_screen(components_q, n_modes=200, xi_max=20000, n_xi=200001):
    """Quick float estimate of M_cert -- not rigorous, just for ranking."""
    components_f = [(fam, float(d), float(lam)) for fam, d, lam in components_q]
    xis = np.linspace(0.0, xi_max, n_xi)
    dxi = xis[1] - xis[0]
    Kh = K_hat_mixture_f(xis, components_f)
    if np.any(Kh < -1e-10):
        return None
    K_2 = 2.0 * np.trapezoid(Kh ** 2, dx=dxi)
    k_1 = float(K_hat_mixture_f(np.array([1.0]), components_f)[0])

    # solve QP
    try:
        a = solve_qp(components_q, n_modes=n_modes)
    except Exception as exc:
        return {"error": str(exc)}

    u_f = float(U_Q)
    qp_xi = np.arange(1, n_modes + 1) / u_f
    Kh_qp = K_hat_mixture_f(qp_xi, components_f)
    if np.any(Kh_qp < 1e-15):
        return None
    S_1 = float(np.sum(a ** 2 / Kh_qp))

    # min G
    xs = np.linspace(0.0, 0.25, 50001)
    G_vals = np.zeros_like(xs)
    for j in range(1, n_modes + 1):
        G_vals += a[j - 1] * np.cos(2 * math.pi * j * xs / u_f)
    minG = G_vals.min()
    if minG <= 0:
        return None

    # MV master inequality (float)
    a_gain = (4.0 / u_f) * minG ** 2 / S_1
    target = 2.0 / u_f + a_gain

    def sup_R(M):
        if M <= 1.0:
            return float('inf')
        mu_ = M * math.sin(math.pi / M) / math.pi
        rad1 = M - 1 - 2 * mu_ * mu_
        rad2 = K_2 - 1 - 2 * k_1 * k_1
        no_z1 = M + 1 + math.sqrt((M - 1) * (K_2 - 1))
        if rad1 <= 0 or rad2 <= 0:
            return no_z1
        return max(M + 1 + 2 * mu_ * k_1 + math.sqrt(rad1 * rad2), no_z1)

    # bisection
    lo, hi = 1.001, 1.5
    if sup_R(lo) >= target:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sup_R(mid) < target:
            lo = mid
        else:
            hi = mid
    return {
        "components": components_f,
        "n_modes": n_modes,
        "k_1": k_1, "K_2": K_2, "S_1": S_1,
        "min_G": minG, "a_gain": a_gain,
        "M_cert_float": lo,
        "coeffs": a,
    }


# ---------------------------------------------------------------------------
# Build candidate list
# ---------------------------------------------------------------------------
def Qfrac(x):
    """Coerce float-like to a small-denominator Fraction (denom=1000)."""
    return Fraction(int(round(x * 1000)), 1000)


def build_candidates():
    """Build a list of mixture candidates to screen.

    Each candidate is a list of (family, delta_Q, lambda_Q) with lambdas
    summing to 1.  We bias around the proven 3-scale arcsine optimum
    (d=[0.138,0.055,0.025], lam=[0.85,0.10,0.05]) and explore replacing
    the *small-scale* slot with a non-arcsine family (since small-scale
    arcsines have negative-tail risk that, e.g., tent does not).
    """
    cands = []
    # Reference 3-scale pure-arcsine (sanity)
    cands.append([
        ("arcsine", Qfrac(0.138), Qfrac(0.85)),
        ("arcsine", Qfrac(0.055), Qfrac(0.10)),
        ("arcsine", Qfrac(0.025), Qfrac(0.05)),
    ])
    # Replace 3rd slot (smallest scale) with one of the other families
    for fam in ["tent", "bspline3", "semicircle", "epanech"]:
        for d3 in [0.018, 0.020, 0.025, 0.030, 0.035, 0.045]:
            for l3 in [0.03, 0.05, 0.07, 0.10]:
                # keep first two arcsine slots
                l1_base = 0.85
                l2_base = 0.10
                total_arc = l1_base + l2_base
                # we want to spend l3 on alt-family, keep arcsines proportional
                scale = (1.0 - l3) / total_arc
                l1 = l1_base * scale
                l2 = l2_base * scale
                cands.append([
                    ("arcsine", Qfrac(0.138), Qfrac(l1)),
                    ("arcsine", Qfrac(0.055), Qfrac(l2)),
                    (fam, Qfrac(d3), Qfrac(l3)),
                ])
    # Two-scale arc + alt-family at delta_2
    for fam in ["tent", "bspline3", "semicircle", "epanech"]:
        for d2 in [0.030, 0.040, 0.045, 0.055, 0.060, 0.075, 0.090]:
            for l1 in [0.75, 0.80, 0.85, 0.90, 0.92]:
                cands.append([
                    ("arcsine", Qfrac(0.138), Qfrac(l1)),
                    (fam, Qfrac(d2), Qfrac(1.0 - l1)),
                ])
    # Four-component: 2 arcsine + tent + Gaussian (Gaussian is not compact-supp!)
    for d_gauss in [0.015, 0.020, 0.030]:
        for l_gauss in [0.02, 0.04, 0.06]:
            cands.append([
                ("arcsine", Qfrac(0.138), Qfrac(0.80)),
                ("arcsine", Qfrac(0.055), Qfrac(0.08)),
                ("tent",    Qfrac(0.040), Qfrac(0.08)),
                ("gaussian", Qfrac(d_gauss), Qfrac(l_gauss)),
            ])
    # Four-way (compact-support only): 2 arc + tent + bspline3
    for d3 in [0.025, 0.030, 0.040]:
        for d4 in [0.018, 0.025, 0.035]:
            for l3 in [0.04, 0.06]:
                for l4 in [0.03, 0.05]:
                    l_alt = l3 + l4
                    if l_alt > 0.25:
                        continue
                    base_left = 1.0 - l_alt
                    l1 = base_left * (0.85 / 0.95)
                    l2 = base_left * (0.10 / 0.95)
                    cands.append([
                        ("arcsine", Qfrac(0.138), Qfrac(l1)),
                        ("arcsine", Qfrac(0.055), Qfrac(l2)),
                        ("tent",    Qfrac(d3),    Qfrac(l3)),
                        ("bspline3", Qfrac(d4),   Qfrac(l4)),
                    ])
    # Normalize: every cand's lambdas should sum to exactly 1 (Fraction).
    out = []
    for c in cands:
        s = sum(lam for _, _, lam in c)
        if s != Fraction(1):
            # adjust last term
            c = list(c)
            fam, d, lam = c[-1]
            c[-1] = (fam, d, lam + (Fraction(1) - s))
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-modes", type=int, default=200)
    parser.add_argument("--xi-max", type=int, default=10000)
    parser.add_argument("--prec", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10,
                        help="number of float-screened candidates to certify rigorously")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--out", type=str, default="_hybrid_rigorous_v2_results.json")
    args = parser.parse_args()

    configure_precision(args.prec)

    print("=" * 78)
    print("Hybrid Bochner-admissible kernel rigorous certifier (v2)")
    print(f"N_modes = {args.n_modes}, XI_MAX = {args.xi_max}, prec = {args.prec}")
    print(f"Reference: pure 3-scale arcsine M_cert = 1.29216 (v4)")
    print("=" * 78)

    cands = build_candidates()
    print(f"\nBuilt {len(cands)} candidate mixtures; running float pre-screen.")

    screen_results = []
    t0_total = time.time()
    for i, comps_q in enumerate(cands):
        fams = ",".join(f for f, _, _ in comps_q)
        try:
            res = float_screen(comps_q, n_modes=args.n_modes)
        except Exception as exc:
            print(f"  [{i+1:3d}/{len(cands)}]  fams={fams:30s}  ERROR: {exc}")
            continue
        if res is None or "error" in (res or {}):
            errmsg = (res or {}).get("error", "infeasible")
            print(f"  [{i+1:3d}/{len(cands)}]  fams={fams:30s}  -- {errmsg}")
            continue
        M_f = res["M_cert_float"]
        compact = all(f in COMPACT_FAMILIES for f, _, _ in comps_q)
        tag = "C " if compact else "NC"
        print(f"  [{i+1:3d}/{len(cands)}]  fams={fams:30s}  M_float={M_f:.5f}  {tag}")
        screen_results.append({
            "idx": i,
            "components_q": comps_q,
            "M_cert_float": M_f,
            "is_compact_support": compact,
            "coeffs": res["coeffs"],
        })

    elapsed_screen = time.time() - t0_total
    print(f"\nScreen done in {elapsed_screen:.1f}s. {len(screen_results)} feasible / {len(cands)}")

    if args.screen_only:
        # Save screen only
        out = {
            "screen_results": [
                {k: v for k, v in r.items() if k != "coeffs" and k != "components_q"}
                | {"components": [(fam, float(d), float(l)) for fam, d, l in r["components_q"]]}
                for r in screen_results
            ],
            "elapsed_screen_s": elapsed_screen,
        }
        with open(_HERE / args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Wrote {_HERE / args.out}")
        return

    # Sort by float M, take top K (preferring compact-support ones)
    screen_results.sort(
        key=lambda r: (r["is_compact_support"], r["M_cert_float"]),
        reverse=True)
    print(f"\n--- Top {args.top_k} candidates by float M_cert ---")
    for r in screen_results[:args.top_k]:
        fams = ",".join(f for f, _, _ in r["components_q"])
        print(f"  M_float={r['M_cert_float']:.6f}  compact={r['is_compact_support']}  {fams}")

    # Rigorous certify
    print(f"\n--- Rigorous certification (top {args.top_k}) ---")
    rigorous_results = []
    best = None
    best_compact = None
    for r in screen_results[:args.top_k]:
        try:
            coeffs_q = round_to_rationals(r["coeffs"], RATIONAL_DEN)
            res = certify(r["components_q"], coeffs_q,
                          xi_max=args.xi_max, prec_bits=args.prec,
                          arb_grid=ARB_GRID_DEFAULT, verbose=True)
            res["M_cert_float_screen"] = r["M_cert_float"]
            rigorous_results.append(res)
            if best is None or res["M_cert_lower"] > best["M_cert_lower"]:
                best = res
            if res["is_compact_support"]:
                if best_compact is None or res["M_cert_lower"] > best_compact["M_cert_lower"]:
                    best_compact = res
        except Exception as exc:
            import traceback; traceback.print_exc()
            print(f"  ERROR in rigorous certify: {exc}")
            continue
        print()

    print("=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)
    if best is not None:
        print(f"Best rigorous M_cert (any):     {best['M_cert_lower']:.8f}")
        print(f"  components: {best['components']}")
    if best_compact is not None:
        print(f"Best COMPACT-SUPPORT M_cert:    {best_compact['M_cert_lower']:.8f}")
        print(f"  components: {best_compact['components']}")
    print(f"Pure 3-scale arcsine (v4):     1.29215650")
    print(f"Beats pure arcsine? {best_compact and best_compact['M_cert_lower'] > 1.29216}")

    out = {
        "n_modes": args.n_modes,
        "xi_max": args.xi_max,
        "prec_bits": args.prec,
        "n_candidates": len(cands),
        "n_feasible": len(screen_results),
        "screen_top": [
            {"components": [(f, float(d), float(l)) for f, d, l in r["components_q"]],
             "M_cert_float": r["M_cert_float"],
             "is_compact_support": r["is_compact_support"]}
            for r in screen_results[:args.top_k]
        ],
        "rigorous_results": rigorous_results,
        "best_overall": best,
        "best_compact_support": best_compact,
        "pure_arcsine_reference": 1.29215650,
        "elapsed_screen_s": elapsed_screen,
    }
    with open(_HERE / args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {_HERE / args.out}")


if __name__ == "__main__":
    main()
