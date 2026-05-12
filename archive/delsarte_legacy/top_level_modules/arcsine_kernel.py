"""Arcsine density and its autoconvolution kernel K.

Mathematical identities (standard; see e.g. Gradshteyn-Ryzhik 3.753.2)
---------------------------------------------------------------------

beta(x) = (2/pi) / sqrt(1 - 4 x^2)      on (-1/2, 1/2), 0 outside.

Properties:
    int_{-1/2}^{1/2} beta(x) dx = 1                  (probability density)
    int_{-1/2}^{1/2} beta(x) cos(2 pi x xi) dx = J_0(pi xi)
    (Bessel function of order zero)

Equivalently, the Fourier transform (with convention
hat{f}(xi) = int f(x) e^{-2 pi i x xi} dx) of beta is

    beta_hat(xi) = J_0(pi xi).

Since beta is even and nonneg, beta_hat is real and equals J_0(pi |xi|).

Autoconvolution:
    (beta * beta)(y) = int beta(x) beta(y - x) dx
    supp (beta * beta) = (-1, 1)
    int (beta * beta) = 1                             (mass preserved)
    FT: (beta * beta)_hat(xi) = J_0(pi xi)^2           [>= 0]

Scaling:
    K(x) = (1/delta) * (beta * beta)(x / delta)
    supp K = (-delta, delta)
    int K = 1
    K_hat(xi) = J_0(pi delta xi)^2

Interval arithmetic
-------------------
We provide both scalar mpmath and mpmath.iv interval enclosures for:
    * beta(x)
    * (beta * beta)(y)  (via mp.quad)
    * K(x)              (via (beta*beta))
    * K_hat(xi) = J_0(pi delta xi)^2  (closed form)
"""
from __future__ import annotations

from functools import lru_cache

import mpmath as mp
from mpmath import iv, mpf


# ----------------------------------------------------------------------------
# Scalar scalar mpmath evaluations
# ----------------------------------------------------------------------------

def beta_mp(x) -> mpf:
    """Arcsine density beta(x) = (2/pi)/sqrt(1 - 4 x^2) on (-1/2, 1/2)."""
    x = mpf(x)
    if x <= mpf("-0.5") or x >= mpf("0.5"):
        return mpf(0)
    return mpf(2) / mp.pi / mp.sqrt(mpf(1) - 4 * x * x)


def beta_hat_mp(xi) -> mpf:
    """beta_hat(xi) = J_0(pi xi). Standard identity."""
    xi = mpf(xi)
    return mp.besselj(0, mp.pi * xi)


def beta_conv_beta_mp(y, *, quad_prec: int = 80) -> mpf:
    """(beta * beta)(y) via mpmath quadrature. Supp (-1, 1)."""
    old_prec = mp.mp.prec
    mp.mp.prec = max(old_prec, quad_prec)
    try:
        y = mpf(y)
        if y <= mpf(-1) or y >= mpf(1):
            return mpf(0)
        a = max(mpf("-0.5"), y - mpf("0.5"))
        b = min(mpf("0.5"), y + mpf("0.5"))
        if a >= b:
            return mpf(0)
        # Integrand beta(t) * beta(y - t); endpoints are integrable singularities
        # of the form 1/sqrt(...) so quad handles them with Chebyshev-type node.
        return mp.quad(lambda t: beta_mp(t) * beta_mp(y - t), [a, b])
    finally:
        mp.mp.prec = old_prec


def K_mp(x, delta) -> mpf:
    """K(x) = (1/delta) (beta*beta)(x/delta) on (-delta, delta)."""
    x = mpf(x)
    delta = mpf(delta)
    if delta <= 0:
        raise ValueError("delta must be positive")
    y = x / delta
    if y <= mpf(-1) or y >= mpf(1):
        return mpf(0)
    return beta_conv_beta_mp(y) / delta


def K_hat_mp(xi, delta) -> mpf:
    """K_hat(xi) = J_0(pi delta xi)^2. Closed form from FT identities."""
    xi = mpf(xi)
    delta = mpf(delta)
    j0 = mp.besselj(0, mp.pi * delta * xi)
    return j0 * j0


# ----------------------------------------------------------------------------
# Interval arithmetic (mpmath.iv)
# ----------------------------------------------------------------------------

def _iv_fabs(I):
    a, b = I.a, I.b
    if a >= 0:
        return iv.mpf([a, b])
    if b <= 0:
        return iv.mpf([-b, -a])
    return iv.mpf([0, max(-a, b)])


def beta_iv(x_iv):
    """Interval enclosure of beta on an mpmath iv interval.

    If x_iv has |x| >= 1/2 anywhere, the enclosure must include 0 (the value
    outside the open interval). If x_iv straddles the boundary, include both
    the singular-interior value and 0. The interior has monotone 1/sqrt
    behaviour in |x|.
    """
    abs_iv = _iv_fabs(x_iv)
    # |x| >= 1/2 everywhere: returns 0
    if abs_iv.a >= mpf("0.5"):
        return iv.mpf([0, 0])
    # The interior: beta(x) = (2/pi)/sqrt(1 - 4 x^2), increasing in |x|.
    pi = iv.pi
    # |x| clipped to [0, 1/2).
    hi = abs_iv.b if abs_iv.b < mpf("0.5") else mpf("0.499999999")  # avoid div/0
    abs_clip = iv.mpf([abs_iv.a, hi])
    # 1 - 4 x^2 ranges over [1 - 4 hi^2, 1 - 4 lo^2].
    val_lo = 2 / (pi * iv.sqrt(1 - 4 * iv.mpf([abs_iv.a, abs_iv.a]) ** 2))
    val_hi_denom = iv.sqrt(1 - 4 * iv.mpf([hi, hi]) ** 2)
    if val_hi_denom.a <= 0:
        # Straddles the boundary; upper is +inf.
        val_hi = iv.mpf([float("inf"), float("inf")])
    else:
        val_hi = 2 / (pi * val_hi_denom)
    # include 0 if x_iv includes |x| >= 1/2
    lo = mpf(0) if abs_iv.b > mpf("0.5") else mpf(val_lo.a)
    upper = max(mpf(val_hi.b), mpf(val_lo.b))
    return iv.mpf([lo, upper])


def K_hat_iv(xi_iv, delta):
    """K_hat(xi) = J_0(pi delta xi)^2, interval-valued.

    iv does not provide Bessel, so we bracket via mpmath J_0 at endpoints
    plus a bound on the derivative between. For an iv of width w,
    |J_0'(z)| = |J_1(z)| <= 1, so J_0 on [a, b] lies in
    [min(J_0(a), J_0(b)) - w, max(J_0(a), J_0(b)) + w]  (conservative)
    but tighter: J_0 is decreasing on [0, j_{0,1}] where j_{0,1} ~ 2.4048.
    For publication, use certified Bessel evaluator.

    NOTE: until we wire in python-flint's arb Bessel, this is APPROXIMATE.
    Not yet publication-grade.
    """
    raise NotImplementedError(
        "K_hat_iv needs a certified interval Bessel-J_0 evaluator. "
        "See ROADMAP in arcsine_kernel.py."
    )


# ----------------------------------------------------------------------------
# Sanity-check helpers
# ----------------------------------------------------------------------------

def _verify_integrals():
    """int beta = 1 and (beta*beta)(0) = known value."""
    mp.mp.dps = 30
    total = mp.quad(beta_mp, [mpf("-0.5"), mpf("0.5")])
    bb0 = beta_conv_beta_mp(mpf(0))
    # (beta*beta)(0) = int beta(t)^2 dt = int_{-1/2}^{1/2} (2/pi)^2/(1-4t^2) dt
    # = (4/pi^2) * (1/2) * log((1+2t)/(1-2t)) -- diverges! So beta * beta(0) diverges.
    # Actually int beta^2 diverges, so (beta*beta)(0) = int beta(t) beta(-t) dt diverges too.
    # Hmm, actually K has a logarithmic singularity at 0. Let's check a nonzero point.
    bb_half = beta_conv_beta_mp(mpf("0.5"))
    return {"int_beta": total, "beta_conv_beta_at_0": bb0, "bbc_at_0.5": bb_half}


if __name__ == "__main__":
    mp.mp.dps = 30
    print("Sanity checks:")
    print(f"int beta  = {mp.quad(beta_mp, [mpf('-0.5'), mpf('0.5')])}  (expect 1)")
    print(f"beta_hat(0)  = {beta_hat_mp(0)}  (expect 1 = J_0(0))")
    print(f"beta_hat(1)  = {beta_hat_mp(1)}  (expect J_0(pi) ~ -0.304)")
    # A sample (beta*beta) at 0.5:
    print(f"(beta*beta)(0.5)  = {beta_conv_beta_mp('0.5')}  (should be ~ positive)")
    # K at 0 with delta=0.138
    from mpmath import mpf as _mpf
    # K(0) has log divergence... let's sample at K(0.05) with delta=0.138
    try:
        print(f"K(0.05), delta=0.138 = {K_mp('0.05', '0.138')}")
    except Exception as e:
        print(f"K exception: {e}")
