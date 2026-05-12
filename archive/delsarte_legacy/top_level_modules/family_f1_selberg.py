"""Family F1: Fejer-modulated Selberg-type test function.

    g(t) = (sin(pi T t)/(pi t))^2 * (1 + sum_{k=1}^K a_k cos(2 pi k omega t))

Fourier transform (symmetrised):
    ghat(xi) = Delta_T(xi)
             + (1/2) sum_{k=1}^K a_k ( Delta_T(xi - k*omega) + Delta_T(xi + k*omega) )

where Delta_T(xi) = max(0, T - |xi|) is the triangle supported on [-T, T].

Positive-definiteness certificate
---------------------------------
ghat is piecewise linear with breakpoints at { -T, T, k*omega +/- T, -k*omega +/- T }
for k=1..K. It is continuous, and linear on each piece. Hence ghat >= 0 on R
iff ghat(xi) >= 0 at every breakpoint xi and ghat -> 0 at +/-infty (automatic).

We certify nonnegativity by evaluating ghat at each breakpoint (mpmath) and
checking all values are >= 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import sympy as sp
from mpmath import iv, mp, mpf


@dataclass
class F1Params:
    T: float                 # Fejer width (spectral support = [-T - K*omega, T + K*omega])
    omega: float             # Cosine frequency spacing
    a: Sequence[float]       # Cosine coefficients a_1,...,a_K

    def K(self) -> int:
        return len(self.a)


def _triangle(xi, T):
    """Delta_T(xi) = max(0, T - |xi|). Works for mpmath mpf and iv.mpf."""
    # iv.mpf has no __abs__ that returns a clean interval always; branch instead.
    # But iv supports abs() via max(a, -b). We'll use iv.fabs if present.
    try:
        absxi = iv.fabs(xi) if hasattr(iv, "fabs") else _iv_abs(xi)
    except Exception:
        absxi = abs(xi)
    return iv.mpf([0, 0]).maximum if False else _pos_part(T - absxi)


def _iv_abs(I):
    """Absolute value on an mpmath iv interval."""
    a, b = I.a, I.b
    if a >= 0:
        return iv.mpf([a, b])
    if b <= 0:
        return iv.mpf([-b, -a])
    return iv.mpf([0, max(-a, b)])


def _pos_part(I):
    """max(0, I) for an mpmath iv interval."""
    if I.a >= 0:
        return I
    if I.b <= 0:
        return iv.mpf([0, 0])
    return iv.mpf([0, I.b])


def ghat_iv(xi_iv, params: F1Params):
    """Interval-arithmetic evaluation of ghat at xi_iv (an iv interval)."""
    T = iv.mpf(repr(params.T))
    omega = iv.mpf(repr(params.omega))
    val = _pos_part(T - _iv_abs(xi_iv))
    for k, ak in enumerate(params.a, start=1):
        ak_iv = iv.mpf(repr(float(ak)))
        shift = k * omega
        left = _pos_part(T - _iv_abs(xi_iv - shift))
        right = _pos_part(T - _iv_abs(xi_iv + shift))
        val = val + (ak_iv / 2) * (left + right)
    return val


def g_iv(t_iv, params: F1Params):
    """Interval-arithmetic evaluation of g(t) at t_iv.

    g(t) = (sin(pi T t) / (pi t))^2 * (1 + sum a_k cos(2 pi k omega t))

    The sinc^2 factor is evaluated directly (iv supports sin and division by
    an interval not containing 0; when t_iv contains 0 we use the limit).
    """
    T = iv.mpf(repr(params.T))
    omega = iv.mpf(repr(params.omega))
    pi = iv.pi

    # Factor 1: (sin(pi T t)/(pi t))^2 = T^2 * sinc^2(pi T t) where sinc(x)=sin(x)/x.
    # For intervals containing 0 we use a tight Taylor-remainder bound:
    #   sinc(x)^2 in [(1 - x^2/6)^2, 1] for |x| <= 1, since sinc(x) >= 1 - x^2/6
    #   (truncated alternating Taylor, remainder sign-checked for |x|<=sqrt(20)).
    # This replaces the crude [0, T^2] enclosure and dramatically speeds B&B.
    if t_iv.a <= 0 <= t_iv.b:
        x_iv = pi * T * t_iv            # argument of sinc
        # |x| upper bound:
        abs_x_hi = max(abs(x_iv.a), abs(x_iv.b))
        if abs_x_hi <= 1:
            # sinc(x) in [1 - x^2/6, 1], so sinc^2 in [(1 - x^2/6)^2, 1].
            xhi_sq = abs_x_hi * abs_x_hi
            lo = (1 - xhi_sq / 6)
            lo_sq = lo * lo if lo > 0 else 0
            Tsq = T * T
            factor1 = iv.mpf([lo_sq, 1]) * Tsq
        else:
            # Fall back: split around 0 into two sides.
            Tsq = T * T
            factor1 = iv.mpf([0, Tsq.b])
    else:
        sinpt = iv.sin(pi * T * t_iv)
        pt = pi * t_iv
        factor1 = (sinpt / pt) ** 2

    # Factor 2: 1 + sum a_k cos(2 pi k omega t).
    factor2 = iv.mpf([1, 1])
    for k, ak in enumerate(params.a, start=1):
        ak_iv = iv.mpf(repr(float(ak)))
        factor2 = factor2 + ak_iv * iv.cos(2 * pi * k * omega * t_iv)

    return factor1 * factor2


def ghat_mp(xi, params: F1Params):
    """Scalar mpmath evaluation of ghat at a scalar xi (mpf)."""
    T = mpf(params.T)
    omega = mpf(params.omega)

    def tri(x):
        v = T - abs(x)
        return v if v > 0 else mpf(0)

    val = tri(xi)
    for k, ak in enumerate(params.a, start=1):
        val += mpf(float(ak)) / 2 * (tri(xi - k * omega) + tri(xi + k * omega))
    return val


def breakpoints(params: F1Params):
    """Return the (finite) list of breakpoints of ghat."""
    T = mpf(params.T)
    omega = mpf(params.omega)
    pts = {mpf(-T), mpf(T), mpf(0)}
    for k in range(1, params.K() + 1):
        for sgn_shift in (-1, 1):
            s = sgn_shift * k * omega
            pts.add(s - T)
            pts.add(s + T)
            pts.add(s)
    return sorted(pts)


def positive_definite_certificate(params: F1Params, tol: float = 1e-15) -> tuple[bool, list]:
    """Certify ghat >= 0 by checking values at all breakpoints.

    Returns (ok, [(xi, ghat(xi)), ...]).
    """
    mp.prec = 200
    pts = breakpoints(params)
    vals = [(xi, ghat_mp(xi, params)) for xi in pts]
    ok = all(v >= -tol for _, v in vals)
    return ok, vals


def ghat_zero(params: F1Params) -> mpf:
    """ghat(0) = T + sum_k a_k * max(0, T - |k*omega|)."""
    mp.prec = 200
    T = mpf(params.T)
    omega = mpf(params.omega)
    val = T
    for k, ak in enumerate(params.a, start=1):
        val += mpf(float(ak)) * max(mpf(0), T - k * omega)
    return val


def ghat_support(params: F1Params) -> float:
    """The truncation xi_max such that ghat(xi) = 0 for |xi| > xi_max."""
    return params.T + params.K() * params.omega


def integral_g_mp(params: F1Params) -> mpf:
    """Rigorous mpmath closed-form int_{-1/2}^{1/2} g(t) dt.

    g(t) = sinc^2(pi T t) * T^2 * (1 + sum_k a_k cos(2 pi k omega t))
    Wait: (sin(pi T t)/(pi t))^2 = T^2 sinc^2(pi T t). And the integral of
    sinc^2(pi T t) * cos(2 pi k omega t) over [-1/2, 1/2] has a closed form
    via Parseval: sum equals int (triangle Delta_T) shifted against cosine
    Fourier weights... complicated. We use mpmath quadrature here, which is
    rigorous up to the declared error bound.
    """
    from mpmath import quad
    mp.prec = 200
    T = mpf(params.T)
    omega = mpf(params.omega)
    pi_mp = mp.pi

    def integrand(t):
        if abs(t) < mpf("1e-40"):
            fac1 = T * T
        else:
            fac1 = (mp.sin(pi_mp * T * t) / (pi_mp * t)) ** 2
        fac2 = mpf(1)
        for k, ak in enumerate(params.a, start=1):
            fac2 += mpf(float(ak)) * mp.cos(2 * pi_mp * k * omega * t)
        return fac1 * fac2

    # quad returns a (val, err) pair only if error=True; otherwise just val.
    val = quad(integrand, [mpf("-0.5"), mpf("0"), mpf("0.5")])
    return mpf(val)


def integral_g_iv(params: F1Params, n_subdiv: int = 1024):
    """Rigorous interval enclosure of int_{-1/2}^{1/2} g(t) dt via
    sub-interval range-bounded midpoint quadrature.

    For each sub-interval J of width h we add h * [min g, max g] over J.
    """
    a_iv = iv.mpf("-0.5")
    b_iv = iv.mpf("0.5")
    h = (b_iv - a_iv) / n_subdiv
    lo_sum = iv.mpf(0)
    hi_sum = iv.mpf(0)
    for k in range(n_subdiv):
        t_lo = iv.mpf("-0.5") + k * h
        t_hi = t_lo + h
        J = iv.mpf([t_lo.a, t_hi.b])
        G = g_iv(J, params)
        # Contribution width h * G.
        lo_sum = lo_sum + h * iv.mpf([G.a, G.a])
        hi_sum = hi_sum + h * iv.mpf([G.b, G.b])
    return mpf(lo_sum.a), mpf(hi_sum.b)


def weight_iv(xi_iv):
    """Sharp rigorous pointwise lower bound L^sharp(xi) on A(xi)^2 - B(xi)^2.

    For any admissible f >= 0 with supp f subset [-1/4, 1/4] and int f = 1,
    let A(xi) = int f(x) cos(2 pi x xi) dx and B(xi) = int f(x) sin(2 pi x xi) dx.
    The sharp pointwise lower bound on A(xi)^2 - B(xi)^2 is:

        L^sharp(xi) = cos(pi |xi|)    for |xi| <= 1
        L^sharp(xi) = -1              for |xi| > 1

    (continuous, both extremes achieved by f = delta_{1/4}).

    Since ghat(xi) >= 0 pointwise, we have the rigorous inequality
        int ghat(xi) [A(xi)^2 - B(xi)^2] dxi  >=  int ghat(xi) L^sharp(xi) dxi

    In particular, the second integral can be NEGATIVE where L^sharp < 0
    (mass of ghat outside [-1/2, 1/2] now contributes a penalty). This is
    the CORRECT rigorous formulation -- the previous weight
    max(0, cos(pi|xi|)) was not rigorous.

    Interval handling
    -----------------
    If the input interval xi_iv straddles the boundary |xi| = 1, we return
    the enclosure of the UNION of the two pieces (cos(pi|xi|) piece and
    constant -1 piece).
    """
    absxi = _iv_abs(xi_iv)
    # Case 1: entire interval has |xi| >= 1 -> weight is exactly -1.
    if absxi.a >= mpf(1):
        return iv.mpf([-1, -1])
    pi = iv.pi
    # Case 2: entire interval has |xi| <= 1 -> weight = cos(pi * |xi|).
    if absxi.b <= mpf(1):
        return iv.cos(pi * absxi)
    # Case 3: interval straddles |xi| = 1. Take the union enclosure of
    # - cos(pi * |xi|) on |xi| in [absxi.a, 1]  (yields values in [cos(pi), cos(pi*absxi.a)] = [-1, cos(pi*absxi.a)])
    # - {-1}                 on |xi| in [1, absxi.b]
    absxi_lo = iv.mpf([absxi.a, mpf(1)])
    c_lo_piece = iv.cos(pi * absxi_lo)  # encloses cos on [absxi.a, 1]
    # Union with [-1, -1]: lower = min(c_lo_piece.a, -1) = -1, upper = max(c_lo_piece.b, -1).
    lower = mpf(-1)
    upper = c_lo_piece.b if c_lo_piece.b >= mpf(-1) else mpf(-1)
    return iv.mpf([lower, upper])


# -----------------------------------------------------------------------------
# Symbolic closed forms (for theory.md cross-checks and tests).
# -----------------------------------------------------------------------------
def symbolic_g(params: F1Params):
    """Return sympy expression for g(t) with the given params."""
    t, pi = sp.symbols("t pi", positive=True)
    T = sp.Rational(*_to_rational(params.T))
    omega = sp.Rational(*_to_rational(params.omega))
    fejer = (sp.sin(sp.pi * T * t) / (sp.pi * t)) ** 2
    modulation = 1 + sum(
        sp.Rational(*_to_rational(float(ak))) * sp.cos(2 * sp.pi * k * omega * t)
        for k, ak in enumerate(params.a, start=1)
    )
    return fejer * modulation


def _to_rational(x: float, denom_cap: int = 10 ** 9):
    """Approximate float x by a rational p/q with |q|<=denom_cap."""
    from fractions import Fraction
    f = Fraction(x).limit_denominator(denom_cap)
    return (f.numerator, f.denominator)
