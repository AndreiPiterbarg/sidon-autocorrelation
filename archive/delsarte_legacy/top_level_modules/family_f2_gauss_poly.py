"""Family F2: Gaussian-modulated even polynomial.

    g(t) = exp(-alpha t^2) * P(t^2),    P(u) = sum_{m=0}^N c_m u^m.

Fourier transform:
    ghat(xi) = sqrt(pi/alpha) * e^{-pi^2 xi^2 / alpha} * Q(xi^2)

where Q is an explicit polynomial in xi^2 derived from P by the operator
calculus
    FT[ t^{2m} e^{-alpha t^2} ](xi) = sqrt(pi/alpha) * (-1)^m d^m/d(alpha)^m  e^{-pi^2 xi^2 / alpha}.

The closed form for low m:
    FT[ e^{-alpha t^2} ]          = sqrt(pi/alpha) * e^{-pi^2 xi^2/alpha}
    FT[ t^2 e^{-alpha t^2} ]      = sqrt(pi/alpha) * e^{-pi^2 xi^2/alpha} * (1/(2 alpha) - pi^2 xi^2 / alpha^2)
    FT[ t^4 e^{-alpha t^2} ]      = sqrt(pi/alpha) * e^{-pi^2 xi^2/alpha}
                                    * (3/(4 alpha^2) - 3 pi^2 xi^2 / alpha^3 + pi^4 xi^4 / alpha^4)

These are hardcoded for N=0,1,2. Higher N is possible via sympy; for the
present implementation we stop at N=2 which already spans a 3-parameter
search (c_0, c_1, c_2, alpha).

Positive-definiteness certificate
---------------------------------
ghat >= 0 on R iff Q(u) >= 0 for all u >= 0, a univariate polynomial
non-negativity problem. For deg Q <= 2 we have the closed-form test
(Q = a u^2 + b u + c, u >= 0): Q >= 0 on [0,inf) iff
  (i) a >= 0, c >= 0, and
  (ii) if b < 0 and u_vertex = -b/(2a) > 0, then Q(u_vertex) >= 0,
       i.e. c - b^2/(4a) >= 0.
For deg Q > 2 we use sympy polys + Sturm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import sympy as sp
from mpmath import iv, mp, mpf


@dataclass
class F2Params:
    alpha: float
    c: Sequence[float]   # c_0, c_1, ..., c_N (coefficients of P(u) = sum c_m u^m)

    def N(self) -> int:
        return len(self.c) - 1


def _poly_eval_iv(coefs, u_iv):
    """Horner on an iv.mpf variable."""
    acc = iv.mpf([0, 0])
    for c in reversed(coefs):
        acc = acc * u_iv + iv.mpf(repr(float(c)))
    return acc


def _poly_eval_mp(coefs, u_mp):
    acc = mpf(0)
    for c in reversed(coefs):
        acc = acc * u_mp + mpf(float(c))
    return acc


def Q_coefs(params: F2Params) -> list[mpf]:
    """Return the coefficients q_0, q_1, ..., q_N of Q(xi^2) =
    sum_m c_m * (m-th derivative factor of the gaussian).

    For our implementation with N <= 2:
      m=0:  contributes c_0
      m=1:  contributes c_1 * (1/(2 alpha) - pi^2 xi^2/alpha^2)
      m=2:  contributes c_2 * (3/(4 alpha^2) - 3 pi^2 xi^2/alpha^3 + pi^4 xi^4/alpha^4)
    """
    mp.prec = 200
    alpha = mpf(params.alpha)
    pi2 = mp.pi ** 2
    N = params.N()
    q = [mpf(0)] * (N + 1)

    if N >= 0:
        q[0] += mpf(params.c[0])
    if N >= 1:
        q[0] += mpf(params.c[1]) * (1 / (2 * alpha))
        q[1] += mpf(params.c[1]) * (-pi2 / alpha ** 2)
    if N >= 2:
        q[0] += mpf(params.c[2]) * (3 / (4 * alpha ** 2))
        q[1] += mpf(params.c[2]) * (-3 * pi2 / alpha ** 3)
        q[2] += mpf(params.c[2]) * (pi2 ** 2 / alpha ** 4)
    if N >= 3:
        raise NotImplementedError("F2 implemented for N <= 2 only.")
    return q


def ghat_zero(params: F2Params) -> mpf:
    """ghat(0) = sqrt(pi/alpha) * Q(0) = sqrt(pi/alpha) * q_0."""
    mp.prec = 200
    q = Q_coefs(params)
    return mp.sqrt(mp.pi / mpf(params.alpha)) * q[0]


def ghat_iv(xi_iv, params: F2Params):
    alpha = iv.mpf(repr(params.alpha))
    pi = iv.pi
    xi_sq = xi_iv * xi_iv
    gauss = iv.exp(-pi * pi * xi_sq / alpha)
    q = Q_coefs(params)
    Qv = _poly_eval_iv([float(qi) for qi in q], xi_sq)
    return iv.sqrt(pi / alpha) * gauss * Qv


def g_iv(t_iv, params: F2Params):
    alpha = iv.mpf(repr(params.alpha))
    t_sq = t_iv * t_iv
    gauss = iv.exp(-alpha * t_sq)
    P = _poly_eval_iv([float(ci) for ci in params.c], t_sq)
    return gauss * P


def g_mp(t, params: F2Params):
    mp.prec = 200
    alpha = mpf(params.alpha)
    t_sq = mpf(t) ** 2
    return mp.exp(-alpha * t_sq) * _poly_eval_mp(params.c, t_sq)


def positive_definite_certificate(params: F2Params, tol: float = 1e-15) -> Tuple[bool, str]:
    """Deg-2 closed-form test on Q(u) >= 0 for u >= 0."""
    q = Q_coefs(params)
    N = params.N()
    if N == 0:
        ok = q[0] >= -tol
        return ok, f"Q(u)=q0={float(q[0]):.6g}; ok={ok}"
    if N == 1:
        # Q(u) = q0 + q1 * u, nonneg on [0, inf) iff q0 >= 0 and q1 >= 0
        # (else a linear polynomial with negative slope goes negative).
        ok = q[0] >= -tol and q[1] >= -tol
        return ok, f"q0={float(q[0]):.6g}, q1={float(q[1]):.6g}; ok={ok}"
    if N == 2:
        q0, q1, q2 = q
        # Q(u) = q2 u^2 + q1 u + q0.
        # Required: q0 >= 0, q2 >= 0.
        if q0 < -tol or q2 < -tol:
            return False, f"q0={float(q0):.6g}, q2={float(q2):.6g}: one is negative."
        if q2 == 0:
            # Linear: q1 u + q0.
            ok = q1 >= -tol
            return ok, f"degenerate linear: q0={float(q0):.6g}, q1={float(q1):.6g}; ok={ok}"
        # Minimum at u* = -q1/(2 q2) if q1 < 0.
        if q1 >= 0:
            return True, f"q0>=0, q1>=0, q2>=0; Q increasing on [0,inf)."
        u_star = -q1 / (2 * q2)
        Qmin = q0 - q1 ** 2 / (4 * q2)
        ok = Qmin >= -tol
        return ok, f"q=({float(q0):.3g},{float(q1):.3g},{float(q2):.3g}), Q(u*)={float(Qmin):.6g}; ok={ok}"
    raise NotImplementedError


def weight_iv(xi_iv):
    """Sharp rigorous lower bound on |f_hat|^2; see F1.weight_iv."""
    from .family_f1_selberg import weight_iv as f1_weight
    return f1_weight(xi_iv)


def integral_g_mp(params: F2Params) -> mpf:
    """int_{-1/2}^{1/2} e^{-alpha t^2} P(t^2) dt, closed form in erf/moments.

    With P(u) = sum_m c_m u^m, int_{-1/2}^{1/2} e^{-alpha t^2} t^{2m} dt
    = I_m(alpha) = (1/2) * Gamma(m+1/2)/alpha^{m+1/2} * gamma_incomplete(m+1/2, alpha/4)
    where gamma_incomplete is the lower incomplete gamma (regularised form).

    We use mpmath's mp.quad for robustness; the smoothness of the integrand
    makes this converge to machine precision quickly.
    """
    mp.prec = 200
    alpha = mpf(params.alpha)
    c = [mpf(float(ci)) for ci in params.c]

    def integrand(t):
        u = t * t
        P = mpf(0)
        for ci in reversed(c):
            P = P * u + ci
        return mp.exp(-alpha * u) * P

    return mpf(mp.quad(integrand, [mpf("-0.5"), mpf(0), mpf("0.5")]))


def integral_g_iv(params: F2Params, n_subdiv: int = 1024):
    """Interval enclosure of int_{-1/2}^{1/2} g."""
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
        lo_sum = lo_sum + h * iv.mpf([G.a, G.a])
        hi_sum = hi_sum + h * iv.mpf([G.b, G.b])
    return mpf(lo_sum.a), mpf(hi_sum.b)


def ghat_support(params: F2Params) -> float:
    """ghat has no compact support; we truncate where the Gaussian factor
    < 1e-60 of its peak. e^{-pi^2 xi^2/alpha} < 1e-60 iff
      pi^2 xi^2/alpha > 60 log 10, i.e. xi > sqrt(60 log 10 alpha)/pi.
    """
    import math
    return math.sqrt(60 * math.log(10) * params.alpha) / math.pi


def symbolic_g(params: F2Params):
    t = sp.symbols("t", real=True)
    alpha = sp.nsimplify(params.alpha, rational=True)
    P = sum(sp.nsimplify(float(c), rational=True) * t ** (2 * m) for m, c in enumerate(params.c))
    return sp.exp(-alpha * t ** 2) * P
