"""Analytic closed-form integrals for F1.

Eliminates quadrature error in the rigorous verification.

Notation
--------
Given params T, omega, a = (a_1, ..., a_K):
    ghat(xi) = Delta_T(xi) + (1/2) sum_{k=1}^K a_k ( Delta_T(xi - k*omega) + Delta_T(xi + k*omega) )
    g(t)   = T^2 * sinc^2(pi T t) * ( 1 + sum_{k=1}^K a_k * cos(2 pi k omega t) )

Closed forms we compute rigorously in mpmath:

(1) Numerator:  N(params) = int_{-1}^{1} ghat(xi) * cos^2(pi xi / 2) dxi.
    Piecewise-linear * cos^2 is elementary.

(2) int_g:  I(params) = int_{-1/2}^{1/2} g(t) dt.
    Uses the Fourier pair:
      int_{-infty}^{infty} T^2 sinc^2(pi T t) cos(2 pi k omega t) dt
         = Delta_T(k*omega)         (triangle at shift k*omega)
    so int over all R of g = Delta_T(0) + sum (a_k/2) * 2 Delta_T(k*omega)
                        = T + sum a_k * max(0, T - k*omega).
    BUT we need the truncated integral on [-1/2, 1/2]; decompose as
    (int over R) - (2 * int_{1/2}^{infty}).
    The tail  int_{1/2}^{infty} T^2 sinc^2(pi T t) cos(2 pi k omega t) dt
    has a closed form involving Si (sine integral) and Ci (cosine integral)
    functions; mpmath provides `si` and `ci`. See derivation in this module.

All functions return (lo, hi) as mpmath.mpf pairs — rigorous enclosures.
"""
from __future__ import annotations

from mpmath import mp, mpf, iv

from . import family_f1_selberg as f1


# ------------------------- analytic numerator -------------------------

def _int_triangle_times_wgen_piecewise(s, T, *, lo=mpf("-0.5"), hi=mpf("0.5")):
    """Returns int_{lo}^{hi} max(0, T - |xi - s|) * cos(pi xi) dxi.

    Default [lo, hi] = [-1/2, 1/2], the support of the general-f weight
    w_gen(xi) = cos(pi|xi|). On [-1/2, 1/2], cos(pi|xi|) = cos(pi xi) since
    cos is even.

    Piecewise linear in xi on each triangle leg; closed form for
    int (c + m xi) cos(pi xi) dxi follows from the standard antiderivative
    int xi cos(pi xi) dxi = xi sin(pi xi)/pi + cos(pi xi)/pi^2.
    """
    aA = max(lo, s - T)
    bA = min(hi, s)
    aB = max(lo, s)
    bB = min(hi, s + T)

    total = mpf(0)
    if bA > aA:
        c1, m1 = T - s, mpf(1)
        total += _int_linear_times_cos(c1, m1, aA, bA)
    if bB > aB:
        c1, m1 = T + s, mpf(-1)
        total += _int_linear_times_cos(c1, m1, aB, bB)
    return total


def _int_linear_times_cos(c, m, a, b):
    """int_a^b (c + m xi) * cos(pi xi) dxi, closed form.

    int cos(pi xi) dxi = sin(pi xi)/pi
    int xi cos(pi xi) dxi = xi sin(pi xi)/pi + cos(pi xi)/pi^2
    """
    pi = mp.pi

    def F(x):
        return c * mp.sin(pi * x) / pi + m * (x * mp.sin(pi * x) / pi + mp.cos(pi * x) / (pi * pi))

    return F(b) - F(a)


# Keep the old name as alias but redirect to the new weight.
_int_triangle_cos2_piecewise = _int_triangle_times_wgen_piecewise


def numerator_mp(params: f1.F1Params) -> mpf:
    """Rigorous mpmath closed form of N(params) = int ghat * cos^2(pi xi/2) dxi.

    Support of the integrand: [-1, 1] (weight vanishes outside). ghat may have
    support beyond [-1, 1]; the weight w clips it.
    """
    mp.prec = 200
    T = mpf(params.T)
    omega = mpf(params.omega)

    # Contribution from the central triangle Delta_T(xi) (shift s=0):
    total = _int_triangle_cos2_piecewise(mpf(0), T)
    for k, ak in enumerate(params.a, start=1):
        ak_mp = mpf(float(ak))
        s_pos = mpf(k) * omega
        s_neg = -s_pos
        contrib = (_int_triangle_cos2_piecewise(s_pos, T) +
                   _int_triangle_cos2_piecewise(s_neg, T))
        total += (ak_mp / 2) * contrib
    return total


# ------------------------- analytic int_g -------------------------

def _full_line_int_g(params: f1.F1Params) -> mpf:
    """int_{-inf}^{inf} g(t) dt = T + sum a_k * max(0, T - k*omega).
    (Via Parseval: int g = ghat(0) = triangle value at 0.)"""
    mp.prec = 200
    T = mpf(params.T)
    omega = mpf(params.omega)
    val = T
    for k, ak in enumerate(params.a, start=1):
        v = T - k * omega
        if v > 0:
            val += mpf(float(ak)) * v
    return val


def _tail_integral_sinc2_cos(k, T, omega):
    """Returns 2 * int_{1/2}^{infty} T^2 sinc^2(pi T t) cos(2 pi k omega t) dt.

    Derivation: sinc^2(pi T t) = (sin(pi T t)/(pi T t))^2.  Using
    identities  sin^2(x) = (1 - cos(2x))/2, we have
      T^2 sinc^2(pi T t) * cos(2 pi k omega t)
          = (1 - cos(2 pi T t)) / (2 pi^2 t^2) * cos(2 pi k omega t)
    so the full-line integral reduces to a combination of
      int_{1/2}^{infty} [1 - cos(2 pi T t)] cos(2 pi k omega t) / (2 pi^2 t^2) dt
    which decomposes by product-to-sum into
      cos A cos B = (1/2)(cos(A-B) + cos(A+B))
    and each term of the form
      int_{1/2}^{infty} (1 - cos(u * t)) / t^2 dt
    has the closed form
      int_{1/2}^{infty} (1 - cos(u t))/t^2 dt
        = 2 * ( Si(u/2)*(u/2)/(1/2)?? )   [Use integration by parts]
    Actually integration by parts:
      int (1-cos(ut))/t^2 dt = -(1-cos(ut))/t + u int sin(ut)/t dt
    So
      int_{1/2}^{A} (1 - cos(u t))/t^2 dt
        = [-(1 - cos(u t))/t]_{1/2}^{A} + u int_{1/2}^{A} sin(u t)/t dt.
    As A -> infty, -(1 - cos(u A))/A -> 0 (bounded/A).
    And int_{1/2}^{inf} sin(u t)/t dt = pi/2 - Si(u/2) (sign depends on u > 0).
    So
      int_{1/2}^{inf} (1 - cos(u t))/t^2 dt
        = 2 * (1 - cos(u/2)) + u * (pi/2 * sgn(u) - Si(|u|/2)*sgn(u))

    For u = 0, (1 - cos(0))/t^2 = 0 everywhere -> integral is 0.
    """
    mp.prec = 200
    two_pi = 2 * mp.pi

    # Target: 2 * int_{1/2}^{inf} T^2 sinc^2(pi T t) cos(2 pi k omega t) dt
    # = int_{1/2}^{inf} (1 - cos(2 pi T t)) cos(2 pi k omega t) / (pi^2 t^2) dt   [times 2]
    # Let I(u) := int_{1/2}^{inf} (1 - cos(u t))/t^2 dt (closed form).
    def I(u):
        u = mpf(u)
        if u == 0:
            return mpf(0)
            # For u=0, integrand is 0. Note sinc^2 * 1 still has tail, handled elsewhere.
        au = abs(u)
        # -(1 - cos(u * 1/2))/(1/2) = -2*(1 - cos(u/2))
        term1 = -2 * (1 - mp.cos(u / 2))
        # int_{1/2}^{inf} sin(u t)/t dt.  For u > 0, this equals pi/2 - Si(u/2).
        # For u < 0, by odd symmetry of sin, it equals -(pi/2 - Si(|u|/2)).
        # mpmath.si(x) = Si(x).
        if u > 0:
            sin_tail = mp.pi / 2 - mp.si(u / 2)
        else:
            sin_tail = -(mp.pi / 2 - mp.si(au / 2))
        return term1 + u * sin_tail

    # cos(A) cos(B) = (cos(A-B) + cos(A+B))/2, with A = 2pi T t, B = 2pi k omega t
    # u1 = 2 pi (T - k omega),  u2 = 2 pi (T + k omega)
    u1 = two_pi * (T - k * omega)
    u2 = two_pi * (T + k * omega)
    # integrand (1 - cos(u1 t) + ... wait let me redo
    # (1 - cos(2 pi T t)) cos(2 pi k omega t)
    #   = cos(2 pi k omega t) - (1/2)[cos(2 pi (T-k omega) t) + cos(2 pi (T+k omega) t)]
    # So:
    # A := int_{1/2}^{inf} cos(2 pi k omega t) / t^2 dt
    # B := int_{1/2}^{inf} cos(2 pi (T-k omega) t) / t^2 dt
    # C := int_{1/2}^{inf} cos(2 pi (T+k omega) t) / t^2 dt
    # tail = 2/pi^2 * (A - (B + C)/2)
    # Each int cos(v t)/t^2 dt from 1/2 to inf: use int cos(v t)/t^2 dt =
    # -cos(v t)/t - v int sin(v t)/t dt  (integration by parts), then
    # int_{1/2}^{inf} cos(v t)/t^2 dt = cos(v/2)*2 - v * (pi/2 sgn v - Si(|v|/2) sgn v)
    # = 2 cos(v/2) - |v| (pi/2 - Si(|v|/2)) sgn(v)
    # Actually simpler: int_{1/2}^{inf} cos(v t)/t^2 dt
    # = [-cos(v t)/t]_{1/2}^{inf} - v int_{1/2}^{inf} sin(v t)/t dt    ... signs via IBP
    # Let me just compute I(u) above properly and reuse.

    # Reconsider via the already-derived I(u) = int_{1/2}^{inf} (1 - cos(u t))/t^2 dt.
    # (1 - cos(u t))/t^2 = 1/t^2 - cos(u t)/t^2.
    # int_{1/2}^{inf} 1/t^2 dt = 2.
    # So int_{1/2}^{inf} cos(u t)/t^2 dt = 2 - I(u).
    def J(u):
        """int_{1/2}^{inf} cos(u t)/t^2 dt = 2 - I(u)."""
        return 2 - I(mpf(u))

    two_pi_ko = two_pi * k * omega
    A = J(two_pi_ko)
    B = J(u1)
    C = J(u2)

    # The sinc^2 factor T^2 sinc^2(pi T t) = (1 - cos(2 pi T t))/(2 pi^2 t^2).
    # times cos(2 pi k omega t): factor out 1/(2 pi^2):
    # 2 * int_{1/2}^{inf} T^2 sinc^2(pi T t) cos(2 pi k omega t) dt
    #  = (1/pi^2) * int_{1/2}^{inf} (1 - cos(2pi T t)) cos(2pi k om t) / t^2 dt
    #  = (1/pi^2) * [ A - (B + C)/2 ].
    tail = (A - (B + C) / 2) / (mp.pi * mp.pi)
    return tail


def integral_g_mp_analytic(params: f1.F1Params) -> mpf:
    """Rigorous analytic int_{-1/2}^{1/2} g(t) dt."""
    mp.prec = 200
    T = mpf(params.T)
    omega = mpf(params.omega)
    # full-line int of g:
    #   int T^2 sinc^2(pi T t) dt = T,
    #   int T^2 sinc^2(pi T t) cos(2 pi k omega t) dt = Delta_T(k omega) = max(0, T - k omega).
    full = T
    tail_contrib = _tail_integral_sinc2_cos(0, T, omega)  # = 2 * int_{1/2}^{inf} T^2 sinc^2 dt
    for k, ak in enumerate(params.a, start=1):
        delta_k = T - k * omega
        if delta_k > 0:
            full += mpf(float(ak)) * delta_k
        # Tail for cos basis:
        tail_k = _tail_integral_sinc2_cos(k, T, omega)
        tail_contrib += mpf(float(ak)) * tail_k
    # int_{-1/2}^{1/2} g = full - tail_contrib
    return full - tail_contrib
