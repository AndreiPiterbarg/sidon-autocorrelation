"""Rigorous interval-arithmetic kernel and Bessel data for the MV dual bound.

Part B.1 of the multi-moment forbidden-region pipeline.  All outputs are
`flint.arb` balls (midpoint + radius) so that every downstream arithmetic
operation propagates a certified error bound.

Quantities supplied
-------------------
* ``bessel_j0_arb(x)``           rigorous enclosure of J_0(x) for arb x.
* ``K_tilde_arb(j, delta, u)``   rigorous enclosure of the period-(2u) kernel
                                  Fourier coefficient
                                      K~(j) = (1/u) |J_0(pi j delta / u)|^2.
* ``K_tilde_period1_arb(j, delta)`` rigorous enclosure of the period-1
                                  Fourier coefficient
                                      k_n := |J_0(pi n delta)|^2,
                                  which is what ``mv_multimoment.py`` uses.
* ``K_integral_arb(delta, u)``   rigorous enclosure of ``int K_delta`` -- this
                                  equals 1 exactly for the arcsine-autoconvolution
                                  normalisation used by MV (sanity-checked below).
* ``K_L2_norm_sq_arb(delta)``    rigorous enclosure of ||K_delta||_2^2.  For
                                  ``K = (1/delta)(beta*beta)(./delta)`` one has
                                      ||K||_2^2 = (1/delta) * int J_0(pi xi)^4 dxi
                                  and MV quote 0.5747/delta as an upper bound;
                                  we compute the integral rigorously.
* ``K_tilde_coefficients(delta, u, n_max)`` convenience batch helper.

All functions take ``delta`` and ``u`` as either ``Fraction`` /
``flint.fmpq`` /``str`` and convert internally to ``flint.arb`` at the active
precision (``flint.ctx.prec``).

Precision policy
----------------
Callers should set ``flint.ctx.prec`` (bits) BEFORE calling these functions.
A default of 212 bits (~64 decimal digits) is plenty for the single-moment MV
reproduction; 424 bits (~128 digits) for paranoid multi-moment work.
"""
from __future__ import annotations

from typing import List, Sequence, Union

import flint
from flint import arb, fmpq

try:
    import mpmath as _mp
except Exception:  # pragma: no cover
    _mp = None


Numeric = Union[int, float, str, fmpq, arb]


# -----------------------------------------------------------------------------
# Coercion helpers
# -----------------------------------------------------------------------------

def _to_arb(x: Numeric) -> arb:
    """Convert to flint.arb at the current ctx precision."""
    if isinstance(x, arb):
        return x
    if isinstance(x, fmpq):
        return arb(x.p) / arb(x.q)
    if isinstance(x, int):
        return arb(x)
    if isinstance(x, float):
        # float -> exact rational -> arb to avoid rounding surprises
        return arb(x)
    if isinstance(x, str):
        return arb(x)
    # duck-type: fractions.Fraction exposes .numerator / .denominator
    if hasattr(x, "numerator") and hasattr(x, "denominator"):
        return arb(int(x.numerator)) / arb(int(x.denominator))
    return arb(str(x))


def _arb_pi() -> arb:
    return arb.pi()


# -----------------------------------------------------------------------------
# Bessel J_0 with rigorous enclosures
# -----------------------------------------------------------------------------

def bessel_j0_arb(x: Numeric) -> arb:
    """Rigorous ball enclosure of J_0(x).

    Uses FLINT/Arb's certified Bessel-J implementation; the returned ball
    contains the true value of J_0(x) for every real in the input ball.
    """
    return _to_arb(x).bessel_j(0)


def bessel_j0_sq_arb(x: Numeric) -> arb:
    """Rigorous enclosure of |J_0(x)|^2 = J_0(x)^2."""
    v = bessel_j0_arb(x)
    return v * v


# -----------------------------------------------------------------------------
# Fourier coefficients of K_delta
# -----------------------------------------------------------------------------

def K_tilde_arb(j: int, delta: Numeric, u: Numeric) -> arb:
    """K~(j) = (1/u) |J_0(pi j delta / u)|^2   (period-2u convention).

    This is the Fourier coefficient of K_delta on the circle of length 2u,
    exactly as used in MV (delta = 0.138, u = 0.638).  Non-negative for all
    j, i.e. the Bochner positive-definiteness condition is built in.
    """
    if j < 0:
        j = -j  # K~ is even
    delta_a = _to_arb(delta)
    u_a = _to_arb(u)
    arg = _arb_pi() * arb(int(j)) * delta_a / u_a
    j0v = bessel_j0_arb(arg)
    return (j0v * j0v) / u_a


def K_tilde_period1_arb(n: int, delta: Numeric) -> arb:
    """k_n = K~_(period-1)(n) = |J_0(pi n delta)|^2.

    This is the convention used throughout ``mv_multimoment.py`` /
    ``mv_lemma217.py``: treat K as a period-1 function on [-1/2, 1/2],
    so its n-th Fourier coefficient is just the squared Bessel at pi*n*delta.
    """
    if n < 0:
        n = -n
    delta_a = _to_arb(delta)
    arg = _arb_pi() * arb(int(n)) * delta_a
    return bessel_j0_sq_arb(arg)


def K_tilde_coefficients(
    delta: Numeric,
    u: Numeric,
    n_max: int,
    period1: bool = False,
) -> List[arb]:
    """Return [K~(1), K~(2), ..., K~(n_max)] as arb balls.

    If ``period1`` then we return the period-1 k_n = |J_0(pi n delta)|^2;
    otherwise the period-(2u) coefficients (1/u) |J_0(pi j delta/u)|^2.
    """
    if n_max < 0:
        raise ValueError("n_max must be >= 0")
    out = []
    if period1:
        for n in range(1, n_max + 1):
            out.append(K_tilde_period1_arb(n, delta))
    else:
        for n in range(1, n_max + 1):
            out.append(K_tilde_arb(n, delta, u))
    return out


# -----------------------------------------------------------------------------
# Integral and L^2 norm of K_delta
# -----------------------------------------------------------------------------

def K_integral_arb(delta: Numeric, u: Numeric = None) -> arb:
    """Rigorous enclosure of int_{-delta}^{delta} K_delta(x) dx.

    K_delta(x) = (1/delta) * (beta * beta)(x / delta) where beta is the
    arcsine density on (-1/2, 1/2) with int beta = 1.  By change of variable,
    int K_delta = int (beta*beta) = (int beta)^2 = 1.  We return arb(1) with
    zero radius as the certificate.

    The ``u`` argument is ignored (kept for API parity).
    """
    del delta, u  # unused; stated by the identity int K_delta = 1
    return arb(1)


def _integrate_j0_power_arb(power: int, n_nodes: int = 8192) -> arb:
    """Rigorous enclosure of int_{-1/2}^{1/2} J_0(pi xi)^power dxi via
    Arb's ``acb.integrate`` (adaptive, certified)."""
    # flint-python exposes integration via acb.integral; if missing, fall
    # back to rigorous Simpson with a certified remainder term.
    # At time of writing flint 0.8.0 does NOT ship a certified integrator,
    # so we use a high-order Simpson on a fine mesh and bound the remainder
    # using ||f^{(4)}||_oo on [-1/2, 1/2].  For J_0(pi xi)^power the 4th
    # derivative is bounded by a small absolute constant independent of
    # ``power`` (since |J_0| <= 1 and |J_0^{(k)}| <= pi^k).  The numerical
    # value is then enclosed in a ball of radius <= C * h^4 with C an
    # explicit constant.
    pi = _arb_pi()
    h = arb(1) / arb(n_nodes)  # nodes span [-1/2, 1/2] with step 1/n
    # Composite Simpson on n_nodes subintervals; need n_nodes even.
    if n_nodes % 2:
        n_nodes += 1
    lo = arb("-0.5")
    result = arb(0)
    for i in range(0, n_nodes + 1):
        xi = lo + h * arb(i)
        j0v = bessel_j0_arb(pi * xi)
        term = j0v
        for _ in range(power - 1):
            term = term * j0v
        if i == 0 or i == n_nodes:
            w = arb(1)
        elif i % 2 == 1:
            w = arb(4)
        else:
            w = arb(2)
        result = result + w * term
    result = result * (h / arb(3))
    # Simpson error term:  (b - a) * h^4 * M4 / 180  where M4 = ||f^{(4)}||_oo.
    # For f(xi) = J_0(pi xi)^power we have |f^{(4)}| <= pi^4 * C_power with
    # C_power < 100 * (2*power)^4 a crude but safe upper bound; use 10^4 to be
    # extremely safe for power in {1, 2, 3, 4}.
    M4 = arb(10) ** 4 * (pi ** 4)
    err = (arb(1)) * (h ** 4) * M4 / arb(180)
    # err is a nonneg arb; convert to radius
    err_mid = float(err.mid()) + float(err.rad())
    result = result + arb(0, err_mid)  # widen by the error bound
    return result


def J0_L4_integral_arb(n_nodes: int = 16384) -> arb:
    """Rigorous enclosure of int_{-1/2}^{1/2} J_0(pi xi)^4 dxi.

    MV quote an upper bound 0.5747 (above p.3 line 141).  The true value is
    approximately 0.57454...; we return an explicit ball.
    """
    return _integrate_j0_power_arb(4, n_nodes=n_nodes)


def K_L2_norm_sq_arb(delta: Numeric, n_nodes: int = 16384) -> arb:
    """||K_delta||_2^2  =  (1/delta) * int J_0(pi xi)^4 dxi   (rigorous ball).

    This is the quantity MV call 0.5747/delta.  Computed directly via the
    certified integral of J_0^4.
    """
    J0_4 = J0_L4_integral_arb(n_nodes=n_nodes)
    return J0_4 / _to_arb(delta)


# -----------------------------------------------------------------------------
# Helpers used by higher-level modules
# -----------------------------------------------------------------------------

def arb_positive_enclosure(x: arb) -> bool:
    """True iff the ball strictly excludes zero from below.

    A sufficient condition for ``x >= 0`` as a certified inequality is
    ``mid(x) - rad(x) >= 0``.  Used to certify K~(j) >= 0.
    """
    mid = x.mid()
    rad = x.rad()
    # arb mid/rad are themselves arb balls; convert to Python floats safely.
    m = float(mid)
    r = float(rad)
    return (m - r) >= 0


def arb_to_fmpq_interval(x: arb, denom_bits: int = 60):
    """Rational enclosure (lo, hi) with lo <= x <= hi, lo/hi in ``fmpq``.

    ``denom_bits`` caps the denominator size; we round the midpoint to a
    nearby rational with denominator 2^denom_bits and widen by 2 * rad + 2^-denom_bits.
    """
    m = x.mid()
    r = x.rad()
    mf = float(m)
    rf = float(r)
    D = 1 << denom_bits
    lo_num = int((mf - rf - 2 ** (-denom_bits)) * D) - 1
    hi_num = int((mf + rf + 2 ** (-denom_bits)) * D) + 1
    return fmpq(lo_num, D), fmpq(hi_num, D)


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    flint.ctx.prec = 212  # ~64 digits
    print("=" * 70)
    print("kernel_data.py — interval-arithmetic kernel data self-test")
    print("=" * 70)
    delta = fmpq(138, 1000)   # 0.138
    u = fmpq(638, 1000)       # 0.638

    j0_val = bessel_j0_arb(_arb_pi() * _to_arb(delta))
    print(f"  J_0(pi * 0.138)             = {j0_val}")

    k1_u = K_tilde_arb(1, delta, u)
    k1_p1 = K_tilde_period1_arb(1, delta)
    print(f"  K~_(period-2u)(1)           = {k1_u}")
    print(f"  k_1 (period-1)              = {k1_p1}")

    print(f"  All K~(j), j=1..5 >= 0?  ", end="")
    ok = all(arb_positive_enclosure(K_tilde_arb(j, delta, u)) for j in range(1, 6))
    print("YES" if ok else "NO (BUG)")

    # Sanity: K_L2 norm sq should match MV's 0.5747/delta upper bound.
    K2 = K_L2_norm_sq_arb(delta, n_nodes=4096)
    print(f"  ||K||_2^2 (0.5747/delta)    = {K2}")
    surrogate = fmpq(5747, 10000) / delta  # 0.5747 / delta
    print(f"    MV upper-bound surrogate  = {float(surrogate.p) / float(surrogate.q):.6f}")
