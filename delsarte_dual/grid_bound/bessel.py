"""Rigorous Bessel-function quantities for the MV master inequality.

All transcendental outputs are returned as ``flint.arb`` midpoint-radius
intervals.  Rational inputs (``delta``, ``u``, ``j``) stay as ``fmpq`` so
no rounding is introduced before the transcendental step.

  * :func:`j0_pi_j_delta_over_u`  -- ``J_0(pi j delta / u)`` as ``arb``.
  * :func:`K_tilde_period_u`      -- ``(1/u) |J_0(pi j delta / u)|^2``,
                                     the period-``u`` Fourier coefficient of
                                     the single-scale arcsine kernel.
  * :func:`k1_period_one`         -- ``|J_0(pi delta)|^2``, the period-1
                                     Fourier coefficient ``k_1``.
"""
from __future__ import annotations

from flint import arb, fmpq, ctx


def _arb_pi_j_delta_over_u(j: int, delta: fmpq, u: fmpq) -> arb:
    """Return ``pi * j * delta / u`` as an arb ball at the current precision.

    The rational factor ``j delta / u`` is computed exactly in ``fmpq``,
    then multiplied by ``arb.pi()`` to produce a tight enclosure.
    """
    if j < 0:
        raise ValueError("j must be non-negative")
    if u <= 0:
        raise ValueError("u must be positive")
    q = fmpq(j) * delta / u
    return arb.pi() * arb(q)


def j0_pi_j_delta_over_u(
    j: int, delta: fmpq, u: fmpq, prec_bits: int = 256
) -> arb:
    """Rigorous arb enclosure of ``J_0(pi j delta / u)``."""
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        return _arb_pi_j_delta_over_u(j, delta, u).bessel_j(0)
    finally:
        ctx.prec = old


def K_tilde_period_u(
    j: int, delta: fmpq, u: fmpq, prec_bits: int = 256
) -> arb:
    """Period-``u`` Fourier coefficient ``(1/u) |J_0(pi j delta / u)|^2``.

    This is the ``j``-th Fourier coefficient of the single-scale arcsine
    kernel ``K`` on the period-``u`` torus.  Non-negative by the square,
    as required by admissibility property (K4).
    """
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        j0 = _arb_pi_j_delta_over_u(j, delta, u).bessel_j(0)
        return (j0 * j0) / arb(u)
    finally:
        ctx.prec = old


def k1_period_one(delta: fmpq, prec_bits: int = 256) -> arb:
    """``k_1 = hat K(1) = |J_0(pi delta)|^2`` (period-1 Fourier coefficient).

    Used in the ``z_1``-refined master inequality.
    """
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        arg = arb.pi() * arb(delta)
        j0 = arg.bessel_j(0)
        return j0 * j0
    finally:
        ctx.prec = old


__all__ = [
    "j0_pi_j_delta_over_u",
    "K_tilde_period_u",
    "k1_period_one",
]
