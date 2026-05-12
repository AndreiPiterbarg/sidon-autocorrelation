"""Rigorous lower bound on ``min_{x in [0, 1/4]} G(x)`` via Taylor B&B.

``G(x) = sum_{j=1}^{N} a_j cos(2 pi j x / u)``.

On a closed cell ``[c - r, c + r]`` we use the second-order Taylor
expansion around the centre ``c``:

    ``G(cell) subset G(c) + G'(c) [-r, r] + G''(cell) [-r^2/2, r^2/2]``,

where ``G''(cell)`` is an arb enclosure of ``G''`` over the cell.  The
enclosure radius is ``|G'(c)| r + |G''(cell)| r^2 / 2``, second-order
near the minimum and first-order elsewhere.

We compute a certified lower bound on ``min G`` -- not an estimate of
where the minimum is attained -- by:

  1. Decomposing ``[0, 1/4]`` into ``n_cells`` equal cells.
  2. Forming the Taylor enclosure on each cell.
  3. Taking the minimum of the per-cell lower endpoints.

The returned ``arb`` has ``.lower()`` equal to the certified bound.
"""
from __future__ import annotations

from typing import Sequence

from flint import arb, fmpq, ctx


def _two_pi_over_u(u: fmpq) -> arb:
    return arb(2) * arb.pi() / arb(u)


def _eval_G_at_point(
    coeffs: Sequence[fmpq], x_q: fmpq, u: fmpq
) -> arb:
    """``G(x)`` at an exact rational ``x``; only the cosine evaluations
    introduce arb-width.
    """
    two_pi_over_u = _two_pi_over_u(u)
    x_arb = arb(x_q)
    total = arb(0)
    for j, a_j in enumerate(coeffs, start=1):
        arg = two_pi_over_u * arb(j) * x_arb
        total = total + arb(a_j) * arg.cos()
    return total


def _eval_G_prime_at_point(
    coeffs: Sequence[fmpq], x_q: fmpq, u: fmpq
) -> arb:
    """``G'(x) = -sum_j a_j (2 pi j / u) sin(2 pi j x / u)``."""
    two_pi_over_u = _two_pi_over_u(u)
    x_arb = arb(x_q)
    total = arb(0)
    for j, a_j in enumerate(coeffs, start=1):
        arg = two_pi_over_u * arb(j) * x_arb
        total = total - arb(a_j) * (two_pi_over_u * arb(j)) * arg.sin()
    return total


def _eval_G_second_on_cell(
    coeffs: Sequence[fmpq], cell_arb: arb, u: fmpq
) -> arb:
    """``G''`` enclosure on a cell.

    ``G''(x) = -sum_j a_j (2 pi j / u)^2 cos(2 pi j x / u)``.
    """
    two_pi_over_u = _two_pi_over_u(u)
    total = arb(0)
    for j, a_j in enumerate(coeffs, start=1):
        arg = two_pi_over_u * arb(j) * cell_arb
        w = two_pi_over_u * arb(j)
        total = total - arb(a_j) * (w * w) * arg.cos()
    return total


def G_enclosure_taylor(
    coeffs: Sequence[fmpq],
    c: fmpq,
    r: fmpq,
    u: fmpq,
) -> arb:
    """Rigorous arb enclosure of ``{G(x) : x in [c - r, c + r]}``.

    Second-order Taylor: ``G(c) + G'(c) [-r, r] + G''(cell) [-r^2/2, r^2/2]``.
    """
    G_c = _eval_G_at_point(coeffs, c, u)
    Gp_c = _eval_G_prime_at_point(coeffs, c, u)
    cell_arb = arb(c, r)
    Gpp_cell = _eval_G_second_on_cell(coeffs, cell_arb, u)

    dx_ball = arb(0, r)
    # Conservative superset of the second-order Taylor remainder; the
    # tighter form is on [0, r^2/2] but [-r^2/2, r^2/2] only doubles the
    # remainder width and avoids sign bookkeeping.
    half_r_sq = (arb(r) * arb(r)) / arb(2)
    rem_ball = arb(0, 1) * half_r_sq
    return G_c + Gp_c * dx_ball + Gpp_cell * rem_ball


def min_G_lower_bound(
    coeffs: Sequence[fmpq],
    u: fmpq,
    x_lo: fmpq = fmpq(0),
    x_hi: fmpq = fmpq(1, 4),
    n_cells: int = 4096,
    prec_bits: int = 256,
) -> tuple[arb, fmpq]:
    """Certify a rigorous lower bound on ``min_{x in [x_lo, x_hi]} G(x)``.

    Returns the arb of the worst cell's Taylor enclosure (its ``.lower()``
    is the certified scalar bound) and the rational midpoint of that
    cell.  Cell selection compares scalar floats; rigour comes from the
    arb enclosures themselves.
    """
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        total_width = x_hi - x_lo
        cell_width = total_width / fmpq(n_cells)
        half_width = cell_width / fmpq(2)

        worst_arb = None
        worst_float = None
        worst_center = None

        for k in range(n_cells):
            c = x_lo + (fmpq(2 * k + 1) * half_width)
            encl = G_enclosure_taylor(coeffs, c, half_width, u)
            lo_as_float = float(encl.lower())
            if worst_float is None or lo_as_float < worst_float:
                worst_float = lo_as_float
                worst_arb = encl
                worst_center = c
        return worst_arb, worst_center
    finally:
        ctx.prec = old


__all__ = [
    "G_enclosure_taylor",
    "min_G_lower_bound",
]
