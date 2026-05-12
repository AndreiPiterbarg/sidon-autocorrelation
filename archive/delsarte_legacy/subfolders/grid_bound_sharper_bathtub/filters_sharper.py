"""Filters using the (placeholder) sharper bathtub bound.

STATUS: OBSTRUCTED -- mu_sharper currently equals mu_MV. See ``derivation.md``.

The signature and call surface are set up for future drop-in of a rigorous
mu_sharper(M, n) < mu_MV(M); at that time, F_bathtub_sharper becomes a
strictly tighter admissibility cut.
"""
from __future__ import annotations

from typing import Sequence

from flint import arb, ctx

from delsarte_dual.grid_bound.filters import (
    F_bathtub,
    FilterVerdict,
    _arb_sqr,
    _arb_nonneg,
    filter_all as _filter_all_MV,
)

from .mu_sharper import mu_sharper


def F_bathtub_sharper(
    ab: Sequence[arb],
    N: int,
    M_arb: arb,
    prec_bits: int = 256,
) -> FilterVerdict:
    """z_n^2 <= mu_sharper(M, n) for every n = 1..N.

    STATUS: currently identical to F_bathtub since mu_sharper = mu_MV.
    """
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        verdict = FilterVerdict.ACCEPT
        for n in range(1, N + 1):
            mu_n = mu_sharper(M_arb, n, prec_bits=prec_bits).upper()
            a = ab[2 * (n - 1)]
            b = ab[2 * (n - 1) + 1]
            z_sq = _arb_sqr(a) + _arb_sqr(b)
            v = _arb_nonneg(mu_n - z_sq)
            if v == FilterVerdict.REJECT:
                return FilterVerdict.REJECT
            if v == FilterVerdict.UNCLEAR:
                verdict = FilterVerdict.UNCLEAR
        return verdict
    finally:
        ctx.prec = old


def filter_all_sharper(
    ab: Sequence[arb],
    N: int,
    M_arb: arb,
    *,
    enable_F4_MO217: bool = True,
    enable_F7: bool = True,
    enable_F8: bool = True,
    prec_bits: int = 256,
) -> FilterVerdict:
    """Run all admissibility filters, using F_bathtub_sharper as the bathtub cut.

    Dispatches to the baseline filter_all with mu_arb equal to the (per-n
    uniform) mu_sharper upper bound, to preserve API compatibility.
    """
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        mu_arb = mu_sharper(M_arb, 1, prec_bits=prec_bits).upper()
        return _filter_all_MV(
            ab, N,
            mu_arb=mu_arb,
            enable_F4_MO217=enable_F4_MO217,
            enable_F7=enable_F7,
            enable_F8=enable_F8,
        )
    finally:
        ctx.prec = old


__all__ = ["F_bathtub_sharper", "filter_all_sharper"]
