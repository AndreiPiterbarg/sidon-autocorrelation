"""N-D cell search using filter_all_sharper instead of filter_all.

STATUS: OBSTRUCTED. Identical behaviour to baseline cell_search_nd since
mu_sharper = mu_MV. Shipped as infrastructure for when a rigorous sharper
bound is plugged into mu_sharper.py.
"""
from __future__ import annotations

from typing import Optional

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.cell_search_nd import (
    CellND,
    CellNDResult,
    CellSearchNDResult,
    certify_phi_mm_negative as _certify_baseline,
    initial_box,
)
from delsarte_dual.grid_bound.phi_mm import PhiMMParams

from .mu_sharper import mu_sharper


def certify_phi_mm_negative_sharper(
    M: arb,
    params: PhiMMParams,
    N: int,
    max_cells: int = 200000,
    filter_kwargs: Optional[dict] = None,
    prec_bits: int = 256,
    starting_cell: Optional[CellND] = None,
) -> CellSearchNDResult:
    """Run the N-D cell search with mu_arb = mu_sharper (placeholder = mu_MV).

    The only difference from ``certify_phi_mm_negative`` is the value of
    ``mu_arb`` passed to the filter pipeline; since mu_sharper currently
    equals mu_MV, the result is identical. When a strict mu_sharper is
    proved and plugged into mu_sharper.py, this function automatically
    tightens without further changes.
    """
    old = ctx.prec
    ctx.prec = prec_bits
    filter_kwargs = dict(filter_kwargs or {})
    try:
        mu_arb = mu_sharper(M, 1, prec_bits=prec_bits).upper()
        filter_kwargs.setdefault("mu_arb", mu_arb)
        return _certify_baseline(
            M=M,
            params=params,
            N=N,
            max_cells=max_cells,
            filter_kwargs=filter_kwargs,
            prec_bits=prec_bits,
            starting_cell=starting_cell,
        )
    finally:
        ctx.prec = old


__all__ = [
    "CellND",
    "CellNDResult",
    "CellSearchNDResult",
    "certify_phi_mm_negative_sharper",
    "initial_box",
]
