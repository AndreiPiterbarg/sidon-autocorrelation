"""N-D cell search for Phi_Holder < 0 certification.

Clone of ``delsarte_dual.grid_bound.cell_search_nd`` with phi_holder in
place of phi_mm.  The filter panel is reused (imported, not modified)
because the filters depend only on the admissibility geometry of the
Fourier moments, not on the tail step.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappop
from typing import List, Optional, Sequence

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.filters import filter_all, FilterVerdict
from delsarte_dual.grid_bound.cell_search_nd import CellND, CellNDResult, CellSearchNDResult

from .phi_holder import PhiHolderParams, phi_holder, mu_of_M


def _mu_sqrt_upper_q(M: arb, cushion: fmpq = fmpq(1, 10**10)) -> fmpq:
    mu = mu_of_M(M)
    sq = mu.sqrt()
    f = float(sq.upper())
    num, den = f.as_integer_ratio()
    return fmpq(num, den) + cushion


def initial_box(M: arb, N: int) -> CellND:
    s = _mu_sqrt_upper_q(M)
    lo = tuple(-s for _ in range(2 * N))
    hi = tuple( s for _ in range(2 * N))
    return CellND(lo, hi)


def certify_phi_holder_negative(
    M: arb,
    params: PhiHolderParams,
    N: int,
    max_cells: int = 200000,
    filter_kwargs: dict | None = None,
    prec_bits: int = 256,
    starting_cell: CellND | None = None,
) -> CellSearchNDResult:
    """Adaptive N-D cell-search certificate for Phi_Holder(M, .) < 0.

    Soundness: every terminal cell is either (a) rejected by filter_all
    (rigorously inadmissible) or (b) certified with phi_holder.upper() < 0
    (rigorously forbidden by HM-10).  The union of terminal cells is the
    full admissibility box [-sqrt(mu(M)), sqrt(mu(M))]^{2N}.
    """
    old = ctx.prec
    ctx.prec = prec_bits
    filter_kwargs = dict(filter_kwargs or {})
    mu_arb = mu_of_M(M)
    mu_upper = mu_arb.upper()
    filter_kwargs.setdefault("mu_arb", mu_upper)
    try:
        root = starting_cell if starting_cell is not None else initial_box(M, N)
        terminal: list[CellNDResult] = []
        live: list[tuple[float, int, CellND]] = []
        cells_processed = 0
        worst_live: Optional[CellNDResult] = None

        def classify(cell: CellND) -> CellNDResult:
            ab = cell.as_arb_list()
            fv = filter_all(ab, N, **filter_kwargs)
            if fv == FilterVerdict.REJECT:
                return CellNDResult(cell, "FILTER_REJECT", "filter_all", None)
            try:
                phi_v = phi_holder(M, ab, params)
            except ValueError:
                return CellNDResult(cell, "PHI_REJECT", "phi_radicand_neg", None)
            up = float(phi_v.upper())
            if phi_v.upper() < 0:
                return CellNDResult(cell, "PHI_REJECT", "phi_upper_negative", up)
            return CellNDResult(cell, "LIVE", "needs_refine", up)

        r0 = classify(root)
        cells_processed += 1
        if r0.verdict != "LIVE":
            terminal.append(r0)
            return CellSearchNDResult(
                verdict="CERTIFIED_FORBIDDEN",
                cells_processed=cells_processed,
                terminal_cells=terminal,
                worst_live=None,
            )
        heappush(live, (-r0.phi_upper_float, cells_processed, root))

        while live:
            neg_up, _ser, cell = heappop(live)

            if cells_processed >= max_cells:
                worst_live = classify(cell)
                return CellSearchNDResult(
                    verdict="NOT_CERTIFIED",
                    cells_processed=cells_processed,
                    terminal_cells=terminal,
                    worst_live=worst_live,
                )

            dim = cell.widest_dim()
            left, right = cell.bisect(dim)
            for child in (left, right):
                r = classify(child)
                cells_processed += 1
                if r.verdict == "LIVE":
                    heappush(live, (-r.phi_upper_float, cells_processed, child))
                else:
                    terminal.append(r)

        return CellSearchNDResult(
            verdict="CERTIFIED_FORBIDDEN",
            cells_processed=cells_processed,
            terminal_cells=terminal,
            worst_live=None,
        )
    finally:
        ctx.prec = old


__all__ = [
    "CellND",
    "CellNDResult",
    "CellSearchNDResult",
    "certify_phi_holder_negative",
    "initial_box",
]
