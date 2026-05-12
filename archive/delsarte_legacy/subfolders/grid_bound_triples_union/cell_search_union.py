"""Cell search on (a, b) with UNION-of-triples Phi rejection.

Clone of ``delsarte_dual.grid_bound.cell_search_nd`` where a cell is
rejected when EITHER

  (a) a Phase-2 admissibility filter (F1, F2, F4_MO217, F7, F8, F_bathtub)
      rigorously rejects it, OR
  (b) ANY triple in the family has its Phi_MM.upper() < 0 on the cell.

Otherwise the cell is LIVE and is bisected along its widest dimension.
Efficient pruning: triples are evaluated in supplied order and we
short-circuit the instant any rejects -- put the most productive triple
first.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappop
from typing import List, Optional, Sequence

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.filters import filter_all, FilterVerdict
from delsarte_dual.grid_bound.phi_mm import mu_of_M
from delsarte_dual.grid_bound.cell_search_nd import CellND
from .phi_union import any_triple_rejects
from .triples import Triple


@dataclass
class UnionCellResult:
    cell: CellND
    verdict: str           # "FILTER_REJECT" | "PHI_REJECT" | "LIVE"
    reason: str            # "filter_all" | f"phi_triple_{idx}" | "needs_refine"
    triple_idx: Optional[int]    # which triple rejected (if PHI_REJECT)
    phi_upper_float: Optional[float]   # min Phi.upper() across triples when LIVE


@dataclass
class UnionCellSearchResult:
    verdict: str                             # "CERTIFIED_FORBIDDEN" | "NOT_CERTIFIED"
    cells_processed: int
    terminal_cells: List[UnionCellResult]
    worst_live: Optional[UnionCellResult]


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


def certify_union_negative(
    M: arb,
    triples: Sequence[Triple],
    N: int,
    *,
    max_cells: int = 200000,
    filter_kwargs: dict | None = None,
    prec_bits: int = 256,
    starting_cell: CellND | None = None,
) -> UnionCellSearchResult:
    """Adaptive cell search rejecting when ANY triple or any filter rejects.

    Every triple's PhiMMParams must share the same N_max == N (multi-moment
    level); this is enforced to guarantee the 2N-D ab vector is consistent.
    """
    # Sanity: every triple must have the same N_max = N.
    for t in triples:
        if t.params.N_max != N:
            raise ValueError(
                f"triple idx={t.idx} has N_max={t.params.N_max}, expected {N}"
            )

    old = ctx.prec
    ctx.prec = prec_bits
    filter_kwargs = dict(filter_kwargs or {})
    mu_arb = mu_of_M(M)
    mu_upper = mu_arb.upper()
    filter_kwargs.setdefault("mu_arb", mu_upper)
    try:
        root = starting_cell if starting_cell is not None else initial_box(M, N)
        terminal: list[UnionCellResult] = []
        live: list[tuple[float, int, CellND]] = []
        cells_processed = 0
        worst_live: Optional[UnionCellResult] = None

        def classify(cell: CellND) -> UnionCellResult:
            ab = cell.as_arb_list()
            fv = filter_all(ab, N, **filter_kwargs)
            if fv == FilterVerdict.REJECT:
                return UnionCellResult(
                    cell, "FILTER_REJECT", "filter_all",
                    triple_idx=None, phi_upper_float=None,
                )
            rejected, which, best_up = any_triple_rejects(M, ab, triples)
            if rejected:
                return UnionCellResult(
                    cell, "PHI_REJECT", f"phi_triple_{which}",
                    triple_idx=which, phi_upper_float=best_up,
                )
            # LIVE: prioritise by the MIN upper across triples
            return UnionCellResult(
                cell, "LIVE", "needs_refine",
                triple_idx=None, phi_upper_float=best_up,
            )

        r0 = classify(root)
        cells_processed += 1
        if r0.verdict != "LIVE":
            terminal.append(r0)
            return UnionCellSearchResult(
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
                return UnionCellSearchResult(
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

        return UnionCellSearchResult(
            verdict="CERTIFIED_FORBIDDEN",
            cells_processed=cells_processed,
            terminal_cells=terminal,
            worst_live=None,
        )
    finally:
        ctx.prec = old


__all__ = [
    "UnionCellResult",
    "UnionCellSearchResult",
    "certify_union_negative",
    "initial_box",
]
