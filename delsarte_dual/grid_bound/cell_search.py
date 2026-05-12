"""Adaptive cell B&B certifier for ``Phi(M, y) < 0`` on ``[0, mu(M)]``.

At a fixed ``M``, ``Phi(M, .)`` is smooth in ``y``.  We certify that

    ``sup_{y in [0, mu(M)]} Phi(M, y) < 0``

by subdividing ``[0, mu(M)]`` into cells, evaluating an arb enclosure of
``Phi`` on each, and accepting only when every cell's ``Phi.upper() < 0``.

A priority-queue over live cells is sorted by ``Phi.upper()`` descending.
Each iteration pops the worst cell; if its upper bound is already
negative, the queue invariant proves every other cell is too and we
certify.  Otherwise the cell is bisected and re-queued, until either the
certificate succeeds or the cell budget is exhausted.

The list of terminal cells (every one of which has ``Phi.upper() < 0``)
is returned as the witness for the certificate and is re-checked by the
independent verifier in :mod:`certify`.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappop
from typing import List, Optional

from flint import arb, fmpq, ctx

from .phi import PhiParams, phi_N1, mu_of_M


@dataclass
class Cell:
    """Closed cell ``[lo, hi]`` with exact rational endpoints."""

    lo: fmpq
    hi: fmpq

    @property
    def width(self) -> fmpq:
        return self.hi - self.lo

    @property
    def center_q(self) -> fmpq:
        return (self.lo + self.hi) / fmpq(2)

    @property
    def half_width_q(self) -> fmpq:
        return (self.hi - self.lo) / fmpq(2)

    def as_arb(self) -> arb:
        """Rigorous arb enclosure of the closed cell."""
        return arb(self.center_q, self.half_width_q)

    def bisect(self) -> tuple["Cell", "Cell"]:
        mid = self.center_q
        return Cell(self.lo, mid), Cell(mid, self.hi)

    def to_dict(self) -> dict:
        return {
            "lo": f"{self.lo.p}/{self.lo.q}",
            "hi": f"{self.hi.p}/{self.hi.q}",
        }


@dataclass
class CellResult:
    cell: Cell
    phi_upper_float: float
    phi_arb_str: str


@dataclass
class CellSearchResult:
    verdict: str
    terminal_cells: List[CellResult]
    worst_cell: Optional[CellResult]
    cells_processed: int


def _mu_upper_rational(
    M: arb, extra_cushion: fmpq = fmpq(1, 10**10)
) -> fmpq:
    """Conservative rational upper bound on ``mu(M)``.

    Takes the arb-enclosed ``mu(M).upper()``, converts to its exact
    binary rational, then adds a tiny cushion so the cell domain
    ``[0, mu_rat]`` strictly covers ``[0, mu(M)_true]``.
    """
    mu_up = float(mu_of_M(M).upper())
    num, den = mu_up.as_integer_ratio()
    return fmpq(num, den) + extra_cushion


def certify_phi_negative(
    M: arb,
    params: PhiParams,
    max_cells: int = 20000,
    initial_splits: int = 16,
    prec_bits: int = 256,
) -> CellSearchResult:
    """Certify ``Phi(M, y) < 0`` for all ``y in [0, mu(M)]``.

    Returns ``CERTIFIED_FORBIDDEN`` on success, ``NOT_CERTIFIED`` if the
    cell budget is exhausted before the certificate succeeds.
    """
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        mu_q = _mu_upper_rational(M)

        live: list = []
        terminal: List[CellResult] = []
        cells_processed = 0

        def eval_cell(cell: Cell) -> CellResult:
            y_arb = cell.as_arb()
            phi_v = phi_N1(M, y_arb, params)
            return CellResult(
                cell=cell,
                phi_upper_float=float(phi_v.upper()),
                phi_arb_str=str(phi_v),
            )

        w = mu_q / fmpq(initial_splits)
        for k in range(initial_splits):
            cell = Cell(fmpq(k) * w, fmpq(k + 1) * w)
            r = eval_cell(cell)
            cells_processed += 1
            # heapq is a min-heap; store negated upper bound for max-heap semantics.
            heappush(live, (-r.phi_upper_float, cells_processed, cell, r))

        while live:
            neg_up, _serial, cell, res = heappop(live)
            up_val = -neg_up
            if up_val < 0:
                # The popped cell is the worst remaining; every other cell
                # has Phi.upper() <= up_val < 0.  Collect them and certify.
                terminal.append(res)
                while live:
                    _n, _s, _c, _r = heappop(live)
                    terminal.append(_r)
                return CellSearchResult(
                    verdict="CERTIFIED_FORBIDDEN",
                    terminal_cells=terminal,
                    worst_cell=res,
                    cells_processed=cells_processed,
                )

            if cells_processed >= max_cells:
                heappush(live, (neg_up, _serial, cell, res))
                return CellSearchResult(
                    verdict="NOT_CERTIFIED",
                    terminal_cells=terminal,
                    worst_cell=res,
                    cells_processed=cells_processed,
                )

            left, right = cell.bisect()
            r_left = eval_cell(left)
            r_right = eval_cell(right)
            cells_processed += 2
            heappush(
                live,
                (-r_left.phi_upper_float, cells_processed - 1, left, r_left),
            )
            heappush(
                live,
                (-r_right.phi_upper_float, cells_processed, right, r_right),
            )

        # Unreachable: an empty queue means the loop above already certified.
        return CellSearchResult(
            verdict="CERTIFIED_FORBIDDEN",
            terminal_cells=terminal,
            worst_cell=None,
            cells_processed=cells_processed,
        )
    finally:
        ctx.prec = old


__all__ = [
    "Cell",
    "CellResult",
    "CellSearchResult",
    "certify_phi_negative",
]
