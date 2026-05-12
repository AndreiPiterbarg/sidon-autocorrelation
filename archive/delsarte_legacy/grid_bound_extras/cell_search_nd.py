"""N-D cell search on (a_1, b_1, ..., a_N, b_N) for Phi_MM < 0 certification.

Algorithm
---------
Admissibility domain (starting box):
    a_n, b_n in [-sqrt(mu(M)), sqrt(mu(M))]   for n = 1..N
where mu(M) = M sin(pi/M)/pi  is the Chebyshev-Markov / bathtub bound on z_n^2.
(We over-approximate the disk {a_n^2+b_n^2 <= mu(M)} by the square box; F1
then trims this during the search.)

We adaptively subdivide the 2N-D box and certify Phi_MM(M, cell) < 0 on
every cell.  Each cell is either:
  * REJECTED by a filter (certified infeasible by admissibility) -> terminal
  * REJECTED by Phi_MM.upper() < 0 (certified forbidden)          -> terminal
  * bisected along its widest dimension -> 2 children

If the cell budget is exhausted without certifying every cell, we return
NOT_CERTIFIED and record the worst-case live cell.

Soundness
---------
A terminal cell falls into one of the two certified sets.  All terminal cells
together cover the entire starting box.  Hence every (a, b) consistent with
admissibility at M is either filter-inadmissible or Phi_MM-forbidden: M is
rigorously a lower bound on C_{1a}.
"""
from __future__ import annotations

from dataclasses import dataclass
from heapq import heappush, heappop
from typing import List, Optional, Sequence

from flint import arb, fmpq, ctx

from .filters import filter_all, FilterVerdict
from .phi_mm import PhiMMParams, phi_mm, mu_of_M


@dataclass
class CellND:
    """Hyperrectangle [lo_i, hi_i]_{i=1..2N} with rational endpoints."""
    lo: tuple[fmpq, ...]         # length 2N
    hi: tuple[fmpq, ...]

    @property
    def dim(self) -> int:
        return len(self.lo)

    def center(self, i: int) -> fmpq:
        return (self.lo[i] + self.hi[i]) / fmpq(2)

    def half_width(self, i: int) -> fmpq:
        return (self.hi[i] - self.lo[i]) / fmpq(2)

    def widths(self) -> list[fmpq]:
        return [self.hi[i] - self.lo[i] for i in range(self.dim)]

    def widest_dim(self) -> int:
        widths = self.widths()
        # Stable tie-break: lowest index among dimensions at max width.
        max_w = max(widths)
        return widths.index(max_w)

    def as_arb_list(self) -> list[arb]:
        return [
            arb(self.center(i), self.half_width(i))
            for i in range(self.dim)
        ]

    def bisect(self, dim: int) -> tuple["CellND", "CellND"]:
        mid = (self.lo[dim] + self.hi[dim]) / fmpq(2)
        lo2 = list(self.lo); hi2 = list(self.hi)
        lo2[dim] = mid
        hi1 = list(self.hi); hi1[dim] = mid
        return (
            CellND(tuple(self.lo), tuple(hi1)),
            CellND(tuple(lo2),     tuple(self.hi)),
        )

    def to_dict(self) -> dict:
        return {
            "lo": [f"{q.p}/{q.q}" for q in self.lo],
            "hi": [f"{q.p}/{q.q}" for q in self.hi],
        }


@dataclass
class CellNDResult:
    cell: CellND
    verdict: str           # "FILTER_REJECT" | "PHI_REJECT" | "LIVE"
    reason: str            # "F1" / "F7" / "F4_MO217" / "phi" / etc.
    phi_upper_float: Optional[float]


@dataclass
class CellSearchNDResult:
    verdict: str                        # "CERTIFIED_FORBIDDEN" or "NOT_CERTIFIED"
    cells_processed: int
    terminal_cells: List[CellNDResult]
    worst_live: Optional[CellNDResult]


def _mu_sqrt_upper_q(M: arb, cushion: fmpq = fmpq(1, 10**10)) -> fmpq:
    """Rational upper bound on sqrt(mu(M))."""
    mu = mu_of_M(M)
    sq = mu.sqrt()
    f = float(sq.upper())
    num, den = f.as_integer_ratio()
    return fmpq(num, den) + cushion


def initial_box(M: arb, N: int) -> CellND:
    """Build the 2N-dim starting box [-sqrt(mu), sqrt(mu)]^{2N}."""
    s = _mu_sqrt_upper_q(M)
    lo = tuple(-s for _ in range(2 * N))
    hi = tuple( s for _ in range(2 * N))
    return CellND(lo, hi)


def certify_phi_mm_negative(
    M: arb,
    params: PhiMMParams,
    N: int,
    max_cells: int = 200000,
    filter_kwargs: dict | None = None,
    prec_bits: int = 256,
    starting_cell: "CellND | None" = None,
) -> CellSearchNDResult:
    """Adaptive N-D cell-search certificate for Phi_MM(M, .) < 0 on admissible set.

    If ``starting_cell`` is supplied, the search uses it as the root instead
    of the full [-sqrt(mu), sqrt(mu)]^{2N} box.  This enables box-splitting
    parallelisation: split the full box into P disjoint sub-boxes and run
    P independent certifications; if all succeed the full box is certified.
    """
    old = ctx.prec
    ctx.prec = prec_bits
    filter_kwargs = dict(filter_kwargs or {})
    # Compute mu(M) once as an arb and pass to every filter_all call.
    # Use an upper-bounded arb: take mu.upper() as the effective cut.
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
            # 1. Filters
            fv = filter_all(ab, N, **filter_kwargs)
            if fv == FilterVerdict.REJECT:
                return CellNDResult(cell, "FILTER_REJECT", "filter_all", None)
            # 2. Phi_MM
            try:
                phi_v = phi_mm(M, ab, params)
            except ValueError:
                # Non-physical radicand => safely forbidden
                return CellNDResult(cell, "PHI_REJECT", "phi_radicand_neg", None)
            up = float(phi_v.upper())
            if phi_v.upper() < 0:
                return CellNDResult(cell, "PHI_REJECT", "phi_upper_negative", up)
            return CellNDResult(cell, "LIVE", "needs_refine", up)

        # Seed
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

        # Adaptive bisect
        while live:
            neg_up, _ser, cell = heappop(live)
            up_val = -neg_up
            # Invariant: up_val >= phi_upper of every remaining cell.

            if cells_processed >= max_cells:
                worst_live = classify(cell)
                return CellSearchNDResult(
                    verdict="NOT_CERTIFIED",
                    cells_processed=cells_processed,
                    terminal_cells=terminal,
                    worst_live=worst_live,
                )

            # Bisect
            dim = cell.widest_dim()
            left, right = cell.bisect(dim)
            for child in (left, right):
                r = classify(child)
                cells_processed += 1
                if r.verdict == "LIVE":
                    heappush(live, (-r.phi_upper_float, cells_processed, child))
                else:
                    terminal.append(r)

            # Early termination: if all remaining are already < 0 upper or the
            # pop-order wasn't re-established correctly, continue the loop.

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
    "certify_phi_mm_negative",
    "initial_box",
]
