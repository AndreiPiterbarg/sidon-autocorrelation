"""2-D adaptive cell search and bisection for the F4-marginalized N=2 phi.

Uses ``phi_mm_marg.phi_marg_f4_upper`` to compute a rigorous upper bound on
sup over (a_2, b_2) admissible AND F4_MO217 of Phi_N=2(M, y_1, y_2)
for each (a_1, b_1) cell.  Certifies M as a rigorous lower bound on C_{1a}
when this upper bound is < 0 over the entire 2-D admissibility square.

Algorithm
---------
2-D adaptive B&B on (a_1, b_1) over the box [-sqrt(mu(M)), sqrt(mu(M))]^2.
Filters at the 2-D level:
  * F_bathtub: a_1^2 + b_1^2 <= mu(M).
  * F1:        a_1^2 + b_1^2 <= 1  (subsumed by F_bathtub since mu < 1).

For each cell:
  - If filter rejects (rigorously), terminal_FILTER_REJECT.
  - Else compute phi_marg_f4_upper.  If < 0, terminal_PHI_REJECT.
  - Else bisect along widest dim.

Certificate format
------------------
JSON with the same structure as Phase 2 MM certs, but:
  * 'kind': 'grid_bound_MM_Marg_F4'
  * each terminal cell records:  - cell endpoints in fmpq
                                 - verdict (FILTER_REJECT / PHI_REJECT)
                                 - phi_marg_upper (float)

The independent verifier in ``certify_marg.py`` re-runs phi_marg_f4 on each
cell and re-checks each filter rejection.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from heapq import heappush, heappop
from typing import List, Optional

from flint import arb, fmpq, ctx

from .phi_mm import PhiMMParams, mu_of_M
from .phi_mm_marg import phi_marg_f4_upper
from .coeffs import MV_DELTA, MV_U, MV_K2_NUMERATOR


# ============================================================================
#  Cell type (2-D over (a_1, b_1))
# ============================================================================

@dataclass
class Cell2D:
    """2-D rectangular cell with rational endpoints over (a_1, b_1)."""
    a_lo: fmpq
    a_hi: fmpq
    b_lo: fmpq
    b_hi: fmpq

    def width_a(self) -> fmpq: return self.a_hi - self.a_lo
    def width_b(self) -> fmpq: return self.b_hi - self.b_lo

    def widest_dim(self) -> int:
        """0 for a, 1 for b."""
        return 0 if self.width_a() >= self.width_b() else 1

    def as_arbs(self) -> tuple[arb, arb]:
        ca = (self.a_lo + self.a_hi) / fmpq(2)
        ra = (self.a_hi - self.a_lo) / fmpq(2)
        cb = (self.b_lo + self.b_hi) / fmpq(2)
        rb = (self.b_hi - self.b_lo) / fmpq(2)
        return arb(ca, ra), arb(cb, rb)

    def bisect(self, dim: int) -> tuple["Cell2D", "Cell2D"]:
        if dim == 0:
            mid = (self.a_lo + self.a_hi) / fmpq(2)
            return (
                Cell2D(self.a_lo, mid, self.b_lo, self.b_hi),
                Cell2D(mid, self.a_hi, self.b_lo, self.b_hi),
            )
        else:
            mid = (self.b_lo + self.b_hi) / fmpq(2)
            return (
                Cell2D(self.a_lo, self.a_hi, self.b_lo, mid),
                Cell2D(self.a_lo, self.a_hi, mid, self.b_hi),
            )

    def to_dict(self) -> dict:
        return {
            "a_lo": f"{self.a_lo.p}/{self.a_lo.q}",
            "a_hi": f"{self.a_hi.p}/{self.a_hi.q}",
            "b_lo": f"{self.b_lo.p}/{self.b_lo.q}",
            "b_hi": f"{self.b_hi.p}/{self.b_hi.q}",
        }


@dataclass
class Cell2DResult:
    cell: Cell2D
    verdict: str       # "FILTER_REJECT" | "PHI_REJECT" | "LIVE"
    reason: str
    phi_upper: float


@dataclass
class Search2DResult:
    verdict: str       # "CERTIFIED_FORBIDDEN" or "NOT_CERTIFIED"
    cells_processed: int
    terminal_cells: List[Cell2DResult]
    worst_live: Optional[Cell2DResult]


# ============================================================================
#  Helpers
# ============================================================================

def _mu_sqrt_upper_q(M: arb, cushion: fmpq = fmpq(1, 10**10)) -> fmpq:
    mu = mu_of_M(M)
    sq = mu.sqrt()
    f = float(sq.upper())
    num, den = f.as_integer_ratio()
    return fmpq(num, den) + cushion


def _arb_sqr_local(x: arb) -> arb:
    al = x.abs_lower()
    au = x.abs_upper()
    return (al * al).union(au * au)


def _filter_check_2d(a_arb: arb, b_arb: arb, mu_arb: arb) -> tuple[str, str] | None:
    """Returns (verdict, reason) if filter rigorously rejects; None if not.

    F_bathtub: a^2 + b^2 <= mu.   Reject iff (a^2+b^2).lower() > mu.upper(),
    equivalently (mu - z^2).upper() < 0.
    F1:        a^2 + b^2 <= 1.    Reject iff (a^2+b^2).lower() > 1.
    """
    z_sq = _arb_sqr_local(a_arb) + _arb_sqr_local(b_arb)
    # F_bathtub rejection
    if (mu_arb - z_sq).upper() < 0:
        return ("FILTER_REJECT", "F_bathtub")
    # F1 rejection (looser; subsumed by F_bathtub when mu < 1, but check explicitly).
    if (arb(1) - z_sq).upper() < 0:
        return ("FILTER_REJECT", "F1")
    return None


# ============================================================================
#  Cell search
# ============================================================================

def certify_phi_marg_negative(
    M: arb,
    params: PhiMMParams,
    max_cells: int = 200_000,
    starting_cell: Optional[Cell2D] = None,
    prec_bits: int = 256,
) -> Search2DResult:
    """Certify phi_marg_f4_upper(M, a_1, b_1) < 0 for all (a_1, b_1) in
    the F1+F_bathtub admissibility region.

    If ``starting_cell`` is None, the search starts from
    [-sqrt(mu(M)), sqrt(mu(M))]^2 (with rational cushion).
    """
    if len(params.k_arb) < 2:
        raise ValueError("phi_marg requires N >= 2 (need k_1, k_2)")
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        if starting_cell is None:
            s = _mu_sqrt_upper_q(M)
            starting_cell = Cell2D(-s, s, -s, s)
        mu_arb = mu_of_M(M)

        def eval_cell(c: Cell2D) -> Cell2DResult:
            a_arb, b_arb = c.as_arbs()
            f = _filter_check_2d(a_arb, b_arb, mu_arb)
            if f is not None:
                v, r = f
                return Cell2DResult(c, v, r, float('-inf'))
            up = phi_marg_f4_upper(M, a_arb, b_arb, params)
            if up < 0:
                return Cell2DResult(c, "PHI_REJECT", "phi_marg_negative", up)
            return Cell2DResult(c, "LIVE", "needs_refine", up)

        terminal: list[Cell2DResult] = []
        live: list[tuple[float, int, Cell2D]] = []
        cells_processed = 0

        r0 = eval_cell(starting_cell)
        cells_processed += 1
        if r0.verdict != "LIVE":
            terminal.append(r0)
            return Search2DResult("CERTIFIED_FORBIDDEN", cells_processed, terminal, None)
        heappush(live, (-r0.phi_upper, cells_processed, starting_cell))

        while live:
            neg_up, _ser, cell = heappop(live)
            up_val = -neg_up
            if cells_processed >= max_cells:
                worst = eval_cell(cell)
                return Search2DResult("NOT_CERTIFIED", cells_processed, terminal, worst)
            dim = cell.widest_dim()
            for child in cell.bisect(dim):
                r = eval_cell(child)
                cells_processed += 1
                if r.verdict == "LIVE":
                    heappush(live, (-r.phi_upper, cells_processed, child))
                else:
                    terminal.append(r)

        return Search2DResult("CERTIFIED_FORBIDDEN", cells_processed, terminal, None)
    finally:
        ctx.prec = old


# ============================================================================
#  Bisect on M
# ============================================================================

@dataclass
class CertifiedBoundMarg:
    M_cert_q: fmpq
    cell_search: Search2DResult
    params: PhiMMParams
    prec_bits: int
    bisection_history: list


def _fmpq_to_str(q: fmpq) -> str:
    return f"{q.p}/{q.q}"


def _fmpq_to_float(q: fmpq) -> float:
    return float(q.p) / float(q.q)


def bisect_M_cert_marg(
    params: PhiMMParams,
    *,
    M_lo_init: fmpq = fmpq(127, 100),
    M_hi_init: fmpq = fmpq(1276, 1000),
    tol_q: fmpq = fmpq(1, 10**4),
    max_cells_per_M: int = 200_000,
    prec_bits: int = 256,
    verbose: bool = True,
) -> CertifiedBoundMarg:
    """Bisect on M for the F4-marginalized N=2 cell search."""
    history: list = []

    def _run(M_q: fmpq) -> Search2DResult:
        return certify_phi_marg_negative(
            arb(M_q), params,
            max_cells=max_cells_per_M,
            prec_bits=prec_bits,
        )

    M_lo = M_lo_init
    M_hi = M_hi_init
    if verbose:
        print(f"Initial bracket: [{_fmpq_to_float(M_lo):.6f}, {_fmpq_to_float(M_hi):.6f}]")
    first = _run(M_lo)
    history.append({
        "M_q": _fmpq_to_str(M_lo),
        "M_float": _fmpq_to_float(M_lo),
        "verdict": first.verdict,
        "cells_processed": first.cells_processed,
    })
    if first.verdict != "CERTIFIED_FORBIDDEN":
        raise RuntimeError(
            f"M_lo_init = {_fmpq_to_float(M_lo):.6f} could not be certified "
            f"(worst phi_upper = {first.worst_live.phi_upper if first.worst_live else 'N/A'}); "
            "widen bracket / increase budget."
        )
    last_good = first

    while M_hi - M_lo > tol_q:
        M_mid = (M_lo + M_hi) / fmpq(2)
        res = _run(M_mid)
        history.append({
            "M_q": _fmpq_to_str(M_mid),
            "M_float": _fmpq_to_float(M_mid),
            "verdict": res.verdict,
            "cells_processed": res.cells_processed,
        })
        if verbose:
            wlu = res.worst_live.phi_upper if res.worst_live else None
            print(f"  mid = {_fmpq_to_float(M_mid):.7f} -> {res.verdict:22s}  "
                  f"(cells={res.cells_processed}, worst_live={wlu})")
        if res.verdict == "CERTIFIED_FORBIDDEN":
            M_lo = M_mid
            last_good = res
        else:
            M_hi = M_mid

    return CertifiedBoundMarg(
        M_cert_q=M_lo,
        cell_search=last_good,
        params=params,
        prec_bits=prec_bits,
        bisection_history=history,
    )


# ============================================================================
#  Certificate emit/verify
# ============================================================================

def _arb_to_str(x: arb) -> str:
    return x.str(30)


def emit_certificate_marg(bound: CertifiedBoundMarg, filepath: str) -> str:
    p = bound.params
    body = {
        "format_version": 1,
        "kind": "grid_bound_MM_Marg_F4",
        "note": (
            "F4_MO217-marginalized N=2 certificate.  Each terminal (a_1, b_1) "
            "cell is rigorously certified either by F_bathtub/F1 filter rejection "
            "or by phi_marg_f4_upper(M, a_1, b_1) < 0 where phi_marg is the "
            "analytic sup over (a_2, b_2 | F4) of Phi_N=2(M, y_1, y_2)."
        ),
        "inputs": {
            "delta_q": _fmpq_to_str(MV_DELTA),
            "u_q":     _fmpq_to_str(MV_U),
            "K2_times_delta_q": _fmpq_to_str(MV_K2_NUMERATOR),
            "n_coeffs": p.n_coeffs,
        },
        "compiled": {
            "k_arb_list": [_arb_to_str(k) for k in p.k_arb],
            "K2_arb":     _arb_to_str(p.K2),
            "S1_arb":     _arb_to_str(p.S1),
            "min_G_cert": _arb_to_str(p.min_G),
            "min_G_cell_center_q": _fmpq_to_str(p.min_G_center),
            "gain_a":     _arb_to_str(p.gain_a),
        },
        "M_cert": {
            "rational": _fmpq_to_str(bound.M_cert_q),
            "float":    _fmpq_to_float(bound.M_cert_q),
        },
        "cell_search_at_M_cert": {
            "verdict":     bound.cell_search.verdict,
            "cells_processed": bound.cell_search.cells_processed,
            "n_terminal":  len(bound.cell_search.terminal_cells),
            "terminal_cells": [
                {
                    "cell":    r.cell.to_dict(),
                    "verdict": r.verdict,
                    "reason":  r.reason,
                    "phi_upper": r.phi_upper if r.phi_upper != float('-inf') else "-inf",
                }
                for r in bound.cell_search.terminal_cells
            ],
        },
        "bisection_history": bound.bisection_history,
        "prec_bits": bound.prec_bits,
    }
    body_json = json.dumps(body, indent=2, sort_keys=True)
    digest = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    final = {"sha256_of_body": digest, "body": body}
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, sort_keys=True)
    return digest


__all__ = [
    "Cell2D",
    "Cell2DResult",
    "Search2DResult",
    "CertifiedBoundMarg",
    "certify_phi_marg_negative",
    "bisect_M_cert_marg",
    "emit_certificate_marg",
]
