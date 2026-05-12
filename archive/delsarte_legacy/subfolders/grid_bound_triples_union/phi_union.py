"""Union Phi: evaluate every triple's Phi_MM on the same (M, ab) cell.

A cell is FORBIDDEN by the UNION if ANY single triple's Phi rejects it
(Phi_i.upper() < 0).  Equivalently, the forbidden set in the 2N-D (a, b)
domain is the UNION of each triple's individual forbidden set.

This is sound because each triple independently certifies its own forbidden
region (via Phase 2's MM-10 inequality).  Intersecting the admissible
regions of all triples is the same as uniting the forbidden regions.  If the
union covers the entire admissibility box, M is certified as a lower bound
for the autocorrelation constant.

The efficient caller pattern is to evaluate triples in order and short-circuit
the instant any triple rejects -- see ``any_triple_rejects``.  When scoring
a still-LIVE cell for prioritisation we use the MIN phi_upper across triples
(most rejection-friendly triple's verdict).
"""
from __future__ import annotations

from typing import Iterable, Sequence

from flint import arb

from delsarte_dual.grid_bound.phi_mm import PhiMMParams, phi_mm
from .triples import Triple


def phi_union(
    M: arb,
    ab: Sequence[arb],
    triples: Sequence[Triple],
) -> dict:
    """Evaluate every triple's Phi on the same (M, ab).

    Returns {i: phi_i_arb} with i the triple idx.  A cell is forbidden iff
    ``min_i phi_i.upper() < 0``.  Callers wishing to short-circuit should
    use ``any_triple_rejects`` instead.
    """
    out: dict[int, arb] = {}
    for t in triples:
        try:
            out[t.idx] = phi_mm(M, ab, t.params)
        except ValueError:
            # Non-physical radicand for this triple => treat as infinitely
            # negative (certified rejection by this triple).
            out[t.idx] = arb(-1e300)  # sentinel: its .upper() is < 0
    return out


def any_triple_rejects(
    M: arb,
    ab: Sequence[arb],
    triples: Sequence[Triple],
) -> tuple[bool, int | None, float | None]:
    """Short-circuit evaluation: returns (rejected, which_idx, best_upper_float).

    * ``rejected`` True iff at least one triple's Phi.upper() < 0.
    * ``which_idx`` is the idx of the first rejecting triple.
    * ``best_upper_float`` is the MIN phi_upper seen across triples (most
      "rejection-friendly" value).  Useful for worst-cell prioritisation
      when ``rejected`` is False.

    Triples are evaluated in the order supplied; for efficiency put the
    most-often-rejecting triple first.
    """
    best_upper = None
    for t in triples:
        try:
            phi_v = phi_mm(M, ab, t.params)
        except ValueError:
            return True, t.idx, None
        up = float(phi_v.upper())
        if best_upper is None or up < best_upper:
            best_upper = up
        if phi_v.upper() < 0:
            return True, t.idx, best_upper
    return False, None, best_upper


__all__ = ["phi_union", "any_triple_rejects"]
