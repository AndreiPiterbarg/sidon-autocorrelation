"""Bisection driver + certificate emission for the union-of-triples pipeline.

Given a family of triples (each a fully compiled PhiMMParams at a particular
delta), bisect on M to find the largest M that is certifiably forbidden by
the UNION of the triples' forbidden sets (see cell_search_union).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence, Optional

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.coeffs import MV_K2_NUMERATOR
from delsarte_dual.grid_bound.phi_mm import PhiMMParams
from .triples import Triple, load_qp_coeffs_json
from .cell_search_union import (
    UnionCellSearchResult,
    UnionCellResult,
    certify_union_negative,
)


def _fmpq_to_str(q: fmpq) -> str:
    return f"{q.p}/{q.q}"


def _fmpq_to_float(q: fmpq) -> float:
    return float(q.p) / float(q.q)


def _arb_to_str(a: arb) -> str:
    return json.dumps({
        "repr": str(a),
        "mid_float": float(a.mid()),
        "rad_float": float(a.rad()),
        "lower_float": float(a.lower()),
        "upper_float": float(a.upper()),
    })


@dataclass
class UnionBound:
    M_cert_q: fmpq
    cell_search: UnionCellSearchResult
    triples: tuple[Triple, ...]
    N: int
    filter_kwargs: dict
    prec_bits: int
    bisection_history: list


def bisect_M_cert_union(
    triples: Sequence[Triple],
    N: int,
    *,
    M_lo_init: fmpq = fmpq(127, 100),
    M_hi_init: fmpq = fmpq(1280, 1000),
    tol_q: fmpq = fmpq(1, 10**4),
    max_cells_per_M: int = 500000,
    filter_kwargs: dict | None = None,
    prec_bits: int = 256,
    verbose: bool = True,
) -> UnionBound:
    filter_kwargs = filter_kwargs or {}
    history: list = []

    def _run(M_q: fmpq) -> UnionCellSearchResult:
        return certify_union_negative(
            arb(M_q), triples, N=N,
            max_cells=max_cells_per_M,
            filter_kwargs=filter_kwargs,
            prec_bits=prec_bits,
        )

    M_lo = M_lo_init
    M_hi = M_hi_init
    if verbose:
        print(f"Initial bracket: [{_fmpq_to_float(M_lo):.6f}, "
              f"{_fmpq_to_float(M_hi):.6f}]  (N={N}, T={len(triples)})")

    first = _run(M_lo)
    history.append({
        "M_q": _fmpq_to_str(M_lo),
        "M_float": _fmpq_to_float(M_lo),
        "verdict": first.verdict,
        "cells_processed": first.cells_processed,
    })
    if first.verdict != "CERTIFIED_FORBIDDEN":
        raise RuntimeError(
            f"M_lo_init = {_fmpq_to_float(M_lo):.6f} could not be certified; "
            "widen bracket."
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
            print(
                f"  mid = {_fmpq_to_float(M_mid):.6f} -> {res.verdict:22s}  "
                f"(cells={res.cells_processed})"
            )
        if res.verdict == "CERTIFIED_FORBIDDEN":
            M_lo = M_mid
            last_good = res
        else:
            M_hi = M_mid

    return UnionBound(
        M_cert_q=M_lo,
        cell_search=last_good,
        triples=tuple(triples),
        N=N,
        filter_kwargs=filter_kwargs,
        prec_bits=prec_bits,
        bisection_history=history,
    )


def _load_triple_coeffs(t: Triple) -> tuple[fmpq, ...]:
    """Return the coefficients backing triple `t` as a tuple of fmpq.

    These must be the EXACT rationals that produced the compiled PhiMMParams;
    the verifier recomputes S1, min_G, gain_a from these.
    """
    coeffs, _raw = load_qp_coeffs_json(t.params.delta)
    return coeffs


def emit_certificate_union(bound: UnionBound, filepath: str) -> str:
    triples_data = []
    for t in bound.triples:
        p = t.params
        coeffs = _load_triple_coeffs(t)
        triples_data.append({
            "idx": t.idx,
            "delta_q": _fmpq_to_str(p.delta),
            "u_q":     _fmpq_to_str(p.u),
            "n_coeffs": p.n_coeffs,
            "N_max":   p.N_max,
            "K2_times_delta_q": _fmpq_to_str(MV_K2_NUMERATOR),
            "coeffs_pq": [_fmpq_to_str(q) for q in coeffs],
            "compiled": {
                "k_arb_list":    [_arb_to_str(k) for k in p.k_arb],
                "sum_kn_sq_arb": _arb_to_str(p.sum_kn_sq_arb),
                "K2_arb":        _arb_to_str(p.K2),
                "S1_arb":        _arb_to_str(p.S1),
                "min_G_cert":    _arb_to_str(p.min_G),
                "min_G_cell_center_q": _fmpq_to_str(p.min_G_center),
                "gain_a":        _arb_to_str(p.gain_a),
            },
        })

    cell_list = []
    for r in bound.cell_search.terminal_cells:
        cell_list.append({
            "cell": r.cell.to_dict(),
            "verdict": r.verdict,            # FILTER_REJECT | PHI_REJECT
            "reason":  r.reason,
            "triple_idx": r.triple_idx,
            "phi_upper_float": r.phi_upper_float,
        })

    body = {
        "format_version": 1,
        "kind": "grid_bound_TRIPLES_UNION",
        "note": (
            "Union-of-triples certificate: forbidden iff ANY triple's Phi_MM "
            "rejects OR the filter panel rejects.  MV Remark 2 "
            "(mv_construction_detailed.md lines 493-499) applied in 2N-D."
        ),
        "N": bound.N,
        "T": len(bound.triples),
        "triples": triples_data,
        "filter_panel": {
            "enable_F1":       True,
            "enable_F2":       True,
            "enable_F4_MO217": bound.filter_kwargs.get("enable_F4_MO217", True),
            "enable_F7":       bound.filter_kwargs.get("enable_F7",       True),
            "enable_F8":       bound.filter_kwargs.get("enable_F8",       True),
            "enable_F_bathtub": True,
        },
        "M_cert": {
            "rational": _fmpq_to_str(bound.M_cert_q),
            "float":    _fmpq_to_float(bound.M_cert_q),
        },
        "cell_search_at_M_cert": {
            "verdict":         bound.cell_search.verdict,
            "cells_processed": bound.cell_search.cells_processed,
            "n_terminal":      len(bound.cell_search.terminal_cells),
            "terminal_cells":  cell_list,
        },
        "bisection_history": bound.bisection_history,
        "prec_bits": bound.prec_bits,
    }
    body_json = json.dumps(body, indent=2, sort_keys=True)
    digest = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    final = {"sha256_of_body": digest, "body": body}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, sort_keys=True)
    return digest


__all__ = ["UnionBound", "bisect_M_cert_union", "emit_certificate_union"]
