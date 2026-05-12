"""Multi-moment + Hoelder bisection driver + rational certificate emission.

Mirrors ``delsarte_dual.grid_bound.bisect_mm`` but uses phi_holder and
emits certificates with an additional ``holder_exponents`` field.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Optional

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.coeffs import MV_K2_NUMERATOR

from .phi_holder import PhiHolderParams, mu_of_M
from .cell_search_holder import (
    CellND, CellNDResult, CellSearchNDResult, certify_phi_holder_negative,
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
class CertifiedBoundHolder:
    M_cert_q: fmpq
    cell_search: CellSearchNDResult
    params: PhiHolderParams
    N: int
    filter_kwargs: dict
    prec_bits: int
    bisection_history: list


def bisect_M_cert_holder(
    params: PhiHolderParams,
    N: int,
    *,
    M_lo_init: fmpq = fmpq(127, 100),
    M_hi_init: fmpq = fmpq(1276, 1000),
    tol_q: fmpq = fmpq(1, 10**4),
    max_cells_per_M: int = 500000,
    filter_kwargs: dict | None = None,
    prec_bits: int = 256,
    verbose: bool = True,
) -> CertifiedBoundHolder:
    filter_kwargs = filter_kwargs or {}
    history: list = []

    def _run(M_q: fmpq) -> CellSearchNDResult:
        return certify_phi_holder_negative(
            arb(M_q), params, N=N,
            max_cells=max_cells_per_M,
            filter_kwargs=filter_kwargs,
            prec_bits=prec_bits,
        )

    M_lo = M_lo_init
    M_hi = M_hi_init
    if verbose:
        print(
            f"Initial bracket: [{_fmpq_to_float(M_lo):.6f}, {_fmpq_to_float(M_hi):.6f}]  "
            f"(N={N}, p={_fmpq_to_str(params.p)}, q={_fmpq_to_str(params.q)})"
        )
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

    return CertifiedBoundHolder(
        M_cert_q=M_lo,
        cell_search=last_good,
        params=params,
        N=N,
        filter_kwargs=filter_kwargs,
        prec_bits=prec_bits,
        bisection_history=history,
    )


def emit_certificate_holder(bound: CertifiedBoundHolder, filepath: str) -> str:
    p = bound.params
    body = {
        "format_version": 1,
        "kind": "grid_bound_Holder_Phase2",
        "note": (
            "Phase 2 Hoelder-generalised certificate: phi_holder with "
            "admissibility filters.  Spec sign: Phi_Holder >= 0 means "
            "admissible.  At (p, q) = (2, 2), phi_holder reduces bit-for-bit "
            "to phi_mm (MM-10)."
        ),
        "N": bound.N,
        "holder_exponents": {
            "p": _fmpq_to_str(p.p),
            "q": _fmpq_to_str(p.q),
        },
        "inputs": {
            "delta_q": _fmpq_to_str(p.delta),
            "u_q": _fmpq_to_str(p.u),
            "K2_times_delta_q": _fmpq_to_str(MV_K2_NUMERATOR),
            "n_coeffs": p.n_coeffs,
            "coeffs_source": (
                "arXiv:0907.1379 Appendix (verbatim 8-digit decimals, "
                "treated as exact rationals)"
            ),
        },
        "input_assumptions": {
            "K2_upper_bound": (
                "MV state ||K||_2^2 < 0.5747/delta (MV p.3 line 141); K is "
                "not in L^2 so this is a regularised surrogate inherited "
                "from Martin-O'Bryant arXiv:0807.5121."
            ),
            "K_q_upper_construction": (
                "For q != 2, K_q := sum_j k_j^q is upper-bounded via "
                "2/(pi*delta) + 2*sum_{j=1..J_tail} (k_j^q - k_j); see "
                "derivation.md eq. (K_q-bound).  At q = 2 exactly, we reuse "
                "MV's K_2 = 0.5747/delta verbatim."
            ),
        },
        "filter_panel": {
            "enable_F1":       True,
            "enable_F2":       True,
            "enable_F4_MO217": bound.filter_kwargs.get("enable_F4_MO217", True),
            "enable_F7":       bound.filter_kwargs.get("enable_F7",       True),
            "enable_F8":       bound.filter_kwargs.get("enable_F8",       True),
            "enable_F_bathtub": True,
        },
        "compiled": {
            "k_arb_list":   [_arb_to_str(k) for k in p.k_arb],
            "sum_kn_sq_arb": _arb_to_str(p.sum_kn_sq_arb),
            "sum_kn_q_arb":  _arb_to_str(p.sum_kn_q_arb),
            "K2_arb":       _arb_to_str(p.K2),
            "K_q_upper_arb": _arb_to_str(p.K_q_upper),
            "S1_arb":       _arb_to_str(p.S1),
            "min_G_cert":   _arb_to_str(p.min_G),
            "min_G_cell_center_q": _fmpq_to_str(p.min_G_center),
            "gain_a":       _arb_to_str(p.gain_a),
        },
        "M_cert": {
            "rational": _fmpq_to_str(bound.M_cert_q),
            "float":    _fmpq_to_float(bound.M_cert_q),
        },
        "cell_search_at_M_cert": {
            "verdict":         bound.cell_search.verdict,
            "cells_processed": bound.cell_search.cells_processed,
            "n_terminal":      len(bound.cell_search.terminal_cells),
            "terminal_cells": [
                {
                    "cell":    r.cell.to_dict(),
                    "verdict": r.verdict,
                    "reason":  r.reason,
                    "phi_upper_float": r.phi_upper_float,
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
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, sort_keys=True)
    return digest


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--p-num", type=int, default=2, help="Hoelder exponent numerator")
    parser.add_argument("--p-den", type=int, default=1, help="Hoelder exponent denominator")
    parser.add_argument("--prec-bits", type=int, default=256)
    parser.add_argument("--n-cells-min-G", type=int, default=4096)
    parser.add_argument("--J-tail", type=int, default=512)
    parser.add_argument("--max-cells-per-M", type=int, default=500000)
    parser.add_argument("--tol", type=str, default="1/10000")
    parser.add_argument("--disable-MO217", action="store_true")
    parser.add_argument("--disable-F7", action="store_true")
    parser.add_argument("--disable-F8", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    p_q = fmpq(args.p_num, args.p_den)
    out = args.out or (
        f"delsarte_dual/grid_bound_holder/certificates/holder_N{args.N}"
        f"_p{args.p_num}-{args.p_den}.json"
    )
    tol_parts = args.tol.split("/")
    tol_q = fmpq(int(tol_parts[0]), int(tol_parts[1])) if len(tol_parts) == 2 else fmpq(int(tol_parts[0]))

    print("=" * 70)
    print(f"Hoelder Phase-2 at N={args.N}, p={p_q} (q={p_q/(p_q - fmpq(1))})")
    print("=" * 70)
    params = PhiHolderParams.from_mv(
        N_max=args.N, p=p_q,
        n_cells_min_G=args.n_cells_min_G,
        J_tail=args.J_tail,
        prec_bits=args.prec_bits,
    )
    print(f"Compiled params: gain_a={params.gain_a}, K_q={params.K_q_upper}")

    filter_kwargs = dict(
        enable_F4_MO217=not args.disable_MO217,
        enable_F7=not args.disable_F7,
        enable_F8=not args.disable_F8,
    )

    bound = bisect_M_cert_holder(
        params, N=args.N,
        M_lo_init=fmpq(127, 100),
        M_hi_init=fmpq(1276, 1000),
        tol_q=tol_q,
        max_cells_per_M=args.max_cells_per_M,
        filter_kwargs=filter_kwargs,
        prec_bits=args.prec_bits,
        verbose=True,
    )
    print(f"Certified M_cert = {bound.M_cert_q} (~{_fmpq_to_float(bound.M_cert_q):.6f})")
    digest = emit_certificate_holder(bound, out)
    print(f"Certificate: {out}")
    print(f"SHA-256: {digest}")
    return bound


if __name__ == "__main__":
    main()
