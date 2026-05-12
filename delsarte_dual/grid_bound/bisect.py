"""Bisection on ``M`` for the largest certifiable lower bound ``M_cert``.

Given compiled :class:`PhiParams`, bisect ``M`` in ``[M_lo, M_hi]`` so
that

    ``M_lo``  was certified by ``certify_phi_negative`` (Phi < 0 over the
              admissible box);
    ``M_hi``  was not certified (either truly feasible, or refinement
              budget exhausted).

The largest certifiable ``M_lo`` is the rigorous lower bound ``M_cert``
on ``C_{1a}``.

The CLI in this module emits a single-scale arcsine reproduction
certificate (``C_{1a} >= 1.27481``) using the Matolcsi-Vinuesa baseline
parameters; the production multi-scale certificate is emitted by
:mod:`delsarte_dual.grid_bound_alt_kernel.bisect_alt_kernel`.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

from flint import arb, fmpq, ctx

from .phi import PhiParams
from .cell_search import certify_phi_negative, CellSearchResult
from .coeffs import MV_K2_NUMERATOR


def fmpq_to_str(q: fmpq) -> str:
    """Canonical ``p/q`` serialisation of an exact rational."""
    return f"{q.p}/{q.q}"


def fmpq_to_float(q: fmpq) -> float:
    """Inexact float for display only; never used inside rigorous code."""
    return float(q.p) / float(q.q)


def arb_to_dict(a: arb) -> dict:
    """JSON-serialisable summary of an arb interval (repr + float endpoints)."""
    return {
        "repr": str(a),
        "mid_float": float(a.mid()),
        "rad_float": float(a.rad()),
        "lower_float": float(a.lower()),
        "upper_float": float(a.upper()),
    }


@dataclass
class CertifiedBound:
    M_cert_q: fmpq
    cell_search: CellSearchResult
    params: PhiParams
    prec_bits: int
    bisection_history: list


def bisect_M_cert(
    params: PhiParams,
    M_lo_init: fmpq = fmpq(127, 100),
    M_hi_init: fmpq = fmpq(130, 100),
    tol_q: fmpq = fmpq(1, 10**5),
    max_cells_per_M: int = 100000,
    initial_splits: int = 32,
    prec_bits: int = 256,
    verbose: bool = True,
) -> CertifiedBound:
    """Bisect on ``M`` to find the largest certifiably-forbidden bound.

    The lower bracket invariant is

        ``M_lo`` was certified ``CERTIFIED_FORBIDDEN`` (Phi < 0 on the
        admissible box).

    If ``M_lo_init`` is not certifiable, raises ``RuntimeError``.
    """
    history: list = []
    M_lo = M_lo_init
    M_hi = M_hi_init
    if verbose:
        print(
            f"Initial bracket: [{fmpq_to_float(M_lo):.6f}, "
            f"{fmpq_to_float(M_hi):.6f}]"
        )
    first = certify_phi_negative(
        arb(M_lo),
        params,
        max_cells=max_cells_per_M,
        initial_splits=initial_splits,
        prec_bits=prec_bits,
    )
    history.append(
        {
            "M_q": fmpq_to_str(M_lo),
            "M_float": fmpq_to_float(M_lo),
            "verdict": first.verdict,
            "cells_processed": first.cells_processed,
        }
    )
    if first.verdict != "CERTIFIED_FORBIDDEN":
        raise RuntimeError(
            f"M_lo_init = {fmpq_to_float(M_lo):.6f} could not be certified "
            f"forbidden (verdict: {first.verdict}); widen the bracket."
        )
    last_good_result = first

    while M_hi - M_lo > tol_q:
        M_mid = (M_lo + M_hi) / fmpq(2)
        res = certify_phi_negative(
            arb(M_mid),
            params,
            max_cells=max_cells_per_M,
            initial_splits=initial_splits,
            prec_bits=prec_bits,
        )
        history.append(
            {
                "M_q": fmpq_to_str(M_mid),
                "M_float": fmpq_to_float(M_mid),
                "verdict": res.verdict,
                "cells_processed": res.cells_processed,
            }
        )
        if verbose:
            print(
                f"  mid = {fmpq_to_float(M_mid):.6f}  -> "
                f"{res.verdict:22s}  (cells={res.cells_processed})"
            )
        if res.verdict == "CERTIFIED_FORBIDDEN":
            M_lo = M_mid
            last_good_result = res
        else:
            M_hi = M_mid

    return CertifiedBound(
        M_cert_q=M_lo,
        cell_search=last_good_result,
        params=params,
        prec_bits=prec_bits,
        bisection_history=history,
    )


def emit_certificate(bound: CertifiedBound, filepath: str) -> str:
    """Write a JSON certificate for the single-scale Matolcsi-Vinuesa
    baseline.  Returns the SHA-256 hex of the body.

    The certificate is consumed by the independent verifier
    :mod:`delsarte_dual.grid_bound.certify`.
    """
    p = bound.params
    body = {
        "format_version": 1,
        "kind": "grid_bound_N1_single_scale_arcsine",
        "description": (
            "Single-scale arcsine reproduction of the Matolcsi-Vinuesa "
            "lower bound C_{1a} >= 1.27481.  Sign convention: "
            "Phi >= 0 means admissible."
        ),
        "inputs": {
            "delta_q": fmpq_to_str(p.delta),
            "u_q": fmpq_to_str(p.u),
            "K2_times_delta_q": fmpq_to_str(MV_K2_NUMERATOR),
            "n_coeffs": p.n_coeffs,
            "coeffs_source": (
                "arXiv:0907.1379 Appendix (verbatim 8-digit decimals, "
                "interpreted as exact rationals)"
            ),
        },
        "input_assumptions": {
            "K2_upper_bound": (
                "||K||_2^2 < 0.5747 / delta from Martin-O'Bryant, "
                "arXiv:0807.5121, Lemma 3.2."
            ),
        },
        "compiled": {
            "k1_period1":  arb_to_dict(p.k1),
            "K2_arb":      arb_to_dict(p.K2),
            "S1_arb":      arb_to_dict(p.S1),
            "min_G_cert":  arb_to_dict(p.min_G),
            "min_G_cell_center_q": fmpq_to_str(p.min_G_center),
            "gain_a":      arb_to_dict(p.gain_a),
        },
        "M_cert": {
            "rational": fmpq_to_str(bound.M_cert_q),
            "float":    fmpq_to_float(bound.M_cert_q),
        },
        "cell_search_at_M_cert": {
            "verdict":     bound.cell_search.verdict,
            "n_terminal":  len(bound.cell_search.terminal_cells),
            "worst_terminal_phi_upper": (
                bound.cell_search.worst_cell.phi_upper_float
                if bound.cell_search.worst_cell
                else None
            ),
            "cells_processed": bound.cell_search.cells_processed,
            "terminal_cells":  [
                {
                    "cell":             r.cell.to_dict(),
                    "phi_upper_float":  r.phi_upper_float,
                    "phi_arb":          r.phi_arb_str,
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
    if os.path.dirname(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, sort_keys=True)
    return digest


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the single-scale arcsine Matolcsi-Vinuesa lower "
            "bound C_{1a} >= 1.27481 and emit a JSON certificate."
        )
    )
    parser.add_argument("--prec-bits", type=int, default=256)
    parser.add_argument("--n-cells-min-G", type=int, default=8192)
    parser.add_argument("--max-cells-per-M", type=int, default=100000)
    parser.add_argument("--initial-splits", type=int, default=32)
    parser.add_argument("--tol", type=str, default="1/100000")
    parser.add_argument(
        "--out",
        default="delsarte_dual/grid_bound/certificates/"
        "single_scale_arcsine.json",
    )
    args = parser.parse_args(argv)

    print("=" * 70)
    print(
        "Single-scale arcsine baseline -- rigorous reproduction of "
        "C_{1a} >= 1.27481"
    )
    print("=" * 70)
    print()
    print(
        f"Compiling PhiParams (prec_bits={args.prec_bits}, "
        f"n_cells_min_G={args.n_cells_min_G}) ..."
    )
    params = PhiParams.from_mv(
        n_cells_min_G=args.n_cells_min_G,
        prec_bits=args.prec_bits,
    )
    print(f"  delta     = {params.delta}")
    print(f"  u         = {params.u}")
    print(f"  min G     = {params.min_G}")
    print(f"  gain a    = {params.gain_a}")
    print(f"  k_1       = {params.k1}")
    print(f"  K_2       = {params.K2}")
    print()

    tol_parts = args.tol.split("/")
    if len(tol_parts) == 2:
        tol_q = fmpq(int(tol_parts[0]), int(tol_parts[1]))
    else:
        tol_q = fmpq(int(tol_parts[0]))

    bound = bisect_M_cert(
        params,
        M_lo_init=fmpq(127, 100),
        M_hi_init=fmpq(1276, 1000),
        tol_q=tol_q,
        max_cells_per_M=args.max_cells_per_M,
        initial_splits=args.initial_splits,
        prec_bits=args.prec_bits,
        verbose=True,
    )

    print()
    print(
        f"Certified M_cert = {bound.M_cert_q}  "
        f"(float: {fmpq_to_float(bound.M_cert_q):.6f})"
    )
    print(f"Matolcsi-Vinuesa published: 1.27481")

    digest = emit_certificate(bound, args.out)
    print(f"Certificate written: {args.out}")
    print(f"SHA-256: {digest}")
    return bound


if __name__ == "__main__":
    main()
