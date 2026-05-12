"""Driver: run the union-of-triples certification for a range of families.

Prints the standard report table (T, delta_set, M_cert_union) and writes a
certificate for the full sweep to ``certificates/union_T<T>_N<N>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.coeffs import MV_K2_NUMERATOR
from delsarte_dual.grid_bound_triples_union.triples import (
    DELTA_SWEEP, build_family,
)
from delsarte_dual.grid_bound_triples_union.bisect_union import (
    bisect_M_cert_union, emit_certificate_union,
)
from delsarte_dual.grid_bound_triples_union.certify_union import (
    verify_certificate_union,
)


def _fq(q): return float(q.p) / float(q.q)


def run_family(delta_list, N, *, M_lo, M_hi, tol, max_cells, prec_bits, verbose=True):
    """Build the family, bisect, and return (bound, wall_seconds)."""
    t0 = time.time()
    family = build_family(
        list(delta_list), N_max=N,
        K2_times_delta=MV_K2_NUMERATOR,
        n_cells_min_G=4096, prec_bits=prec_bits,
    )
    if not family:
        raise RuntimeError("No cached QPs available for any delta in the list.")
    bound = bisect_M_cert_union(
        family, N=N,
        M_lo_init=M_lo, M_hi_init=M_hi, tol_q=tol,
        max_cells_per_M=max_cells,
        filter_kwargs=dict(enable_F4_MO217=(N >= 2), enable_F7=True, enable_F8=True),
        prec_bits=prec_bits,
        verbose=verbose,
    )
    wall = time.time() - t0
    return bound, wall, family


def summarise(bound, wall, family, label):
    deltas = [str(_fq(t.delta)) for t in family]
    print(f"\n{label}: T={len(family)}, deltas={deltas}")
    print(f"  M_cert = {bound.M_cert_q}  (~{_fq(bound.M_cert_q):.6f})")
    print(f"  cells = {bound.cell_search.cells_processed}")
    print(f"  wall  = {wall:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1)
    ap.add_argument("--M-lo", type=str, default="1270/1000")
    ap.add_argument("--M-hi", type=str, default="1280/1000")
    ap.add_argument("--tol", type=str, default="1/10000")
    ap.add_argument("--max-cells", type=int, default=500000)
    ap.add_argument("--prec-bits", type=int, default=256)
    ap.add_argument("--skip-T1", action="store_true")
    ap.add_argument("--skip-T2", action="store_true")
    ap.add_argument("--skip-T3", action="store_true")
    ap.add_argument("--skip-T7", action="store_true")
    ap.add_argument("--out-dir", type=str,
                    default=os.path.join(os.path.dirname(__file__), "certificates"))
    args = ap.parse_args()

    def parse_q(s):
        p, q = s.split("/"); return fmpq(int(p), int(q))
    M_lo = parse_q(args.M_lo); M_hi = parse_q(args.M_hi); tol = parse_q(args.tol)

    ctx.prec = args.prec_bits
    os.makedirs(args.out_dir, exist_ok=True)

    results = {}

    # Canonical single triple
    if not args.skip_T1:
        bound, wall, fam = run_family(
            [fmpq(138, 1000)], N=args.N,
            M_lo=M_lo, M_hi=M_hi, tol=tol,
            max_cells=args.max_cells, prec_bits=args.prec_bits,
        )
        summarise(bound, wall, fam, "T=1 (delta=0.138)")
        results["T1"] = {"M_cert_float": _fq(bound.M_cert_q), "wall": wall}

    # Pair
    if not args.skip_T2:
        bound, wall, fam = run_family(
            [fmpq(13, 100), fmpq(138, 1000)], N=args.N,
            M_lo=M_lo, M_hi=M_hi, tol=tol,
            max_cells=args.max_cells, prec_bits=args.prec_bits,
        )
        summarise(bound, wall, fam, "T=2 ({0.13, 0.138})")
        results["T2"] = {"M_cert_float": _fq(bound.M_cert_q), "wall": wall}

    # Triple
    if not args.skip_T3:
        bound, wall, fam = run_family(
            [fmpq(13, 100), fmpq(138, 1000), fmpq(15, 100)], N=args.N,
            M_lo=M_lo, M_hi=M_hi, tol=tol,
            max_cells=args.max_cells, prec_bits=args.prec_bits,
        )
        summarise(bound, wall, fam, "T=3 ({0.13, 0.138, 0.15})")
        results["T3"] = {"M_cert_float": _fq(bound.M_cert_q), "wall": wall}

    # Full sweep
    if not args.skip_T7:
        bound, wall, fam = run_family(
            list(DELTA_SWEEP), N=args.N,
            M_lo=M_lo, M_hi=M_hi, tol=tol,
            max_cells=args.max_cells, prec_bits=args.prec_bits,
        )
        summarise(bound, wall, fam, "T=7 (full sweep)")
        # Emit and verify the certificate
        cert_path = os.path.join(args.out_dir, f"union_T7_N{args.N}.json")
        digest = emit_certificate_union(bound, cert_path)
        print(f"\nCertificate written: {cert_path}")
        print(f"SHA-256: {digest}")
        results["T7"] = {
            "M_cert_float": _fq(bound.M_cert_q),
            "wall": wall,
            "cert_path": cert_path,
            "sha256": digest,
        }

        print("\n--- Verifying certificate ---")
        res = verify_certificate_union(cert_path)
        print(f"\nCertificate accepted: {res.accepted}")
        results["T7"]["cert_accepted"] = bool(res.accepted)

    print("\n============ SWEEP SUMMARY ============")
    for k, v in results.items():
        print(f"{k:4s}  M_cert = {v.get('M_cert_float'):.6f}  wall = {v.get('wall'):.1f}s")
    print("=======================================")


if __name__ == "__main__":
    main()
