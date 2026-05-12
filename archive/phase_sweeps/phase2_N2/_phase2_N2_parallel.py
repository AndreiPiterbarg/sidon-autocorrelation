"""Parallel N=2 multi-moment cell search driver.

Splits the 2N-D root box into K = 2^split_dims sub-boxes by bisecting the first
``split_dims`` coordinates, then certifies each sub-box in parallel.  If all
sub-boxes certify CERTIFIED_FORBIDDEN, the full box is certified, and we
emit a single certificate combining all terminal cells.

Soundness: the union of disjoint sub-boxes equals the full box, so per-sub-box
certification implies global certification.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.bisect_mm import (
    CertifiedBoundMM, emit_certificate_mm, _fmpq_to_float, _fmpq_to_str,
)
from delsarte_dual.grid_bound.cell_search_nd import (
    CellND, CellSearchNDResult, certify_phi_mm_negative, initial_box,
)
from delsarte_dual.grid_bound.phi_mm import PhiMMParams
from delsarte_dual.grid_bound.certify_mm import verify_certificate_mm


def split_root(M: arb, N: int, split_dims: int) -> List[CellND]:
    """Split the initial box by bisecting the first ``split_dims`` coordinates.

    Returns 2**split_dims disjoint sub-boxes whose union equals the root.
    """
    root = initial_box(M, N)
    boxes: List[CellND] = [root]
    for d in range(min(split_dims, root.dim)):
        new_boxes: List[CellND] = []
        for b in boxes:
            l, r = b.bisect(d)
            new_boxes.extend([l, r])
        boxes = new_boxes
    return boxes


def _certify_subbox(args):
    """Worker: certify one sub-box.  Returns (idx, result_dict) for pickling."""
    (idx, M_str, M_rad_str, N, sub_lo, sub_hi, params_args,
     max_cells, filter_kwargs, prec_bits) = args
    ctx.prec = prec_bits
    M = arb(M_str, M_rad_str)
    params = PhiMMParams.from_mv(**params_args)
    sub_cell = CellND(
        lo=tuple(fmpq(*[int(x) for x in s.split("/")]) for s in sub_lo),
        hi=tuple(fmpq(*[int(x) for x in s.split("/")]) for s in sub_hi),
    )
    t0 = time.time()
    res = certify_phi_mm_negative(
        M, params, N=N,
        max_cells=max_cells,
        filter_kwargs=filter_kwargs,
        prec_bits=prec_bits,
        starting_cell=sub_cell,
    )
    return {
        "idx": idx,
        "verdict": res.verdict,
        "cells_processed": res.cells_processed,
        "n_terminal": len(res.terminal_cells),
        "elapsed_s": time.time() - t0,
        "worst_live_up": (
            res.worst_live.phi_upper_float if res.worst_live else None
        ),
        # Return terminal cells in fmpq-string form for the cert
        "terminal_cells": [
            {
                "cell": r.cell.to_dict(),
                "verdict": r.verdict,
                "reason": r.reason,
                "phi_upper_float": r.phi_upper_float,
            }
            for r in res.terminal_cells
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=2)
    p.add_argument("--M", type=str, required=True, help="target M, e.g. 1.275 or 5099/4000")
    p.add_argument("--split-dims", type=int, default=4,
                   help="bisect first K coords -> 2^K sub-boxes (default 4 -> 16)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-cells-per-subbox", type=int, default=2_000_000)
    p.add_argument("--prec-bits", type=int, default=256)
    p.add_argument("--n-cells-min-G", type=int, default=4096)
    p.add_argument("--out", type=str,
                   default="delsarte_dual/grid_bound/certificates/phase2_N2_parallel.json")
    args = p.parse_args()

    ctx.prec = args.prec_bits
    if "/" in args.M:
        a, b = args.M.split("/", 1); M_q = fmpq(int(a), int(b))
        M = arb(M_q)
        M_str = f"{M_q.p}/{M_q.q}"; M_rad_str = "0"
    else:
        M = arb(args.M)
        M_q = fmpq(*[int(x) for x in M.mid().str(24).split("e")[0].replace(".", "")
                     .lstrip("-").lstrip("+").split("/")[:1] or ["0"]]) if False else None
        # simpler: parse decimal string
        from decimal import Decimal as Dec
        d = Dec(args.M); num, denom = d.as_integer_ratio()
        M_q = fmpq(num, denom)
        M_str = args.M; M_rad_str = "0"

    print(f"=== Parallel N={args.N} certification at M={float(M_q.p)/float(M_q.q):.6f} ===")
    print(f"split_dims={args.split_dims} -> {2**args.split_dims} sub-boxes; workers={args.workers}")

    boxes = split_root(M, args.N, args.split_dims)
    print(f"Created {len(boxes)} sub-boxes")

    params_args = dict(N_max=args.N, n_cells_min_G=args.n_cells_min_G,
                       prec_bits=args.prec_bits)

    filter_kwargs = dict(enable_F4_MO217=True, enable_F7=True, enable_F8=True)

    tasks = []
    for i, b in enumerate(boxes):
        sub_lo = [f"{q.p}/{q.q}" for q in b.lo]
        sub_hi = [f"{q.p}/{q.q}" for q in b.hi]
        tasks.append((i, M_str, M_rad_str, args.N, sub_lo, sub_hi, params_args,
                      args.max_cells_per_subbox, filter_kwargs, args.prec_bits))

    t_start = time.time()
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_certify_subbox, t): t[0] for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            results[r["idx"]] = r
            print(f"  box {r['idx']:2d}/{len(tasks)}: {r['verdict']:22s}  "
                  f"cells={r['cells_processed']:>9d}  terminal={r['n_terminal']:>7d}  "
                  f"({r['elapsed_s']:.1f}s)")

    elapsed = time.time() - t_start
    print(f"\nElapsed: {elapsed:.1f}s")

    n_cert = sum(1 for r in results if r["verdict"] == "CERTIFIED_FORBIDDEN")
    n_fail = sum(1 for r in results if r["verdict"] != "CERTIFIED_FORBIDDEN")
    print(f"Certified: {n_cert}/{len(results)}; Failed: {n_fail}")
    if n_fail > 0:
        print("\nNOT all sub-boxes certified — global verdict NOT_CERTIFIED.")
        for r in results:
            if r["verdict"] != "CERTIFIED_FORBIDDEN":
                print(f"  FAIL box {r['idx']}: worst_live_up = {r['worst_live_up']}")
        sys.exit(1)

    # All certified — combine into a single certificate.
    all_cells = []
    total_processed = 0
    for r in results:
        all_cells.extend(r["terminal_cells"])
        total_processed += r["cells_processed"]
    print(f"\nALL sub-boxes CERTIFIED.  Total terminal cells: {len(all_cells)}; "
          f"cells_processed sum: {total_processed}")

    # Build a CertifiedBoundMM-shaped object for emit_certificate_mm
    from delsarte_dual.grid_bound.cell_search_nd import (
        CellNDResult, CellSearchNDResult, CellND as _CellND,
    )
    flat_results = []
    for c in all_cells:
        cell = _CellND(
            lo=tuple(fmpq(*[int(x) for x in s.split("/")]) for s in c["cell"]["lo"]),
            hi=tuple(fmpq(*[int(x) for x in s.split("/")]) for s in c["cell"]["hi"]),
        )
        flat_results.append(CellNDResult(
            cell=cell,
            verdict=c["verdict"],
            reason=c["reason"],
            phi_upper_float=c["phi_upper_float"],
        ))
    cs_result = CellSearchNDResult(
        verdict="CERTIFIED_FORBIDDEN",
        cells_processed=total_processed,
        terminal_cells=flat_results,
        worst_live=None,
    )
    params = PhiMMParams.from_mv(**params_args)
    bound = CertifiedBoundMM(
        M_cert_q=M_q,
        cell_search=cs_result,
        params=params,
        N=args.N,
        filter_kwargs=filter_kwargs,
        prec_bits=args.prec_bits,
        bisection_history=[{
            "M_q": _fmpq_to_str(M_q),
            "M_float": _fmpq_to_float(M_q),
            "verdict": "CERTIFIED_FORBIDDEN",
            "cells_processed": total_processed,
            "parallel_subboxes": len(results),
        }],
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    digest = emit_certificate_mm(bound, args.out)
    print(f"\nCertificate -> {args.out}")
    print(f"SHA-256: {digest}")

    # Independent verifier
    print("\nRunning independent verifier...")
    t0 = time.time()
    res = verify_certificate_mm(args.out)
    print(f"  verified in {time.time()-t0:.1f}s; accepted={res.accepted}")
    if not res.accepted:
        sys.exit(1)


if __name__ == "__main__":
    main()
