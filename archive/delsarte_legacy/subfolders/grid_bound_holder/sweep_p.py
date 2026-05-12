"""Exponent-sweep driver for the Hoelder-generalised Phase 2 pipeline.

For each p in a rational grid, compile PhiHolderParams(N=1, p), run
bisect_M_cert_holder with tol_q = 1/10000, and record:

  * M_cert (rational + float)
  * time
  * #cells processed at M_cert
  * worst terminal phi_upper (float)

Output JSON: ``delsarte_dual/grid_bound_holder/sweep_p_results.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

from flint import arb, fmpq, ctx

from .phi_holder import PhiHolderParams
from .bisect_holder import bisect_M_cert_holder


def _fmpq_str(q: fmpq) -> str:
    return f"{q.p}/{q.q}"


def _fmpq_float(q: fmpq) -> float:
    return float(q.p) / float(q.q)


# Default rational grid per the task spec; p >= 2, q = p/(p - 1)
DEFAULT_P_GRID = [
    fmpq(2, 1),
    fmpq(9, 4),
    fmpq(5, 2),
    fmpq(11, 4),
    fmpq(3, 1),
    fmpq(7, 2),
    fmpq(4, 1),
    fmpq(5, 1),
    fmpq(6, 1),
    fmpq(10, 1),
]


def _sweep_single_p(args: tuple) -> dict:
    """Worker: run bisection for one p and return the result dict."""
    (p_str, N, tol_q_str, M_lo_str, M_hi_str, max_cells_per_M,
     J_tail, n_cells_min_G, prec_bits, filter_kwargs) = args
    from flint import fmpq as _fmpq, ctx as _ctx
    _ctx.prec = prec_bits
    pn, pd = p_str.split("/")
    p = _fmpq(int(pn), int(pd))
    q = p / (p - _fmpq(1))
    tn, td = tol_q_str.split("/")
    tol_q_ = _fmpq(int(tn), int(td))
    ln, ld = M_lo_str.split("/")
    M_lo_ = _fmpq(int(ln), int(ld))
    hn, hd = M_hi_str.split("/")
    M_hi_ = _fmpq(int(hn), int(hd))

    t0 = time.time()
    try:
        params = PhiHolderParams.from_mv(
            N_max=N, p=p,
            n_cells_min_G=n_cells_min_G,
            J_tail=J_tail,
            prec_bits=prec_bits,
        )
        bound = bisect_M_cert_holder(
            params, N=N,
            M_lo_init=M_lo_,
            M_hi_init=M_hi_,
            tol_q=tol_q_,
            max_cells_per_M=max_cells_per_M,
            filter_kwargs=filter_kwargs,
            prec_bits=prec_bits,
            verbose=False,
        )
        M_cert_q = bound.M_cert_q
        worst_phi = max(
            (r.phi_upper_float for r in bound.cell_search.terminal_cells
             if r.phi_upper_float is not None),
            default=None,
        )
        status = "OK"
        result = {
            "p": _fmpq_str(p), "q": _fmpq_str(q),
            "p_float": _fmpq_float(p), "q_float": _fmpq_float(q),
            "status": status,
            "M_cert_q": _fmpq_str(M_cert_q),
            "M_cert_float": _fmpq_float(M_cert_q),
            "cells_processed": bound.cell_search.cells_processed,
            "worst_terminal_phi_upper": worst_phi,
            "time_seconds": time.time() - t0,
            "K_q_upper_float": float(params.K_q_upper.upper()),
        }
    except Exception as e:
        result = {
            "p": _fmpq_str(p), "q": _fmpq_str(q),
            "p_float": _fmpq_float(p), "q_float": _fmpq_float(q),
            "status": f"ERROR: {e!r}",
            "M_cert_q": None, "M_cert_float": None,
            "cells_processed": None, "worst_terminal_phi_upper": None,
            "time_seconds": time.time() - t0,
            "K_q_upper_float": None,
        }
    return result


def sweep_p(
    *,
    N: int = 1,
    p_grid: list[fmpq] | None = None,
    tol_q: fmpq = fmpq(1, 10**4),
    M_lo_init: fmpq = fmpq(127, 100),
    M_hi_init: fmpq = fmpq(1276, 1000),
    max_cells_per_M: int = 500_000,
    J_tail: int = 1024,
    n_cells_min_G: int = 4096,
    prec_bits: int = 256,
    filter_kwargs: dict | None = None,
    out_path: str | None = None,
    verbose: bool = True,
    n_workers: int = 1,
) -> dict:
    """Run the Hoelder exponent sweep."""
    if p_grid is None:
        p_grid = DEFAULT_P_GRID
    if filter_kwargs is None:
        if N == 1:
            filter_kwargs = dict(enable_F4_MO217=False, enable_F7=True, enable_F8=True)
        else:
            filter_kwargs = dict(enable_F4_MO217=True, enable_F7=True, enable_F8=True)

    # Parallel worker path (used only when n_workers > 1)
    if n_workers > 1:
        import multiprocessing as _mp
        work_args = [
            (_fmpq_str(p), N, _fmpq_str(tol_q),
             _fmpq_str(M_lo_init), _fmpq_str(M_hi_init),
             max_cells_per_M, J_tail, n_cells_min_G, prec_bits,
             filter_kwargs)
            for p in p_grid
        ]
        if verbose:
            print(f"[sweep_p] parallel sweep with n_workers={n_workers}, "
                  f"n_p={len(p_grid)}")
        # Use 'spawn' for safety on flint (which stores some process state)
        ctx_mp = _mp.get_context("spawn")
        with ctx_mp.Pool(n_workers) as pool:
            results_ord = pool.map(_sweep_single_p, work_args)
        # Preserve p_grid order
        results = results_ord
        if verbose:
            for r in results:
                if r["M_cert_float"] is not None:
                    print(f"  p={r['p']:>6} q={r['q']:>8} "
                          f"M_cert={r['M_cert_float']:.6f} "
                          f"cells={r['cells_processed']} "
                          f"time={r['time_seconds']:.1f}s")
                else:
                    print(f"  p={r['p']} {r['status']} ({r['time_seconds']:.1f}s)")
        out = {
            "sweep": "grid_bound_holder.sweep_p",
            "N": N, "tol_q": _fmpq_str(tol_q), "prec_bits": prec_bits,
            "J_tail": J_tail, "filter_kwargs": filter_kwargs,
            "n_workers": n_workers,
            "M_lo_init": _fmpq_str(M_lo_init), "M_hi_init": _fmpq_str(M_hi_init),
            "results": results,
        }
        if out_path:
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            if verbose:
                print(f"Wrote {out_path}")
        return out

    # Serial path (original)
    results = []
    for p in p_grid:
        q = p / (p - fmpq(1))
        t0 = time.time()
        if verbose:
            print("-" * 70)
            print(f"[sweep_p] p = {_fmpq_str(p)} ({_fmpq_float(p):.4f}), "
                  f"q = {_fmpq_str(q)} ({_fmpq_float(q):.4f}), N = {N}")
        params = PhiHolderParams.from_mv(
            N_max=N, p=p,
            n_cells_min_G=n_cells_min_G,
            J_tail=J_tail,
            prec_bits=prec_bits,
        )
        try:
            bound = bisect_M_cert_holder(
                params, N=N,
                M_lo_init=M_lo_init,
                M_hi_init=M_hi_init,
                tol_q=tol_q,
                max_cells_per_M=max_cells_per_M,
                filter_kwargs=filter_kwargs,
                prec_bits=prec_bits,
                verbose=False,
            )
            M_cert_q = bound.M_cert_q
            M_cert_f = _fmpq_float(M_cert_q)
            cells = bound.cell_search.cells_processed
            worst_phi = max(
                (r.phi_upper_float for r in bound.cell_search.terminal_cells
                 if r.phi_upper_float is not None),
                default=None,
            )
            status = "OK"
        except Exception as e:
            M_cert_q = None
            M_cert_f = None
            cells = None
            worst_phi = None
            status = f"ERROR: {e!r}"
            bound = None
        elapsed = time.time() - t0
        if verbose:
            if M_cert_q is not None:
                print(f"  M_cert = {_fmpq_str(M_cert_q)} (~{M_cert_f:.6f}), "
                      f"cells = {cells}, time = {elapsed:.2f}s, "
                      f"worst_phi_upper = {worst_phi}")
            else:
                print(f"  {status}, time = {elapsed:.2f}s")
        results.append({
            "p": _fmpq_str(p),
            "q": _fmpq_str(q),
            "p_float": _fmpq_float(p),
            "q_float": _fmpq_float(q),
            "status": status,
            "M_cert_q": _fmpq_str(M_cert_q) if M_cert_q is not None else None,
            "M_cert_float": M_cert_f,
            "cells_processed": cells,
            "worst_terminal_phi_upper": worst_phi,
            "time_seconds": elapsed,
            "K_q_upper_float": (
                float(params.K_q_upper.upper())
                if hasattr(params, "K_q_upper") else None
            ),
        })

    out = {
        "sweep": "grid_bound_holder.sweep_p",
        "N": N,
        "tol_q": _fmpq_str(tol_q),
        "prec_bits": prec_bits,
        "J_tail": J_tail,
        "filter_kwargs": filter_kwargs,
        "M_lo_init": _fmpq_str(M_lo_init),
        "M_hi_init": _fmpq_str(M_hi_init),
        "results": results,
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        if verbose:
            print(f"Wrote {out_path}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--tol", type=str, default="1/10000")
    parser.add_argument("--max-cells-per-M", type=int, default=500_000)
    parser.add_argument("--J-tail", type=int, default=1024)
    parser.add_argument("--prec-bits", type=int, default=256)
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers across p_grid (1=serial)")
    parser.add_argument("--out", type=str,
                        default="delsarte_dual/grid_bound_holder/sweep_p_results.json")
    args = parser.parse_args(argv)
    tol_parts = args.tol.split("/")
    tol_q = fmpq(int(tol_parts[0]), int(tol_parts[1])) if len(tol_parts) == 2 else fmpq(int(tol_parts[0]))
    res = sweep_p(
        N=args.N,
        tol_q=tol_q,
        max_cells_per_M=args.max_cells_per_M,
        J_tail=args.J_tail,
        prec_bits=args.prec_bits,
        n_workers=args.n_workers,
        out_path=args.out,
        verbose=True,
    )
    # Pretty-print table
    print("")
    print("=" * 70)
    print(f"{'p':>6}  {'q':>8}  {'M_cert':>10}  {'cells':>8}  {'time':>7}")
    print("-" * 70)
    for r in res["results"]:
        pstr = r["p"]; qstr = r["q"]
        mstr = (f"{r['M_cert_float']:.6f}" if r["M_cert_float"] is not None else "N/A")
        cstr = (f"{r['cells_processed']}" if r["cells_processed"] is not None else "N/A")
        print(f"{pstr:>6}  {qstr:>8}  {mstr:>10}  {cstr:>8}  {r['time_seconds']:>6.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
