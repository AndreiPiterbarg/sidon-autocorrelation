"""CPU-pod entry point for the Hoelder Phase-2 pipeline.

Invokes ``run_all`` with multiprocessing enabled across the p_grid.
Script form (not a module) so ``cpupod launch`` can drive it directly:

    cpupod launch delsarte_dual/grid_bound_holder/pod_run.py \\
                  --n-workers 10 --N-big 2 --tol 1/10000

Outputs:
    delsarte_dual/grid_bound_holder/sweep_p_results.json
    delsarte_dual/grid_bound_holder/certificates/holder_N2_pstar.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def main(argv=None):
    # Ensure the project root is on sys.path so relative imports work.
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=10,
                        help="Parallel workers across p_grid")
    parser.add_argument("--N-big", type=int, default=2,
                        help="Moment order for the best-p follow-up cert")
    parser.add_argument("--tol", type=str, default="1/10000")
    parser.add_argument("--max-cells-per-M", type=int, default=500_000)
    parser.add_argument("--max-cells-per-M-big", type=int, default=2_000_000,
                        help="Cell budget per M for the N=N_big cert (higher than sweep)")
    parser.add_argument("--M-lo-big", type=str, default="1273/1000")
    parser.add_argument("--M-hi-big", type=str, default="1276/1000")
    # Sweep bracket: wide enough that any p in the grid can certify.
    # Critical-point probes show M_cert(p=10) ~ 1.05, so M_lo = 1.10 is safe
    # for all p in {2, 9/4, ..., 6, 10}.
    parser.add_argument("--M-lo-sweep", type=str, default="11/10")
    parser.add_argument("--M-hi-sweep", type=str, default="1276/1000")
    parser.add_argument("--J-tail", type=int, default=1024)
    parser.add_argument("--prec-bits", type=int, default=256)
    parser.add_argument("--sweep-out", type=str,
                        default="delsarte_dual/grid_bound_holder/sweep_p_results.json")
    parser.add_argument("--cert-out", type=str,
                        default="delsarte_dual/grid_bound_holder/certificates/holder_N2_pstar.json")
    args = parser.parse_args(argv)

    from flint import fmpq
    from delsarte_dual.grid_bound_holder.sweep_p import (
        sweep_p, DEFAULT_P_GRID, _fmpq_str, _fmpq_float,
    )
    from delsarte_dual.grid_bound_holder.phi_holder import PhiHolderParams
    from delsarte_dual.grid_bound_holder.bisect_holder import (
        bisect_M_cert_holder, emit_certificate_holder,
    )
    from delsarte_dual.grid_bound_holder.certify_holder import verify_certificate_holder

    tn, td = args.tol.split("/")
    tol_q = fmpq(int(tn), int(td)) if "/" in args.tol else fmpq(int(args.tol))

    t_all = time.time()

    print("=" * 70)
    print(f"STEP 1: parallel Hoelder sweep (n_workers={args.n_workers}, N=1)")
    print("=" * 70, flush=True)
    # Parse sweep bracket
    _lsn, _lsd = args.M_lo_sweep.split("/")
    _hsn, _hsd = args.M_hi_sweep.split("/")
    M_lo_sweep = fmpq(int(_lsn), int(_lsd))
    M_hi_sweep = fmpq(int(_hsn), int(_hsd))
    sweep_res = sweep_p(
        N=1, p_grid=DEFAULT_P_GRID,
        tol_q=tol_q,
        M_lo_init=M_lo_sweep,
        M_hi_init=M_hi_sweep,
        max_cells_per_M=args.max_cells_per_M,
        J_tail=args.J_tail,
        prec_bits=args.prec_bits,
        n_workers=args.n_workers,
        out_path=args.sweep_out,
        verbose=True,
    )

    # Pick best p
    best_p_str = None; best_M = None
    for r in sweep_res["results"]:
        m = r.get("M_cert_float")
        if m is None: continue
        if best_M is None or m > best_M:
            best_M = m; best_p_str = r["p"]
    if best_p_str is None:
        print("FATAL: no successful p in sweep; aborting.")
        sys.exit(1)
    pn, pd = best_p_str.split("/")
    p_star = fmpq(int(pn), int(pd))
    q_star = p_star / (p_star - fmpq(1))
    print()
    print(f"Best p* = {best_p_str} (M_cert(p*, N=1) = {best_M:.6f})")

    # p=2 reference
    M_at_p2 = None
    for r in sweep_res["results"]:
        if r["p"] == "2/1":
            M_at_p2 = r["M_cert_float"]
    print(f"M_cert(p=2, N=1) = {M_at_p2}")

    print()
    print("=" * 70)
    print(f"STEP 2: N = {args.N_big} bisect at p = {best_p_str}")
    print("=" * 70, flush=True)
    params_big = PhiHolderParams.from_mv(
        N_max=args.N_big, p=p_star,
        n_cells_min_G=4096,
        J_tail=args.J_tail,
        prec_bits=args.prec_bits,
    )
    # Parse big-step bracket
    ln, ld = args.M_lo_big.split("/")
    hn, hd = args.M_hi_big.split("/")
    M_lo_big = fmpq(int(ln), int(ld))
    M_hi_big = fmpq(int(hn), int(hd))
    bound_big = bisect_M_cert_holder(
        params_big, N=args.N_big,
        M_lo_init=M_lo_big,
        M_hi_init=M_hi_big,
        tol_q=tol_q,
        max_cells_per_M=args.max_cells_per_M_big,
        filter_kwargs=dict(
            enable_F4_MO217=(args.N_big >= 2),
            enable_F7=True, enable_F8=True,
        ),
        prec_bits=args.prec_bits,
        verbose=True,
    )
    M_big = _fmpq_float(bound_big.M_cert_q)
    print(f"N = {args.N_big} M_cert = {bound_big.M_cert_q} (~{M_big:.6f})")
    digest = emit_certificate_holder(bound_big, args.cert_out)
    print(f"Certificate: {args.cert_out}")
    print(f"SHA-256: {digest}")

    print()
    print("=" * 70)
    print("STEP 3: independent verifier")
    print("=" * 70, flush=True)
    res = verify_certificate_holder(args.cert_out, prec_bits=args.prec_bits)

    total_time = time.time() - t_all
    print()
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"{'p':>6} {'q':>8} {'M_cert (N=1)':>14} {'cells':>8} {'time':>7}")
    print("-" * 70)
    for r in sweep_res["results"]:
        mstr = f"{r['M_cert_float']:.6f}" if r["M_cert_float"] is not None else "N/A"
        cstr = str(r["cells_processed"]) if r["cells_processed"] is not None else "N/A"
        marker = " <-- best" if r["p"] == best_p_str else ""
        print(f"{r['p']:>6} {r['q']:>8} {mstr:>14} {cstr:>8} {r['time_seconds']:>6.1f}s{marker}")
    print("-" * 70)
    print(f"Best (p*, q*) = ({best_p_str}, {_fmpq_str(q_star)})")
    print(f"M_cert(p=2,  N=1) = {M_at_p2}")
    print(f"M_cert(p*,   N=1) = {best_M:.6f}")
    print(f"M_cert(p*,   N={args.N_big}) = {M_big:.6f}  (rational {bound_big.M_cert_q})")
    print(f"Breaks 1.28? {'YES' if M_big > 1.28 else 'NO'}")
    print(f"Certificate: {args.cert_out}")
    print(f"Cert SHA-256: {digest}")
    print(f"Verifier verdict: {'ACCEPTED' if res.accepted else 'REJECTED'}")
    print(f"Total wall time: {total_time:.1f}s ({total_time/60:.2f} min)")


if __name__ == "__main__":
    main()
