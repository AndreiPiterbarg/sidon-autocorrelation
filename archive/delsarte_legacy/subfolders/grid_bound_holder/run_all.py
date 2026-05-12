"""Orchestrator for the Hoelder-generalised Phase 2 pipeline.

Steps:
  1. Run ``sweep_p`` at N = 1 over the default grid and write
     ``sweep_p_results.json``.
  2. Pick the best p* (largest M_cert).
  3. Run bisect at N = 2 with p = p* and the full filter panel
     (F1 + F2 + F4_MO217 + F7 + F8 + F_bathtub), tol_q = 1/10000,
     max_cells = 500000.  Emit certificate.
  4. Run ``certify_holder`` on the emitted certificate.
  5. Print a final report table summarising the sweep and the
     N = 2 certification.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from flint import fmpq, arb

from .phi_holder import PhiHolderParams
from .bisect_holder import bisect_M_cert_holder, emit_certificate_holder
from .certify_holder import verify_certificate_holder
from .sweep_p import sweep_p, DEFAULT_P_GRID, _fmpq_str, _fmpq_float


def _best_p_from_sweep(sweep_json: dict) -> fmpq:
    """Return the p in the sweep with the largest M_cert."""
    best = None
    best_M = None
    for r in sweep_json["results"]:
        m = r.get("M_cert_float")
        if m is None:
            continue
        if best_M is None or m > best_M:
            best_M = m
            best = r["p"]
    if best is None:
        raise RuntimeError("No successful p in sweep")
    pp, qq = best.split("/")
    return fmpq(int(pp), int(qq))


def run_all(
    *,
    sweep_out: str = "delsarte_dual/grid_bound_holder/sweep_p_results.json",
    cert_out_fmt: str = "delsarte_dual/grid_bound_holder/certificates/holder_N2_pstar.json",
    N_big: int = 2,
    tol_q: fmpq = fmpq(1, 10**4),
    max_cells_per_M: int = 500_000,
    prec_bits: int = 256,
    J_tail: int = 1024,
) -> dict:
    t0 = time.time()
    print("=" * 70)
    print("STEP 1 -- Hoelder exponent sweep at N = 1")
    print("=" * 70)
    sweep_res = sweep_p(
        N=1,
        p_grid=DEFAULT_P_GRID,
        tol_q=tol_q,
        max_cells_per_M=max_cells_per_M,
        J_tail=J_tail,
        prec_bits=prec_bits,
        out_path=sweep_out,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("STEP 2 -- pick best p*")
    print("=" * 70)
    p_star = _best_p_from_sweep(sweep_res)
    q_star = p_star / (p_star - fmpq(1))
    print(f"Best p* = {_fmpq_str(p_star)}, q* = {_fmpq_str(q_star)}")

    # Also print M at p=2 for comparison to Phase-2 MM
    M_at_p2 = None
    M_at_pstar = None
    for r in sweep_res["results"]:
        if r["p"] == "2/1":
            M_at_p2 = r["M_cert_float"]
        if r["p"] == _fmpq_str(p_star):
            M_at_pstar = r["M_cert_float"]

    print(f"M_cert(p=2, N=1)   = {M_at_p2}")
    print(f"M_cert(p*, N=1)    = {M_at_pstar}")
    print(f"Improvement at N=1: {(M_at_pstar - M_at_p2) if M_at_p2 else None}")

    print("\n" + "=" * 70)
    print(f"STEP 3 -- run N = {N_big} bisect at p = {p_star}")
    print("=" * 70)
    params_big = PhiHolderParams.from_mv(
        N_max=N_big, p=p_star,
        n_cells_min_G=4096,
        J_tail=J_tail,
        prec_bits=prec_bits,
    )
    bound_big = bisect_M_cert_holder(
        params_big, N=N_big,
        M_lo_init=fmpq(127, 100),
        M_hi_init=fmpq(1276, 1000),
        tol_q=tol_q,
        max_cells_per_M=max_cells_per_M,
        filter_kwargs=dict(
            enable_F4_MO217=(N_big >= 2),
            enable_F7=True, enable_F8=True,
        ),
        prec_bits=prec_bits,
        verbose=True,
    )
    print(f"N = {N_big} M_cert = {bound_big.M_cert_q} (~{_fmpq_float(bound_big.M_cert_q):.6f})")

    cert_path = cert_out_fmt
    digest = emit_certificate_holder(bound_big, cert_path)
    print(f"Certificate emitted: {cert_path}")
    print(f"SHA-256: {digest}")

    print("\n" + "=" * 70)
    print("STEP 4 -- verify certificate independently")
    print("=" * 70)
    res = verify_certificate_holder(cert_path, prec_bits=prec_bits)
    cert_ok = res.accepted
    print(f"Certificate verification: {'ACCEPTED' if cert_ok else 'REJECTED'}")

    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'p':>6}  {'q':>8}  {'M_cert (N=1)':>14}  {'cells':>8}  {'time':>6}")
    print("-" * 70)
    for r in sweep_res["results"]:
        pstr = r["p"]; qstr = r["q"]
        mstr = (f"{r['M_cert_float']:.6f}" if r["M_cert_float"] is not None else "N/A")
        cstr = (f"{r['cells_processed']}" if r["cells_processed"] is not None else "N/A")
        marker = " <-- best" if pstr == _fmpq_str(p_star) else ""
        print(f"{pstr:>6}  {qstr:>8}  {mstr:>14}  {cstr:>8}  {r['time_seconds']:>5.1f}s{marker}")
    print("-" * 70)
    print(f"Best  p* = {_fmpq_str(p_star)}  q* = {_fmpq_str(q_star)}")
    print(f"M_cert(p=2,   N=1) = {M_at_p2}")
    print(f"M_cert(p*,    N=1) = {M_at_pstar}")
    print(f"M_cert(p*,    N={N_big}) = {_fmpq_float(bound_big.M_cert_q):.6f}  "
          f"(rational {bound_big.M_cert_q})")
    print(f"Breaks 1.28? {_fmpq_float(bound_big.M_cert_q) > 1.28}")
    print(f"Certificate: {cert_path}")
    print(f"Certificate SHA-256: {digest}")
    print(f"Verifier verdict: {'ACCEPTED' if cert_ok else 'REJECTED'}")
    print(f"Total wall time: {total_time:.1f}s  ({total_time/60:.2f} min)")

    return {
        "sweep": sweep_res,
        "p_star": _fmpq_str(p_star),
        "q_star": _fmpq_str(q_star),
        "M_cert_at_p2_N1":    M_at_p2,
        "M_cert_at_pstar_N1": M_at_pstar,
        "M_cert_at_pstar_Nbig": _fmpq_float(bound_big.M_cert_q),
        "M_cert_at_pstar_Nbig_rational": _fmpq_str(bound_big.M_cert_q),
        "N_big": N_big,
        "cert_path": cert_path,
        "cert_sha256": digest,
        "cert_accepted": cert_ok,
        "total_time_seconds": total_time,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--N-big", type=int, default=2)
    parser.add_argument("--tol", type=str, default="1/10000")
    parser.add_argument("--max-cells-per-M", type=int, default=500_000)
    parser.add_argument("--prec-bits", type=int, default=256)
    parser.add_argument("--J-tail", type=int, default=1024)
    parser.add_argument("--sweep-out", type=str,
                        default="delsarte_dual/grid_bound_holder/sweep_p_results.json")
    parser.add_argument("--cert-out", type=str,
                        default="delsarte_dual/grid_bound_holder/certificates/holder_N2_pstar.json")
    args = parser.parse_args(argv)
    tol_parts = args.tol.split("/")
    tol_q = fmpq(int(tol_parts[0]), int(tol_parts[1])) if len(tol_parts) == 2 else fmpq(int(tol_parts[0]))
    return run_all(
        sweep_out=args.sweep_out,
        cert_out_fmt=args.cert_out,
        N_big=args.N_big,
        tol_q=tol_q,
        max_cells_per_M=args.max_cells_per_M,
        prec_bits=args.prec_bits,
        J_tail=args.J_tail,
    )


if __name__ == "__main__":
    main()
