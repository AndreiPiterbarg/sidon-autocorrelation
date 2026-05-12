"""Parameter sweep over (delta, n, N, MO) for the forbidden-region bound.

Part B.6.  Writes JSON to ``delsarte_dual/results/sweep_YYYYMMDD_HHMMSS.json``
plus a ``delsarte_dual/certificates/<cert_hash>.json`` for the best result.

Usage:
    python -m delsarte_dual.sweep [--quick]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mpmath as mp
from mpmath import mpf

from .mv_bound import (
    MVMultiMomentBound,
    MVMultiMomentBoundWithMO,
    MV_COEFFS_119,
    MV_K2_BOUND_OVER_DELTA,
)
from .forbidden_region import (
    certified_forbidden_max,
    round_to_rational_certificate,
    CertifiedBisectionResult,
)


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CERTS_DIR = HERE / "certificates"


def _ensure_dirs():
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    CERTS_DIR.mkdir(exist_ok=True, parents=True)


def _cert_to_json(cert) -> Dict[str, Any]:
    return {
        "M_cert_p": str(cert.M_cert_fmpq.p),
        "M_cert_q": str(cert.M_cert_fmpq.q),
        "M_cert_float": cert.extras["M_cert_float"],
        "target_lo_p": str(cert.target_lo_fmpq.p),
        "target_lo_q": str(cert.target_lo_fmpq.q),
        "rhs_upper_p": str(cert.rhs_upper_fmpq.p),
        "rhs_upper_q": str(cert.rhs_upper_fmpq.q),
        "extras": cert.extras,
    }


def _cert_hash(cert_json: Dict[str, Any]) -> str:
    buf = json.dumps(cert_json, sort_keys=True).encode()
    return hashlib.sha256(buf).hexdigest()[:16]


def _run_single(
    delta, u, N, use_mo, mo_strong, G_coeffs, K2_bod,
    tol=mpf("1e-9"), max_iter=120,
) -> Tuple[CertifiedBisectionResult, Dict[str, Any]]:
    if use_mo and N >= 2 and mo_strong:
        bound = MVMultiMomentBoundWithMO(
            delta=delta, u=u, G_coeffs=G_coeffs,
            K2_bound_over_delta=K2_bod, N=N, mo_strong=True,
        )
    else:
        bound = MVMultiMomentBound(
            delta=delta, u=u, G_coeffs=G_coeffs,
            K2_bound_over_delta=K2_bod, N=N,
        )
    res = certified_forbidden_max(
        bound, use_mo=use_mo, mo_strong=mo_strong,
        M_lo=mpf("1.0001"), M_hi=mpf("1.45"),
        tol=tol, max_iter=max_iter,
    )
    meta = {
        "delta": str(delta),
        "u": str(u),
        "N": N,
        "use_mo": bool(use_mo),
        "mo_strong": bool(mo_strong),
        "n_G": len(G_coeffs),
        "gain_a": float(bound.a_gain),
        "min_G": float(bound.min_G),
        "K2": float(bound.K2),
        "M_cert": float(res.M_cert),
        "M_upper": float(res.M_upper),
        "target": float(res.target),
        "n_iterations": res.n_iterations,
    }
    return res, meta


def sweep(quick: bool = False) -> Dict[str, Any]:
    mp.mp.dps = 30
    _ensure_dirs()

    if quick:
        deltas = [mpf("0.138")]
        u_vals = [mpf("0.638")]
        Ns = [1, 2, 3]
        mo_vals = [False, True]
    else:
        deltas = [mpf(str(d)) for d in (0.10, 0.138, 0.18, 0.22)]
        u_vals = [mpf("0.638")]
        Ns = [1, 2, 3, 5]
        mo_vals = [False, True]

    all_rows: List[Dict[str, Any]] = []
    best_row: Dict[str, Any] = None
    best_res: CertifiedBisectionResult = None

    for delta in deltas:
        # Rebuild the G coefficients at MV's 119-term approximation (delta-agnostic
        # but they are optimised only for delta=0.138; far from 0.138 the gain a
        # drops sharply).
        G = list(MV_COEFFS_119)
        for u in u_vals:
            for N in Ns:
                for use_mo in mo_vals:
                    if use_mo and N < 2:
                        continue  # MO needs >=2 moments
                    try:
                        res, meta = _run_single(
                            delta=delta, u=u, N=N, use_mo=use_mo,
                            mo_strong=True, G_coeffs=G,
                            K2_bod=MV_K2_BOUND_OVER_DELTA,
                        )
                    except Exception as e:
                        meta = {"error": str(e), "delta": str(delta), "u": str(u), "N": N, "use_mo": use_mo}
                        all_rows.append(meta)
                        continue
                    all_rows.append(meta)
                    print(f"  delta={float(delta):.3f} u={float(u):.3f} N={N} MO={use_mo} "
                          f"=> M_cert={meta['M_cert']:.6f}  (gain_a={meta['gain_a']:.4e})")
                    if best_row is None or meta["M_cert"] > best_row["M_cert"]:
                        best_row = meta
                        best_res = res

    # Certify the best
    best_cert = round_to_rational_certificate(best_res, denom_bits=60) if best_res else None
    cert_json = _cert_to_json(best_cert) if best_cert else None
    cert_hash = _cert_hash(cert_json) if cert_json else "NONE"

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "timestamp": stamp,
        "quick": bool(quick),
        "best": {
            "M_cert_float": best_row["M_cert"] if best_row else None,
            "M_cert_rational": f"{best_cert.M_cert_fmpq.p}/{best_cert.M_cert_fmpq.q}" if best_cert else None,
            "params": {k: v for k, v in (best_row or {}).items() if k != "M_cert"},
            "cert_hash": cert_hash,
        },
        "all": all_rows,
    }

    out_path = RESULTS_DIR / f"sweep_{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")

    if cert_json is not None:
        cpath = CERTS_DIR / f"{cert_hash}.json"
        cpath.write_text(json.dumps(cert_json, indent=2))
        print(f"Wrote {cpath}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="short sweep")
    args = parser.parse_args()
    sweep(quick=args.quick)
