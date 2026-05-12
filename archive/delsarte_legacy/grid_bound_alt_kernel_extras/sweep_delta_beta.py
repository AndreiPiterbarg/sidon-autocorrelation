"""Sweep (delta, beta) jointly for the Chebyshev-beta auto-conv family.

For each delta in a scan and each beta, re-solve the kernel-specific QP and
run the N=1 Phi bisection.  u = 1/2 + delta follows MV's convention.

Output: ``sweep_delta_beta_results.json``.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict

from flint import fmpq

from .kernels import ChebyshevBetaKernel, ArcsineKernel
from .bisect_alt_kernel import run_single_kernel


def delta_u_pair(delta_num: int, delta_den: int) -> tuple[fmpq, fmpq]:
    """delta = delta_num/delta_den; u = 1/2 + delta  (rational)."""
    delta = fmpq(delta_num, delta_den)
    u = fmpq(1, 2) + delta
    return delta, u


def run_delta_beta_sweep(
    deltas: list[fmpq],
    betas: list[tuple[fmpq, str]],
    out_path: str = "delsarte_dual/grid_bound_alt_kernel/sweep_delta_beta_results.json",
    verbose: bool = True,
    quick: bool = True,
) -> list[dict]:
    """Full grid of (delta, beta) evaluations."""
    results = []
    for delta in deltas:
        u = fmpq(1, 2) + delta
        delta_f = float(delta.p) / float(delta.q)
        u_f = float(u.p) / float(u.q)
        if verbose:
            print(f"\n### delta = {delta_f:.4f}  u = {u_f:.4f} ###")

        for beta, lab in betas:
            K = ChebyshevBetaKernel(delta=delta, beta=beta, label=lab)
            t0 = time.time()
            try:
                res = run_single_kernel(
                    K,
                    u=u,
                    n_coeffs=119,
                    n_grid_qp=3001 if quick else 5001,
                    n_cells_min_G=2048 if quick else 4096,
                    M_lo_init=fmpq(120, 100),
                    M_hi_init=fmpq(1276, 1000),
                    tol_q=fmpq(1, 10**3) if quick else fmpq(1, 10**4),
                    max_cells_per_M=10000 if quick else 30000,
                    initial_splits=32,
                    bochner_max=40,
                    prec_bits=128,
                    verbose=False,
                )
            except Exception as e:
                res = None
                if verbose:
                    print(f"  beta={lab}  EXC: {e}")
                results.append({
                    "delta": f"{delta.p}/{delta.q}",
                    "delta_float": delta_f,
                    "u": f"{u.p}/{u.q}",
                    "u_float": u_f,
                    "beta": f"{beta.p}/{beta.q}",
                    "beta_float": float(beta.p) / float(beta.q),
                    "M_cert": None,
                    "k1": None, "K_norm_sq": None, "S1": None, "gain_a": None,
                    "note": f"exception: {str(e)[:120]}",
                    "wall_time_sec": time.time() - t0,
                })
                continue

            entry = {
                "delta": f"{delta.p}/{delta.q}",
                "delta_float": delta_f,
                "u": f"{u.p}/{u.q}",
                "u_float": u_f,
                "beta": f"{beta.p}/{beta.q}",
                "beta_float": float(beta.p) / float(beta.q),
                "M_cert": res.M_cert,
                "k1": res.tilde_K_1,
                "K_norm_sq": res.K_norm_sq,
                "S1": res.S1,
                "gain_a": res.gain_a,
                "note": res.note,
                "wall_time_sec": res.wall_time_sec,
            }
            results.append(entry)
            if verbose:
                mc = f"{res.M_cert:.5f}" if res.M_cert is not None else "SKIPPED"
                print(f"  beta={lab:5s}  M_cert={mc:>10s}  "
                      f"k1={res.tilde_K_1:.5f} K2={res.K_norm_sq:.3f} "
                      f"({res.wall_time_sec:.1f}s)")

    # Find best
    best = None
    for r in results:
        if r["M_cert"] is None:
            continue
        if best is None or r["M_cert"] > best["M_cert"]:
            best = r

    body = {
        "kind": "grid_bound_alt_kernel_delta_beta_sweep",
        "results": results,
        "summary": {
            "best": best,
            "beats_MV_1_2748": (best is not None and best["M_cert"] > 1.2748),
            "breaks_1_28": (best is not None and best["M_cert"] > 1.28),
        },
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    body_json = json.dumps(body, indent=2, sort_keys=True, default=str)
    digest = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    final = {"sha256_of_body": digest, "body": body}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, sort_keys=True, default=str)
    if verbose:
        print(f"\nResults: {out_path}")
        print(f"SHA-256: {digest}")
        print(f"BEST: {best}")
    return results


if __name__ == "__main__":
    # delta from 0.10 to 0.17 in steps of 0.005 (gives 15 values);
    # conserve compute: use 8 values coarser.
    deltas = [
        fmpq(10, 100), fmpq(11, 100), fmpq(12, 100),
        fmpq(128, 1000), fmpq(138, 1000), fmpq(148, 1000),
        fmpq(16, 100), fmpq(17, 100),
    ]
    betas = [
        (fmpq(45, 100),   "0.45"),
        (fmpq(48, 100),   "0.48"),
        (fmpq(1, 2),      "0.50"),
        (fmpq(52, 100),   "0.52"),
        (fmpq(55, 100),   "0.55"),
    ]
    run_delta_beta_sweep(deltas, betas, verbose=True, quick=True)
