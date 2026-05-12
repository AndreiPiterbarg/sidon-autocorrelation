"""Multi-moment (MM-N) kernel sweep.

For each kernel, we run the MM-10 Phi bisection at N_max in {1, 2, 3} using
the existing ``grid_bound/bisect_mm.py`` driver.  The hypothesis: kernels
with larger k_2, k_3 (not just k_1) may benefit differently from the
multi-moment refinement than arcsine, potentially overtaking it.

Output: ``sweep_mm_results.json``.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict

from flint import arb, fmpq, ctx

from delsarte_dual.grid_bound.phi_mm import PhiMMParams
from delsarte_dual.grid_bound.bisect_mm import bisect_M_cert_mm, _fmpq_to_float, _fmpq_to_str
from delsarte_dual.grid_bound.G_min import min_G_lower_bound

from .kernels import (
    Kernel,
    ArcsineKernel,
    TriangularKernel,
    ChebyshevBetaKernel,
)
from .optimize_G import solve_qp_for_kernel


def compile_phi_mm_params_for_kernel(
    kernel: Kernel,
    coeffs: list,
    N_max: int,
    u: fmpq = fmpq(638, 1000),
    n_cells_min_G: int = 4096,
    prec_bits: int = 192,
) -> PhiMMParams:
    """Build a PhiMMParams object using kernel-specific k_n, K_2, S_1.

    Replicates ``PhiMMParams.from_mv`` but with generic kernel k_n via
    kernel.K_tilde(n) and generic K_2 via kernel.K_norm_sq().
    """
    delta = kernel.supp_halfwidth
    old = ctx.prec
    ctx.prec = prec_bits
    try:
        K2_arb = kernel.K_norm_sq(prec_bits=prec_bits)

        # k_n, n = 1..N_max
        k_list = tuple(
            kernel.K_tilde(n, prec_bits=prec_bits)
            for n in range(1, N_max + 1)
        )
        sum_kn_sq = arb(0)
        for k in k_list:
            sum_kn_sq = sum_kn_sq + k * k

        # S_1 (per-kernel weighted objective): w_j = hat_K_R(j/u)
        S1 = arb(0)
        for j, a_j in enumerate(coeffs, start=1):
            xi = arb(fmpq(j)) / arb(u)
            w_j = kernel.K_tilde_real(xi, prec_bits=prec_bits)
            if w_j.lower() <= 0:
                a_j_f = float(a_j.p) / float(a_j.q)
                if a_j_f != 0.0:
                    raise ValueError(
                        f"{kernel.name}: Bochner violated at j={j}"
                    )
                continue
            S1 = S1 + (arb(a_j) * arb(a_j)) / w_j

        # min_G via Taylor B&B
        min_G_encl, min_G_center = min_G_lower_bound(
            coeffs, u, n_cells=n_cells_min_G, prec_bits=prec_bits
        )
        min_G_cert = min_G_encl.lower()
        if min_G_cert.upper() <= 0:
            raise ValueError(
                f"{kernel.name}: min_G lower bound is non-positive: {min_G_cert}"
            )
        gain_a = (arb(4) / arb(u)) * (min_G_cert * min_G_cert) / S1

        return PhiMMParams(
            delta=delta,
            u=u,
            K2=K2_arb,
            k_arb=k_list,
            sum_kn_sq_arb=sum_kn_sq,
            gain_a=gain_a,
            min_G=min_G_cert,
            S1=S1,
            n_coeffs=len(coeffs),
            N_max=N_max,
            min_G_center=min_G_center,
        )
    finally:
        ctx.prec = old


def run_mm_single_kernel(
    kernel: Kernel,
    N_max: int,
    u: fmpq = fmpq(638, 1000),
    n_coeffs: int = 119,
    n_grid_qp: int = 3001,
    n_cells_min_G: int = 4096,
    M_lo_init: fmpq = fmpq(127, 100),
    M_hi_init: fmpq = fmpq(1276, 1000),
    tol_q: fmpq = fmpq(1, 10**3),
    max_cells_per_M: int = 300000,
    prec_bits: int = 192,
    verbose: bool = True,
) -> dict:
    t0 = time.time()
    note_parts = []

    # QP re-optimise for this kernel
    try:
        qp_res = solve_qp_for_kernel(
            kernel, n=n_coeffs, u=u, n_grid=n_grid_qp, verbose=False,
        )
    except Exception as e:
        return {"kernel": kernel.name, "N_max": N_max, "M_cert": None,
                "note": f"QP failed: {e}", "wall_time_sec": time.time() - t0}

    # Build PhiMMParams
    try:
        params = compile_phi_mm_params_for_kernel(
            kernel, qp_res.a_opt_fmpq, N_max=N_max, u=u,
            n_cells_min_G=n_cells_min_G, prec_bits=prec_bits,
        )
    except Exception as e:
        return {"kernel": kernel.name, "N_max": N_max, "M_cert": None,
                "note": f"PhiMM compile failed: {e}",
                "wall_time_sec": time.time() - t0}

    if verbose:
        print(f"  {kernel.name} N={N_max}: k_arb[0]={float(params.k_arb[0].mid()):.5f}, "
              f"K2={float(params.K2.mid()):.4f}, S1={float(params.S1.mid()):.3f}, "
              f"gain={float(params.gain_a.mid()):.5f}")

    # Bisection with MM cell search
    try:
        bound = bisect_M_cert_mm(
            params, N=N_max,
            M_lo_init=M_lo_init,
            M_hi_init=M_hi_init,
            tol_q=tol_q,
            max_cells_per_M=max_cells_per_M,
            prec_bits=prec_bits,
            verbose=False,
        )
    except Exception as e:
        return {"kernel": kernel.name, "N_max": N_max, "M_cert": None,
                "note": f"bisect_mm failed: {e}",
                "wall_time_sec": time.time() - t0}

    return {
        "kernel": kernel.name,
        "N_max": N_max,
        "M_cert": _fmpq_to_float(bound.M_cert_q),
        "M_cert_q": _fmpq_to_str(bound.M_cert_q),
        "k_n_arb_mid": [float(k.mid()) for k in params.k_arb],
        "K_norm_sq": float(params.K2.mid()),
        "S1": float(params.S1.mid()),
        "gain_a": float(params.gain_a.mid()),
        "min_G_cert": float(params.min_G.mid()),
        "note": "ok",
        "wall_time_sec": time.time() - t0,
    }


def run_mm_sweep(out_path: str = "delsarte_dual/grid_bound_alt_kernel/sweep_mm_results.json",
                 verbose: bool = True) -> list[dict]:
    # Curated subset: top kernels from v2 sweep
    kernels = [
        ArcsineKernel(delta=fmpq(138, 1000)),
        ChebyshevBetaKernel(delta=fmpq(138, 1000), beta=fmpq(45, 100), label="0.45"),
        ChebyshevBetaKernel(delta=fmpq(138, 1000), beta=fmpq(55, 100), label="0.55"),
        ChebyshevBetaKernel(delta=fmpq(138, 1000), beta=fmpq(60, 100), label="0.60"),
        TriangularKernel(delta=fmpq(138, 1000)),
    ]
    # N_max levels
    N_levels = [1, 2, 3]

    results = []
    for K in kernels:
        if verbose:
            print(f"\n### KERNEL: {K.name} ###")
        for N in N_levels:
            if verbose:
                print(f"  N_max = {N}:")
            # N=2, 3 B&B blow up memory for large N.  Tight budget for N>=2.
            max_cells = 100000 if N == 1 else (50000 if N == 2 else 30000)
            tol = fmpq(1, 10**3)
            if N == 1:
                tol = fmpq(1, 10**3)
            else:
                tol = fmpq(1, 10**3)   # still 1e-3 for speed
            r = run_mm_single_kernel(
                K, N_max=N,
                tol_q=tol,
                max_cells_per_M=max_cells,
                verbose=verbose,
            )
            results.append(r)
            if verbose:
                mc = f"{r['M_cert']:.5f}" if r.get("M_cert") is not None else "SKIPPED"
                print(f"    -> M_cert(N={N}) = {mc}  ({r['wall_time_sec']:.1f}s)"
                      f"   ({r.get('note', 'ok')[:80]})")

    body = {
        "kind": "grid_bound_alt_kernel_mm_sweep",
        "results": results,
        "summary": None,
    }
    # Best by (kernel, N) -> M_cert
    best = {"kernel": None, "N": None, "M_cert": -1.0}
    for r in results:
        if r.get("M_cert") is not None and r["M_cert"] > best["M_cert"]:
            best = {"kernel": r["kernel"], "N": r["N_max"], "M_cert": r["M_cert"]}
    body["summary"] = {
        "best": best,
        "beats_MV_1_2748": (best["M_cert"] > 1.2748),
        "breaks_1_28": (best["M_cert"] > 1.28),
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
    run_mm_sweep()
