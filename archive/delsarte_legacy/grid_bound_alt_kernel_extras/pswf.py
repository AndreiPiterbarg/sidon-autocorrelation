"""K12: Prolate Spheroidal Wave Function (PSWF) auto-convolution kernel.

K = phi * phi where phi(x) = c_0 * psi_0^c(2x/delta) is the leading PSWF
psi_0^c rescaled to [-delta/2, delta/2].

PSWF psi_n^c is the eigenfunction of the prolate operator
    L_c psi  =  -d/dx[(1-x^2) d/dx psi]  +  c^2 x^2 psi  =  chi_n(c) psi
on [-1, 1], with chi_0 < chi_1 < .... psi_0 is positive on (-1, 1) and
even, so phi >= 0, K = phi * phi >= 0, and K_hat = phi_hat^2 >= 0
(Bochner).  Support of K is [-delta, delta] by phi-support.

Bandwidth parameter c is the only family parameter.  c -> 0: psi_0
constant -> phi uniform -> K = triangle (= K2).  c -> oo: psi_0
concentrated at 0 -> phi narrow -> K narrower / faster Fourier decay.

This module is the only "genuinely untested family" identified by the
2026-04-28 audit in poss_ideas.md (after the 31-kernel + 2D (delta, beta)
+ 9-D Chebyshev DE all left arcsine as strict local maximum, and after
the 3-point Lasserre SDP died structurally).

Implementation
--------------
1. Bouwkamp tridiagonal in normalized Legendre basis (even parity only):
       \\bar P_k(u) = sqrt((2k+1)/2) P_k(u),  k = 0, 2, 4, ...
   Matrix entries (symmetric):
       M[j, j]   = k(k+1) + c^2 * B_k
       M[j, j+1] = c^2 * A_k * sqrt((2k+1)/(2k+5))
   with k = 2j and
       A_k = (k+1)(k+2) / ((2k+1)(2k+3))
       B_k = (2k^2 + 2k - 1) / ((2k-1)(2k+3))   (=> 1/3 at k=0).
2. Smallest eigenvalue chi_0(c), eigenvector beta gives
       psi_0(u) = sum_j beta[j] * \\bar P_{2j}(u).
3. Pipeline integration (matches variational_pod.compute_quantities):
   theta-grid x = (delta/2) sin(theta),  integrand_theta = psi_0(sin theta) * cos(theta).
   All downstream (k_1, K_2, S_1, M_cert) is identical to the existing
   variational pipeline.

Run:
    python -m delsarte_dual.grid_bound_alt_kernel.pswf  --include-fine
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq
from scipy.special import eval_legendre


# ---- Match variational_pod.py constants exactly so M_cert is comparable ----
DELTA = 0.138
U = 0.5 + DELTA
N_QP = 119
N_THETA = 2001
N_XI = 8001


def _mv_coeffs_float():
    from delsarte_dual.grid_bound.coeffs import mv_coeffs_fmpq
    return np.array([float(c.p) / float(c.q) for c in mv_coeffs_fmpq()])


MV_COEFFS = _mv_coeffs_float()

_THETA = np.linspace(-np.pi / 2, np.pi / 2, N_THETA)
_DTHETA = _THETA[1] - _THETA[0]
_SIN_T = np.sin(_THETA)
_COS_T = np.cos(_THETA)
_X_OF_T = (DELTA / 2.0) * _SIN_T

_KN_J = np.arange(1, 6)
_KN_COS = np.cos(2 * np.pi * _KN_J[:, None] * _X_OF_T[None, :])
_QP_XI = np.arange(1, N_QP + 1) / U
_QP_COS = np.cos(2 * np.pi * _QP_XI[:, None] * _X_OF_T[None, :])

_XI = np.linspace(0.0, 100.0 / DELTA, N_XI)
_DXI = _XI[1] - _XI[0]


# -----------------------------------------------------------------------------
# PSWF eigensolver
# -----------------------------------------------------------------------------

def pswf_psi0_legendre_coeffs(c: float, n_basis: int = 120):
    """Leading PSWF psi_0^c expansion coefficients in normalized Legendre basis.

    Returns
    -------
    beta : (n_basis,) array
        psi_0(u) = sum_j beta[j] * \\bar P_{2j}(u).  L^2-normalized
        (sum beta_j^2 = 1).  Sign chosen so beta[0] > 0  ->  psi_0(0) > 0.
    chi_0 : float
        Smallest eigenvalue of the prolate operator.
    """
    if c < 0:
        raise ValueError("PSWF bandwidth c must be >= 0")
    if n_basis < 2:
        raise ValueError("n_basis must be >= 2")
    diag = np.empty(n_basis, dtype=float)
    off = np.empty(n_basis - 1, dtype=float)
    cc = float(c) * float(c)
    for j in range(n_basis):
        k = 2 * j
        # B_k = (2k^2 + 2k - 1)/((2k-1)(2k+3));  evaluates to 1/3 at k=0.
        num_b = 2.0 * k * k + 2.0 * k - 1.0
        den_b = (2.0 * k - 1.0) * (2.0 * k + 3.0)
        B_k = num_b / den_b
        diag[j] = k * (k + 1) + cc * B_k
        if j < n_basis - 1:
            A_k = (k + 1.0) * (k + 2.0) / ((2.0 * k + 1.0) * (2.0 * k + 3.0))
            off[j] = cc * A_k * np.sqrt((2.0 * k + 1.0) / (2.0 * k + 5.0))
    chi, vec = eigh_tridiagonal(diag, off, select='i', select_range=(0, 0))
    beta = np.ascontiguousarray(vec[:, 0], dtype=float)
    if beta[0] < 0:
        beta = -beta
    return beta, float(chi[0])


def pswf_psi0_evaluate(beta_even, u):
    """Evaluate psi_0(u) = sum_j beta_even[j] * sqrt((4j+1)/2) * P_{2j}(u)."""
    u = np.asarray(u, dtype=float)
    n = len(beta_even)
    out = np.zeros_like(u)
    for j in range(n):
        k = 2 * j
        norm = np.sqrt(0.5 * (2 * k + 1))
        out = out + beta_even[j] * norm * eval_legendre(k, u)
    return out


# -----------------------------------------------------------------------------
# MV pipeline integration (mirror of variational_pod.compute_quantities)
# -----------------------------------------------------------------------------

def compute_quantities_pswf(c: float, n_basis: int = 120):
    """Compute M_cert via the MV master inequality with PSWF phi."""
    beta, chi = pswf_psi0_legendre_coeffs(c, n_basis=n_basis)
    psi0_sin = pswf_psi0_evaluate(beta, _SIN_T)

    # phi >= 0 required for K = phi*phi >= 0 (auto-conv admissibility).
    if np.min(psi0_sin) < -1e-9:
        return {
            "c": float(c),
            "n_basis": int(n_basis),
            "chi_0": chi,
            "M_cert": None,
            "reason": f"psi_0 negative; min={float(np.min(psi0_sin))!r}",
        }

    integrand_theta = psi0_sin * _COS_T
    if np.any(integrand_theta < -1e-9):
        return {
            "c": float(c),
            "n_basis": int(n_basis),
            "chi_0": chi,
            "M_cert": None,
            "reason": "integrand negative",
        }

    Z_scaled = np.trapezoid(integrand_theta, dx=_DTHETA)
    Z = (DELTA / 2.0) * Z_scaled
    if Z <= 0 or not np.isfinite(Z):
        return {"c": float(c), "n_basis": int(n_basis), "chi_0": chi, "M_cert": None, "reason": "Z<=0"}
    w_theta = (DELTA / 2.0) * integrand_theta / Z

    phi_hat_kn = _KN_COS @ w_theta * _DTHETA
    k_1 = float(phi_hat_kn[0] ** 2)

    phi_hat_qp = _QP_COS @ w_theta * _DTHETA
    if np.any(phi_hat_qp ** 2 < 1e-18):
        return {"c": float(c), "n_basis": int(n_basis), "chi_0": chi, "M_cert": None,
                "reason": "phi_hat_qp near zero (S_1 blow-up)"}
    S_1 = float(np.sum((MV_COEFFS ** 2) / (phi_hat_qp ** 2)))
    if not np.isfinite(S_1) or S_1 <= 0:
        return {"c": float(c), "n_basis": int(n_basis), "chi_0": chi, "M_cert": None,
                "reason": "S_1 invalid"}

    batch = 1000
    K2_pos = 0.0
    for s in range(0, N_XI, batch):
        xi_chunk = _XI[s:s + batch]
        cos_mat = np.cos(2 * np.pi * xi_chunk[:, None] * _X_OF_T[None, :])
        phi_hat_chunk = cos_mat @ w_theta * _DTHETA
        K2_pos += np.sum(phi_hat_chunk ** 4) * _DXI
    K_2 = float(2.0 * K2_pos)

    rad2 = K_2 - 1 - 2 * k_1 ** 2
    if rad2 <= 0 or not np.isfinite(K_2):
        return {"c": float(c), "n_basis": int(n_basis), "chi_0": chi, "M_cert": None,
                "k_1": k_1, "K_2": K_2, "S_1": S_1,
                "reason": f"K2 radicand invalid: rad2={rad2}"}

    a = (4.0 / U) / S_1
    target = 2.0 / U + a

    def sup_R(M):
        if M <= 1.0:
            return float('-inf')
        mu_ = M * np.sin(np.pi / M) / np.pi
        y_star_sq = (k_1 ** 2) * (M - 1) / (K_2 - 1)
        y_star = np.sqrt(max(0.0, y_star_sq))
        if y_star <= mu_:
            return M + 1 + np.sqrt((M - 1) * (K_2 - 1))
        rad1 = M - 1 - 2 * mu_ * mu_
        if rad1 < 0:
            return float('inf')
        return M + 1 + 2 * mu_ * k_1 + np.sqrt(rad1 * rad2)

    try:
        f = lambda M: sup_R(M) - target
        f_lo = f(1.0 + 1e-10)
        f_hi = f(2.0)
        if f_lo >= 0 or f_hi <= 0:
            return {"c": float(c), "n_basis": int(n_basis), "chi_0": chi, "M_cert": None,
                    "k_1": k_1, "K_2": K_2, "S_1": S_1, "a": float(a),
                    "reason": f"bracket fail: f(1+)={f_lo}, f(2)={f_hi}"}
        M_cert = brentq(f, 1.0 + 1e-10, 2.0, xtol=1e-7)
    except Exception as exc:
        return {"c": float(c), "n_basis": int(n_basis), "chi_0": chi, "M_cert": None,
                "reason": f"brentq exception: {exc!r}"}

    return {
        "c": float(c),
        "n_basis": int(n_basis),
        "chi_0": chi,
        "k_1": k_1,
        "K_2": K_2,
        "S_1": S_1,
        "a": float(a),
        "Z": float(Z),
        "M_cert": float(M_cert),
        "psi0_min": float(np.min(psi0_sin)),
        "psi0_max": float(np.max(psi0_sin)),
        "beta_l2_norm_sq": float(np.sum(beta ** 2)),
    }


# -----------------------------------------------------------------------------
# Sweep driver
# -----------------------------------------------------------------------------

def sweep(c_values, n_basis=120, verbose=True):
    results = []
    for c in c_values:
        t0 = time.time()
        q = compute_quantities_pswf(float(c), n_basis=n_basis)
        dt = time.time() - t0
        q["wall_time_sec"] = dt
        results.append(q)
        if verbose:
            if q.get("M_cert") is not None:
                print(
                    f"  c={float(c):8.4f}  M_cert={q['M_cert']:.7f}  "
                    f"k_1={q['k_1']:.6f}  K_2={q['K_2']:.5f}  "
                    f"S_1={q['S_1']:.4f}  chi_0={q['chi_0']:.4f}  ({dt:.2f}s)",
                    flush=True,
                )
            else:
                print(
                    f"  c={float(c):8.4f}  INFEASIBLE  reason={q.get('reason')!r}  "
                    f"chi_0={q.get('chi_0', 'n/a')}  ({dt:.2f}s)",
                    flush=True,
                )
    return results


def _arcsine_baseline():
    """Arcsine baseline via the existing pipeline."""
    try:
        from delsarte_dual.grid_bound_alt_kernel.variational_pod import compute_quantities
        return compute_quantities(np.array([0.0]))
    except Exception as e:
        return {"error": repr(e)}


def main():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--c-min", type=float, default=0.0)
    p.add_argument("--c-max", type=float, default=30.0)
    p.add_argument("--n-c", type=int, default=61)
    p.add_argument("--n-basis", type=int, default=120)
    p.add_argument("--out", default="data/pswf_sweep.json")
    p.add_argument("--include-fine", action="store_true")
    p.add_argument("--fine-half-width", type=float, default=1.0)
    p.add_argument("--fine-n", type=int, default=41)
    args = p.parse_args()

    print("=" * 72, flush=True)
    print("PSWF (K12) auto-convolution kernel sweep", flush=True)
    print("=" * 72, flush=True)
    print(f"  delta = {DELTA}", flush=True)
    print(f"  n_basis = {args.n_basis}  (Legendre even-parity truncation)", flush=True)
    print(f"  coarse c-grid: {args.n_c} points in [{args.c_min}, {args.c_max}]", flush=True)
    print()

    print(">>> Sanity checks:", flush=True)
    q_c0 = compute_quantities_pswf(0.0, n_basis=args.n_basis)
    if q_c0.get("M_cert") is not None:
        print(
            f"  PSWF c=0 (psi_0 = const  =>  phi = uniform  =>  K = triangle):",
            flush=True,
        )
        print(
            f"    M_cert={q_c0['M_cert']:.7f}  k_1={q_c0['k_1']:.6f}  "
            f"K_2={q_c0['K_2']:.5f}  S_1={q_c0['S_1']:.4f}  chi_0={q_c0['chi_0']:.4e}",
            flush=True,
        )
    else:
        print(f"  PSWF c=0 INFEASIBLE: {q_c0}", flush=True)

    arc = _arcsine_baseline()
    if isinstance(arc, dict) and arc.get("M_cert") is not None:
        print(
            f"  Arcsine baseline (alpha=0, no poly):  M_cert={arc['M_cert']:.7f}  "
            f"k_1={arc['k_1']:.6f}  K_2={arc['K_2']:.5f}  S_1={arc['S_1']:.4f}",
            flush=True,
        )
    else:
        print(f"  arcsine baseline failed: {arc}", flush=True)
    print()

    print(">>> Coarse sweep:", flush=True)
    c_grid = np.linspace(args.c_min, args.c_max, args.n_c)
    coarse = sweep(c_grid, n_basis=args.n_basis)
    feasible = [r for r in coarse if r.get("M_cert") is not None]
    if not feasible:
        print("\nERROR: no feasible PSWF candidates in coarse sweep", flush=True)
        return
    best = max(feasible, key=lambda r: r["M_cert"])
    print()
    print(f">>> Best from coarse sweep: c={best['c']:.4f}  M_cert={best['M_cert']:.7f}", flush=True)

    fine = []
    if args.include_fine:
        c_lo = max(args.c_min, best["c"] - args.fine_half_width)
        c_hi = best["c"] + args.fine_half_width
        c_grid_fine = np.linspace(c_lo, c_hi, args.fine_n)
        print()
        print(f">>> Fine sweep around c={best['c']:.4f} in [{c_lo:.4f}, {c_hi:.4f}]:",
              flush=True)
        fine = sweep(c_grid_fine, n_basis=args.n_basis)
        feasible_f = [r for r in fine if r.get("M_cert") is not None]
        if feasible_f:
            best_f = max(feasible_f, key=lambda r: r["M_cert"])
            if best_f["M_cert"] > best["M_cert"]:
                best = best_f
            print()
            print(
                f">>> Best from fine sweep: c={best_f['c']:.5f}  "
                f"M_cert={best_f['M_cert']:.7f}",
                flush=True,
            )

    print()
    print("=" * 72, flush=True)
    print("FINAL VERDICT", flush=True)
    print("=" * 72, flush=True)
    print(f"  Best PSWF M_cert: {best['M_cert']:.7f} at c={best['c']:.5f}", flush=True)
    if isinstance(arc, dict) and arc.get("M_cert") is not None:
        delta_arc = best["M_cert"] - arc["M_cert"]
        print(f"  Arcsine baseline: {arc['M_cert']:.7f}", flush=True)
        print(f"  Delta vs arcsine: {delta_arc:+.7f}", flush=True)
        print(f"  Beats arcsine?    {'YES' if delta_arc > 0 else 'NO'}", flush=True)
    print(f"  vs MV-published 1.27481: {best['M_cert'] - 1.27481:+.7f}", flush=True)
    print(f"  vs CS-record   1.28020: {best['M_cert'] - 1.28020:+.7f}", flush=True)
    print(f"  Beats MV-published 1.27481?  {'YES' if best['M_cert'] > 1.27481 else 'NO'}",
          flush=True)
    print(f"  Beats CS-record   1.28020?  {'YES' if best['M_cert'] > 1.28020 else 'NO'}",
          flush=True)

    out_data = {
        "delta": DELTA,
        "n_basis": args.n_basis,
        "coarse_c_grid": c_grid.tolist(),
        "coarse_results": coarse,
        "fine_results": fine,
        "best": best,
        "arcsine_baseline": arc if isinstance(arc, dict) else None,
        "psw_c0_baseline": q_c0,
        "mv_published_M_cert": 1.27481,
        "cs_record_M_cert": 1.28020,
    }
    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2, default=str)
    print()
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
