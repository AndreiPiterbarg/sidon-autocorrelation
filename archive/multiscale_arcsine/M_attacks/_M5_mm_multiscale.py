"""Agent M5: Multi-moment (MM-N) evaluator on multi-scale arcsine kernels.

We use the MV multi-moment master inequality (Theorem 2 / MM-10) at N in {1, 2, 3}:

    2/u + a <= M + 1 + 2 Sum_{n=1..N} y_n k_n
             + sqrt(M - 1 - 2 Sum y_n^2) sqrt(K_2 - 1 - 2 Sum k_n^2)

subject to 0 <= y_n <= mu(M) := M sin(pi/M)/pi, where y_n = |hat_f(n)|^2 = z_n^2
and k_n is the period-1 Fourier coefficient of the multi-scale arcsine K.

For multi-scale K(x) = sum_i lambda_i K_arc(x; delta_i):
    k_n = K_hat(n) = sum_i lambda_i * J_0(pi n delta_i)^2
    K_2 = int K_hat(xi)^2 dxi  (numerical, via scipy.quad)
    S_1 = sum_{j=1..119} a_j^2 / K_hat(j/u)  (a_j from MV)
    min_G = QP-reoptimized lower envelope min for the multi-scale weights

At fixed M, sup_y RHS over y_n in [0, mu(M)] is concave; in the unclipped regime
    y_n* = k_n * sqrt((M - 1) / (K_2 - 1))
and the value reduces to MV-7's M + 1 + sqrt((M-1)(K_2-1)) (INDEPENDENT of N).
N > 1 only HELPS over N=1 when the clipping y_n <= mu(M) is BINDING for at least
one n (since binding clipping reduces the inner sum k_n y_n LESS than the radicand
penalty for the corresponding y_n^2 — yielding a STRICTLY larger sup R, hence a
LARGER M_cert).

For this script:
  - Evaluate sup_y RHS at any (M, k_n, K_2) numerically over the box [0, mu(M)]^N
    via a fast deterministic strategy: KKT (unclipped) then iterative clipping.
  - Bisect on M to find M_cert.
  - Sanity-check: at N=1, reproduce K26's 1.29005 on the 2-scale (0.138, 0.045, 0.85).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.special import j0
from scipy.integrate import quad
from scipy.optimize import brentq, minimize

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

# Reuse the K26 pipeline for K_2 integration / QP / k_n compute.
from _K26_full_sweep_reopt import (  # noqa: E402
    DELTA,
    U,
    N_QP,
    K_hat_ms,
    K_2_quad,
    solve_QP,
)


def k_n_of(n, deltas, lambdas):
    """Period-1 FT coefficient of multi-scale K at integer n.

    K_hat(n) = sum_i lambda_i * J_0(pi n delta_i)^2.
    """
    out = 0.0
    for lam, d in zip(lambdas, deltas):
        out += lam * j0(np.pi * n * d) ** 2
    return float(out)


# ---------------------------------------------------------------------------
# Sup-R(M) at level N: maximize over y_n in [0, mu(M)] subject to global radicand
# non-negativity.
# ---------------------------------------------------------------------------

def sup_R_mm(M, ks, K_2):
    """Compute sup_y RHS of MM-10 at level N = len(ks).

    R(M, y) := M + 1 + 2 Sum y_n k_n + sqrt((M - 1 - 2 Sum y_n^2)_+)
                                       * sqrt((K_2 - 1 - 2 Sum k_n^2)_+)

    y_n in [0, mu(M)] for each n. Returns the sup (a float; +inf if radicand
    M-1-2 Sum y_n^2 < 0 at the supremum which is treated as the "trivial"
    saturation — but for our regime this never happens since M < 1.51 and
    y_n is bounded).
    """
    N = len(ks)
    if M <= 1.0:
        return float('-inf')
    A = sum(k * k for k in ks)
    B = K_2 - 1.0 - 2.0 * A
    if B <= 0:
        return float('inf')
    mu_M = M * np.sin(np.pi / M) / np.pi

    # Step 1: unclipped (interior) KKT solution.
    #     y_n* = k_n * sqrt((M - 1) / (K_2 - 1))      (derived in docstring)
    common = np.sqrt(max(0.0, (M - 1.0) / (K_2 - 1.0)))
    y_unclipped = [k * common for k in ks]

    # If all y_unclipped <= mu, we're in the interior regime, with value
    #     M + 1 + sqrt((M-1)(K_2-1))     (= MV-7 form).
    if all(y <= mu_M + 1e-15 for y in y_unclipped):
        return M + 1.0 + np.sqrt((M - 1.0) * (K_2 - 1.0))

    # Step 2: at least one y_n is clipped. Use coordinate-descent / iterative
    # solver on which subset to clip: pin the over-budget coords to mu, then
    # re-solve the KKT system on the remaining free coords.
    #
    # If S = {n : y_n* > mu}, pin y_n = mu for n in S. For free coords F = N\S:
    #   y_n = k_n * sqrt((M - 1 - 2 (|S| mu^2 + Sum_{m in F} y_m^2)) /
    #                    (K_2 - 1 - 2 Sum_n k_n^2))
    # Squaring & summing over F:
    #   Sum_F y_n^2 = (Sum_F k_n^2) * (M - 1 - 2|S| mu^2 - 2 Sum_F y_n^2) / B
    # Solve for s_F = Sum_F y_n^2:
    #   s_F * B = A_F (M - 1 - 2|S| mu^2 - 2 s_F)
    #   s_F (B + 2 A_F) = A_F (M - 1 - 2|S| mu^2)
    #   s_F = A_F (M - 1 - 2|S| mu^2) / (B + 2 A_F) = A_F (M - 1 - 2|S| mu^2) / (K_2 - 1 - 2 A_S)
    #         where A_S = Sum_S k_n^2 (since K_2 - 1 - 2A + 2A_F = K_2 - 1 - 2A_S).
    # Then y_n = k_n * sqrt((M - 1 - 2|S| mu^2 - 2 s_F) / B).
    #
    # We iterate: start with S = {n : y_unclipped[n] > mu}; verify each
    # re-solved y_n; expand S if any new y_n exceeds mu; repeat until stable.
    # In practice for N <= 3 this is O(2^N) cases worst-case; we just enumerate.
    best_val = -np.inf
    for mask in range(2 ** N):
        S = [n for n in range(N) if (mask >> n) & 1]
        F = [n for n in range(N) if n not in S]
        kS = [ks[n] for n in S]
        kF = [ks[n] for n in F]
        AS = sum(k * k for k in kS)
        AF = sum(k * k for k in kF)
        BF_denom = K_2 - 1.0 - 2.0 * AS  # = B + 2 AF
        m1m_2Smu2 = (M - 1.0) - 2.0 * len(S) * mu_M * mu_M
        if m1m_2Smu2 < 0:
            continue
        if BF_denom <= 0:
            continue
        # Re-solve free y_n:
        if AF > 0:
            s_F = AF * m1m_2Smu2 / BF_denom
            radF = m1m_2Smu2 - 2.0 * s_F
            if radF < 0:
                continue
            sqrt_factor = np.sqrt(radF / (K_2 - 1.0 - 2.0 * A))
            y_F = [k * sqrt_factor for k in kF]
        else:
            s_F = 0.0
            radF = m1m_2Smu2
            if radF < 0:
                continue
            y_F = []

        # Verify clipping consistency: every y in F must be <= mu, every n in S
        # must be the active boundary (we accept it).
        if any(y > mu_M + 1e-12 for y in y_F):
            continue
        # Build full y vector.
        y = [0.0] * N
        for i, n in enumerate(S):
            y[n] = mu_M
        for i, n in enumerate(F):
            y[n] = y_F[i]

        # Compute R.
        sum_yk = sum(y[n] * ks[n] for n in range(N))
        sum_y2 = sum(y[n] * y[n] for n in range(N))
        rad1 = M - 1.0 - 2.0 * sum_y2
        if rad1 < 0:
            continue
        val = M + 1.0 + 2.0 * sum_yk + np.sqrt(rad1) * np.sqrt(B)
        if val > best_val:
            best_val = val
    if best_val == -np.inf:
        # Fall back to interior (no consistent mask found) — should not happen.
        return M + 1.0 + np.sqrt(max(0.0, (M - 1.0) * (K_2 - 1.0)))
    return best_val


def M_cert_mm(ks, K_2, S_1, min_G, M_lo=1.001, M_hi=1.5):
    """MM-N M_cert via brentq.

    target = 2/u + a;  a = (4/u) * min_G^2 / S_1.
    M_cert = M where sup_R(M) = target.
    """
    a_gain = (4.0 / U) * (min_G ** 2) / S_1
    target = 2.0 / U + a_gain

    def g(M):
        return sup_R_mm(M, ks, K_2) - target

    try:
        g_lo = g(M_lo)
        g_hi = g(M_hi)
        if g_lo >= 0 or g_hi <= 0:
            return None
        return brentq(g, M_lo, M_hi, xtol=1e-9)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Evaluation harness: build kernel, compute k_1..k_N, K_2, S_1, min_G, M_cert(N).
# ---------------------------------------------------------------------------

def evaluate(deltas, lambdas, N_max=3, label=""):
    """Run the full pipeline at multiple N levels and return M_cert per N."""
    qp_a, S_1, min_G, status = solve_QP(deltas, lambdas)
    if status != "ok":
        return {"label": label, "status": status, "deltas": deltas,
                "lambdas": lambdas}
    # k_n
    ks = [k_n_of(n, deltas, lambdas) for n in range(1, N_max + 1)]
    # K_2
    K_2 = K_2_quad(deltas, lambdas)
    # M_cert per N
    M_per_N = {}
    for N in range(1, N_max + 1):
        M_per_N[N] = M_cert_mm(ks[:N], K_2, S_1, min_G)
    return {
        "label": label,
        "deltas": [float(d) for d in deltas],
        "lambdas": [float(l) for l in lambdas],
        "ks": ks,
        "K_2": float(K_2),
        "S_1": float(S_1),
        "min_G": float(min_G),
        "M_cert_per_N": {str(k): (float(v) if v is not None else None)
                          for k, v in M_per_N.items()},
        "status": status,
    }


def evaluate_default_2scale():
    """Best 2-scale K from K26 sweep: (0.138, 0.045, 0.85)."""
    return evaluate([DELTA, 0.045], [0.85, 0.15], N_max=3,
                    label="2scale-K26-best")


def evaluate_sweep():
    """Coarse sweep around the K26 best to see if N=2,3 shift the optimum."""
    results = []
    deltas_1 = DELTA  # fixed
    delta_2_list = [0.030, 0.040, 0.045, 0.050, 0.060, 0.080]
    lambda_1_list = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
    print("\n=== 2-scale sweep at N=1,2,3 ===")
    print(f"{'d2':>7} {'l1':>6} {'k1':>7} {'k2':>7} {'k3':>7} {'K2':>7} "
          f"{'M_N1':>7} {'M_N2':>7} {'M_N3':>7} {'gain_2v1':>9}")
    for d2 in delta_2_list:
        for l1 in lambda_1_list:
            r = evaluate([deltas_1, d2], [l1, 1.0 - l1], N_max=3,
                         label=f"d2={d2},l1={l1}")
            if r.get("status") != "ok":
                continue
            M1 = r["M_cert_per_N"]["1"]
            M2 = r["M_cert_per_N"]["2"]
            M3 = r["M_cert_per_N"]["3"]
            gain_21 = (M2 - M1) if (M1 is not None and M2 is not None) else None
            ks_str = [f"{k:.4f}" for k in r["ks"]]
            print(f"{d2:>7.4f} {l1:>6.3f} {ks_str[0]:>7} {ks_str[1]:>7} "
                  f"{ks_str[2]:>7} {r['K_2']:>7.4f} "
                  f"{M1 if M1 is None else f'{M1:.5f}':>7} "
                  f"{M2 if M2 is None else f'{M2:.5f}':>7} "
                  f"{M3 if M3 is None else f'{M3:.5f}':>7} "
                  f"{gain_21 if gain_21 is None else f'{gain_21:+.5f}':>9}")
            results.append(r)
    return results


def main():
    t0 = time.time()
    out = {
        "agent": "M5",
        "description": "MM-N (N=1,2,3) on multi-scale arcsine kernels",
        "baselines": {
            "MV_arcsine_N1": 1.27481,
            "K26_2scale_N1_best": 1.29005,
        },
        "DELTA": DELTA,
        "U": U,
        "N_QP": N_QP,
    }

    # --- 1. Sanity: N=1 on the K26 best (0.138, 0.045, 0.85) ---
    print("=" * 78)
    print("M5 Step 1: Sanity-check N=1 on (delta_1=0.138, delta_2=0.045, lambda_1=0.85)")
    print("=" * 78)
    sanity = evaluate([DELTA, 0.045], [0.85, 0.15], N_max=3,
                      label="sanity-2scale-K26-best")
    print(json.dumps({k: v for k, v in sanity.items()
                       if k not in ("status",)}, indent=2, default=str))
    out["sanity"] = sanity

    # --- 2. Coarse sweep at N=1, 2, 3 ---
    print("\n" + "=" * 78)
    print("M5 Step 2: 2-scale sweep, N=1 vs N=2 vs N=3")
    print("=" * 78)
    sweep = evaluate_sweep()
    out["sweep_2scale"] = sweep
    # Identify best at each N
    best_per_N = {1: None, 2: None, 3: None}
    for r in sweep:
        for N_str, Mc in r["M_cert_per_N"].items():
            N = int(N_str)
            if Mc is None:
                continue
            if best_per_N[N] is None or Mc > best_per_N[N]["M_cert"]:
                best_per_N[N] = {"M_cert": float(Mc),
                                  "deltas": r["deltas"],
                                  "lambdas": r["lambdas"],
                                  "ks": r["ks"],
                                  "K_2": r["K_2"],
                                  "S_1": r["S_1"],
                                  "min_G": r["min_G"]}
    out["best_per_N"] = best_per_N
    print("\n--- Best per N across sweep ---")
    for N in [1, 2, 3]:
        b = best_per_N[N]
        if b is None:
            print(f"N={N}: no valid points")
        else:
            print(f"N={N}: M_cert={b['M_cert']:.5f} at deltas={b['deltas']}, "
                   f"lambdas={b['lambdas']}")

    # --- 3. Report ---
    out["total_elapsed_s"] = float(time.time() - t0)
    out_path = "_M5_mm_multiscale_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x)
                   if hasattr(x, "item") else str(x))
    print(f"\nWrote {out_path}")
    print(f"Total elapsed: {out['total_elapsed_s']:.1f}s")


if __name__ == "__main__":
    main()
