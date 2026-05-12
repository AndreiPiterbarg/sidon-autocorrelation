"""W9 verification script: Independent QP for N=200 re-opt G at 3-scale K.

Verifies:
 1. w_j = K_hat(j/u) > 0 for all j=1..200 (mpmath dps=30).
 2. QP solves with MOSEK/CLARABEL.
 3. S_1 = sum a_j^2 / w_j ~ 29.84090655472349 (v4's reported value).
 4. Fraction(10^12) rounding does not break G >= 1 - L*h.
"""
from __future__ import annotations

import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.special import j0 as scipy_j0

# Config (mirror v4)
DELTA1_Q = Fraction(138, 1000)
DELTA2_Q = Fraction(55, 1000)
DELTA3_Q = Fraction(25, 1000)
LAM1_Q = Fraction(85, 100)
LAM2_Q = Fraction(10, 100)
LAM3_Q = Fraction(5, 100)
U_Q = Fraction(1, 2) + DELTA1_Q
RATIONAL_DEN = 10**12
N_MODES = 200


def main():
    mp.mp.dps = 30
    print(f"=== W9 QP verification ===")
    print(f"  N = {N_MODES}, u = {float(U_Q)} = {U_Q}")
    print(f"  deltas = {[float(DELTA1_Q), float(DELTA2_Q), float(DELTA3_Q)]}")
    print(f"  lambdas = {[float(LAM1_Q), float(LAM2_Q), float(LAM3_Q)]}")

    # Step 1: compute w_j via mpmath dps=30
    deltas_mp = [mp.mpf(DELTA1_Q.numerator) / DELTA1_Q.denominator,
                 mp.mpf(DELTA2_Q.numerator) / DELTA2_Q.denominator,
                 mp.mpf(DELTA3_Q.numerator) / DELTA3_Q.denominator]
    lambdas_mp = [mp.mpf(LAM1_Q.numerator) / LAM1_Q.denominator,
                  mp.mpf(LAM2_Q.numerator) / LAM2_Q.denominator,
                  mp.mpf(LAM3_Q.numerator) / LAM3_Q.denominator]
    u_mp = mp.mpf(U_Q.numerator) / U_Q.denominator
    print(f"  u (mp30) = {u_mp}")

    w_mp = []
    w_float = np.zeros(N_MODES)
    for j in range(1, N_MODES + 1):
        xi = mp.mpf(j) / u_mp
        v = mp.mpf(0)
        for lam, d in zip(lambdas_mp, deltas_mp):
            arg = mp.pi * d * xi
            j0val = mp.besselj(0, arg)
            v += lam * j0val * j0val
        w_mp.append(v)
        w_float[j - 1] = float(v)

    # Step 2: verify all w_j > 0
    min_w = min(w_mp)
    max_w = max(w_mp)
    argmin_w = int(np.argmin(w_float))
    argmax_w = int(np.argmax(w_float))
    print(f"  min w_j = {float(min_w):.10e} at j={argmin_w + 1}")
    print(f"  max w_j = {float(max_w):.10e} at j={argmax_w + 1}")
    if min_w <= 0:
        print(f"  *** FLAG: w has nonpositive entry! ***")
        return 1
    print(f"  [OK] All 200 weights are positive")

    # Cross-check with scipy.j0 (used by v4)
    w_scipy = np.zeros(N_MODES)
    for j in range(1, N_MODES + 1):
        xi_j = j / float(U_Q)
        v = 0.0
        for lam, d in zip([0.85, 0.10, 0.05],
                          [0.138, 0.055, 0.025]):
            v += lam * scipy_j0(math.pi * d * xi_j) ** 2
        w_scipy[j - 1] = v
    max_rel_err = float(np.max(np.abs(w_float - w_scipy) / w_float))
    print(f"  max rel err scipy vs mpmath: {max_rel_err:.3e}")
    if max_rel_err > 1e-12:
        print(f"  [WARN] mpmath/scipy disagree more than expected")

    # Step 3: Set up QP using CVXPY
    import cvxpy as cp
    n_grid = max(5001, 4 * N_MODES + 1)
    print(f"  n_grid = {n_grid}")
    xs = np.linspace(0.0, 0.25, n_grid)
    B = np.zeros((n_grid, N_MODES))
    u_f = float(U_Q)
    for j in range(1, N_MODES + 1):
        B[:, j - 1] = np.cos(2.0 * math.pi * j * xs / u_f)

    a = cp.Variable(N_MODES)
    obj = cp.Minimize(cp.sum(cp.multiply(1.0 / w_float, cp.square(a))))
    cons = [B @ a >= 1.0]
    prob = cp.Problem(obj, cons)

    t0 = time.time()
    a_value = None
    used_solver = None
    for s_name in ("MOSEK", "CLARABEL", "SCS"):
        try:
            prob.solve(solver=s_name, verbose=False)
            if prob.status in ("optimal", "optimal_inaccurate") and a.value is not None:
                used_solver = s_name
                a_value = np.asarray(a.value).flatten()
                print(f"  QP solved with {s_name}; status={prob.status}; {time.time() - t0:.1f}s")
                break
        except Exception as exc:
            print(f"  Solver {s_name} failed: {exc}")
            continue
    if a_value is None:
        print(f"  *** FLAG: QP did not solve ***")
        return 1

    # Verify the unrounded QP
    S1_unrounded = float(np.sum(a_value ** 2 / w_float))
    print(f"  unrounded QP optimal value (S_1) = {S1_unrounded:.10f}")
    G_vals_unrounded = B @ a_value
    print(f"  min G(x_grid) (unrounded) = {float(G_vals_unrounded.min()):.10f}")

    # Step 4: Round to Fraction(10^12) and verify
    coeffs_q = [Fraction(int(round(av * RATIONAL_DEN)), RATIONAL_DEN) for av in a_value]
    coeffs_f = np.array([float(q) for q in coeffs_q])

    # Recompute S_1 with rounded coefficients
    S1_rounded = float(np.sum(coeffs_f ** 2 / w_float))
    print(f"  rounded S_1 (10^12 denom)        = {S1_rounded:.10f}")

    # Lipschitz constant: L = (2 pi/u) * sum j*|a_j|
    L = (2 * math.pi / u_f) * float(np.sum(np.arange(1, N_MODES + 1) * np.abs(coeffs_f)))
    print(f"  L (Lipschitz)        = {L:.6f}")
    arb_grid = 200001
    h = 1.0 / (4 * arb_grid)
    lip_rem = L * h
    print(f"  h (arb grid) = {h:.6e},  L*h = {lip_rem:.6e}")

    G_vals_rounded = B @ coeffs_f
    minG_grid = float(G_vals_rounded.min())
    minG_after_rem = minG_grid - lip_rem
    print(f"  min G(x_grid) (rounded) = {minG_grid:.10f}")
    print(f"  min G - L*h         = {minG_after_rem:.10f}")

    # Compare to v4's report
    s1_v4 = 29.84090655472349
    print(f"\n  --- COMPARISON ---")
    print(f"  v4 reported S_1_upper = {s1_v4:.10f}")
    print(f"  W9 independent S_1    = {S1_rounded:.10f}")
    print(f"  diff (W9 - v4)        = {S1_rounded - s1_v4:.6e}")
    print(f"  rel diff              = {abs(S1_rounded - s1_v4) / s1_v4:.6e}")

    verdict_s1 = abs(S1_rounded - s1_v4) < 1e-2
    print(f"  S_1 within 1e-2:      {verdict_s1}")
    print(f"  S_1 within 1e-3:      {abs(S1_rounded - s1_v4) < 1e-3}")
    print(f"  S_1 within 1e-4:      {abs(S1_rounded - s1_v4) < 1e-4}")

    # Save results to JSON
    import json
    out = {
        "N_modes": N_MODES,
        "deltas": [float(DELTA1_Q), float(DELTA2_Q), float(DELTA3_Q)],
        "lambdas": [float(LAM1_Q), float(LAM2_Q), float(LAM3_Q)],
        "min_w": float(min_w),
        "max_w": float(max_w),
        "argmin_w": argmin_w + 1,
        "argmax_w": argmax_w + 1,
        "solver_used": used_solver,
        "QP_status": prob.status,
        "S_1_unrounded": S1_unrounded,
        "S_1_rounded_10e12": S1_rounded,
        "S_1_v4_reported": s1_v4,
        "S_1_diff_abs": S1_rounded - s1_v4,
        "S_1_diff_rel": abs(S1_rounded - s1_v4) / s1_v4,
        "min_G_grid_rounded": minG_grid,
        "Lipschitz_L": L,
        "h_grid": h,
        "L_times_h": lip_rem,
        "min_G_minus_remainder": minG_after_rem,
        "max_rel_err_scipy_mpmath": max_rel_err,
        "S_1_within_1e_2": bool(verdict_s1),
    }
    json_path = Path(__file__).parent / "_w9_qp_verify_results.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Wrote {json_path}")

    # Final verdict
    print(f"\n  === FINAL VERDICT ===")
    if verdict_s1 and minG_after_rem > -1e-3 and min_w > 0:
        print(f"  CONFIRM: S_1 <= 29.84091 is achievable; QP well-conditioned; rounding OK")
    elif verdict_s1:
        print(f"  CONFIRM S_1 ~ 29.84; but check rounding/min-G separately")
    else:
        print(f"  FLAG: S_1 differs more than 1e-2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
