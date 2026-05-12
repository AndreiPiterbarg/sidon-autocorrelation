"""V7 verification: Independent QP for multi-scale K (delta_1=0.138, lambda_1=0.85;
                                                       delta_2=0.045, lambda_2=0.15).

Goal: confirm S_1 = 31.441956302611327 from the QP-reoptimized G.

QP:
  minimize sum_{j=1}^{119} a_j^2 / w_j
  subject to G(x) = sum_{j=1}^{119} a_j cos(2 pi j x / u) >= 1 on x in [0, 1/4]
  where u = 638/1000, w_j = 0.85 * J_0(pi*0.138*j/u)^2 + 0.15 * J_0(pi*0.045*j/u)^2.

w_j computed via mpmath at dps=30 (independent from flint.arb in optimize_G.py).
"""
from __future__ import annotations

import math
import sys
import numpy as np
import mpmath as mp

mp.mp.dps = 30

# Problem parameters
N = 119
U_NUM = 638
U_DEN = 1000
u = mp.mpf(U_NUM) / mp.mpf(U_DEN)
u_f = float(u)

delta1 = mp.mpf("0.138")
lam1 = mp.mpf("0.85")
delta2 = mp.mpf("0.045")
lam2 = mp.mpf("0.15")

print(f"V7 verification of S_1 for QP-reoptimized G under multi-scale kernel")
print(f"  delta_1 = {float(delta1):.6f}, lambda_1 = {float(lam1):.6f}")
print(f"  delta_2 = {float(delta2):.6f}, lambda_2 = {float(lam2):.6f}")
print(f"  u = {u_f}, n = {N}")
print()

# -------- Compute w_j via mpmath ----------
print("Computing w_j via mpmath J_0 at dps=30 ...")
w_mp = []
for j in range(1, N + 1):
    arg1 = mp.pi * delta1 * j / u
    arg2 = mp.pi * delta2 * j / u
    j0_1 = mp.besselj(0, arg1)
    j0_2 = mp.besselj(0, arg2)
    w_j = lam1 * j0_1 ** 2 + lam2 * j0_2 ** 2
    w_mp.append(w_j)

w = np.array([float(x) for x in w_mp])
min_w = float(w.min())
max_w = float(w.max())
print(f"  min_j w_j = {min_w:.6e}  (j={int(np.argmin(w))+1})")
print(f"  max_j w_j = {max_w:.6e}  (j={int(np.argmax(w))+1})")
print(f"  All w_j > 0: {np.all(w > 0)}")
print(f"  w_1 = {w[0]:.10f}")
print(f"  w_119 = {w[118]:.6e}")
print()

# Sanity: for single-scale (lambda_1=1) reproduce MV at delta=0.138
print("Sanity check: single-scale (delta=0.138, lambda=1) for a few j:")
for j in [1, 10, 50, 100, 119]:
    sj = float(mp.besselj(0, mp.pi * delta1 * j / u) ** 2)
    print(f"  j={j:3d}: J_0(pi*0.138*j/u)^2 = {sj:.6e}")
print()

# -------- Build B matrix on n_grid=5001 points ----------
n_grid = 5001
xs = np.linspace(0.0, 0.25, n_grid)
B = np.zeros((n_grid, N))
for j in range(1, N + 1):
    B[:, j - 1] = np.cos(2.0 * math.pi * j * xs / u_f)

# -------- Solve QP ----------
print(f"Solving QP with n={N}, n_grid={n_grid} ...")
try:
    import cvxpy as cp
    a_var = cp.Variable(N)
    obj = cp.Minimize(cp.sum(cp.multiply(1.0 / w, cp.square(a_var))))
    cons = [B @ a_var >= 1.0]
    prob = cp.Problem(obj, cons)
    final_solver = None
    final_status = None
    for solver_name in ["MOSEK", "CLARABEL", "SCS", "ECOS"]:
        try:
            prob.solve(solver=solver_name, verbose=False)
            if a_var.value is not None and prob.status in (
                "optimal", "optimal_inaccurate"
            ):
                final_solver = solver_name
                final_status = prob.status
                break
        except Exception as e:
            print(f"  {solver_name} failed: {str(e)[:80]}")
            continue
    if a_var.value is None:
        print("FATAL: All solvers failed.")
        sys.exit(2)

    a_opt = np.asarray(a_var.value).flatten()
    S1 = float(np.sum((a_opt ** 2) / w))
    min_G = float((B @ a_opt).min())
    print(f"  Solver: {final_solver}  Status: {final_status}")
    print(f"  S_1 (multi-scale, mpmath weights)  = {S1:.15f}")
    print(f"  min_x G(x) on grid                  = {min_G:.10f}")
    print(f"  asserted S_1                        = 31.441956302611327")
    rel_diff = abs(S1 - 31.441956302611327) / 31.441956302611327
    abs_diff = abs(S1 - 31.441956302611327)
    print(f"  |abs diff| = {abs_diff:.3e}  rel diff = {rel_diff:.3e}")
    print()

except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(2)

# -------- Comparison: single-scale arcsine (delta=0.138) ----------
print("Comparison: single-scale K_arc(delta=0.138) (lambda=1):")
w_ss = []
for j in range(1, N + 1):
    j0 = float(mp.besselj(0, mp.pi * delta1 * j / u))
    w_ss.append(j0 * j0)
w_ss = np.array(w_ss)

a_var2 = cp.Variable(N)
obj2 = cp.Minimize(cp.sum(cp.multiply(1.0 / w_ss, cp.square(a_var2))))
cons2 = [B @ a_var2 >= 1.0]
prob2 = cp.Problem(obj2, cons2)
for solver_name in ["MOSEK", "CLARABEL", "SCS", "ECOS"]:
    try:
        prob2.solve(solver=solver_name, verbose=False)
        if a_var2.value is not None and prob2.status in (
            "optimal", "optimal_inaccurate"
        ):
            break
    except Exception:
        continue
a_ss = np.asarray(a_var2.value).flatten()
S1_ss = float(np.sum((a_ss ** 2) / w_ss))
min_G_ss = float((B @ a_ss).min())
print(f"  S_1 (single-scale delta=0.138)      = {S1_ss:.10f}")
print(f"  min_G_ss                            = {min_G_ss:.10f}")
print(f"  multi-scale S_1 lower than single?   {S1 < S1_ss}")
print(f"  delta S_1 (single - multi)          = {S1_ss - S1:.6f}")
print()

# -------- Summary line ----------
PASS = (abs_diff <= 1e-3) and bool(np.all(w > 0))
print(f"PASS = {PASS}")
print(f"Computed S_1   = {S1:.15f}")
print(f"min_j w_j      = {min_w:.6e}")
