"""Experiment 4: Moment-3/4 with product localizing matrices x_i*x_j >= 0.

Building on Exp 3, add the product constraints that the Lasserre code uses:
For each pair (i,j), x_i * x_j >= 0 gives a localizing matrix:
  L_{ij}[a,b] = E[x_i x_j x_a x_b] = y4[i,j,a,b]

which must be PSD. These are the "simplex cuts" that improved Lasserre bounds.

This is essentially reconstructing the Lasserre Level-2 but with a FLAT
moment structure (no full degree-4 moment matrix, just individual PSD blocks).
The question: how close does this get to Lasserre-2?
"""
import numpy as np
import cvxpy as cp
import time
import warnings
from itertools import combinations_with_replacement
warnings.filterwarnings('ignore')

SOLVER = 'MOSEK' if 'MOSEK' in cp.installed_solvers() else 'CLARABEL'
print(f"Solver: {SOLVER}")


def solve_product_localizing(P, eta_tol=1e-4, use_conv_localizing=True, use_product_loc=True):
    """Full moment-3/4 with product + convolution localizing matrices."""
    t0 = time.time()
    n_diags = 2 * P - 1

    x = cp.Variable(P, nonneg=True)
    Y = cp.Variable((P, P), symmetric=True)
    eta_param = cp.Parameter(nonneg=True)

    # Degree-3 and degree-4 moments
    basis3 = list(combinations_with_replacement(range(P), 3))
    n3 = len(basis3)
    y3_vars = cp.Variable(n3, nonneg=True)
    y3_idx = {tuple(b): k for k, b in enumerate(basis3)}
    def y3(i, j, l):
        return y3_vars[y3_idx[tuple(sorted([i, j, l]))]]

    basis4 = list(combinations_with_replacement(range(P), 4))
    n4 = len(basis4)
    y4_vars = cp.Variable(n4, nonneg=True)
    y4_idx = {tuple(b): k for k, b in enumerate(basis4)}
    def y4(i, j, k, l):
        return y4_vars[y4_idx[tuple(sorted([i, j, k, l]))]]

    # Moment matrix PSD
    M = cp.bmat([[np.ones((1, 1)), cp.reshape(x, (1, P))],
                  [cp.reshape(x, (P, 1)), Y]])
    constraints = [M >> 0, cp.sum(x) == 1, Y >= 0]

    # Simplex on Y
    for i in range(P):
        constraints.append(cp.sum(Y[i, :]) == x[i])

    # Consistency y3 -> Y and y4 -> y3
    for i in range(P):
        for j in range(i, P):
            constraints.append(sum(y3(i, j, l) for l in range(P)) == Y[i, j])
    for b3 in basis3:
        constraints.append(sum(y4(*b3, l) for l in range(P)) == y3(*b3))

    # x_i^2 <= x_i
    for i in range(P):
        for j in range(P):
            constraints.append(y3(i, i, j) <= Y[min(i,j), max(i,j)])

    # Convolution constraints
    for k in range(n_diags):
        conv_k = sum(Y[i, k-i] for i in range(max(0, k-P+1), min(P, k+1)))
        constraints.append(2 * P * conv_k <= eta_param)

    # Multiply convolution by x_l
    for k in range(n_diags):
        for l in range(P):
            conv_k_l = sum(y3(i, k-i, l)
                           for i in range(max(0, k-P+1), min(P, k+1)))
            constraints.append(2 * P * conv_k_l <= eta_param * x[l])

    # NONNEG LOCALIZING: E[x_i * x_a * x_b] matrix is PSD for each i
    for i in range(P):
        L_i = cp.bmat([[y3(i, a, b) for b in range(P)] for a in range(P)])
        constraints.append(L_i >> 0)

    if use_conv_localizing:
        # Convolution localizing: eta*Y - 2P*sum_{i+j=k} y4[.,.,i,j] PSD
        for k in range(min(n_diags, P)):
            L_k = cp.bmat([[
                eta_param * Y[a, b] - 2 * P * sum(
                    y4(a, b, i, k-i) for i in range(max(0, k-P+1), min(P, k+1)))
                for b in range(P)] for a in range(P)])
            constraints.append(L_k >> 0)

    if use_product_loc:
        # Product localizing: E[x_i x_j x_a x_b] = y4[i,j,a,b] PSD in (a,b) for each (i,j)
        for i in range(P):
            for j in range(i, P):
                L_ij = cp.bmat([[y4(i, j, a, b) for b in range(P)] for a in range(P)])
                constraints.append(L_ij >> 0)

    prob = cp.Problem(cp.Minimize(0), constraints)

    shor = 2 * P / (2 * P - 1)
    eta_lo = shor
    eta_hi = 2.0

    for _ in range(40):
        if eta_hi - eta_lo < eta_tol:
            break
        eta_mid = (eta_lo + eta_hi) / 2
        eta_param.value = eta_mid
        try:
            prob.solve(solver=SOLVER, verbose=False, warm_start=True)
            if prob.status in ('optimal', 'optimal_inaccurate'):
                eta_hi = eta_mid
            else:
                eta_lo = eta_mid
        except Exception:
            eta_lo = eta_mid

    elapsed = time.time() - t0
    return eta_hi, elapsed


print("\n" + "=" * 72)
print("EXP 4: Product localizing matrices")
print("=" * 72)

known_ub = {2: 1.777778, 3: 1.706667, 4: 1.644465, 5: 1.633817,
            6: 1.600883, 7: 1.591746, 8: 1.580150}
known_lb = {2: 1.777778, 3: 1.706667, 4: 1.644465, 5: 1.632651,
            6: 1.585612, 7: 1.581746, 8: 1.548319}

print(f"{'P':>3} | {'Shor':>8} | {'Conv+Prod':>10} | {'Conv only':>10} | {'Lass-2':>10} | {'V(P)UB':>10} | {'Time':>6}")
print("-" * 80)

for P in range(2, 8):
    shor = 2 * P / (2 * P - 1)
    v_both, t_both = solve_product_localizing(P, use_conv_localizing=True, use_product_loc=True)
    v_conv, t_conv = solve_product_localizing(P, use_conv_localizing=True, use_product_loc=False)
    ub = known_ub.get(P, float('nan'))
    lb = known_lb.get(P, float('nan'))
    print(f"{P:>3} | {shor:>8.4f} | {v_both:>10.6f} | {v_conv:>10.6f} | {lb:>10.6f} | {ub:>10.6f} | {t_both:>5.1f}s/{t_conv:.1f}s")
    if t_both > 120:
        print("  (stopping)")
        break
