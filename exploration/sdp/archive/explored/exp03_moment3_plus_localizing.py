"""Experiment 3: Moment-3 + localizing matrix constraints.

Building on Exp 2 (which beats Shor!), add localizing matrix constraints:
For each k, the constraint g_k(x) = eta - 2P*sum_{i+j=k} x_i x_j >= 0
should be certified via a localizing matrix L_k PSD, where:
  L_k[a,b] = E[x_a * x_b * g_k(x)] >= 0  as a matrix.

This requires degree-4 moments for the quadratic part. But we can use a
PARTIAL degree-4: only the moments needed for the localizing matrices,
not the full Lasserre Level-2 moment matrix.

Also try: product moment matrix for pairs (i,j):
  E[x_i * x_j * x_a * x_b] for the simplex constraints x_i(1 - sum x_j) = 0.
"""
import numpy as np
import cvxpy as cp
import time
import warnings
from itertools import combinations_with_replacement
warnings.filterwarnings('ignore')

SOLVER = 'MOSEK' if 'MOSEK' in cp.installed_solvers() else 'CLARABEL'
print(f"Solver: {SOLVER}")


def solve_moment3_localizing(P, eta_tol=1e-4):
    """Moment-3 + localizing matrices for convolution constraints.

    For each convolution diagonal k, we form:
      L_k[a,b] = eta * E[x_a x_b] - 2P * sum_{i+j=k} E[x_a x_b x_i x_j]

    This requires degree-4 moments E[x_a x_b x_i x_j] but ONLY for specific
    combinations (those appearing in convolution diagonals).
    """
    t0 = time.time()
    n_diags = 2 * P - 1

    # Variables
    x = cp.Variable(P, nonneg=True)
    Y = cp.Variable((P, P), symmetric=True)
    eta_param = cp.Parameter(nonneg=True)

    # Degree-3 moments
    basis3 = list(combinations_with_replacement(range(P), 3))
    n3 = len(basis3)
    y3_vars = cp.Variable(n3, nonneg=True)
    y3_idx = {tuple(b): k for k, b in enumerate(basis3)}
    def y3(i, j, l):
        return y3_vars[y3_idx[tuple(sorted([i, j, l]))]]

    # Degree-4 moments (only those needed)
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

    # Consistency y3 -> Y
    for i in range(P):
        for j in range(i, P):
            s = sum(y3(i, j, l) for l in range(P))
            constraints.append(s == Y[i, j])

    # Consistency y4 -> y3
    for b3 in basis3:
        s = sum(y4(*b3, l) for l in range(P))
        constraints.append(s == y3(*b3))

    # x_i^2 <= x_i => y3[i,i,j] <= Y[ij], y4[i,i,j,k] <= y3[j,k,?]...
    for i in range(P):
        for j in range(P):
            constraints.append(y3(i, i, j) <= Y[min(i,j), max(i,j)])

    # Convolution constraints
    for k in range(n_diags):
        conv_k = sum(Y[i, k-i] for i in range(max(0, k-P+1), min(P, k+1)))
        constraints.append(2 * P * conv_k <= eta_param)

    # Strengthened: multiply by x_l
    for k in range(n_diags):
        for l in range(P):
            conv_k_l = sum(y3(i, k-i, l)
                           for i in range(max(0, k-P+1), min(P, k+1)))
            constraints.append(2 * P * conv_k_l <= eta_param * x[l])

    # LOCALIZING MATRICES: For each convolution constraint k,
    # L_k[a,b] = eta*Y[a,b] - 2P * sum_{i+j=k} y4[a,b,i,j]  must be PSD
    # Use symmetry: only k = 0,...,P-1 (k and 2P-2-k give same by reflection)
    for k in range(min(n_diags, P)):
        L_k_entries = []
        for a in range(P):
            row = []
            for b in range(P):
                entry = eta_param * Y[a, b] - 2 * P * sum(
                    y4(a, b, i, k - i)
                    for i in range(max(0, k - P + 1), min(P, k + 1)))
                row.append(entry)
            L_k_entries.append(row)
        # Build matrix using vstack/hstack
        L_k = cp.bmat(L_k_entries)
        constraints.append(L_k >> 0)

    # Degree-2 moment matrix for nonneg constraints
    # L_i[a,b] = E[x_i * x_a * x_b] = y3[i,a,b] must form PSD matrix
    for i in range(P):
        L_i_entries = [[y3(i, a, b) for b in range(P)] for a in range(P)]
        L_i = cp.bmat(L_i_entries)
        constraints.append(L_i >> 0)

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
    return eta_hi, elapsed, n3 + n4


print("\n" + "=" * 72)
print("EXP 3: Moment-3 + localizing matrices")
print("=" * 72)
print(f"{'P':>3} | {'Shor':>10} | {'M3+Loc':>10} | {'M3 only':>10} | {'Lass-2':>10} | {'V(P)UB':>10} | {'Time':>6}")
print("-" * 85)

known_ub = {2: 1.777778, 3: 1.706667, 4: 1.644465, 5: 1.633817,
            6: 1.600883, 7: 1.591746, 8: 1.580150}
known_lb = {2: 1.777778, 3: 1.706667, 4: 1.644465, 5: 1.632651,
            6: 1.585612, 7: 1.581746, 8: 1.548319}
m3_results = {2: 1.600016, 3: 1.532227, 4: 1.383196, 5: 1.342502,
              6: 1.300038, 7: 1.273381, 8: 1.255623}

for P in range(2, 9):
    shor = 2 * P / (2 * P - 1)
    val, t_elapsed, n_vars = solve_moment3_localizing(P)
    ub = known_ub.get(P, float('nan'))
    lb = known_lb.get(P, float('nan'))
    m3 = m3_results.get(P, float('nan'))
    print(f"{P:>3} | {shor:>10.6f} | {val:>10.6f} | {m3:>10.6f} | {lb:>10.6f} | {ub:>10.6f} | {t_elapsed:>5.1f}s ({n_vars} vars)")
    if t_elapsed > 120:
        print("  (stopping, getting slow)")
        break
