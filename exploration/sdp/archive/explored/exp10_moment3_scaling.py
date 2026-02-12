"""Experiment 10: Push moment-3 + localizing approach to larger P.

From Exp 2-4, the moment-3 approach beats Shor at every P tested (up to 10).
The key result: adding degree-3 moments with consistency constraints gives
nontrivial bounds that improve upon the Shor relaxation.

Here we scale up to P=15-25 to see how the bounds behave and compare with
known Lasserre-2 results. Also test: which constraints contribute the most?
"""
import numpy as np
import cvxpy as cp
import time
import warnings
from itertools import combinations_with_replacement
warnings.filterwarnings('ignore')

SOLVER = 'MOSEK' if 'MOSEK' in cp.installed_solvers() else 'CLARABEL'
print(f"Solver: {SOLVER}")


def solve_m3_fast(P, eta_tol=1e-4, use_x2_cuts=True, use_conv_mult=True):
    """Fast moment-3 relaxation.

    Variables: x_i, Y_{ij} = E[x_i x_j], y3_{ijk} = E[x_i x_j x_k]
    Constraints:
    - [1,x;x,Y] PSD, simplex on x and Y
    - sum_k y3[i,j,k] = Y[i,j]
    - y3 >= 0
    - x_i^2 <= x_i cuts: y3[i,i,j] <= Y[ij]
    - Multiply convolution by x_l: 2P*sum_{i+j=k} y3[i,j,l] <= eta*x[l]
    """
    t0 = time.time()
    n_diags = 2 * P - 1

    x = cp.Variable(P, nonneg=True)
    Y = cp.Variable((P, P), symmetric=True)
    eta_param = cp.Parameter(nonneg=True)

    basis3 = list(combinations_with_replacement(range(P), 3))
    n3 = len(basis3)
    y3_vars = cp.Variable(n3, nonneg=True)
    y3_idx = {tuple(b): k for k, b in enumerate(basis3)}
    def y3(i, j, l):
        return y3_vars[y3_idx[tuple(sorted([i, j, l]))]]

    M = cp.bmat([[np.ones((1, 1)), cp.reshape(x, (1, P))],
                  [cp.reshape(x, (P, 1)), Y]])
    constraints = [M >> 0, cp.sum(x) == 1, Y >= 0]

    for i in range(P):
        constraints.append(cp.sum(Y[i, :]) == x[i])

    for i in range(P):
        for j in range(i, P):
            constraints.append(sum(y3(i, j, l) for l in range(P)) == Y[i, j])

    if use_x2_cuts:
        for i in range(P):
            for j in range(P):
                constraints.append(y3(i, i, j) <= Y[min(i,j), max(i,j)])

    for k in range(n_diags):
        conv_k = sum(Y[i, k-i] for i in range(max(0, k-P+1), min(P, k+1)))
        constraints.append(2 * P * conv_k <= eta_param)

    if use_conv_mult:
        for k in range(n_diags):
            for l in range(P):
                conv_k_l = sum(y3(i, k-i, l)
                               for i in range(max(0, k-P+1), min(P, k+1)))
                constraints.append(2 * P * conv_k_l <= eta_param * x[l])

    prob = cp.Problem(cp.Minimize(0), constraints)

    shor = 2 * P / (2 * P - 1)
    eta_lo = shor
    eta_hi = 2.0

    for _ in range(30):
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
    return eta_hi, elapsed, n3


print("=" * 80)
print("EXP 10: Moment-3 scaling to larger P")
print("=" * 80)

known_lb = {2: 1.777778, 3: 1.706667, 4: 1.644465, 5: 1.632651,
            6: 1.585612, 7: 1.581746, 8: 1.548319, 9: 1.545626,
            10: 1.524823, 11: 1.520012, 12: 1.507730, 13: 1.503642,
            14: 1.493577, 15: 1.490516, 16: 1.483826, 17: 1.481782, 18: 1.475327}
known_ub = {2: 1.777778, 3: 1.706667, 4: 1.644465, 5: 1.633817,
            6: 1.600883, 7: 1.591746, 8: 1.580150, 9: 1.578073,
            10: 1.566436, 11: 1.562873, 12: 1.559773, 13: 1.559581,
            14: 1.552608, 15: 1.550623}

# Part 1: Full moment-3 scaling
print(f"\n{'P':>3} | {'Shor':>8} | {'M3 LB':>10} | {'M3-Shor':>10} | {'Lass-2 LB':>10} | {'V(P) UB':>10} | {'#y3':>6} | {'Time':>6}")
print("-" * 90)

for P in list(range(2, 21)):
    shor = 2 * P / (2 * P - 1)
    m3, t3, n3 = solve_m3_fast(P)
    lb = known_lb.get(P, float('nan'))
    ub = known_ub.get(P, float('nan'))
    diff = m3 - shor
    print(f"{P:>3} | {shor:>8.4f} | {m3:>10.6f} | {diff:>+10.6f} | {lb:>10.6f} | {ub:>10.6f} | {n3:>6} | {t3:>5.1f}s")
    if t3 > 120:
        print("  (stopping, getting slow)")
        break

# Part 2: Ablation - which constraints matter?
print("\n" + "=" * 80)
print("Ablation: which constraints contribute?")
print("=" * 80)
print(f"{'P':>3} | {'Full M3':>10} | {'No x^2 cuts':>12} | {'No conv mult':>12} | {'Neither':>10} | {'Shor':>8}")
print("-" * 70)

for P in [5, 8, 10]:
    shor = 2 * P / (2 * P - 1)
    full, _, _ = solve_m3_fast(P, use_x2_cuts=True, use_conv_mult=True)
    no_x2, _, _ = solve_m3_fast(P, use_x2_cuts=False, use_conv_mult=True)
    no_conv, _, _ = solve_m3_fast(P, use_x2_cuts=True, use_conv_mult=False)
    neither, _, _ = solve_m3_fast(P, use_x2_cuts=False, use_conv_mult=False)
    print(f"{P:>3} | {full:>10.6f} | {no_x2:>12.6f} | {no_conv:>12.6f} | {neither:>10.6f} | {shor:>8.4f}")
