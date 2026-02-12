"""Experiment 14: Cross-diagonal product inequalities.

For two diagonals k1, k2, on the simplex:
  (x^T A_{k1} x)(x^T A_{k2} x) <= (eta/(2P))^2

since each factor <= eta/(2P). This gives:
  sum_{i+j=k1, a+b=k2} y4[i,j,a,b] <= (eta/(2P))^2

This is QUADRATIC in eta but linear in y4. With eta as parameter, it's linear.

More generally, for any subset S of diagonals:
  prod_{k in S} (x^T A_k x) <= (eta/(2P))^|S|

For |S|=2 this uses degree-4 moments. For |S|=3 it needs degree-6.

Also try: sum of products. For any weights w_k >= 0 with sum w_k = 1:
  (sum_k w_k q_k)^2 <= sum_k w_k q_k^2  (by Jensen/convexity)
  => 1/(2P-1)^2 <= sum_k q_k^2 / (2P-1) ... hmm, same as CS bound.

Better: use the AM-QM inequality on subsets.

Actually, the cleanest new cut: for each pair (k1,k2),
  sum_{i+j=k1, a+b=k2} y4[i,j,a,b] <= eta^2 / (4P^2)

These are called "RLT product constraints" in the optimization literature.
"""
import numpy as np
import cvxpy as cp
import time
import warnings
from itertools import combinations_with_replacement, combinations
warnings.filterwarnings('ignore')

SOLVER = 'MOSEK' if 'MOSEK' in cp.installed_solvers() else 'CLARABEL'
print(f"Solver: {SOLVER}")


def solve_cross_diagonal(P, eta_tol=1e-4, use_cross=True, use_self=True):
    """Moment-3/4 with cross-diagonal product cuts."""
    t0 = time.time()
    n_diags = 2 * P - 1

    x = cp.Variable(P, nonneg=True)
    Y = cp.Variable((P, P), symmetric=True)
    eta_param = cp.Parameter(nonneg=True)

    basis3 = list(combinations_with_replacement(range(P), 3))
    y3_vars = cp.Variable(len(basis3), nonneg=True)
    y3_idx = {tuple(b): k for k, b in enumerate(basis3)}
    def y3(i, j, l):
        return y3_vars[y3_idx[tuple(sorted([i, j, l]))]]

    basis4 = list(combinations_with_replacement(range(P), 4))
    y4_vars = cp.Variable(len(basis4), nonneg=True)
    y4_idx = {tuple(b): k for k, b in enumerate(basis4)}
    def y4(i, j, k, l):
        return y4_vars[y4_idx[tuple(sorted([i, j, k, l]))]]

    M = cp.bmat([[np.ones((1, 1)), cp.reshape(x, (1, P))],
                  [cp.reshape(x, (P, 1)), Y]])
    constraints = [M >> 0, cp.sum(x) == 1, Y >= 0]

    for i in range(P):
        constraints.append(cp.sum(Y[i, :]) == x[i])

    for i in range(P):
        for j in range(i, P):
            constraints.append(sum(y3(i, j, l) for l in range(P)) == Y[i, j])

    for b3 in basis3:
        constraints.append(sum(y4(*b3, l) for l in range(P)) == y3(*b3))

    # Original convolution
    for k in range(n_diags):
        conv_k = sum(Y[i, k-i] for i in range(max(0, k-P+1), min(P, k+1)))
        constraints.append(2 * P * conv_k <= eta_param)

    # Conv * x_l
    for k in range(n_diags):
        for l in range(P):
            conv_k_l = sum(y3(i, k-i, l) for i in range(max(0, k-P+1), min(P, k+1)))
            constraints.append(2 * P * conv_k_l <= eta_param * x[l])

    # Scalar all (from exp 11)
    for k in range(n_diags):
        for l in range(P):
            for m in range(l, P):
                s = sum(y4(i, k-i, l, m) for i in range(max(0, k-P+1), min(P, k+1)))
                constraints.append(eta_param * Y[l, m] >= 2 * P * s)

    if use_self:
        # Self-product: (x^T A_k x)^2 <= (eta/(2P))^2
        # => sum_{i+j=k, a+b=k} y4[i,j,a,b] <= eta^2/(4P^2)
        for k in range(n_diags):
            pairs_k = [(i, k-i) for i in range(max(0, k-P+1), min(P, k+1))]
            s = sum(y4(i1, j1, i2, j2) for i1, j1 in pairs_k for i2, j2 in pairs_k)
            # eta^2/(4P^2) is quadratic in eta, but eta is a parameter
            constraints.append(4 * P**2 * s <= eta_param * eta_param)
            # Wait, eta_param * eta_param is not DCP. Use: (2P)^2 * s <= eta^2
            # => 2P * sqrt(s) <= eta ... no, not right either.
            # Actually with binary search, eta is FIXED, so eta^2 is a constant.
            # constraints.append(4 * P**2 * s <= eta_param**2)
            # But eta_param**2 is convex, and this is <= convex, which is not DCP.
            # Workaround: since eta is a Parameter, eta**2 is just a number.
            pass  # Will handle below

    if use_cross:
        # Cross-product: (x^T A_{k1} x)(x^T A_{k2} x) <= (eta/(2P))^2
        # => sum_{(i1,j1) in k1, (i2,j2) in k2} y4[i1,j1,i2,j2] <= eta^2/(4P^2)
        # Use symmetry: only k1 <= k2
        for k1 in range(n_diags):
            pairs_k1 = [(i, k1-i) for i in range(max(0, k1-P+1), min(P, k1+1))]
            for k2 in range(k1, n_diags):
                pairs_k2 = [(i, k2-i) for i in range(max(0, k2-P+1), min(P, k2+1))]
                s = sum(y4(i1, j1, i2, j2) for i1, j1 in pairs_k1 for i2, j2 in pairs_k2)
                # This needs eta^2 which is a parameter value
                pass  # Handle via eta_param.value**2

    # For the cross/self-product constraints, since eta is a parameter,
    # we need to pass eta^2 as a parameter too.
    eta2_param = cp.Parameter(nonneg=True)

    if use_self:
        for k in range(n_diags):
            pairs_k = [(i, k-i) for i in range(max(0, k-P+1), min(P, k+1))]
            s = sum(y4(i1, j1, i2, j2) for i1, j1 in pairs_k for i2, j2 in pairs_k)
            constraints.append(4 * P**2 * s <= eta2_param)

    if use_cross:
        for k1 in range(n_diags):
            pairs_k1 = [(i, k1-i) for i in range(max(0, k1-P+1), min(P, k1+1))]
            for k2 in range(k1 + 1, n_diags):
                pairs_k2 = [(i, k2-i) for i in range(max(0, k2-P+1), min(P, k2+1))]
                s = sum(y4(i1, j1, i2, j2) for i1, j1 in pairs_k1 for i2, j2 in pairs_k2)
                constraints.append(4 * P**2 * s <= eta2_param)

    prob = cp.Problem(cp.Minimize(0), constraints)

    shor = 2 * P / (2 * P - 1)
    eta_lo = shor
    eta_hi = 2.0

    for _ in range(30):
        if eta_hi - eta_lo < eta_tol:
            break
        eta_mid = (eta_lo + eta_hi) / 2
        eta_param.value = eta_mid
        eta2_param.value = eta_mid**2
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


print("=" * 80)
print("EXP 14: Cross-diagonal product cuts")
print("=" * 80)

known_lb = {2: 1.777778, 3: 1.706667, 4: 1.644465, 5: 1.632651,
            6: 1.585612, 7: 1.581746, 8: 1.548319}

print(f"{'P':>3} | {'Shor':>8} | {'Self+Cross':>10} | {'Self only':>10} | {'Neither':>10} | {'Lass-2':>10}")
print("-" * 70)

for P in range(2, 8):
    shor = 2 * P / (2 * P - 1)
    both, t1 = solve_cross_diagonal(P, use_cross=True, use_self=True)
    self_only, t2 = solve_cross_diagonal(P, use_cross=False, use_self=True)
    neither, t3 = solve_cross_diagonal(P, use_cross=False, use_self=False)
    lb = known_lb.get(P, float('nan'))
    print(f"{P:>3} | {shor:>8.4f} | {both:>10.6f} | {self_only:>10.6f} | {neither:>10.6f} | {lb:>10.6f} [{t1:.1f}/{t2:.1f}/{t3:.1f}s]")
    if t1 > 60:
        print("  (stopping)")
        break
