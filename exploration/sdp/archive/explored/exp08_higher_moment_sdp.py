"""Experiment 8: SDP-certified higher moment bounds.

From Exp 5: max_k q_k >= (sum q_k^p)^{1/(p-1)} where q_k = x^T A_k x.

For p=2: sum q_k^2 = sum_k (x^T A_k x)^2.
This is a degree-4 polynomial in x. We can get a CERTIFIED lower bound
on min_{x in simplex} sum q_k^2 using Lasserre or SOS.

The bound then gives: V(P) >= 2P * (min S2)^{1/1} = 2P * min_S2.
For p=3: sum q_k^3 is degree 6, harder.

KEY INSIGHT: sum q_k^2 = sum_k sum_{i+j=k,a+b=k} x_i x_j x_a x_b.
This can be written as x^{otimes 2} M x^{otimes 2} where M encodes the
cross-diagonal structure. We can bound this with Lasserre Level-2!

Actually, sum q_k^2 = sum_{i+j = a+b} x_i x_j x_a x_b.
This is a quartic form. We want min over the simplex.

Shor relaxation for this: replace x_i x_j x_a x_b with degree-4 moment
variables, subject to PSD moment matrix constraint.

But this is expensive. Let me try a simpler approach first:
can we get an ANALYTIC lower bound on sum q_k^2?

Claim: sum q_k^2 >= 1/(2P-1) + delta(P) for some computable delta > 0.
By Cauchy-Schwarz on {q_k}: sum q_k^2 >= (sum q_k)^2 / (2P-1) = 1/(2P-1).
Equality iff all q_k equal. But can all q_k be equal on the simplex?
If all x^T A_k x = 1/(2P-1), the resulting x would be a specific point.
Let's check if this is achievable.
"""
import numpy as np
import cvxpy as cp
import time
import warnings
from itertools import combinations_with_replacement
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

SOLVER = 'MOSEK' if 'MOSEK' in cp.installed_solvers() else 'CLARABEL'
print(f"Solver: {SOLVER}")


def build_A(P):
    n_diags = 2 * P - 1
    A = []
    for k in range(n_diags):
        Ak = np.zeros((P, P))
        for i in range(max(0, k - P + 1), min(P, k + 1)):
            Ak[i, k - i] = 1
        A.append(Ak)
    return A


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


print("=" * 72)
print("EXP 8: SDP-certified higher moment bounds")
print("=" * 72)


# Part 1: Check if equal diagonal values are achievable
print("\n--- Part 1: Can all q_k be equal? ---")
for P in [3, 5, 8, 10]:
    A_mats = build_A(P)
    n_diags = 2 * P - 1
    target = 1.0 / n_diags

    # Try to find x on simplex with all q_k = target
    def deviation(z):
        x = softmax(z)
        qs = np.array([x @ A @ x for A in A_mats])
        return np.sum((qs - target)**2)

    best = np.inf
    for seed in range(50):
        z0 = np.random.RandomState(seed).randn(P) * 0.3
        res = minimize(deviation, z0, method='L-BFGS-B', options={'maxiter': 1000})
        if res.fun < best:
            best = res.fun
    print(f"  P={P}: min deviation from uniform = {best:.2e} "
          f"({'ACHIEVABLE' if best < 1e-10 else 'NOT achievable'})")


# Part 2: SDP lower bound on sum q_k^2 (Shor-level)
print("\n--- Part 2: Shor-level SDP for min sum q_k^2 ---")

def sdp_min_S2(P, eta_tol=1e-5):
    """Lower bound on min_{x in simplex} sum_k (x^T A_k x)^2 using moment relaxation.

    sum q_k^2 = sum_k (sum_{i+j=k} y_{ij})^2  where y_{ij} = x_i x_j.

    In terms of degree-4 moments:
    sum q_k^2 = sum_k sum_{i+j=k, a+b=k} y_{ijab}
    where y_{ijab} = E[x_i x_j x_a x_b].

    We need degree-4 moments. Use: Shor on degree-2 reformulation.
    Let z_k = x^T A_k x (linear in Y). Then sum z_k^2 is convex in z.
    min sum z_k^2 s.t. z_k = tr(A_k Y), [1,x;x,Y] PSD, simplex constraints.
    """
    t0 = time.time()
    A_mats = build_A(P)
    n_diags = 2 * P - 1

    x = cp.Variable(P, nonneg=True)
    Y = cp.Variable((P, P), symmetric=True)

    # Moment matrix PSD
    M = cp.bmat([[np.ones((1, 1)), cp.reshape(x, (1, P))],
                  [cp.reshape(x, (P, 1)), Y]])
    constraints = [M >> 0, cp.sum(x) == 1, Y >= 0]
    for i in range(P):
        constraints.append(cp.sum(Y[i, :]) == x[i])

    # z_k = tr(A_k @ Y) = sum_{i+j=k} Y[i,j]
    z = []
    for k in range(n_diags):
        zk = sum(Y[i, k-i] for i in range(max(0, k-P+1), min(P, k+1)))
        z.append(zk)

    # Minimize sum z_k^2 (this is convex! sum of squares of linear functions)
    obj = sum(zk**2 for zk in z)
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=SOLVER, verbose=False)

    elapsed = time.time() - t0
    val = float(prob.value) if prob.value is not None else None
    return val, elapsed


for P in range(2, 16):
    s2_lb, t_s2 = sdp_min_S2(P)
    n_diags = 2 * P - 1
    uniform_s2 = 1.0 / n_diags
    shor = 2 * P / (2 * P - 1)

    if s2_lb is not None:
        cs_lb = 2 * P * s2_lb  # V(P) >= 2P * S2
        ratio = s2_lb / uniform_s2
        print(f"  P={P:2d}: SDP min S2 = {s2_lb:.8f}, uniform = {uniform_s2:.8f}, "
              f"ratio = {ratio:.6f}, CS bound = {cs_lb:.6f} (Shor={shor:.6f}) "
              f"[{t_s2:.1f}s]")
    else:
        print(f"  P={P:2d}: FAILED")


# Part 3: Full quartic moment relaxation for sum q_k^2
print("\n--- Part 3: Quartic moment relaxation for min sum q_k^2 ---")
print("Using degree-4 moments with consistency constraints.")

def sdp_quartic_S2(P, eta_tol=1e-5):
    """Better lower bound using degree-4 moments directly.

    min sum_k (sum_{i+j=k} sum_{a+b=k} y4_{ijab})
    s.t. consistency, PSD, simplex constraints.
    """
    t0 = time.time()
    A_mats = build_A(P)
    n_diags = 2 * P - 1

    x = cp.Variable(P, nonneg=True)
    Y = cp.Variable((P, P), symmetric=True)

    # Degree-3 moments
    basis3 = list(combinations_with_replacement(range(P), 3))
    y3_vars = cp.Variable(len(basis3), nonneg=True)
    y3_idx = {tuple(b): k for k, b in enumerate(basis3)}
    def y3(i, j, l):
        return y3_vars[y3_idx[tuple(sorted([i, j, l]))]]

    # Degree-4 moments
    basis4 = list(combinations_with_replacement(range(P), 4))
    y4_vars = cp.Variable(len(basis4), nonneg=True)
    y4_idx = {tuple(b): k for k, b in enumerate(basis4)}
    def y4(i, j, k, l):
        return y4_vars[y4_idx[tuple(sorted([i, j, k, l]))]]

    # PSD moment matrix
    M = cp.bmat([[np.ones((1, 1)), cp.reshape(x, (1, P))],
                  [cp.reshape(x, (P, 1)), Y]])
    constraints = [M >> 0, cp.sum(x) == 1, Y >= 0]

    # Simplex
    for i in range(P):
        constraints.append(cp.sum(Y[i, :]) == x[i])

    # Consistency y3 -> Y
    for i in range(P):
        for j in range(i, P):
            constraints.append(sum(y3(i, j, l) for l in range(P)) == Y[i, j])

    # Consistency y4 -> y3
    for b3 in basis3:
        constraints.append(sum(y4(*b3, l) for l in range(P)) == y3(*b3))

    # x_i^2 <= x_i
    for i in range(P):
        for j in range(P):
            constraints.append(y3(i, i, j) <= Y[min(i,j), max(i,j)])

    # Nonneg localizing: y3[i,a,b] PSD as (a,b) matrix for each i
    for i in range(P):
        L_i = cp.bmat([[y3(i, a, b) for b in range(P)] for a in range(P)])
        constraints.append(L_i >> 0)

    # Product localizing: y4[i,j,a,b] PSD for each (i,j)
    for i in range(P):
        for j in range(i, P):
            L_ij = cp.bmat([[y4(i, j, a, b) for b in range(P)] for a in range(P)])
            constraints.append(L_ij >> 0)

    # Degree-2 moment matrix with Y as the core:
    # M2[alpha, beta] = y_{alpha+beta} for |alpha|,|beta| <= 2
    # This would be the full Lasserre Level-2 matrix. Too expensive to build here.
    # Instead, just use the above constraints.

    # Objective: min sum_k (sum_{i+j=k, a+b=k} y4[i,j,a,b])
    obj = 0
    for k in range(n_diags):
        pairs = [(i, k-i) for i in range(max(0, k-P+1), min(P, k+1))]
        for i1, j1 in pairs:
            for i2, j2 in pairs:
                obj = obj + y4(i1, j1, i2, j2)

    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=SOLVER, verbose=False)
    elapsed = time.time() - t0
    val = float(prob.value) if prob.value is not None else None
    return val, elapsed


for P in range(2, 10):
    s2_q, t_q = sdp_quartic_S2(P)
    n_diags = 2 * P - 1
    uniform_s2 = 1.0 / n_diags
    shor = 2 * P / (2 * P - 1)

    if s2_q is not None:
        cs_lb = 2 * P * s2_q
        ratio = s2_q / uniform_s2
        print(f"  P={P:2d}: quartic min S2 = {s2_q:.8f}, CS bound = {cs_lb:.6f} "
              f"(Shor={shor:.6f}, ratio={ratio:.6f}) [{t_q:.1f}s]")
    else:
        print(f"  P={P:2d}: FAILED")
    if t_q > 60:
        print("  (stopping)")
        break
