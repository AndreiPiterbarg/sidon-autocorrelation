"""Experiment 15: Aggregated localizing matrix.

Instead of requiring L_k PSD for EACH k (expensive: 2P-1 PSD constraints),
aggregate: find weights w_k >= 0 with sum w_k = 1 such that
  L_w = sum_k w_k L_k  is PSD

where L_k[a,b] = eta*Y[a,b] - 2P*sum_{i+j=k} y4[a,b,i,j].

Since L_k PSD for all k => L_w PSD for any w, this is WEAKER than per-k.
But it's MUCH cheaper (1 PSD constraint instead of 2P-1).

The optimal w_k can be found by alternating:
1. Fix w, solve for moments → get lower bound
2. Fix moments, optimize w to get tightest aggregate

Actually, we can make w a variable! If w is a variable:
  sum_k w_k * [eta*Y[a,b] - 2P*sum_{ij:i+j=k} y4[a,b,i,j]] PSD
  = eta * Y[a,b] * (sum w_k) - 2P * sum_k w_k sum_{ij} y4[a,b,i,j]
  = eta * Y[a,b] - 2P * sum_{ij} y4[a,b,i,j] * w_{i+j}

Hmm this is bilinear in (w, y4). Not convex.

Alternative: use w_k = lambda_k from the minimax dual. Or use uniform w.
Or: use a small number of "representative" localizing matrices (e.g., k=P-1
the main diagonal, and k=0, k=2P-2 the corners).

Let's test: how much do we lose by using only a few localizing matrices?
"""
import numpy as np
import cvxpy as cp
import time
import warnings
from itertools import combinations_with_replacement
warnings.filterwarnings('ignore')

SOLVER = 'MOSEK' if 'MOSEK' in cp.installed_solvers() else 'CLARABEL'
print(f"Solver: {SOLVER}")


def solve_sparse_localizing(P, which_k='all', eta_tol=1e-4):
    """Moment-3/4 with a subset of localizing matrices.

    which_k options:
    - 'all': all 2P-1 (baseline)
    - 'center': only k = P-1 (center diagonal)
    - 'edges': k=0, P-1, 2P-2
    - 'sparse3': 3 evenly spaced
    - 'sparse5': 5 evenly spaced
    - 'half': k = 0,...,P-1 (using symmetry)
    """
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

    # Convolution constraints
    for k in range(n_diags):
        conv_k = sum(Y[i, k-i] for i in range(max(0, k-P+1), min(P, k+1)))
        constraints.append(2 * P * conv_k <= eta_param)

    # Conv multiplier (always include — it's cheap)
    for k in range(n_diags):
        for l in range(P):
            conv_k_l = sum(y3(i, k-i, l) for i in range(max(0, k-P+1), min(P, k+1)))
            constraints.append(2 * P * conv_k_l <= eta_param * x[l])

    # Nonneg localizing (always include — relatively cheap)
    for i in range(P):
        L_i = cp.bmat([[y3(i, a, b) for b in range(P)] for a in range(P)])
        constraints.append(L_i >> 0)

    # Select which convolution localizing matrices to include
    if which_k == 'all':
        k_list = list(range(n_diags))
    elif which_k == 'half':
        k_list = list(range(P))
    elif which_k == 'center':
        k_list = [P - 1]
    elif which_k == 'edges':
        k_list = [0, P - 1, n_diags - 1]
    elif which_k == 'sparse3':
        k_list = [0, (n_diags - 1) // 2, n_diags - 1]
    elif which_k == 'sparse5':
        k_list = np.linspace(0, n_diags - 1, 5, dtype=int).tolist()
    elif which_k == 'none':
        k_list = []
    else:
        k_list = list(range(n_diags))

    n_loc = len(k_list)

    for k in k_list:
        L_k = cp.bmat([[
            eta_param * Y[a, b] - 2 * P * sum(
                y4(a, b, i, k - i) for i in range(max(0, k - P + 1), min(P, k + 1)))
            for b in range(P)] for a in range(P)])
        constraints.append(L_k >> 0)

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
    return eta_hi, elapsed, n_loc


print("=" * 80)
print("EXP 15: Sparse/aggregated localizing matrices")
print("=" * 80)

known_lb = {2: 1.777778, 3: 1.706667, 4: 1.644465, 5: 1.632651,
            6: 1.585612, 7: 1.581746, 8: 1.548319}
m3_loc = {2: 1.735514, 3: 1.632812, 4: 1.501011, 5: 1.472873,
          6: 1.424272, 7: 1.406851, 8: 1.377360}

configs = ['none', 'center', 'sparse3', 'sparse5', 'half', 'all']

for P in [3, 5, 7]:
    print(f"\n  P={P} (Shor={2*P/(2*P-1):.4f}, Lass-2={known_lb.get(P, 0):.6f})")
    print(f"  {'Config':>10} | {'Bound':>10} | {'#Loc':>5} | {'Time':>6}")
    print(f"  {'-'*40}")
    for cfg in configs:
        val, t, n_loc = solve_sparse_localizing(P, which_k=cfg)
        print(f"  {cfg:>10} | {val:>10.6f} | {n_loc:>5} | {t:>5.1f}s")
