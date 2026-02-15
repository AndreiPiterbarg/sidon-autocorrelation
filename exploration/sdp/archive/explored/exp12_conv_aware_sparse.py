"""Experiment 12: Convolution-aware sparse basis for mini-Lasserre.

Instead of diagonal basis or bandwidth-based, choose degree-2 monomials
based on which are most relevant to the convolution structure.

For diagonal k, the convolution involves x_i * x_{k-i}. So the most
relevant degree-2 monomials are those that appear in the convolution.

Strategy: include x_i * x_j in the basis if |i+j - (P-1)| is small,
i.e., the pair contributes to the central (most constrained) diagonals.

Also try: include pairs that appear in the ACTIVE diagonals from the
primal solution (from Exp 5: 3-5 diagonals are typically active).
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


def softmax(z):
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)


def find_active_diags(P, n_top=5):
    """Find the most active diagonals from primal optimization."""
    n_diags = 2 * P - 1
    A_mats = []
    for k in range(n_diags):
        A = np.zeros((P, P))
        for i in range(max(0, k-P+1), min(P, k+1)):
            A[i, k-i] = 1
        A_mats.append(A)

    best_val = np.inf
    best_x = None
    for seed in range(30):
        z0 = np.random.RandomState(seed).randn(P) * 0.5
        for beta in [5, 20, 100, 500]:
            def obj(z, b=beta):
                x = softmax(z)
                conv = np.convolve(x, x, mode='full')
                mx = np.max(conv)
                return 2*P*(mx + (1/b)*np.log(np.sum(np.exp(b*(conv-mx)))))
            res = minimize(obj, z0, method='L-BFGS-B', options={'maxiter': 300})
            z0 = res.x
        def exact(z):
            x = softmax(z)
            return 2*P*np.max(np.convolve(x, x, mode='full'))
        res = minimize(exact, z0, method='L-BFGS-B', options={'maxiter': 500})
        if res.fun < best_val:
            best_val = res.fun
            best_x = softmax(res.x)

    diag_vals = np.array([best_x @ A @ best_x for A in A_mats])
    top_k = np.argsort(diag_vals)[-n_top:]
    return sorted(top_k), best_val, best_x


def solve_conv_aware(P, extra_pairs=None, eta_tol=1e-4):
    """Mini-Lasserre with convolution-aware basis.

    Always include diagonal pairs (i,i). Then add extra_pairs.
    """
    t0 = time.time()
    n_diags = 2 * P - 1

    basis = [()]
    for i in range(P):
        basis.append((i,))
    # Diagonal
    for i in range(P):
        basis.append((i, i))
    # Extra pairs
    if extra_pairs:
        for i, j in extra_pairs:
            if i != j:
                pair = (min(i,j), max(i,j))
                if pair not in [(b[0], b[1]) if len(b) == 2 else (None, None) for b in basis]:
                    basis.append(pair)
    d = len(basis)

    loc_basis = [()]
    for i in range(P):
        loc_basis.append((i,))
    loc_d = len(loc_basis)

    # Moments
    moments_set = set()
    for a in range(d):
        for b in range(a, d):
            moments_set.add(tuple(sorted(basis[a] + basis[b])))
    for i in range(P):
        for a in range(loc_d):
            for b in range(a, loc_d):
                moments_set.add(tuple(sorted((i,) + loc_basis[a] + loc_basis[b])))
    for k in range(n_diags):
        for a in range(loc_d):
            for b in range(a, loc_d):
                mu_base = tuple(sorted(loc_basis[a] + loc_basis[b]))
                moments_set.add(mu_base)
                for i in range(max(0, k-P+1), min(P, k+1)):
                    j = k - i
                    moments_set.add(tuple(sorted(mu_base + (i, j))))
    for deg in range(5):
        for beta in combinations_with_replacement(range(P), deg):
            beta = tuple(sorted(beta))
            moments_set.add(beta)
            for i in range(P):
                moments_set.add(tuple(sorted(beta + (i,))))
    for i in range(P):
        for j in range(i, P):
            moments_set.add((i, j))

    moments_list = sorted(moments_set, key=lambda m: (len(m), m))
    n_mom = len(moments_list)
    moment_idx = {m: i for i, m in enumerate(moments_list)}

    y = cp.Variable(n_mom)
    eta_param = cp.Parameter(nonneg=True)
    constraints = [y[moment_idx[()]] == 1]

    for m in moments_list:
        if len(m) >= 1:
            constraints.append(y[moment_idx[m]] >= 0)

    # Moment matrix PSD
    B_M = {}
    for a in range(d):
        for b in range(a, d):
            mu = tuple(sorted(basis[a] + basis[b]))
            idx = moment_idx[mu]
            if idx not in B_M:
                B_M[idx] = np.zeros((d, d))
            B_M[idx][a, b] += 1
            if a != b:
                B_M[idx][b, a] += 1
    constraints.append(sum(y[idx] * mat for idx, mat in B_M.items()) >> 0)

    # Simplex
    for deg in range(4):
        for beta in combinations_with_replacement(range(P), deg):
            beta = tuple(sorted(beta))
            if beta not in moment_idx:
                continue
            lhs = []
            for i in range(P):
                mu = tuple(sorted(beta + (i,)))
                if mu in moment_idx:
                    lhs.append(moment_idx[mu])
            if lhs:
                constraints.append(sum(y[i] for i in lhs) == y[moment_idx[beta]])

    # Nonneg localizing
    for i in range(P):
        B_Li = {}
        for a in range(loc_d):
            for b in range(a, loc_d):
                mu = tuple(sorted((i,) + loc_basis[a] + loc_basis[b]))
                if mu in moment_idx:
                    idx = moment_idx[mu]
                    if idx not in B_Li:
                        B_Li[idx] = np.zeros((loc_d, loc_d))
                    B_Li[idx][a, b] += 1
                    if a != b:
                        B_Li[idx][b, a] += 1
        if B_Li:
            constraints.append(sum(y[idx] * mat for idx, mat in B_Li.items()) >> 0)

    # Convolution + localizing
    for k in range(n_diags):
        conv_k = sum(y[moment_idx[tuple(sorted((i, k-i)))]]
                     for i in range(max(0, k-P+1), min(P, k+1)))
        constraints.append(2 * P * conv_k <= eta_param)

    B_M1 = {}
    for a in range(loc_d):
        for b in range(a, loc_d):
            mu = tuple(sorted(loc_basis[a] + loc_basis[b]))
            if mu in moment_idx:
                idx = moment_idx[mu]
                if idx not in B_M1:
                    B_M1[idx] = np.zeros((loc_d, loc_d))
                B_M1[idx][a, b] += 1
                if a != b:
                    B_M1[idx][b, a] += 1
    M1_expr = sum(y[idx] * mat for idx, mat in B_M1.items())

    for k in range(n_diags):
        B_pk = {}
        for a in range(loc_d):
            for b in range(a, loc_d):
                for i in range(max(0, k-P+1), min(P, k+1)):
                    j = k - i
                    mu = tuple(sorted(loc_basis[a] + loc_basis[b] + (i, j)))
                    if mu in moment_idx:
                        idx = moment_idx[mu]
                        if idx not in B_pk:
                            B_pk[idx] = np.zeros((loc_d, loc_d))
                        B_pk[idx][a, b] += 1
                        if a != b:
                            B_pk[idx][b, a] += 1
        if B_pk:
            pk_expr = sum(y[idx] * mat for idx, mat in B_pk.items())
            constraints.append(eta_param * M1_expr - 2 * P * pk_expr >> 0)

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
    return eta_hi, elapsed, d


print("=" * 80)
print("EXP 12: Convolution-aware sparse basis")
print("=" * 80)

known_lb = {5: 1.632651, 7: 1.581746, 8: 1.548319, 10: 1.524823}

for P in [5, 7, 8, 10]:
    print(f"\n  P={P} (Shor={2*P/(2*P-1):.4f}, Lasserre-2={known_lb.get(P, 0):.6f})")

    # Find active diagonals
    active_k, vp, xstar = find_active_diags(P)
    print(f"  Active diags: {active_k}, V(P)~{vp:.6f}")

    # Strategies for extra pairs
    strategies = {
        'diag_only': [],
        'active_pairs': [(i, k-i) for k in active_k
                         for i in range(max(0, k-P+1), min(P, k+1)) if i != k-i],
        'center_pairs': [(i, P-1-i) for i in range(P//2)],  # central diagonal
        'near_diag': [(i, i+1) for i in range(P-1)],  # bandwidth-1
        'conv_pairs': [(i, j) for k in range(P-1, min(P+2, 2*P-1))  # k near P-1
                        for i in range(max(0, k-P+1), min(P, k+1))
                        for j in [k-i] if i < j],
    }

    print(f"  {'Strategy':>15} | {'Bound':>10} | {'d':>4} | {'Time':>6}")
    print(f"  {'-'*45}")
    for name, pairs in strategies.items():
        val, t, d = solve_conv_aware(P, extra_pairs=pairs)
        print(f"  {name:>15} | {val:>10.6f} | {d:>4} | {t:>5.1f}s")
