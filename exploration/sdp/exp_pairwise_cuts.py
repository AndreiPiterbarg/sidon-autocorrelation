"""Experiment: Pairwise simplex cuts x_i(1-x_i-x_j) >= 0.

These are stronger than x_i(1-x_i) >= 0 because they capture pairwise
correlations on the simplex. For P variables on the simplex,
x_i + x_j <= 1 for all i != j (since all x_k >= 0 and sum = 1).

So g_{ij}(x) = x_i(1 - x_i - x_j) >= 0 for all i != j.

The localizing matrix for g_{ij} is degree 2 in x, so fits within
the Level-2 moment framework (degree-4 moments).

Expected: stronger than simplex cuts at P=5-8 where solver is fast.
"""

import numpy as np
import cvxpy as cp
import time
import warnings
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from exploration.sdp.baseline_results import BASELINE, compare_with_baseline
from exploration.sdp.core_utils import (
    monomial_basis, combine, canonical_index, build_symmetry_map, PRIMARY_SOLVER
)


def build_enhanced_problem(P, use_simplex_cuts=True, use_pairwise=True):
    """Build Lasserre Level-2 with simplex and pairwise cuts."""
    t0 = time.time()

    basis_2 = monomial_basis(P, 2)
    d = len(basis_2)
    basis_1 = monomial_basis(P, 1)
    loc_d = len(basis_1)
    all_beta = monomial_basis(P, 3)

    # Collect moments
    moments_set = set()
    for i in range(d):
        for j in range(i, d):
            moments_set.add(combine(basis_2[i], basis_2[j]))
    for k in range(P):
        for a in range(loc_d):
            for b in range(a, loc_d):
                moments_set.add(combine((k,), basis_1[a], basis_1[b]))
    for beta in all_beta:
        moments_set.add(beta)
        for i in range(P):
            moments_set.add(combine((i,), beta))
    for k_conv in range(2 * P - 1):
        for a in range(loc_d):
            for b in range(a, loc_d):
                moments_set.add(combine(basis_1[a], basis_1[b]))
                for i in range(P):
                    j_val = k_conv - i
                    if 0 <= j_val < P:
                        moments_set.add(combine(basis_1[a], basis_1[b], (i,), (j_val,)))
    for i in range(P):
        for j in range(i + 1, P):
            for a in range(loc_d):
                for b in range(a, loc_d):
                    moments_set.add(combine((i,), (j,), basis_1[a], basis_1[b]))
    # Simplex and pairwise cuts use same degree-4 moments
    for i in range(P):
        for a in range(loc_d):
            for b in range(a, loc_d):
                moments_set.add(combine((i,), (i,), basis_1[a], basis_1[b]))
    if use_pairwise:
        for i in range(P):
            for j in range(P):
                if i != j:
                    for a in range(loc_d):
                        for b in range(a, loc_d):
                            moments_set.add(combine((i,), (j,), basis_1[a], basis_1[b]))

    moments_list = sorted(moments_set, key=lambda m: (len(m), m))

    canon_moments, canon_idx_map, full_map, equiv_classes = \
        build_symmetry_map(moments_list, P)
    n_mom = len(canon_moments)
    moment_idx = full_map
    for m, idx in canon_idx_map.items():
        moment_idx[m] = idx

    # Build indicator matrices (standard)
    B_M = {}
    for i in range(d):
        for j in range(i, d):
            mu = combine(basis_2[i], basis_2[j])
            idx = moment_idx[mu]
            if idx not in B_M:
                B_M[idx] = np.zeros((d, d))
            B_M[idx][i, j] += 1
            if i != j:
                B_M[idx][j, i] += 1

    nonneg_indices = list(range((P + 1) // 2))
    B_Locs = []
    for k in nonneg_indices:
        B_L = {}
        for a in range(loc_d):
            for b in range(a, loc_d):
                mu = combine((k,), basis_1[a], basis_1[b])
                idx = moment_idx[mu]
                if idx not in B_L:
                    B_L[idx] = np.zeros((loc_d, loc_d))
                B_L[idx][a, b] += 1
                if a != b:
                    B_L[idx][b, a] += 1
        B_Locs.append(B_L)

    B_prod_Locs = []
    for i in range(P):
        for j in range(i + 1, P):
            i2, j2 = P - 1 - j, P - 1 - i
            if (i2, j2) < (i, j):
                continue
            B_pL = {}
            for a in range(loc_d):
                for b in range(a, loc_d):
                    mu = combine((i,), (j,), basis_1[a], basis_1[b])
                    idx = moment_idx[mu]
                    if idx not in B_pL:
                        B_pL[idx] = np.zeros((loc_d, loc_d))
                    B_pL[idx][a, b] += 1
                    if a != b:
                        B_pL[idx][b, a] += 1
            B_prod_Locs.append(B_pL)

    # Simplex cuts: x_i(1-x_i) = x_i - x_i^2
    B_simplex = []
    if use_simplex_cuts:
        for k in nonneg_indices:
            B_gi = {}
            for a in range(loc_d):
                for b in range(a, loc_d):
                    mu1 = combine((k,), basis_1[a], basis_1[b])
                    idx1 = moment_idx[mu1]
                    if idx1 not in B_gi:
                        B_gi[idx1] = np.zeros((loc_d, loc_d))
                    B_gi[idx1][a, b] += 1
                    if a != b:
                        B_gi[idx1][b, a] += 1
                    mu2 = combine((k,), (k,), basis_1[a], basis_1[b])
                    idx2 = moment_idx[mu2]
                    if idx2 not in B_gi:
                        B_gi[idx2] = np.zeros((loc_d, loc_d))
                    B_gi[idx2][a, b] -= 1
                    if a != b:
                        B_gi[idx2][b, a] -= 1
            B_simplex.append(B_gi)

    # Pairwise cuts: x_i(1-x_i-x_j) = x_i - x_i^2 - x_i*x_j
    B_pairwise = []
    if use_pairwise:
        for i in nonneg_indices:
            for j in range(P):
                if i == j:
                    continue
                i2, j2 = P - 1 - i, P - 1 - j
                if i2 in nonneg_indices and (i2, j2) < (i, j):
                    continue
                B_gij = {}
                for a in range(loc_d):
                    for b in range(a, loc_d):
                        mu1 = combine((i,), basis_1[a], basis_1[b])
                        idx1 = moment_idx[mu1]
                        if idx1 not in B_gij:
                            B_gij[idx1] = np.zeros((loc_d, loc_d))
                        B_gij[idx1][a, b] += 1
                        if a != b:
                            B_gij[idx1][b, a] += 1
                        mu2 = combine((i,), (i,), basis_1[a], basis_1[b])
                        idx2 = moment_idx[mu2]
                        if idx2 not in B_gij:
                            B_gij[idx2] = np.zeros((loc_d, loc_d))
                        B_gij[idx2][a, b] -= 1
                        if a != b:
                            B_gij[idx2][b, a] -= 1
                        mu3 = combine((i,), (j,), basis_1[a], basis_1[b])
                        idx3 = moment_idx[mu3]
                        if idx3 not in B_gij:
                            B_gij[idx3] = np.zeros((loc_d, loc_d))
                        B_gij[idx3][a, b] -= 1
                        if a != b:
                            B_gij[idx3][b, a] -= 1
                B_pairwise.append(B_gij)

    simplex_data = []
    for beta in all_beta:
        lhs_indices = [moment_idx[combine((i,), beta)] for i in range(P)]
        rhs_idx = moment_idx[beta]
        simplex_data.append((lhs_indices, rhs_idx))

    B_M1 = {}
    for a in range(loc_d):
        for b in range(a, loc_d):
            mu = combine(basis_1[a], basis_1[b])
            idx = moment_idx[mu]
            if idx not in B_M1:
                B_M1[idx] = np.zeros((loc_d, loc_d))
            B_M1[idx][a, b] += 1
            if a != b:
                B_M1[idx][b, a] += 1

    conv_indices = list(range(P))
    B_pks = []
    for k_conv in conv_indices:
        B_pk = {}
        for a in range(loc_d):
            for b in range(a, loc_d):
                for i in range(P):
                    j_val = k_conv - i
                    if 0 <= j_val < P:
                        mu = combine(basis_1[a], basis_1[b], (i,), (j_val,))
                        idx = moment_idx[mu]
                        if idx not in B_pk:
                            B_pk[idx] = np.zeros((loc_d, loc_d))
                        B_pk[idx][a, b] += 1
                        if a != b:
                            B_pk[idx][b, a] += 1
        B_pks.append(B_pk)

    # Build CVXPY problem
    y = cp.Variable(n_mom)
    eta_param = cp.Parameter(nonneg=True)
    constraints = []

    M_expr = sum(y[idx] * mat for idx, mat in B_M.items())
    constraints.append(M_expr >> 0)
    constraints.append(y[moment_idx[()]] == 1)

    for B_L in B_Locs:
        constraints.append(sum(y[idx] * mat for idx, mat in B_L.items()) >> 0)
    for B_pL in B_prod_Locs:
        constraints.append(sum(y[idx] * mat for idx, mat in B_pL.items()) >> 0)
    for B_gi in B_simplex:
        constraints.append(sum(y[idx] * mat for idx, mat in B_gi.items()) >> 0)
    for B_gij in B_pairwise:
        constraints.append(sum(y[idx] * mat for idx, mat in B_gij.items()) >> 0)

    for lhs_indices, rhs_idx in simplex_data:
        constraints.append(sum(y[i] for i in lhs_indices) == y[rhs_idx])

    M1_expr = sum(y[idx] * mat for idx, mat in B_M1.items())
    for B_pk in B_pks:
        pk_expr = sum(y[idx] * mat for idx, mat in B_pk.items())
        constraints.append(eta_param * M1_expr - 2 * P * pk_expr >> 0)

    prob = cp.Problem(cp.Minimize(0), constraints)
    setup_time = time.time() - t0

    return {
        'prob': prob, 'y': y, 'eta_param': eta_param,
        'P': P, 'd': d, 'n_mom': n_mom, 'setup_time': setup_time,
        'n_products': len(B_prod_Locs), 'n_simplex': len(B_simplex),
        'n_pairwise': len(B_pairwise),
    }


def solve_binary_search(pd, eta_lo, eta_hi, eta_tol=1e-6, max_time=300):
    """Binary search."""
    prob = pd['prob']
    y = pd['y']
    eta_param = pd['eta_param']
    t0 = time.time()
    best_y = None
    n_iter = 0

    def try_solve():
        try:
            kwargs = {'verbose': False}
            if PRIMARY_SOLVER == 'MOSEK':
                kwargs['mosek_params'] = {
                    'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-9,
                    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-9,
                    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-9,
                }
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Solution may be inaccurate')
                prob.solve(solver=PRIMARY_SOLVER, warm_start=True, **kwargs)
            return prob.status
        except Exception:
            return 'solver_error'

    eta_param.value = eta_hi
    status = try_solve()
    if status not in ('optimal', 'optimal_inaccurate'):
        return None, time.time() - t0, 0

    best_y = y.value.copy()

    while eta_hi - eta_lo > eta_tol and time.time() - t0 < max_time:
        eta_mid = (eta_lo + eta_hi) / 2
        eta_param.value = eta_mid
        status = try_solve()
        n_iter += 1
        if status in ('optimal', 'optimal_inaccurate'):
            eta_hi = eta_mid
            best_y = y.value.copy()
        else:
            eta_lo = eta_mid

    return eta_hi, time.time() - t0, n_iter


if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("EXPERIMENT: Pairwise Simplex Cuts x_i(1-x_i-x_j) >= 0", flush=True)
    print("=" * 80, flush=True)

    # Compare at P=5-8: base+products vs +simplex vs +pairwise
    print("\n--- Comparison at P=5-8 ---", flush=True)
    for P in range(5, 9):
        primal_ub = BASELINE[P]['primal_ub']
        shor = 2 * P / (2 * P - 1)

        # Simplex only
        print(f"\nP={P}:", flush=True)
        pd_s = build_enhanced_problem(P, use_simplex_cuts=True, use_pairwise=False)
        print(f"  Simplex only: n_mom={pd_s['n_mom']}, n_simplex={pd_s['n_simplex']}, "
              f"setup={pd_s['setup_time']:.1f}s", flush=True)
        b_s, t_s, n_s = solve_binary_search(pd_s, shor, primal_ub + 0.001)
        if b_s:
            compare_with_baseline(P, b_s, pd_s['setup_time'] + t_s, 'L2+simplex')

        # Pairwise
        pd_p = build_enhanced_problem(P, use_simplex_cuts=True, use_pairwise=True)
        print(f"  +Pairwise: n_mom={pd_p['n_mom']}, n_pairwise={pd_p['n_pairwise']}, "
              f"setup={pd_p['setup_time']:.1f}s", flush=True)
        b_p, t_p, n_p = solve_binary_search(pd_p, shor, primal_ub + 0.001)
        if b_p:
            compare_with_baseline(P, b_p, pd_p['setup_time'] + t_p, 'L2+pairwise')

        if b_s and b_p:
            diff = b_p - b_s
            print(f"  Pairwise improvement: {diff:+.8f}", flush=True)
