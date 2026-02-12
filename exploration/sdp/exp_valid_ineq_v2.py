"""Experiment 3v2: Valid Inequalities — simplified and robust.

Test simplex-specific cuts at P=5..8 where the relaxation gap emerges.
Uses core_utils.solve_lasserre_2_v2 with added x_i(1-x_i)>=0 constraints.

Approach: Instead of rebuilding the entire solver, modify the binary search
loop to add extra localizing constraints.
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
    monomial_basis, combine, canonical_index, build_symmetry_map,
    solve_primal_improved, PRIMARY_SOLVER
)


def solve_with_simplex_cuts(P, primal_hint=None, eta_tol=1e-6):
    """Lasserre Level-2 with x_i(1-x_i) >= 0 simplex cuts.

    x_i(1-x_i) = x_i * sum_{j!=i} x_j >= 0 on the simplex.
    The localizing matrix for g_i = x_i - x_i^2 is:
    L_gi[a,b] = E[g_i * b_a * b_b] = E[x_i * b_a * b_b] - E[x_i^2 * b_a * b_b]
    """
    t0 = time.time()

    basis_2 = monomial_basis(P, 2)
    d = len(basis_2)
    basis_1 = monomial_basis(P, 1)
    loc_d = len(basis_1)
    all_beta = monomial_basis(P, 3)

    # Collect all moments needed
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
    # Products
    for i in range(P):
        for j in range(i + 1, P):
            for a in range(loc_d):
                for b in range(a, loc_d):
                    moments_set.add(combine((i,), (j,), basis_1[a], basis_1[b]))
    # Simplex cuts: x_i^2 * basis_1[a] * basis_1[b] — already degree 4, should be covered
    # but let's make sure
    for i in range(P):
        for a in range(loc_d):
            for b in range(a, loc_d):
                moments_set.add(combine((i,), (i,), basis_1[a], basis_1[b]))

    moments_list = sorted(moments_set, key=lambda m: (len(m), m))

    # Symmetry
    canon_moments, canon_idx_map, full_map, equiv_classes = \
        build_symmetry_map(moments_list, P)
    n_mom = len(canon_moments)
    moment_idx = full_map
    for m, idx in canon_idx_map.items():
        moment_idx[m] = idx

    print(f"  P={P}: d={d}, n_moments={n_mom} (sym) +simplex_cuts")

    # Build indicator matrices
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

    # Product constraints (symmetry-reduced)
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

    # Simplex cut: g_i = x_i(1-x_i) = x_i - x_i^2
    B_simplex_cuts = []
    for k in nonneg_indices:
        B_gi = {}
        for a in range(loc_d):
            for b in range(a, loc_d):
                # +x_k term
                mu1 = combine((k,), basis_1[a], basis_1[b])
                idx1 = moment_idx[mu1]
                if idx1 not in B_gi:
                    B_gi[idx1] = np.zeros((loc_d, loc_d))
                B_gi[idx1][a, b] += 1
                if a != b:
                    B_gi[idx1][b, a] += 1

                # -x_k^2 term
                mu2 = combine((k,), (k,), basis_1[a], basis_1[b])
                idx2 = moment_idx[mu2]
                if idx2 not in B_gi:
                    B_gi[idx2] = np.zeros((loc_d, loc_d))
                B_gi[idx2][a, b] -= 1
                if a != b:
                    B_gi[idx2][b, a] -= 1
        B_simplex_cuts.append(B_gi)

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
        L_k = sum(y[idx] * mat for idx, mat in B_L.items())
        constraints.append(L_k >> 0)

    for B_pL in B_prod_Locs:
        L_ij = sum(y[idx] * mat for idx, mat in B_pL.items())
        constraints.append(L_ij >> 0)

    # EXTRA: simplex cuts
    for B_gi in B_simplex_cuts:
        L_gi = sum(y[idx] * mat for idx, mat in B_gi.items())
        constraints.append(L_gi >> 0)

    for lhs_indices, rhs_idx in simplex_data:
        constraints.append(sum(y[i] for i in lhs_indices) == y[rhs_idx])

    M1_expr = sum(y[idx] * mat for idx, mat in B_M1.items())
    for B_pk in B_pks:
        pk_expr = sum(y[idx] * mat for idx, mat in B_pk.items())
        L_gk = eta_param * M1_expr - 2 * P * pk_expr
        constraints.append(L_gk >> 0)

    prob = cp.Problem(cp.Minimize(0), constraints)

    def try_solve(tight=True):
        try:
            kwargs = {'verbose': False}
            if PRIMARY_SOLVER == 'MOSEK':
                if tight:
                    kwargs['mosek_params'] = {
                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-10,
                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-10,
                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-10,
                        'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-12,
                    }
                else:
                    kwargs['mosek_params'] = {
                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-7,
                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-7,
                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-7,
                    }
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Solution may be inaccurate')
                prob.solve(solver=PRIMARY_SOLVER, warm_start=True, **kwargs)
            return prob.status
        except Exception:
            return 'solver_error'

    # Binary search
    shor_bound = 2 * P / (2 * P - 1)
    eta_lo = shor_bound
    eta_hi = (primal_hint + 0.001) if primal_hint is not None else 2.0

    best_y = None
    n_iter = 0

    eta_param.value = eta_hi
    status = try_solve()
    if status not in ('optimal', 'optimal_inaccurate'):
        eta_hi = 2.0 * P
        eta_param.value = eta_hi
        status = try_solve()
        if status not in ('optimal', 'optimal_inaccurate'):
            return {'P': P, 'bound': None, 'status': status, 'time': time.time() - t0}
    best_y = y.value.copy()

    while eta_hi - eta_lo > 1e-3:
        eta_mid = (eta_lo + eta_hi) / 2
        eta_param.value = eta_mid
        status = try_solve()
        n_iter += 1
        if status in ('optimal', 'optimal_inaccurate'):
            eta_hi = eta_mid
            best_y = y.value.copy()
        else:
            status2 = try_solve(tight=False)
            if status2 in ('optimal', 'optimal_inaccurate'):
                eta_hi = eta_mid
                best_y = y.value.copy()
            else:
                eta_lo = eta_mid

    while eta_hi - eta_lo > eta_tol:
        eta_mid = (eta_lo + eta_hi) / 2
        eta_param.value = eta_mid
        status = try_solve()
        n_iter += 1
        if status in ('optimal', 'optimal_inaccurate'):
            eta_hi = eta_mid
            best_y = y.value.copy()
        else:
            eta_lo = eta_mid

    solve_time = time.time() - t0
    print(f"  Binary search: {n_iter} iters, eta*=[{eta_lo:.8f}, {eta_hi:.8f}], {solve_time:.1f}s")

    return {
        'P': P,
        'bound': eta_hi,
        'status': 'optimal',
        'time': solve_time,
        'n_iter': n_iter,
        'n_simplex_cuts': len(B_simplex_cuts),
        'n_products': len(B_prod_Locs),
        'n_moments': n_mom,
    }


if __name__ == '__main__':
    print("=" * 80)
    print("EXPERIMENT 3v2: Simplex Cuts x_i(1-x_i) >= 0")
    print("=" * 80)
    print(f"Solver: {PRIMARY_SOLVER}\n")

    print("--- Comparison: baseline vs baseline+simplex_cuts ---")
    for P in range(5, 11):
        primal_ub = BASELINE[P]['primal_ub'] if P in BASELINE else None

        res = solve_with_simplex_cuts(P, primal_hint=primal_ub)
        if res['bound'] is not None:
            compare_with_baseline(P, res['bound'], res['time'], 'L2+simplex_cuts')
        else:
            print(f"  P={P}: FAILED ({res['status']})")

        if res['time'] > 300:
            print(f"  Stopping: P={P} took {res['time']:.0f}s")
            break
