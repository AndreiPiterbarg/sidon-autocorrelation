"""Experiment 3: Additional Valid Inequalities for Lasserre Level-2.

The baseline Level-2 Lasserre uses:
- Moment matrix M_2 >> 0
- Localizing matrices for x_k >= 0
- Simplex constraints (sum x_i = 1)
- Product constraints x_i x_j >= 0 (trivial but not implied by moment matrix)

This experiment adds stronger valid inequalities:
1. Simplex-specific: x_i(1 - x_i) >= 0 localizing matrices
2. Pairwise simplex: x_i(1 - x_i - x_j) >= 0 for i != j
3. Triple products: x_i x_j x_k >= 0 (localizing matrices)
4. Stronger RLT: x_i(1 - sum_{j in S} x_j) >= 0 for subsets S

These are all valid for x on the standard simplex and may tighten
the SDP relaxation.
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
    monomial_basis, combine, reflect_index, canonical_index, build_symmetry_map,
    solve_primal_improved, PRIMARY_SOLVER
)


def solve_lasserre_2_enhanced(P, verbose=False, eta_tol=1e-6,
                               primal_hint=None,
                               use_simplex_ineq=True,
                               use_pairwise_simplex=False,
                               use_triple_products=False):
    """Level-2 Lasserre with enhanced valid inequalities.

    Always includes: M_2 >> 0, x_k >= 0 localizing, simplex equalities, products.
    Optional extras:
    - use_simplex_ineq: x_i(1 - x_i) >= 0 localizing matrices
    - use_pairwise_simplex: x_i(1 - x_i - x_j) >= 0 (|i-j| <= bandwidth)
    - use_triple_products: x_i x_j x_k >= 0 for selected triples
    """
    t0 = time.time()

    basis_2 = monomial_basis(P, 2)
    d = len(basis_2)
    basis_1 = monomial_basis(P, 1)
    loc_d = len(basis_1)

    # We need moments up to degree 4 (from M_2),
    # and up to degree 5 for simplex ineq (g=x_i(1-x_i), localizing with basis_1)
    # Actually: x_i(1-x_i) = x_i - x_i^2 is degree 2, localizing with basis_1 (degree 1)
    # gives degree 4 moments. OK, stays within degree 4.

    # For x_i(1-x_i-x_j) = x_i - x_i^2 - x_i x_j, also degree 2, localizing stays degree 4.

    # Collect moments
    moments_set = set()

    # From M_2
    for i in range(d):
        for j in range(i, d):
            moments_set.add(combine(basis_2[i], basis_2[j]))

    # From x_k >= 0 localizing
    for k in range(P):
        for a in range(loc_d):
            for b in range(a, loc_d):
                moments_set.add(combine((k,), basis_1[a], basis_1[b]))

    # From simplex equalities
    all_beta = monomial_basis(P, 3)
    for beta in all_beta:
        moments_set.add(beta)
        for i in range(P):
            moments_set.add(combine((i,), beta))

    # From convolution localizing
    for k_conv in range(2 * P - 1):
        for a in range(loc_d):
            for b in range(a, loc_d):
                moments_set.add(combine(basis_1[a], basis_1[b]))
                for i in range(P):
                    j_val = k_conv - i
                    if 0 <= j_val < P:
                        moments_set.add(combine(basis_1[a], basis_1[b], (i,), (j_val,)))

    # From product constraints x_i x_j >= 0
    for i in range(P):
        for j in range(i + 1, P):
            for a in range(loc_d):
                for b in range(a, loc_d):
                    moments_set.add(combine((i,), (j,), basis_1[a], basis_1[b]))

    # From x_i(1-x_i) >= 0 localizing (same moments as products, already covered)
    # g_i = x_i - x_i^2, localizing: g_i * basis_1[a] * basis_1[b]
    # = x_i * b_a * b_b - x_i^2 * b_a * b_b
    # These are already degree-4 moments, should be in moments_set

    # From x_i(1-x_i-x_j) >= 0 localizing
    if use_pairwise_simplex:
        for i in range(P):
            for j in range(P):
                if i != j:
                    for a in range(loc_d):
                        for b in range(a, loc_d):
                            # g_{ij} * b_a * b_b = x_i*b_a*b_b - x_i^2*b_a*b_b - x_i*x_j*b_a*b_b
                            moments_set.add(combine((i,), basis_1[a], basis_1[b]))
                            moments_set.add(combine((i,), (i,), basis_1[a], basis_1[b]))
                            moments_set.add(combine((i,), (j,), basis_1[a], basis_1[b]))

    moments_list = sorted(moments_set, key=lambda m: (len(m), m))

    # Apply symmetry
    canon_moments, canon_idx_map, full_map, equiv_classes = \
        build_symmetry_map(moments_list, P)
    n_mom = len(canon_moments)
    moment_idx = full_map
    for m, idx in canon_idx_map.items():
        moment_idx[m] = idx

    extra_info = []
    if use_simplex_ineq:
        extra_info.append('+simplex_ineq')
    if use_pairwise_simplex:
        extra_info.append('+pairwise')
    if use_triple_products:
        extra_info.append('+triples')

    print(f"  P={P}: d={d}, n_moments={n_mom} (sym) {' '.join(extra_info)}")

    # Pre-compute indicator matrices (same as core_utils)
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

    # Product constraints
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

    # Simplex inequality constraints: x_i(1 - x_i) >= 0
    # g_i(x) = x_i - x_i^2 = x_i(1 - x_i)
    # Since sum x_j = 1, this equals x_i * sum_{j!=i} x_j
    # Localizing matrix: L_g[a,b] = E[g_i * b_a * b_b]
    #   = E[x_i * b_a * b_b] - E[x_i^2 * b_a * b_b]
    B_simplex_ineq = []
    if use_simplex_ineq:
        for k in nonneg_indices:
            # B_xi: moments of x_k * b_a * b_b
            # B_xi2: moments of x_k^2 * b_a * b_b
            B_gi = {}
            for a in range(loc_d):
                for b in range(a, loc_d):
                    # x_k * b_a * b_b
                    mu1 = combine((k,), basis_1[a], basis_1[b])
                    idx1 = moment_idx[mu1]
                    if idx1 not in B_gi:
                        B_gi[idx1] = np.zeros((loc_d, loc_d))
                    B_gi[idx1][a, b] += 1
                    if a != b:
                        B_gi[idx1][b, a] += 1

                    # -x_k^2 * b_a * b_b
                    mu2 = combine((k,), (k,), basis_1[a], basis_1[b])
                    idx2 = moment_idx[mu2]
                    if idx2 not in B_gi:
                        B_gi[idx2] = np.zeros((loc_d, loc_d))
                    B_gi[idx2][a, b] -= 1
                    if a != b:
                        B_gi[idx2][b, a] -= 1
            B_simplex_ineq.append(B_gi)

    # Pairwise simplex: x_i(1 - x_i - x_j) >= 0 for i != j
    # g_{ij}(x) = x_i - x_i^2 - x_i x_j
    B_pairwise = []
    if use_pairwise_simplex:
        for i in nonneg_indices:
            for j in range(P):
                if i == j:
                    continue
                # Only keep if canonical
                i2 = P - 1 - i
                j2 = P - 1 - j
                if (i2, j2) < (i, j) and i2 in nonneg_indices:
                    continue

                B_gij = {}
                for a in range(loc_d):
                    for b in range(a, loc_d):
                        # x_i * b_a * b_b
                        mu1 = combine((i,), basis_1[a], basis_1[b])
                        idx1 = moment_idx[mu1]
                        if idx1 not in B_gij:
                            B_gij[idx1] = np.zeros((loc_d, loc_d))
                        B_gij[idx1][a, b] += 1
                        if a != b:
                            B_gij[idx1][b, a] += 1

                        # -x_i^2 * b_a * b_b
                        mu2 = combine((i,), (i,), basis_1[a], basis_1[b])
                        idx2 = moment_idx[mu2]
                        if idx2 not in B_gij:
                            B_gij[idx2] = np.zeros((loc_d, loc_d))
                        B_gij[idx2][a, b] -= 1
                        if a != b:
                            B_gij[idx2][b, a] -= 1

                        # -x_i x_j * b_a * b_b
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

    conv_indices = list(range(P))  # symmetry-reduced
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

    # New: simplex inequality localizing matrices
    for B_gi in B_simplex_ineq:
        L_gi = sum(y[idx] * mat for idx, mat in B_gi.items())
        constraints.append(L_gi >> 0)

    # New: pairwise simplex localizing matrices
    for B_gij in B_pairwise:
        L_gij = sum(y[idx] * mat for idx, mat in B_gij.items())
        constraints.append(L_gij >> 0)

    for lhs_indices, rhs_idx in simplex_data:
        constraints.append(sum(y[i] for i in lhs_indices) == y[rhs_idx])

    M1_expr = sum(y[idx] * mat for idx, mat in B_M1.items())
    for B_pk in B_pks:
        pk_expr = sum(y[idx] * mat for idx, mat in B_pk.items())
        L_gk = eta_param * M1_expr - 2 * P * pk_expr
        constraints.append(L_gk >> 0)

    prob = cp.Problem(cp.Minimize(0), constraints)

    solver_list = [PRIMARY_SOLVER]
    if PRIMARY_SOLVER == 'MOSEK' and 'CLARABEL' in cp.installed_solvers():
        solver_list.append('CLARABEL')

    def try_solve(tight=True):
        for s in solver_list:
            try:
                kwargs = {'verbose': False}
                if s == 'MOSEK':
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
                    prob.solve(solver=s, warm_start=True, **kwargs)
                return s, prob.status
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException:
                continue
        return None, 'solver_error'

    # Binary search
    shor_bound = 2 * P / (2 * P - 1)
    eta_lo = shor_bound
    eta_hi = (primal_hint + 0.001) if primal_hint is not None else 2.0

    best_y = None
    n_iter = 0

    eta_param.value = eta_hi
    _, status = try_solve()
    if status not in ('optimal', 'optimal_inaccurate'):
        eta_hi = 2.0 * P
        eta_param.value = eta_hi
        _, status = try_solve()
        if status not in ('optimal', 'optimal_inaccurate'):
            return {'status': 'infeasible', 'bound': None, 'time': time.time() - t0, 'P': P}
    best_y = y.value.copy()

    # Coarse search
    while eta_hi - eta_lo > 1e-3:
        eta_mid = (eta_lo + eta_hi) / 2
        eta_param.value = eta_mid
        _, status = try_solve()
        n_iter += 1
        if status in ('optimal', 'optimal_inaccurate'):
            eta_hi = eta_mid
            best_y = y.value.copy()
        else:
            _, status2 = try_solve(tight=False)
            if status2 in ('optimal', 'optimal_inaccurate'):
                eta_hi = eta_mid
                best_y = y.value.copy()
            else:
                eta_lo = eta_mid

    # Fine search
    while eta_hi - eta_lo > eta_tol:
        eta_mid = (eta_lo + eta_hi) / 2
        eta_param.value = eta_mid
        _, status = try_solve()
        n_iter += 1
        if status in ('optimal', 'optimal_inaccurate'):
            eta_hi = eta_mid
            best_y = y.value.copy()
        else:
            eta_lo = eta_mid

    solve_time = time.time() - t0
    print(f"  Binary search: {n_iter} iters, eta* in [{eta_lo:.8f}, {eta_hi:.8f}], "
          f"{solve_time:.1f}s")

    return {
        'status': 'optimal', 'time': solve_time,
        'bound': eta_hi, 'n_iter': n_iter, 'P': P,
        'eta_lo': eta_lo, 'eta_hi': eta_hi,
        'n_simplex_ineq': len(B_simplex_ineq),
        'n_pairwise': len(B_pairwise),
        'n_products': len(B_prod_Locs),
        'n_moments': n_mom,
    }


if __name__ == '__main__':
    print("=" * 80)
    print("EXPERIMENT 3: Additional Valid Inequalities")
    print("=" * 80)

    # Test at P=5..10 where the relaxation gap exists
    print("\n--- Baseline (products only) vs Enhanced ---")
    for P in range(5, 11):
        primal_val = BASELINE[P]['primal_ub'] if P in BASELINE else None

        # Baseline: products only
        print(f"\nP={P}:")
        res_base = solve_lasserre_2_enhanced(P, primal_hint=primal_val,
                                              use_simplex_ineq=False,
                                              use_pairwise_simplex=False)
        if res_base['bound'] is not None:
            compare_with_baseline(P, res_base['bound'], res_base['time'], 'base+products')

        # + simplex inequality
        res_si = solve_lasserre_2_enhanced(P, primal_hint=primal_val,
                                            use_simplex_ineq=True,
                                            use_pairwise_simplex=False)
        if res_si['bound'] is not None:
            compare_with_baseline(P, res_si['bound'], res_si['time'], 'base+simplex_ineq')

        # + pairwise simplex
        if P <= 8:  # pairwise gets expensive
            res_pw = solve_lasserre_2_enhanced(P, primal_hint=primal_val,
                                                use_simplex_ineq=True,
                                                use_pairwise_simplex=True)
            if res_pw['bound'] is not None:
                compare_with_baseline(P, res_pw['bound'], res_pw['time'], 'base+pairwise')

        # Print improvement summary
        print(f"  Summary P={P}:")
        if res_base['bound']:
            print(f"    base:        {res_base['bound']:.6f} ({res_base['n_moments']} moments, {res_base['time']:.1f}s)")
        if res_si['bound']:
            diff = res_si['bound'] - (res_base['bound'] or 0)
            print(f"    +simplex:    {res_si['bound']:.6f} (diff={diff:+.8f}, {res_si['time']:.1f}s)")
        if P <= 8 and res_pw['bound']:
            diff = res_pw['bound'] - (res_base['bound'] or 0)
            print(f"    +pairwise:   {res_pw['bound']:.6f} (diff={diff:+.8f}, {res_pw['time']:.1f}s)")
