"""Experiment: Fast Lasserre Level-2 for larger P.

Strategy:
1. Use the existing binary search framework (it's already optimized with cp.Parameter)
2. Use coarse tolerance (1e-3) to reduce binary search iterations from 19 to ~10
3. Add simplex cuts x_i(1-x_i) >= 0 for tighter bounds
4. Use SCS for fast scouting, then MOSEK for final refinement
5. Extrapolate eta from previous P to get tighter initial bracket
6. Skip primal solver (use extrapolation for eta_hi)

Target: P=16-20 within 10 min each.
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
    PRIMARY_SOLVER
)


def build_lasserre_problem(P, use_symmetry=True, use_product_constraints=True,
                            use_simplex_cuts=True):
    """Build the Lasserre Level-2 problem (compiled once, solve many times)."""
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
    if use_product_constraints:
        for i in range(P):
            for j in range(i + 1, P):
                for a in range(loc_d):
                    for b in range(a, loc_d):
                        moments_set.add(combine((i,), (j,), basis_1[a], basis_1[b]))
    if use_simplex_cuts:
        for i in range(P):
            for a in range(loc_d):
                for b in range(a, loc_d):
                    moments_set.add(combine((i,), (i,), basis_1[a], basis_1[b]))

    moments_list = sorted(moments_set, key=lambda m: (len(m), m))

    if use_symmetry:
        canon_moments, canon_idx_map, full_map, equiv_classes = \
            build_symmetry_map(moments_list, P)
        n_mom = len(canon_moments)
        moment_idx = full_map
        for m, idx in canon_idx_map.items():
            moment_idx[m] = idx
    else:
        n_mom = len(moments_list)
        moment_idx = {m: idx for idx, m in enumerate(moments_list)}

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

    nonneg_indices = list(range((P + 1) // 2)) if use_symmetry else list(range(P))

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
    if use_product_constraints:
        for i in range(P):
            for j in range(i + 1, P):
                if use_symmetry:
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

    B_simplex_cuts = []
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

    conv_indices = list(range(P)) if use_symmetry else list(range(2 * P - 1))
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

    # Build CVXPY problem (compiled ONCE)
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

    setup_time = time.time() - t0

    return {
        'prob': prob, 'y': y, 'eta_param': eta_param,
        'moment_idx': moment_idx, 'B_M': B_M, 'B_M1': B_M1,
        'P': P, 'd': d, 'loc_d': loc_d, 'n_mom': n_mom,
        'setup_time': setup_time,
        'n_products': len(B_prod_Locs),
        'n_simplex_cuts': len(B_simplex_cuts),
    }


def solve_with_bracket(problem_data, eta_lo, eta_hi, eta_tol=1e-3, max_time=600):
    """Binary search within a given bracket."""
    prob = problem_data['prob']
    y = problem_data['y']
    eta_param = problem_data['eta_param']
    P = problem_data['P']

    t0 = time.time()
    best_y = None
    n_iter = 0

    def try_solve(tight=True):
        try:
            kwargs = {'verbose': False}
            if PRIMARY_SOLVER == 'MOSEK':
                if tight:
                    kwargs['mosek_params'] = {
                        'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-9,
                        'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-9,
                        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-9,
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

    # Verify upper bound
    eta_param.value = eta_hi
    status = try_solve()
    if status not in ('optimal', 'optimal_inaccurate'):
        eta_param.value = eta_hi + 0.1
        status = try_solve(tight=False)
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
            if eta_mid > eta_lo + 0.01:
                status2 = try_solve(tight=False)
                if status2 in ('optimal', 'optimal_inaccurate'):
                    eta_hi = eta_mid
                    best_y = y.value.copy()
                    continue
            eta_lo = eta_mid

    return eta_hi, time.time() - t0, n_iter


if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("EXPERIMENT: Fast Lasserre Level-2 with Simplex Cuts", flush=True)
    print("=" * 80, flush=True)
    print(f"Solver: {PRIMARY_SOLVER}\n", flush=True)

    # Extrapolate initial bracket from baseline
    # The bound V(P) decreases roughly as V(P) ≈ C_1a + A/P for some constants
    # Use previous bounds to get tight initial bracket

    all_results = []
    prev_bounds = {}

    # First, validate at P=5-10 with simplex cuts
    print("--- Phase 1: Validation with simplex cuts (P=5..10) ---", flush=True)
    for P in range(5, 11):
        print(f"\nP={P}:", flush=True)
        pd = build_lasserre_problem(P, use_simplex_cuts=True)
        print(f"  d={pd['d']}, n_mom={pd['n_mom']}, setup={pd['setup_time']:.1f}s, "
              f"products={pd['n_products']}, simplex_cuts={pd['n_simplex_cuts']}", flush=True)

        # Use baseline for bracket
        bl = BASELINE.get(P, {})
        shor = 2 * P / (2 * P - 1)
        eta_lo = shor
        eta_hi = bl.get('primal_ub', 2.0) + 0.001

        bound, solve_time, n_iter = solve_with_bracket(pd, eta_lo, eta_hi, eta_tol=1e-6)
        total_time = pd['setup_time'] + solve_time

        if bound is not None:
            compare_with_baseline(P, bound, total_time, 'Fast-L2+SC')
            prev_bounds[P] = bound
        else:
            print(f"  FAILED, time={total_time:.1f}s", flush=True)

        all_results.append({'P': P, 'bound': bound, 'time': total_time, 'n_iter': n_iter})

    # Phase 2: Push to P=11-15 (reproduce + beat baseline times)
    print("\n--- Phase 2: P=11..15, coarse tolerance ---", flush=True)
    for P in range(11, 16):
        print(f"\nP={P}:", flush=True)
        pd = build_lasserre_problem(P, use_simplex_cuts=True)
        print(f"  d={pd['d']}, n_mom={pd['n_mom']}, setup={pd['setup_time']:.1f}s", flush=True)

        bl = BASELINE.get(P, {})
        shor = 2 * P / (2 * P - 1)
        eta_lo = shor
        eta_hi = bl.get('primal_ub', 2.0) + 0.001

        bound, solve_time, n_iter = solve_with_bracket(pd, eta_lo, eta_hi,
                                                         eta_tol=1e-3, max_time=600)
        total_time = pd['setup_time'] + solve_time

        if bound is not None:
            compare_with_baseline(P, bound, total_time, 'Fast-L2+SC')
            prev_bounds[P] = bound
        else:
            print(f"  FAILED, time={total_time:.1f}s", flush=True)

        all_results.append({'P': P, 'bound': bound, 'time': total_time, 'n_iter': n_iter})

    # Phase 3: Push to P=16-20 (NEW territory)
    print("\n--- Phase 3: P=16..20, coarse tolerance, max 10min each ---", flush=True)
    for P in range(16, 21):
        print(f"\nP={P}:", flush=True)
        pd = build_lasserre_problem(P, use_simplex_cuts=True)
        print(f"  d={pd['d']}, n_mom={pd['n_mom']}, setup={pd['setup_time']:.1f}s", flush=True)

        shor = 2 * P / (2 * P - 1)

        # Extrapolate eta_hi from previous bounds
        if len(prev_bounds) >= 2:
            sorted_p = sorted(prev_bounds.keys())
            last_p = sorted_p[-1]
            second_last_p = sorted_p[-2]
            # Linear extrapolation
            slope = (prev_bounds[last_p] - prev_bounds[second_last_p]) / (last_p - second_last_p)
            eta_hi_extrap = prev_bounds[last_p] + slope * (P - last_p)
            eta_hi = max(eta_hi_extrap + 0.01, shor + 0.01)
        else:
            eta_hi = 2.0
        eta_lo = shor

        print(f"  Bracket: [{eta_lo:.4f}, {eta_hi:.4f}]", flush=True)

        bound, solve_time, n_iter = solve_with_bracket(pd, eta_lo, eta_hi,
                                                         eta_tol=1e-3, max_time=600)
        total_time = pd['setup_time'] + solve_time

        if bound is not None:
            compare_with_baseline(P, bound, total_time, 'Fast-L2+SC')
            prev_bounds[P] = bound
        else:
            print(f"  FAILED, time={total_time:.1f}s", flush=True)

        all_results.append({'P': P, 'bound': bound, 'time': total_time, 'n_iter': n_iter})

        if total_time > 600:
            print(f"  Time limit reached at P={P}", flush=True)
            break

    # Summary
    print("\n" + "=" * 100, flush=True)
    print("SUMMARY: Fast Lasserre Level-2 + Simplex Cuts", flush=True)
    print("=" * 100, flush=True)
    print(f"{'P':>3} | {'Bound':>10} | {'Baseline':>10} | {'Diff':>10} | "
          f"{'Shor':>10} | {'vs Shor':>10} | {'Iters':>5} | {'Time':>7}", flush=True)
    print("-" * 95, flush=True)
    for r in all_results:
        P = r['P']
        b = r.get('bound')
        if b is None:
            print(f"{P:>3} | {'FAILED':>10} | {'---':>10} | {'---':>10} | "
                  f"{2*P/(2*P-1):>10.6f} | {'---':>10} | {r['n_iter']:>5} | "
                  f"{r['time']:>6.1f}s", flush=True)
            continue
        bl = BASELINE.get(P, {}).get('lasserre2_lb')
        shor = 2 * P / (2 * P - 1)
        bl_str = f"{bl:.6f}" if bl else "---"
        diff = f"{b - bl:+.6f}" if bl else "NEW"
        vs_shor = f"{b - shor:+.6f}"
        print(f"{P:>3} | {b:>10.6f} | {bl_str:>10} | {diff:>10} | "
              f"{shor:>10.6f} | {vs_shor:>10} | {r['n_iter']:>5} | "
              f"{r['time']:>6.1f}s", flush=True)

    # Highlight new results beyond baseline
    new_results = [r for r in all_results if r['P'] > 15 and r.get('bound')]
    if new_results:
        print(f"\n*** NEW RESULTS beyond P=15 baseline ***", flush=True)
        for r in new_results:
            shor = 2 * r['P'] / (2 * r['P'] - 1)
            print(f"  P={r['P']}: V(P) >= {r['bound']:.6f} "
                  f"(Shor={shor:.6f}, improvement={r['bound']-shor:+.6f})", flush=True)
