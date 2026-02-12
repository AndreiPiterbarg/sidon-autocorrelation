"""Experiment: Single-Solve Lasserre Level-2 via lambda reformulation.

KEY INSIGHT: The bilinear constraint  eta * M_1(y) - 2P * pk(y) >> 0
can be linearized by dividing by eta (which is > 0):

  M_1(y) - (2P/eta) * pk(y) >> 0

Setting lambda = 2P/eta, we MAXIMIZE lambda subject to:
  M_1(y) - lambda * pk(y) >> 0  for each conv index k

Then eta* = 2P / lambda*.

This is a standard SDP in (y, lambda) — a SINGLE solve instead of 19!

Expected speedup: 10-19x over binary search, enabling P=20+ within budget.
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


def solve_lasserre_2_direct(P, use_symmetry=True, use_product_constraints=True,
                             use_simplex_cuts=False, verbose=False):
    """Level-2 Lasserre SDP in a SINGLE MOSEK call via lambda reformulation.

    maximize lambda (= 2P/eta)
    s.t.  M_2(y) >> 0
          L_k(y) >> 0  for all k (nonnegativity)
          L_ij(y) >> 0 for all canonical i<j (products)
          simplex(y)
          y_0 = 1
          M_1(y) - lambda * pk(y) >> 0  for each conv index k

    Returns eta* = 2P / lambda*.
    """
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

    print(f"  P={P}: d={d}, loc_d={loc_d}, n_moments={n_mom}", flush=True)

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

    if use_symmetry:
        nonneg_indices = list(range((P + 1) // 2))
    else:
        nonneg_indices = list(range(P))

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

    if use_symmetry:
        conv_indices = list(range(P))
    else:
        conv_indices = list(range(2 * P - 1))

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

    setup_time = time.time() - t0
    print(f"  Setup: {setup_time:.1f}s", flush=True)

    # Build CVXPY problem — SINGLE optimization
    y = cp.Variable(n_mom)
    lam = cp.Variable()  # lambda = 2P/eta, maximize this
    constraints = []

    # M_2 >> 0
    M_expr = sum(y[idx] * mat for idx, mat in B_M.items())
    constraints.append(M_expr >> 0)
    constraints.append(y[moment_idx[()]] == 1)

    # Nonnegativity localizing
    for B_L in B_Locs:
        L_k = sum(y[idx] * mat for idx, mat in B_L.items())
        constraints.append(L_k >> 0)

    # Product localizing
    for B_pL in B_prod_Locs:
        L_ij = sum(y[idx] * mat for idx, mat in B_pL.items())
        constraints.append(L_ij >> 0)

    # Simplex cuts
    for B_gi in B_simplex_cuts:
        L_gi = sum(y[idx] * mat for idx, mat in B_gi.items())
        constraints.append(L_gi >> 0)

    # Simplex equalities
    for lhs_indices, rhs_idx in simplex_data:
        constraints.append(sum(y[i] for i in lhs_indices) == y[rhs_idx])

    # Convolution constraints: M_1(y) - lambda * pk(y) >> 0 for each k
    M1_expr = sum(y[idx] * mat for idx, mat in B_M1.items())
    for B_pk in B_pks:
        pk_expr = sum(y[idx] * mat for idx, mat in B_pk.items())
        # M_1(y) - lambda * pk(y) >> 0
        # This is affine in (y, lambda)!
        constraints.append(M1_expr - lam * pk_expr >> 0)

    # lambda > 0 (implies eta > 0)
    constraints.append(lam >= 0)

    prob = cp.Problem(cp.Maximize(lam), constraints)

    build_time = time.time() - t0 - setup_time
    print(f"  Problem build: {build_time:.1f}s", flush=True)

    kwargs = {'verbose': verbose}
    if PRIMARY_SOLVER == 'MOSEK':
        kwargs['mosek_params'] = {
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-10,
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-10,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-10,
            'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-12,
        }

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Solution may be inaccurate')
        try:
            prob.solve(solver=PRIMARY_SOLVER, **kwargs)
        except Exception as e:
            # Fallback to CLARABEL
            try:
                prob.solve(solver='CLARABEL', verbose=verbose)
            except Exception as e2:
                return {'P': P, 'bound': None, 'status': str(e2),
                        'time': time.time() - t0}

    solve_time = time.time() - t0

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        print(f"  FAILED: {prob.status}, time={solve_time:.1f}s", flush=True)
        return {'P': P, 'bound': None, 'status': prob.status, 'time': solve_time}

    lam_val = float(lam.value)
    eta_val = 2 * P / lam_val if lam_val > 1e-10 else float('inf')

    # Diagnostics
    first_mom = np.array([y.value[moment_idx[(i,)]] for i in range(P)])
    M_val = sum(y.value[idx] * mat for idx, mat in B_M.items())
    M_eigvals = np.linalg.eigvalsh(M_val)
    M_rank = int(np.sum(M_eigvals > 1e-6 * max(M_eigvals.max(), 1e-12)))
    M1_val = M_val[:loc_d, :loc_d]
    M1_eigvals = np.linalg.eigvalsh(M1_val)
    M1_rank = int(np.sum(M1_eigvals > 1e-6 * max(M1_eigvals.max(), 1e-12)))
    flat = M_rank == M1_rank

    print(f"  Solved: lambda={lam_val:.8f}, eta={eta_val:.8f}, time={solve_time:.1f}s "
          f"(setup={setup_time:.1f}s, build={build_time:.1f}s, solve={solve_time-setup_time-build_time:.1f}s)",
          flush=True)
    print(f"  M2 rank={M_rank}/{d}, M1 rank={M1_rank}, flat={'YES' if flat else 'NO'}", flush=True)

    return {
        'P': P,
        'bound': eta_val,
        'lambda': lam_val,
        'status': prob.status,
        'time': solve_time,
        'setup_time': setup_time,
        'build_time': build_time,
        'solve_time_mosek': solve_time - setup_time - build_time,
        'd': d, 'n_moments': n_mom,
        'n_products': len(B_prod_Locs),
        'n_simplex_cuts': len(B_simplex_cuts),
        'M_rank': M_rank, 'M1_rank': M1_rank,
        'flat': flat,
        'first_moments': first_mom,
    }


if __name__ == '__main__':
    print("=" * 80, flush=True)
    print("EXPERIMENT: Single-Solve Lasserre Level-2 (lambda reformulation)", flush=True)
    print("=" * 80, flush=True)
    print(f"Solver: {PRIMARY_SOLVER}\n", flush=True)

    results = []

    # Validation at small P
    print("--- Validation (P=2..8) ---", flush=True)
    for P in range(2, 9):
        res = solve_lasserre_2_direct(P)
        if res['bound'] is not None:
            compare_with_baseline(P, res['bound'], res['time'], 'Direct-L2')
        else:
            print(f"  P={P}: FAILED ({res['status']})", flush=True)
        results.append(res)

    # Scale up
    print("\n--- Scaling (P=9..20) ---", flush=True)
    for P in range(9, 21):
        res = solve_lasserre_2_direct(P)
        if res['bound'] is not None:
            compare_with_baseline(P, res['bound'], res['time'], 'Direct-L2')
        else:
            print(f"  P={P}: FAILED ({res['status']})", flush=True)
        results.append(res)
        if res['time'] > 600:
            print(f"  Stopping: P={P} took {res['time']:.0f}s > 600s", flush=True)
            break

    # With simplex cuts
    print("\n--- With simplex cuts (P=5..12) ---", flush=True)
    for P in range(5, 13):
        res = solve_lasserre_2_direct(P, use_simplex_cuts=True)
        if res['bound'] is not None:
            compare_with_baseline(P, res['bound'], res['time'], 'Direct-L2+SC')
        else:
            print(f"  P={P}: FAILED ({res['status']})", flush=True)
        results.append(res)
        if res['time'] > 300:
            break

    # Summary table
    print("\n" + "=" * 100, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 100, flush=True)
    print(f"{'P':>3} | {'Direct-L2':>10} | {'Baseline':>10} | {'Shor':>10} | "
          f"{'vs BL':>10} | {'Flat':>5} | {'Solve(s)':>8} | {'Total(s)':>8}", flush=True)
    print("-" * 100, flush=True)
    for res in results:
        P = res['P']
        b = res.get('bound')
        if b is None:
            continue
        bl = BASELINE.get(P, {}).get('lasserre2_lb')
        shor = 2 * P / (2 * P - 1)
        bl_str = f"{bl:.6f}" if bl else "---"
        diff = f"{b - bl:+.6f}" if bl else "---"
        flat_str = 'YES' if res.get('flat') else 'no'
        mosek_t = res.get('solve_time_mosek', res['time'])
        print(f"{P:>3} | {b:>10.6f} | {bl_str:>10} | {shor:>10.6f} | "
              f"{diff:>10} | {flat_str:>5} | {mosek_t:>7.1f}s | {res['time']:>7.1f}s", flush=True)
