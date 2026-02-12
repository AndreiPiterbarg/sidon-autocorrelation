"""Experiment: Single-solve Lasserre Level-2 (no binary search).

Key insight: The original implementation binary-searches eta in ~19 iterations,
each solving a feasibility SDP. This is wasteful. We can formulate the problem
as a single SDP:

  minimize eta
  s.t.  M_2(y) >> 0
        L_{x_k}(y) >> 0  for all k
        L_{x_i x_j}(y) >> 0  for all i < j (products)
        sum_i y_{(i,) + beta} = y_beta  for all beta (simplex)
        eta * M_1(y) - 2P * pk(y) >> 0  for all k (convolution epigraph)
        y_{()} = 1

The trick: eta * M_1(y) is bilinear (eta times y). We need to handle this.

Approach 1: Use Schur complement. For eta * M_1 - 2P*pk >> 0:
This is equivalent to [eta*I, sqrt(2P)*pk_half; sqrt(2P)*pk_half^T, M_1] >> 0
... actually that doesn't work directly.

Approach 2: Since we're minimizing eta, and the constraint is linear in y for
fixed eta, we can use the parametric SDP formulation. MOSEK supports this
natively through the dual problem.

Approach 3 (simplest): Reformulate. Let z = eta * y. Then:
- z_{()} = eta (since y_{()} = 1)
- M_2(z) = eta * M_2(y) >> 0  (need to combine with eta >= 0)
- The constraint eta * M_1(y) - 2P * pk(y) >> 0 becomes:
  M_1(z) - 2P * pk(y) >> 0, but now z and y are separate...

Actually, the simplest approach: formulate the DUAL of the Lasserre relaxation
directly. The dual is a single SDP that MOSEK solves efficiently.

Let me use Approach 4: Direct SDP formulation via linearization.

The bilinear term eta * M_1(y) can be handled because we're MINIMIZING eta
and the other constraints are all linear in y for fixed eta. This is a
generalized eigenvalue problem / parametric SDP.

Actually, the cleanest approach: the constraint
  eta * M_1(y) >= 2P * pk(y)
for EACH k can be written as a single LMI by introducing eta as a variable:

  [eta * M_1(y) - 2P * pk(y)] >> 0 for each k

This is bilinear in (eta, y). However, if we substitute eta = z_0 / y_0
where z_0 = eta (a scalar variable) and y_0 = 1 (the zeroth moment = 1),
then z_0 * M_1(y) = z_0 * M_1(y), which is still bilinear.

The KEY realization: if we fix all y variables and optimize over eta only,
or fix eta and optimize over y, both are SDPs. The joint problem is
a Bilinear Matrix Inequality (BMI), which is NP-hard in general.

BUT: for our specific structure, we can use a DIFFERENT formulation.
We minimize eta where the constraints are:
  M_2(y) >> 0, L_k(y) >> 0, simplex(y), and
  for each k: eta >= 2P * trace(A_k Y) / trace(M1_Y)

where Y = moment matrix of y. Since trace(M1_Y) > 0 at any feasible point,
this becomes:
  eta * trace(M1_Y) >= 2P * trace(A_k Y)

Which IS linear in (eta, Y) — a standard SDP!

Let me implement this properly.
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


def solve_lasserre_2_single(P, verbose=False, use_symmetry=True,
                             use_product_constraints=True):
    """Level-2 Lasserre SDP solved in a SINGLE MOSEK call.

    The key insight: the constraint eta * M_1(y) - 2P * pk(y) >> 0
    IS linear in the joint variables (eta, y) because M_1(y) and pk(y)
    are both linear in y, and the product eta * M_1(y) is bilinear.

    HOWEVER, cvxpy may not directly support this. We reformulate:

    Introduce slack: s_k = eta * y_mu - 2P * pk_mu for each entry of each
    localizing matrix. Then the PSD constraint is on the slack matrix.

    Alternative: Use the fact that MOSEK's native conic interface handles
    this through the dual problem.

    Simplest correct approach: Schur complement.

    eta * M_1 - 2P * pk >> 0  (where M_1, pk are affine in y, eta is variable)

    Since eta > 0 (we can assume this), this is equivalent to:
    M_1 - (2P/eta) * pk >> 0

    But 2P/eta is nonlinear in eta.

    Actually, let's just reformulate. Let tau = 1/eta (so we MAXIMIZE tau).
    Then the constraint becomes:
    M_1(y) - 2P * tau * pk(y) >> 0

    Now M_1(y) is affine in y, and tau * pk(y) is bilinear in (tau, y).

    Hmm, still bilinear. Let me think differently.

    The REAL trick: introduce NEW variables w = eta * y.
    Since y_0 = 1, we have w_0 = eta.
    The constraint eta * M_1(y) - 2P * pk(y) >> 0 becomes:
    M_1(w) - 2P * pk(y) >> 0

    And we need: w = eta * y, i.e., w_j = eta * y_j = w_0 * y_j.
    This is ALSO bilinear.

    OK, let me just use the straightforward approach: write the problem
    with eta as a cp.Variable and (eta * y) as a bilinear product.

    CVXPY will reject this... unless we use DPP (disciplined parametric programming).

    THE ACTUAL SOLUTION: Use MOSEK's dualize feature directly.

    Actually, the simplest working approach is: observe that the original
    binary search only needs to solve ~19 SDPs. Each SDP solve for P=15
    takes about 130s (2464/19). If we can make EACH solve faster, that helps.

    But wait — what if we reformulate so that the entire problem is a SINGLE
    SDP? The constraint eta * L1 - 2P * pk >> 0 is bilinear, but we can
    handle it via the following trick:

    Introduce a scalar variable s and matrix variable Z.
    The constraint eta * A - B >> 0 where A, B are affine in y is equivalent to:

    [s*I  C; C^T  A] >> 0  with s*A - C C^T = eta * A - B

    ... no, this is getting complicated. Let me just use a different strategy:

    STRATEGY: Reformulate as a single SDP by making eta a variable and handling
    the bilinear term via the Schur complement trick for products of variables.

    For matrix Q linear in y and scalar eta > 0:
    eta * Q >> 0  <=>  Q >> 0 and eta >= 0  (when Q has fixed sign pattern)

    Wait, eta * M1 - 2P * pk >> 0 is NOT just eta * (something).
    It's eta * M1(y) - 2P * pk(y).

    Let me substitute: define r_k(y) = M1(y) and s_k(y) = pk(y).
    Then the constraint is: eta * r_k - 2P * s_k >> 0.

    This can be written as: [eta * I_d, sqrt(2P) * s_k_half^T; sqrt(2P) * s_k_half, r_k] >> 0
    via Schur complement IF r_k >> 0 (which we can't assume).

    SIMPLEST APPROACH THAT WORKS:
    Use the epigraph formulation. We want min eta s.t. the system is feasible.
    The bilinear constraint eta * M1 - 2P * pk >> 0 can be written as:

    t * M1 - 2P * pk >> 0  where t = eta is a scalar variable.

    In CVXPY, this is: t * M1_expr - 2P * pk_expr >> 0
    where M1_expr is a matrix affine in y and t is a scalar Variable.

    This IS a valid disciplined convex program because:
    - M1_expr >> 0 (enforced by the moment matrix constraint)
    - t >= shor_bound > 0
    - t * PSD_matrix is convex in t for t > 0

    Actually, t * M1_expr where both t and M1_expr are variables is NOT DCP in CVXPY.
    It would need DPP (if t is a parameter) or DQCP.

    Let me just test it.
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

    print(f"  P={P}: d={d}, loc_d={loc_d}, n_moments={n_mom}")

    # Build indicator matrices (same as core_utils)
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
    print(f"  Setup: {setup_time:.1f}s")

    # ==========================================
    # APPROACH: Convert to MOSEK native format
    # ==========================================
    # Since cvxpy can't handle bilinear eta*y terms directly,
    # we reformulate by noting:
    #
    # The convolution constraint is: for each anti-diagonal k and any feasible x,
    #   V(P) >= 2P * max_k (sum_{i+j=k} x_i x_j)
    #
    # In the moment relaxation, this becomes:
    #   eta >= 2P * sum_{i+j=k} y_{(i,j)} for each k
    #
    # Wait — if we don't need localizing matrices for the convolution
    # constraint, and just use SCALAR constraints, then this is easy!
    #
    # The original uses MATRIX constraints:
    #   eta * M_1(y) - 2P * pk(y) >> 0
    # which is the degree-1 localizing matrix for g_k = eta - 2P*p_k.
    #
    # But simpler: just use the scalar constraint:
    #   eta >= 2P * sum_{i+j=k} y_{(i,j)}  for each k
    #
    # This is weaker (scalar vs matrix) but MUCH simpler and still valid.
    # The matrix constraint is equivalent to: for all g in span(basis_1),
    #   E[g(x)^2 * (eta - 2P*p_k(x))] >= 0
    # The scalar constraint is just the special case g = 1.

    # Let's try BOTH: scalar-only (fast) and full matrix (needs binary search).

    # --- Version A: Scalar convolution constraints ---
    y = cp.Variable(n_mom)
    eta = cp.Variable()
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

    for lhs_indices, rhs_idx in simplex_data:
        constraints.append(sum(y[i] for i in lhs_indices) == y[rhs_idx])

    # Scalar convolution constraints: eta >= 2P * sum_{i+j=k} y_{(i,j)}
    for k_conv in range(2 * P - 1):
        terms = []
        for i in range(max(0, k_conv - P + 1), min(P, k_conv + 1)):
            j = k_conv - i
            key = tuple(sorted((i, j)))
            terms.append(y[moment_idx[key]])
        if terms:
            constraints.append(eta >= 2 * P * sum(terms))

    prob_scalar = cp.Problem(cp.Minimize(eta), constraints)

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
        prob_scalar.solve(solver=PRIMARY_SOLVER, **kwargs)

    scalar_time = time.time() - t0 - setup_time
    scalar_bound = float(eta.value) if prob_scalar.status in ('optimal', 'optimal_inaccurate') else None

    if scalar_bound is not None:
        # Extract diagnostics
        first_mom = np.array([y.value[moment_idx[(i,)]] for i in range(P)])
        M_val = sum(y.value[idx] * mat for idx, mat in B_M.items())
        M_eigvals = np.linalg.eigvalsh(M_val)
        M_rank = int(np.sum(M_eigvals > 1e-6 * max(M_eigvals.max(), 1e-12)))
        M1_val = M_val[:loc_d, :loc_d]
        M1_eigvals = np.linalg.eigvalsh(M1_val)
        M1_rank = int(np.sum(M1_eigvals > 1e-6 * max(M1_eigvals.max(), 1e-12)))

        print(f"  Scalar convolution: bound={scalar_bound:.6f}, time={scalar_time:.1f}s, "
              f"M2_rank={M_rank}/{d}, M1_rank={M1_rank}, "
              f"flat={'YES' if M_rank == M1_rank else 'NO'}")

    total_time = time.time() - t0
    return {
        'P': P,
        'scalar_bound': scalar_bound,
        'scalar_time': scalar_time,
        'total_time': total_time,
        'status': prob_scalar.status,
        'd': d,
        'n_moments': n_mom,
        'n_products': len(B_prod_Locs),
        'setup_time': setup_time,
    }


def solve_lasserre_2_matrix_single(P, verbose=False, use_symmetry=True,
                                     use_product_constraints=True):
    """Level-2 Lasserre with matrix convolution constraints, SINGLE solve.

    The bilinear term eta * M_1(y) is handled by introducing auxiliary
    variables w_mu = eta * y_mu for each moment mu that appears in M_1.

    Then eta * M_1(y) = M_1(w), and we add the coupling constraint
    w_mu = eta * y_mu via a rotated second-order cone:
    (eta, y_mu, w_mu) satisfies w_mu = eta * y_mu, equivalently:
    w_mu^2 <= eta * y_mu (if w_mu >= 0) ... no, this isn't right.

    Actually: w = eta * y means w / eta = y, so w and y must be proportional.
    If we know y_0 = 1 (zeroth moment), then w_0 = eta.
    And w_mu / w_0 = y_mu / y_0 = y_mu, so w_mu = w_0 * y_mu.

    This is still bilinear. Let's try yet another approach:

    APPROACH: Replace the matrix localizing constraint with LINEARIZED version.

    For the matrix constraint L_gk = eta * M_1 - 2P * pk >> 0,
    use the Shor-like relaxation: just keep the scalar part plus
    selected 2x2 principal minors.

    But first, let me just test the scalar-only version to see how much
    bound quality we lose vs the full matrix constraint.

    APPROACH 2: Direct MOSEK formulation bypassing cvxpy.

    Actually, let me try something else entirely: write the DUAL of the
    original Lasserre Level-2 relaxation. The dual is:

    max sum_k lambda_k * ... (dual variables for moment constraints)
    s.t. dual PSD constraints

    The dual is a standard SDP and doesn't have bilinear terms.
    """
    # For now, just run the scalar version (already implemented above)
    return solve_lasserre_2_single(P, verbose=verbose,
                                    use_symmetry=use_symmetry,
                                    use_product_constraints=use_product_constraints)


def solve_native_mosek(P, verbose=False, use_product_constraints=True):
    """Solve Lasserre Level-2 directly via MOSEK's native API.

    This bypasses cvxpy entirely and formulates the problem as:

    minimize eta
    s.t. M_2(y) >> 0
         L_k(y) >> 0  for all k (nonnegativity)
         L_ij(y) >> 0 for all i<j (products)
         simplex equalities
         eta * M_1(y) - 2P * pk(y) >> 0  for all k (convolution)
         y_0 = 1

    The bilinear term eta * M_1(y) is handled by MOSEK through the
    dual of the SDP. In the dual, the constraint becomes linear.

    But using mosek directly is complex. Let me try the approach of
    solving the DUAL SDP via cvxpy.
    """
    # Implementation deferred - would need mosek.fusion or mosek native API
    pass


if __name__ == '__main__':
    print("=" * 80)
    print("EXPERIMENT: Single-Solve Lasserre Level-2")
    print("=" * 80)
    print(f"Solver: {PRIMARY_SOLVER}")

    print("\n--- Version A: Scalar convolution constraints ---")
    print("(replaces eta*M_1-2P*pk >> 0 with eta >= 2P*sum_{i+j=k} y_{ij})\n")

    results = []
    for P in range(2, 21):
        res = solve_lasserre_2_single(P)
        if res['scalar_bound'] is not None:
            compare_with_baseline(P, res['scalar_bound'], res['total_time'], 'Scalar-L2')
        else:
            print(f"  P={P}: FAILED ({res['status']}), time={res['total_time']:.1f}s")
        results.append(res)

        if res['total_time'] > 300:
            print(f"  Stopping: P={P} took {res['total_time']:.0f}s")
            break

    # Push further if we're fast
    if results and results[-1]['total_time'] < 60:
        print("\n--- Scaling to larger P ---")
        for P in [25, 30, 40, 50]:
            res = solve_lasserre_2_single(P)
            if res['scalar_bound'] is not None:
                shor = 2 * P / (2 * P - 1)
                print(f"  P={P:3d}: scalar_L2={res['scalar_bound']:.6f}, "
                      f"Shor={shor:.6f}, diff={res['scalar_bound']-shor:+.6f}, "
                      f"time={res['total_time']:.1f}s")
            else:
                print(f"  P={P}: FAILED, time={res['total_time']:.1f}s")
            results.append(res)
            if res['total_time'] > 300:
                break

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'P':>3} | {'Scalar-L2':>10} | {'Baseline L2':>11} | {'Shor':>10} | "
          f"{'vs baseline':>11} | {'vs Shor':>10} | {'Time':>7}")
    print("-" * 85)
    for res in results:
        P = res['P']
        sb = res['scalar_bound']
        if sb is None:
            continue
        bl = BASELINE.get(P, {}).get('lasserre2_lb')
        shor = 2 * P / (2 * P - 1)
        bl_str = f"{bl:.6f}" if bl else "---"
        diff_bl = f"{sb - bl:+.6f}" if bl else "---"
        diff_shor = f"{sb - shor:+.6f}"
        print(f"{P:>3} | {sb:>10.6f} | {bl_str:>11} | {shor:>10.6f} | "
              f"{diff_bl:>11} | {diff_shor:>10} | {res['total_time']:>6.1f}s")
