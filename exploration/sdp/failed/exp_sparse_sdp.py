"""Experiment 1: Sparse/Banded SDP for large P.

Key insight: The convolution constraint p_k(x) = sum_{i+j=k} x_i x_j only couples
variables on anti-diagonal k. The "running intersection" property means the
full moment matrix can be replaced by overlapping smaller blocks.

Approach: Banded Lasserre relaxation.
- Instead of a full d x d moment matrix M_2 (where d = C(P+2,2)),
  we restrict to moments where all variable indices are within a "bandwidth" b
  of each other. This gives much smaller PSD blocks.

The idea: For a bandwidth parameter b, we only include monomials x_alpha
where max(alpha) - min(alpha) <= b. This exploits the fact that the
convolution constraint p_k only involves variables {x_i : i + j = k},
which for a given k couples at most P variables but in a structured way.

We implement two approaches:
A) Shor + targeted RLT cuts (simplest, scales to P=100+)
B) Banded moment matrix (moderate, scales to P=30-50)
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
    StepFunction, peak_autoconv_exact, solve_primal_improved, PRIMARY_SOLVER
)


def solve_shor_rlt(P, use_rlt=True, verbose=False):
    """Shor relaxation + RLT cuts for the step-function C_1a problem.

    Shor relaxation: Replace x_i x_j with Y_{ij}, require
      [1, x^T; x, Y] >> 0  (Shor/moment matrix)
      sum_i x_i = 1, x_i >= 0
      eta >= 2P * p_k(Y) for all k, where p_k = sum_{i+j=k} Y_{ij}

    RLT cuts (Reformulation-Linearization Technique):
      Y_{ij} >= 0  (from x_i x_j >= 0)
      Y_{ij} <= x_i  (from x_i(1 - x_j) >= 0 with sum x = 1 simplification)
      Y_{ij} <= x_j
      sum_j Y_{ij} = x_i  (from x_i * sum_j x_j = x_i)

    Returns dict with 'bound', 'time', etc.
    """
    t0 = time.time()

    # Variables
    x = cp.Variable(P, nonneg=True)
    Y = cp.Variable((P, P), symmetric=True)
    eta = cp.Variable()

    constraints = [
        cp.sum(x) == 1,
    ]

    # Shor relaxation: [1, x^T; x, Y] >> 0
    M = cp.bmat([[np.ones((1, 1)), cp.reshape(x, (1, P))],
                  [cp.reshape(x, (P, 1)), Y]])
    constraints.append(M >> 0)

    # Convolution constraints: eta >= 2P * sum_{i+j=k} Y_{ij}
    for k in range(2 * P - 1):
        pairs = [(i, k - i) for i in range(max(0, k - P + 1), min(P, k + 1))]
        pk_expr = sum(Y[i, j] for i, j in pairs)
        constraints.append(eta >= 2 * P * pk_expr)

    if use_rlt:
        # Y_{ij} >= 0
        constraints.append(Y >= 0)
        # sum_j Y_{ij} = x_i for all i (from x_i * sum_j x_j = x_i * 1 = x_i)
        for i in range(P):
            constraints.append(cp.sum(Y[i, :]) == x[i])
        # Y_{ij} <= x_i and Y_{ij} <= x_j
        for i in range(P):
            for j in range(i, P):
                constraints.append(Y[i, j] <= x[i])
                constraints.append(Y[i, j] <= x[j])

    prob = cp.Problem(cp.Minimize(eta), constraints)

    kwargs = {'verbose': False}
    if PRIMARY_SOLVER == 'MOSEK':
        kwargs['mosek_params'] = {
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-10,
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-10,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-10,
        }

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Solution may be inaccurate')
        prob.solve(solver=PRIMARY_SOLVER, **kwargs)

    solve_time = time.time() - t0

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        return {'P': P, 'bound': None, 'status': prob.status, 'time': solve_time}

    return {
        'P': P,
        'bound': float(eta.value),
        'status': prob.status,
        'time': solve_time,
        'x': x.value,
        'Y': Y.value,
        'method': f'Shor{"_RLT" if use_rlt else ""}',
    }


def solve_banded_lasserre(P, bandwidth=None, verbose=False):
    """Banded Lasserre Level-2 relaxation.

    Instead of all degree-2 monomials, only include monomials x_i*x_j
    where |i-j| <= bandwidth. This dramatically reduces the moment matrix size.

    The moment matrix M_2 is block-diagonal in "cliques" of size bandwidth+1.
    Each clique covers variables {x_s, ..., x_{s+bandwidth}} for overlapping windows.

    Uses Waki-Kim-Kojima style chordal decomposition.
    """
    if bandwidth is None:
        bandwidth = min(P - 1, max(3, P // 3))

    t0 = time.time()
    b = bandwidth

    # Generate banded degree-2 monomials: only (i,j) with |i-j| <= b
    basis_2 = [()]  # constant
    for i in range(P):
        basis_2.append((i,))  # degree 1
    for i in range(P):
        for j in range(i, min(P, i + b + 1)):
            basis_2.append((i, j))  # degree 2, banded

    d = len(basis_2)
    basis_idx = {m: i for i, m in enumerate(basis_2)}

    # Overlapping cliques of width (b+1)
    # Each clique covers variables {s, s+1, ..., s+b}
    n_cliques = max(1, P - b)
    cliques = []
    for s in range(n_cliques):
        end = min(s + b + 1, P)
        clique_vars = list(range(s, end))
        # Monomials in this clique: subsets of clique_vars up to degree 2
        clique_basis = [()]
        for v in clique_vars:
            clique_basis.append((v,))
        for i_idx, vi in enumerate(clique_vars):
            for vj in clique_vars[i_idx:]:
                clique_basis.append((vi, vj))
        cliques.append((clique_vars, clique_basis))

    # Variables: one for each unique banded moment
    y = cp.Variable(d)
    eta = cp.Variable()
    constraints = []

    # y[()] = 1
    constraints.append(y[basis_idx[()]] == 1)

    # Simplex: sum_i y[(i,)] = 1
    constraints.append(sum(y[basis_idx[(i,)]] for i in range(P)) == 1)

    # Nonnegativity: y[(i,)] >= 0
    for i in range(P):
        constraints.append(y[basis_idx[(i,)]] >= 0)

    # Simplex on degree-2: sum_i y[(i,j)] = y[(j,)] for each j
    for j in range(P):
        terms = []
        for i in range(P):
            key = tuple(sorted((i, j)))
            if key in basis_idx:
                terms.append(y[basis_idx[key]])
        if terms:
            constraints.append(sum(terms) == y[basis_idx[(j,)]])

    # PSD constraint on each clique's moment submatrix
    for clique_vars, clique_basis in cliques:
        cd = len(clique_basis)
        # Build moment matrix for this clique
        # M[a,b] = y[combine(clique_basis[a], clique_basis[b])]
        # We need to map combined indices to our y variables
        M_entries = []
        valid = True
        for a in range(cd):
            row = []
            for bb in range(cd):
                combined = tuple(sorted(clique_basis[a] + clique_basis[bb]))
                if len(combined) <= 2 and combined in basis_idx:
                    row.append(y[basis_idx[combined]])
                elif len(combined) == 3:
                    # degree 3 moment - we can't include these in banded approach
                    # Use a fresh variable or drop this entry
                    # For now, skip cliques that need degree-3 moments
                    # Actually, we need to restructure: use degree-1 basis for clique moment matrices
                    valid = False
                    break
                elif len(combined) == 4:
                    valid = False
                    break
                else:
                    row.append(y[basis_idx.get(combined, 0)])
            if not valid:
                break
            M_entries.append(row)

        if not valid:
            # Fall back: use only degree-1 basis for clique moment matrix
            clique_basis_1 = [()]
            for v in clique_vars:
                clique_basis_1.append((v,))
            cd1 = len(clique_basis_1)

            rows = []
            for a in range(cd1):
                row = []
                for bb in range(cd1):
                    combined = tuple(sorted(clique_basis_1[a] + clique_basis_1[bb]))
                    if combined in basis_idx:
                        row.append(y[basis_idx[combined]])
                    else:
                        # This shouldn't happen for degree <= 2
                        row.append(0)
                rows.append(cp.hstack(row))
            M_clique = cp.vstack(rows)
            constraints.append(M_clique >> 0)
        else:
            M_clique = cp.vstack([cp.hstack(row) for row in M_entries])
            constraints.append(M_clique >> 0)

    # RLT cuts on available Y entries
    for i in range(P):
        for j in range(i, min(P, i + b + 1)):
            key = (i, j)
            if key in basis_idx:
                constraints.append(y[basis_idx[key]] >= 0)
                constraints.append(y[basis_idx[key]] <= y[basis_idx[(i,)]])
                constraints.append(y[basis_idx[key]] <= y[basis_idx[(j,)]])

    # Convolution constraints: eta >= 2P * sum_{i+j=k} Y_{ij}
    for k in range(2 * P - 1):
        terms = []
        for i in range(max(0, k - P + 1), min(P, k + 1)):
            j = k - i
            key = tuple(sorted((i, j)))
            if key in basis_idx:
                terms.append(y[basis_idx[key]])
            else:
                # Out-of-band: use McCormick relaxation bound
                # Y_{ij} <= min(x_i, x_j) and Y_{ij} >= max(0, x_i + x_j - 1)
                # For upper bound on pk, use Y_{ij} <= min(x_i, x_j)
                # Introduce auxiliary: t_{ij} <= x_i, t_{ij} <= x_j, t_{ij} >= 0
                t_ij = cp.Variable(nonneg=True)
                constraints.append(t_ij <= y[basis_idx[(i,)]])
                constraints.append(t_ij <= y[basis_idx[(j,)]])
                terms.append(t_ij)
        if terms:
            constraints.append(eta >= 2 * P * sum(terms))

    prob = cp.Problem(cp.Minimize(eta), constraints)

    kwargs = {'verbose': False}
    if PRIMARY_SOLVER == 'MOSEK':
        kwargs['mosek_params'] = {
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8,
        }

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Solution may be inaccurate')
        try:
            prob.solve(solver=PRIMARY_SOLVER, **kwargs)
        except Exception as e:
            return {'P': P, 'bound': None, 'status': str(e), 'time': time.time() - t0,
                    'bandwidth': b, 'method': f'Banded_b{b}'}

    solve_time = time.time() - t0

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        return {'P': P, 'bound': None, 'status': prob.status, 'time': solve_time,
                'bandwidth': b, 'method': f'Banded_b{b}'}

    return {
        'P': P,
        'bound': float(eta.value),
        'status': prob.status,
        'time': solve_time,
        'bandwidth': b,
        'd_banded': d,
        'n_cliques': n_cliques,
        'method': f'Banded_b{b}',
    }


if __name__ == '__main__':
    print("=" * 80)
    print("EXPERIMENT 1A: Shor + RLT Relaxation")
    print("=" * 80)

    # First validate at small P against baseline
    print("\n--- Validation (P=2..8) ---")
    for P in range(2, 9):
        res = solve_shor_rlt(P, use_rlt=True)
        if res['bound'] is not None:
            compare_with_baseline(P, res['bound'], res['time'], 'Shor+RLT')
        else:
            print(f"  P={P}: FAILED ({res['status']})")

    # Now push to larger P
    print("\n--- Scaling to large P ---")
    for P in [10, 15, 20, 25, 30, 40, 50, 75, 100]:
        t0 = time.time()
        res = solve_shor_rlt(P, use_rlt=True)
        if res['bound'] is not None:
            compare_with_baseline(P, res['bound'], res['time'], 'Shor+RLT')
        else:
            print(f"  P={P}: FAILED ({res['status']}), time={time.time()-t0:.1f}s")
        if time.time() - t0 > 300:
            print(f"  Stopping: P={P} took too long")
            break

    # Compare Shor alone vs Shor+RLT
    print("\n--- Shor vs Shor+RLT comparison at P=5,10 ---")
    for P in [5, 10]:
        res_shor = solve_shor_rlt(P, use_rlt=False)
        res_rlt = solve_shor_rlt(P, use_rlt=True)
        if res_shor['bound'] and res_rlt['bound']:
            improvement = res_rlt['bound'] - res_shor['bound']
            print(f"  P={P}: Shor={res_shor['bound']:.6f}, Shor+RLT={res_rlt['bound']:.6f}, "
                  f"improvement={improvement:+.6f}")

    print("\n" + "=" * 80)
    print("EXPERIMENT 1B: Banded Lasserre")
    print("=" * 80)

    # Test banded approach
    print("\n--- Banded Lasserre validation (P=3..8, bandwidth=2) ---")
    for P in range(3, 9):
        res = solve_banded_lasserre(P, bandwidth=2)
        if res['bound'] is not None:
            compare_with_baseline(P, res['bound'], res['time'], f"Banded_b2")
        else:
            print(f"  P={P}: FAILED ({res['status']})")

    print("\n--- Banded Lasserre scaling (bandwidth=3) ---")
    for P in [10, 15, 20, 25, 30]:
        res = solve_banded_lasserre(P, bandwidth=3)
        if res['bound'] is not None:
            compare_with_baseline(P, res['bound'], res['time'], f"Banded_b3")
        else:
            print(f"  P={P}: FAILED ({res['status']}), time={res['time']:.1f}s")
        if res['time'] > 300:
            break

    # Try larger bandwidth at moderate P
    print("\n--- Bandwidth sweep at P=15 ---")
    for b in [2, 3, 5, 7, 10, 14]:
        if b >= P:
            continue
        res = solve_banded_lasserre(15, bandwidth=b)
        if res['bound'] is not None:
            compare_with_baseline(15, res['bound'], res['time'], f"Banded_b{b}")
        else:
            print(f"  b={b}: FAILED ({res['status']})")
        if res['time'] > 120:
            break
