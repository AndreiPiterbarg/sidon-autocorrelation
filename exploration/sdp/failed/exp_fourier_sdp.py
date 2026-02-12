"""Experiment 2: Fourier/Trigonometric SDP for direct C_1a lower bound.

The previous attempt (lower_bound_sdp.ipynb) used:
  max C s.t. M(c) - C * vv^T >= 0 (PSD)
This got C* ≈ 1.0 because the PSD relaxation of the copositive constraint
is too weak — the binding eigenvectors change sign.

NEW APPROACH: Discretized dual formulation.
Instead of working with continuous functions, discretize on a fine grid and
solve a finite-dimensional copositive program directly.

The key identity: for f ≥ 0 with supp(f) ⊆ [-1/4, 1/4], ||f||_1 = 1:
  ||f*f||_∞ = max_t ∫∫ f(x)f(y) δ(x+y-t) dx dy

The dual asks: find φ: [-1/2, 1/2] → R with ∫φ = 1 maximizing C such that
  ∫∫ f(x)f(y) φ(x+y) dx dy ≥ C for all f ≥ 0 with ∫f = 1.

This means the kernel matrix K(x,y) = φ(x+y) must be C-copositive:
  v^T K v ≥ C ||v||_1^2  for all v ≥ 0.

Equivalently: K - C * (11^T / n^2) is copositive (on the simplex).

Strategy:
1) Discretize on M-point grid in [-1/4, 1/4]
2) Build K_{ij} = φ(x_i + x_j) using degree-N cosine polynomial
3) Enforce copositivity via Parrilo's inner approximations:
   - Level 0: K - C*J ∈ S (doubly nonneg) — S = PSD ∩ Nonneg
   - Level 1: K - C*J = S + N with S ≥ 0 (PSD), N ≥ 0 (entrywise)
   — same as copositivity-1 from the original notebook
   - Level 2: K - C*J = Σ D_i S_i D_i with D_i = diag(e_i), S_i PSD
   (THESE ARE STRONGER AND SHOULD BEAT 1.0!)

The critical insight the original notebook missed:
Level-0 copositivity (doubly nonnegative relaxation) should already beat 1.0
because it combines PSD + entrywise nonnegativity. The original used just PSD.

Also: we should try MOSEK instead of CLARABEL for better numerics.
"""

import numpy as np
import cvxpy as cp
import time
import warnings
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from exploration.sdp.baseline_results import compare_with_baseline
from exploration.sdp.core_utils import PRIMARY_SOLVER


def solve_doubly_nonneg(N, M, solver=None, verbose=False):
    """Doubly nonnegative relaxation of copositivity.

    Discretize f on M uniform grid points in [-1/4, 1/4].
    Represent phi as degree-N cosine polynomial.
    Require K - C*J to be doubly nonnegative (PSD + entrywise nonneg).

    N: cosine polynomial degree for phi
    M: discretization points for f
    """
    t0 = time.time()

    # Grid points for f
    x_pts = np.linspace(-0.25, 0.25, M, endpoint=False) + 0.5 / (2 * M)

    # Precompute: K_{ij} = phi(x_i + x_j) = sum_k c_k cos(2*pi*k*(x_i+x_j))
    sums = x_pts[:, None] + x_pts[None, :]  # M x M
    CosK = np.zeros((N + 1, M, M))
    for k in range(N + 1):
        CosK[k] = np.cos(2 * np.pi * k * sums)

    J = np.ones((M, M)) / (M * M)  # normalized 11^T

    # Decision variables
    C_var = cp.Variable(name='C')
    c_var = cp.Variable(N + 1, name='c')  # Fourier coefficients
    Q = cp.Variable((N + 1, N + 1), symmetric=True, name='Q')  # Fejer-Riesz for phi >= 0

    constraints = [
        c_var[0] == 1,  # normalization: integral phi = 1
        Q >> 0,
        cp.trace(Q) == 1,
    ]

    # Fejer-Riesz: c_k = 2 * sum of k-th superdiagonal of Q
    for k in range(1, N + 1):
        constraints.append(c_var[k] == 2 * cp.sum(cp.diag(Q, k)))

    # Build kernel matrix K = sum c_k * CosK[k]
    CosK_flat = CosK.reshape(N + 1, M * M)
    K_expr = cp.reshape(c_var @ CosK_flat, (M, M))

    # Doubly nonnegative: K - C*J is PSD AND entrywise nonneg
    diff = K_expr - C_var * J
    constraints.append(diff >> 0)
    constraints.append(diff >= 0)  # entrywise

    prob = cp.Problem(cp.Maximize(C_var), constraints)

    solver_to_use = solver or PRIMARY_SOLVER
    kwargs = {'verbose': verbose}
    if solver_to_use == 'MOSEK':
        kwargs['mosek_params'] = {
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8,
        }

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Solution may be inaccurate')
        prob.solve(solver=solver_to_use, **kwargs)

    elapsed = time.time() - t0

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        return {'N': N, 'M': M, 'C': None, 'status': prob.status, 'time': elapsed}

    c_vals = c_var.value.copy()
    c_vals[0] = 1.0

    return {
        'N': N, 'M': M,
        'C': float(C_var.value),
        'status': prob.status,
        'time': elapsed,
        'c': c_vals,
        'method': 'doubly_nonneg',
    }


def solve_copositive_parrilo_1(N, M, solver=None, verbose=False):
    """Parrilo's level-1 inner approximation to copositivity.

    K - C*J = S + N_mat where S PSD, N_mat entrywise nonneg.
    (This is the standard copositivity-1 from the original notebook,
    but now with MOSEK.)
    """
    t0 = time.time()

    x_pts = np.linspace(-0.25, 0.25, M, endpoint=False) + 0.5 / (2 * M)
    sums = x_pts[:, None] + x_pts[None, :]
    CosK = np.zeros((N + 1, M, M))
    for k in range(N + 1):
        CosK[k] = np.cos(2 * np.pi * k * sums)

    J = np.ones((M, M)) / (M * M)

    C_var = cp.Variable(name='C')
    c_var = cp.Variable(N + 1, name='c')
    Q = cp.Variable((N + 1, N + 1), symmetric=True, name='Q')
    S = cp.Variable((M, M), symmetric=True, name='S')
    N_mat = cp.Variable((M, M), name='N_mat')

    constraints = [
        c_var[0] == 1,
        Q >> 0,
        cp.trace(Q) == 1,
        S >> 0,
        N_mat >= 0,
    ]

    for k in range(1, N + 1):
        constraints.append(c_var[k] == 2 * cp.sum(cp.diag(Q, k)))

    CosK_flat = CosK.reshape(N + 1, M * M)
    K_expr = cp.reshape(c_var @ CosK_flat, (M, M))

    constraints.append(K_expr - C_var * J == S + N_mat)

    prob = cp.Problem(cp.Maximize(C_var), constraints)

    solver_to_use = solver or PRIMARY_SOLVER
    kwargs = {'verbose': verbose}
    if solver_to_use == 'MOSEK':
        kwargs['mosek_params'] = {
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8,
        }

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Solution may be inaccurate')
        prob.solve(solver=solver_to_use, **kwargs)

    elapsed = time.time() - t0

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        return {'N': N, 'M': M, 'C': None, 'status': prob.status, 'time': elapsed}

    return {
        'N': N, 'M': M,
        'C': float(C_var.value),
        'status': prob.status,
        'time': elapsed,
        'method': 'copositive_1',
    }


def solve_discretized_direct(M, solver=None, verbose=False):
    """Direct discretized dual: no Fourier parameterization of phi.

    Directly optimize phi values at grid points on [-1/2, 1/2].
    phi_j >= 0, sum phi_j * w = 1 (quadrature weight w = 1/M_phi).

    Kernel: K_{ij} = phi(x_i + x_j) where phi is interpolated from the grid.

    This avoids the Fourier truncation issue entirely.
    """
    t0 = time.time()

    # Grid for f
    x_pts = np.linspace(-0.25, 0.25, M, endpoint=False) + 0.5 / (2 * M)

    # Grid for phi (finer, covers [-1/2, 1/2])
    M_phi = 4 * M  # finer grid for phi
    t_pts = np.linspace(-0.5, 0.5, M_phi, endpoint=False) + 0.5 / (2 * M_phi)
    w_phi = 1.0 / M_phi  # quadrature weight

    # For each (x_i, x_j), find the nearest phi grid point for x_i + x_j
    sums = x_pts[:, None] + x_pts[None, :]  # M x M, range [-1/2, 1/2]
    # Map to phi grid indices
    phi_indices = np.round((sums - t_pts[0]) / (t_pts[1] - t_pts[0])).astype(int)
    phi_indices = np.clip(phi_indices, 0, M_phi - 1)

    # Decision variables
    C_var = cp.Variable(name='C')
    phi_var = cp.Variable(M_phi, nonneg=True, name='phi')  # phi >= 0

    constraints = [
        cp.sum(phi_var) * w_phi == 1,  # normalization
    ]

    # Build kernel matrix: K_{ij} = phi[phi_indices[i,j]]
    # Express K as a linear map from phi_var
    # K = A @ phi_var reshaped, where A maps phi values to K entries
    A = np.zeros((M * M, M_phi))
    for i in range(M):
        for j in range(M):
            A[i * M + j, phi_indices[i, j]] = 1

    K_flat = A @ phi_var  # (M*M,)
    K_expr = cp.reshape(K_flat, (M, M))

    J = np.ones((M, M)) / (M * M)

    # Doubly nonneg: K - C*J is PSD + entrywise nonneg
    diff = K_expr - C_var * J
    constraints.append(diff >> 0)
    # K is already nonneg since phi >= 0, but K - C*J might not be
    # For copositivity we actually need: v^T K v >= C for all v >= 0 with sum v = 1
    # That's equivalent to K - C * ones_normalized being copositive
    # Doubly nonneg is inner approximation
    constraints.append(diff >= 0)

    prob = cp.Problem(cp.Maximize(C_var), constraints)

    solver_to_use = solver or PRIMARY_SOLVER
    kwargs = {'verbose': verbose}
    if solver_to_use == 'MOSEK':
        kwargs['mosek_params'] = {
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-7,
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-7,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-7,
        }

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Solution may be inaccurate')
        prob.solve(solver=solver_to_use, **kwargs)

    elapsed = time.time() - t0

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        return {'M': M, 'C': None, 'status': prob.status, 'time': elapsed}

    return {
        'M': M, 'M_phi': M_phi,
        'C': float(C_var.value),
        'status': prob.status,
        'time': elapsed,
        'phi': phi_var.value,
        'method': 'direct_discretized',
    }


def solve_simplex_copositive(M, solver=None, verbose=False):
    """Reformulated copositive program on the simplex.

    Key insight: We can reformulate the problem directly.

    For step functions with P bins, x = (x_1,...,x_P) on the simplex,
    the autoconvolution peak is max_k 2P * sum_{i+j=k} x_i x_j.

    The dual problem for C_1a asks:
      max C s.t. max_k 2P * sum_{i+j=k} x_i x_j >= C for all x on simplex

    Equivalently: C_1a(P) = min_{x on simplex} max_k 2P * sum_{i+j=k} x_i x_j

    For a lower bound on the continuous C_1a, we need the limit as P -> inf.
    But we can compute V(P) = C_1a(P) exactly for moderate P via the
    Lasserre approach. Here we try a different route:

    Use the dual of the minimax: find weights lambda_k >= 0 with sum = 1 such that
      C = min_{x >= 0, sum x = 1} sum_k lambda_k * 2P * sum_{i+j=k} x_i x_j

    The inner min is a convex quadratic on the simplex:
      min x^T (2P * sum_k lambda_k A_k) x  s.t. sum x = 1, x >= 0

    where A_k is the anti-diagonal matrix for index k.

    The dual of this inner problem gives an SDP:
      max alpha
      s.t. 2P * sum_k lambda_k A_k - alpha * J' >= 0 (copositive)

    where J' = ee^T/P^2... Actually this needs more care.

    Let's just directly compute the discretized dual for various P.
    """
    P = M  # Use M as P for step functions
    t0 = time.time()

    # Anti-diagonal matrices A_k: (A_k)_{ij} = 1 if i+j=k, else 0
    n_diags = 2 * P - 1
    A_mats = []
    for k in range(n_diags):
        A = np.zeros((P, P))
        for i in range(max(0, k - P + 1), min(P, k + 1)):
            j = k - i
            A[i, j] = 1
        A_mats.append(A)

    # Variables: lambda_k >= 0 weights, C is the lower bound
    lam = cp.Variable(n_diags, nonneg=True)
    C_var = cp.Variable()

    # Weighted quadratic form: Q = 2P * sum_k lambda_k A_k
    A_flat = np.array([A.flatten() for A in A_mats])  # (n_diags, P*P)
    Q_flat = 2 * P * (lam @ A_flat)
    Q_expr = cp.reshape(Q_flat, (P, P))

    # The constraint is: x^T Q x >= C for all x on simplex (x >= 0, sum x = 1)
    # This is: Q - C * ee^T is copositive
    # Doubly nonneg relaxation: Q - C * ee^T is PSD + entrywise nonneg
    ee = np.ones((P, 1))
    eeT = ee @ ee.T

    diff = Q_expr - C_var * eeT
    constraints = [
        cp.sum(lam) == 1,
        diff >> 0,
        diff >= 0,  # entrywise nonneg
    ]

    prob = cp.Problem(cp.Maximize(C_var), constraints)

    solver_to_use = solver or PRIMARY_SOLVER
    kwargs = {'verbose': verbose}
    if solver_to_use == 'MOSEK':
        kwargs['mosek_params'] = {
            'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-9,
            'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-9,
            'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-9,
        }

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Solution may be inaccurate')
        prob.solve(solver=solver_to_use, **kwargs)

    elapsed = time.time() - t0

    if prob.status not in ('optimal', 'optimal_inaccurate'):
        return {'P': P, 'C': None, 'status': prob.status, 'time': elapsed}

    return {
        'P': P,
        'C': float(C_var.value),
        'status': prob.status,
        'time': elapsed,
        'lambda': lam.value,
        'method': 'simplex_copositive',
    }


if __name__ == '__main__':
    print("=" * 80)
    print("EXPERIMENT 2A: Doubly Nonnegative Relaxation")
    print("=" * 80)

    for N in [5, 10, 20]:
        for M in [20, 50]:
            res = solve_doubly_nonneg(N, M)
            if res['C'] is not None:
                print(f"  N={N:3d}, M={M:3d}: C*={res['C']:.8f}, time={res['time']:.1f}s")
            else:
                print(f"  N={N:3d}, M={M:3d}: FAILED ({res['status']}), time={res['time']:.1f}s")

    print("\n" + "=" * 80)
    print("EXPERIMENT 2B: Copositive Level-1 (with MOSEK)")
    print("=" * 80)

    for N in [5, 10, 20]:
        for M in [20, 50]:
            res = solve_copositive_parrilo_1(N, M)
            if res['C'] is not None:
                print(f"  N={N:3d}, M={M:3d}: C*={res['C']:.8f}, time={res['time']:.1f}s")
            else:
                print(f"  N={N:3d}, M={M:3d}: FAILED ({res['status']}), time={res['time']:.1f}s")

    print("\n" + "=" * 80)
    print("EXPERIMENT 2C: Direct Discretized Dual")
    print("=" * 80)

    for M in [10, 20, 30, 50]:
        res = solve_discretized_direct(M)
        if res['C'] is not None:
            print(f"  M={M:3d}: C*={res['C']:.8f}, time={res['time']:.1f}s")
        else:
            print(f"  M={M:3d}: FAILED ({res['status']}), time={res['time']:.1f}s")

    print("\n" + "=" * 80)
    print("EXPERIMENT 2D: Simplex Copositive Dual (bounds V(P))")
    print("=" * 80)

    print("\n--- Small P validation ---")
    for P in range(2, 16):
        res = solve_simplex_copositive(P)
        if res['C'] is not None:
            compare_with_baseline(P, res['C'], res['time'], 'Simplex_COP')
        else:
            print(f"  P={P}: FAILED ({res['status']})")

    print("\n--- Large P scaling ---")
    for P in [20, 30, 50, 75, 100, 150, 200]:
        res = solve_simplex_copositive(P)
        if res['C'] is not None:
            shor = 2 * P / (2 * P - 1)
            print(f"  P={P:3d}: C*={res['C']:.6f}, Shor={shor:.6f}, diff={res['C']-shor:+.6f}, "
                  f"time={res['time']:.1f}s")
        else:
            print(f"  P={P:3d}: FAILED ({res['status']}), time={res['time']:.1f}s")
        if res['time'] > 300:
            break
