"""Experiment 1: Primal copositivity approach.

Instead of the failed minimax dual, apply copositivity directly to the primal problem.

V(P) = min_{x in Delta_P} max_{k} 2P * x^T A_k x

Lower bound idea: Find largest eta such that for ALL k=0,...,2P-2:
  eta * (sum x_i)^2 - 2P * x^T A_k x  is copositive (nonneg on R^P_+)

Level 0 (DNN): matrix is PSD + entrywise nonneg
Level 1: SOS certificate with Parrilo's hierarchy

This avoids the minimax gap because we require the bound to hold for EACH k separately.
"""
import numpy as np
import cvxpy as cp
import time
import warnings
warnings.filterwarnings('ignore')

SOLVER = 'MOSEK' if 'MOSEK' in cp.installed_solvers() else 'CLARABEL'
print(f"Solver: {SOLVER}")


def build_A_matrices(P):
    """Build convolution matrices A_k for k=0,...,2P-2."""
    n_diags = 2 * P - 1
    A_mats = []
    for k in range(n_diags):
        A = np.zeros((P, P))
        for i in range(max(0, k - P + 1), min(P, k + 1)):
            j = k - i
            A[i, j] = 1
        A_mats.append(A)
    return A_mats


def solve_primal_copositive_L0(P):
    """Level-0 (DNN): eta*(ee^T) - 2P*A_k must be PSD + entrywise nonneg for all k."""
    t0 = time.time()
    A_mats = build_A_matrices(P)
    ee = np.ones((P, P))

    eta = cp.Variable()
    constraints = []
    for k, A in enumerate(A_mats):
        M = eta * ee - 2 * P * A
        constraints.append(M >> 0)      # PSD
        constraints.append(M >= 0)       # entrywise nonneg

    prob = cp.Problem(cp.Maximize(eta), constraints)
    prob.solve(solver=SOLVER, verbose=False)
    elapsed = time.time() - t0
    val = float(eta.value) if eta.value is not None else None
    return val, elapsed


def solve_primal_copositive_L1(P):
    """Level-1 Parrilo: eta*(sum x_i)^2 - 2P*x^T A_k x = z^T N_k z
    where z = (x_i * x_j) Kronecker-lifted, N_k is PSD + entrywise nonneg.

    Actually, for the simplex we need: for each k,
      eta*(sum x_i)^2 - 2P * x^T A_k x = sum_{i} x_i * (x^T R_{k,i} x)
    where each R_{k,i} is PSD (this gives copositivity on the simplex).
    """
    t0 = time.time()
    A_mats = build_A_matrices(P)

    eta = cp.Variable()
    constraints = []

    for k, A in enumerate(A_mats):
        # We need: eta * (sum x_i)^2 - 2P * x^T A_k x >= 0 for x >= 0
        # Parrilo Level-1: decompose as sum_i x_i * (x^T R_{k,i} x) with R_{k,i} PSD
        # The coefficient of x_i * x_j * x_l in the expansion must match.
        #
        # Simpler: use the DNN + one layer of multiplication by (sum x_i)
        # For homogeneous degree-2 polynomial p(x) copositive:
        # Level-1: (sum x_i) * p(x) = sum_i x_i * p(x) is a sum of terms x_i * p(x)
        # where p(x) = eta*(sum x_j)^2 - 2P*x^T A_k x
        # We need this degree-3 form to be nonneg on R^P_+.
        # Write as: sum_i x_i * [sum_{j,l} (eta - 2P*A_k[j,l]) x_j x_l]
        # = sum_{i,j,l} (eta - 2P*A_k[j,l]) x_i x_j x_l
        #
        # This is a cubic form. Parrilo: write it as x^T M_k x where M_k(x) is
        # a matrix-valued linear function. For Level-1, we need:
        # The form (sum x_i) * [eta*ee - 2P*A_k] to be "SOS on nonneg orthant"
        #
        # Let's use a simpler approach: for each k, introduce P matrices S_{k,i} (PxP, PSD)
        # such that eta*ee - 2P*A_k = sum_i S_{k,i} (entrywise) and each S_{k,i} is PSD.
        # Then p(x) = sum_i x^T S_{k,i} x and x_i * p(x) = sum_i x_i * x^T S_{k,i} x >= 0.
        # Wait, that's not right either.
        #
        # Actually the standard Parrilo Level-1 for copositivity of Q:
        # Q = N + sum_i x_i * S_i  where N is PSD+nonneg, S_i are PSD
        # But we need it on the simplex (sum x_i = 1, x >= 0), not just nonneg orthant.
        #
        # For the simplex, Q copositive on simplex iff Q_ij >= min_{x in Delta} x^T Q x.
        # Let's just use the simplex-adapted version:
        # Q(x) >= eta for x in Delta iff there exist sigma_0 (SOS), sigma_i (SOS) such that
        # x^T Q x - eta = sigma_0(x) * (1 - sum x_i)^2 + sum_i sigma_i(x) * x_i
        #
        # For degree-2 Q and degree-0 sigma_i, this gives:
        # x^T Q x - eta = c*(1-sum x_i)^2 + sum_i d_i * x_i
        # Expanding: x^T Q x - eta = c - 2c*sum x_i + c*(sum x_i)^2 + sum d_i * x_i
        # On the simplex (sum x_i = 1): x^T Q x - eta = c - 2c + c + sum d_i * x_i = sum d_i * x_i
        # So this just gives x^T Q x >= eta + sum d_i * x_i with d_i >= 0.
        # This is the LP relaxation, not strong enough.
        #
        # Let's try degree-1 multipliers: sigma_i(x) = sum_j s_{ij} x_j with s_{ij} >= 0
        # Then: x^T (2P*A_k) x <= eta on simplex
        # iff eta - x^T (2P*A_k) x = sum_i x_i * (sum_j s_{ij} x_j) + terms with (1-sum x_i)
        # = sum_{i,j} s_{ij} x_i x_j + ...
        # So eta*ee - 2P*A_k = S + c*ee + ... where S >= 0 entrywise.

        # Let me just try: for each k, require eta*ee - 2P*A_k = S_k + D_k
        # where S_k is PSD, D_k is diagonal with nonneg entries
        # This is a Parrilo-style decomposition adapted to the simplex.
        S_k = cp.Variable((P, P), symmetric=True)
        d_k = cp.Variable(P, nonneg=True)
        Q_k = eta * np.ones((P, P)) - 2 * P * A
        constraints.append(S_k >> 0)
        constraints.append(Q_k == S_k + cp.diag(d_k))

    prob = cp.Problem(cp.Maximize(eta), constraints)
    prob.solve(solver=SOLVER, verbose=False)
    elapsed = time.time() - t0
    val = float(eta.value) if eta.value is not None else None
    return val, elapsed


def solve_primal_copositive_L1b(P):
    """Level-1b: eta*ee - 2P*A_k = S_k + N_k where S_k PSD, N_k entrywise nonneg.
    This is exactly the DNN inner approximation applied per-k on the primal side.
    """
    t0 = time.time()
    A_mats = build_A_matrices(P)

    eta = cp.Variable()
    constraints = []

    for k, A in enumerate(A_mats):
        S_k = cp.Variable((P, P), symmetric=True)
        N_k = cp.Variable((P, P), symmetric=True)
        Q_k = eta * np.ones((P, P)) - 2 * P * A
        constraints.append(S_k >> 0)
        constraints.append(N_k >= 0)
        constraints.append(Q_k == S_k + N_k)

    prob = cp.Problem(cp.Maximize(eta), constraints)
    prob.solve(solver=SOLVER, verbose=False)
    elapsed = time.time() - t0
    val = float(eta.value) if eta.value is not None else None
    return val, elapsed


print("\n" + "=" * 72)
print("EXP 1: Primal copositivity — lower bound on V(P)")
print("=" * 72)
print(f"{'P':>3} | {'Shor':>10} | {'DNN (L0)':>10} | {'PSD+diag':>10} | {'PSD+NN':>10} | {'V(P) UB':>10}")
print("-" * 72)

known_ub = {2: 1.777778, 3: 1.706667, 4: 1.644465, 5: 1.633817,
            6: 1.600883, 7: 1.591746, 8: 1.580150, 9: 1.578073,
            10: 1.566436}

for P in range(2, 11):
    shor = 2 * P / (2 * P - 1)
    l0, t0 = solve_primal_copositive_L0(P)
    l1, t1 = solve_primal_copositive_L1(P)
    l1b, t1b = solve_primal_copositive_L1b(P)
    ub = known_ub.get(P, float('nan'))
    print(f"{P:>3} | {shor:>10.6f} | {l0:>10.6f} | {l1:>10.6f} | {l1b:>10.6f} | {ub:>10.6f}")

print("\nKey question: Do any of these beat the Shor bound 2P/(2P-1)?")
