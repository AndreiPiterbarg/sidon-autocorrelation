"""Experiment: Copositivity on the PRIMAL side.

Instead of the minimax dual (which gives Shor bound), we try:

  V(P) = min_{x in simplex} max_k (2P * x^T A_k x)

Reformulate as:
  V(P) = max c  s.t.  2P * x^T A_k x >= c  for all k, for all x in simplex

For each k, the constraint "x^T A_k x >= c/(2P) for all x in simplex"
is equivalent to "A_k - (c/(2P)) * 11^T is copositive on the simplex".

Using Parrilo's Level-r: A_k - c*J is copositive iff there exists N_k PSD s.t.
  (diag(x) * 1)^r * (A_k - c*J) * (diag(x) * 1)^r = sum of squares

For r=0: A_k - c*J is PSD + nonneg (DNN)
For r=1: multiply by x_i to get degree-4 form, must be SOS

But we need ALL k simultaneously. So we need:
  max c s.t. for all k: A_k - (c/(2P)) * J is copositive

This is a joint copositivity certification problem.

Actually, let me think more carefully. The primal is min_x max_k (2P * x^T A_k x).
The "max_k" is crucial - we need the WORST k for each x.

Alternative: For a given c, V(P) >= c iff for all x in simplex, exists k s.t.
2P * x^T A_k x >= c. This is NOT "for all k" - it's existential in k.

So direct copositivity on individual A_k is too strong. We'd need:
  For all x in simplex: max_k x^T A_k x >= c/(2P)

This is: simplex is covered by the union of sets {x: x^T A_k x >= c/(2P)}.

Hmm, this is harder. Let me try the S-procedure / joint approach instead.

Actually, consider this: if we can find weights w_k >= 0 with sum w_k = 1 such that
sum_k w_k A_k - (c/(2P)) * J is copositive, then for all x in simplex:
  sum_k w_k (x^T A_k x) >= c/(2P)
  => max_k (x^T A_k x) >= c/(2P)
  => max_k (2P * x^T A_k x) >= c

So the BEST such bound over w is:
  max c s.t. exists w in simplex: sum_k w_k A_k - (c/(2P)) * J is copositive

This is EXACTLY the minimax dual again! With copositivity instead of PSD.

Wait - but this time we're not requiring Q(w) PSD, we're requiring Q(w) - c*J
copositive, which is WEAKER than PSD. So higher Parrilo levels MIGHT help here.

Let me test this. The question is whether Parrilo Level-1 can beat the DNN bound.

Actually, let me reconsider. The notebook says the minimax dual gives:
  max_w min_x x^T Q(w) x  where Q(w) = 2P * sum w_k A_k

On the simplex, min_x x^T Q x = min_{1^T x = 1, x >= 0} x^T Q x.

If Q is DNN (PSD + entrywise nonneg), then min_x x^T Q x >= 0 on simplex.
If Q - c*J is DNN, then x^T Q x >= c * x^T J x = c*(1^T x)^2 = c.

But we want Q - c*J to be COPOSITIVE, not DNN. Copositive is weaker.
For r=0 (DNN): Q - c*J >> 0, Q - c*J >= 0 entrywise.
For r=1: multiply by x_i x_j factors and check SOS. Strictly stronger.

Let me just test whether Parrilo Level-1 for copositivity of
  Q(w) - c*J  on the simplex
gives anything better than DNN.

Key insight: The notebook says the optimal w is uniform and Q = (2P/(2P-1)) J.
So Q - c*J = ((2P/(2P-1)) - c) * J. This is DNN iff c <= 2P/(2P-1).
Is it copositive for larger c? Well, J = 11^T is PSD, so J is copositive.
((2P/(2P-1)) - c) * J is copositive iff (2P/(2P-1)) - c >= 0.

So for uniform w, copositivity also gives exactly 2P/(2P-1). But maybe
non-uniform w with copositivity gives a better bound?

Let me test numerically for small P.
"""

import numpy as np
import cvxpy as cp
from itertools import combinations_with_replacement
from collections import defaultdict
import time

SOLVER = 'MOSEK' if 'MOSEK' in cp.installed_solvers() else 'CLARABEL'
print(f'Solver: {SOLVER}')


def build_A_matrices(P):
    """Build the convolution matrices A_k for k=0,...,2P-2."""
    n_diags = 2 * P - 1
    A_mats = []
    for k in range(n_diags):
        A = np.zeros((P, P))
        for i in range(max(0, k - P + 1), min(P, k + 1)):
            j = k - i
            A[i, j] = 1
        A_mats.append(A)
    return A_mats


def solve_cop_primal_level0(P):
    """DNN copositivity: max c s.t. exists w: Q(w) - c*J is DNN."""
    A_mats = build_A_matrices(P)
    n_diags = len(A_mats)

    w = cp.Variable(n_diags, nonneg=True)
    c_var = cp.Variable()

    A_flat = np.array([A.flatten() for A in A_mats])
    Q_flat = 2 * P * (w @ A_flat)
    Q_expr = cp.reshape(Q_flat, (P, P), order='C')
    J = np.ones((P, P))

    M = Q_expr - c_var * J

    constraints = [
        cp.sum(w) == 1,
        M >> 0,  # PSD
        M >= 0,  # entrywise nonneg
    ]

    prob = cp.Problem(cp.Maximize(c_var), constraints)
    prob.solve(solver=SOLVER, verbose=False)
    return float(c_var.value) if c_var.value is not None else None


def solve_cop_primal_level1(P):
    """Parrilo Level-1 copositivity on primal side.

    Q(w) - c*J is copositive iff there exist:
    - N PSD (P x P) entrywise nonneg
    - Gram matrix G PSD of size D x D (D = dim of {x_i * x_j * x_k})
    such that:
    (Q(w) - c*J)_{ij} = N_{ij} + [contribution from x_i * (Q-cJ) * x_j being SOS with nonneg multiplier]

    Actually, Parrilo Level-1: M is copositive iff
    diag(x) * M * diag(x) is a sum of (nonneg)*(sos) forms of appropriate degree.

    Simpler formulation: M is copositive iff
    sum_{ij} M_ij * x_i * x_j >= 0 for all x >= 0.

    Level-0 (DNN): M = N + S where N >= 0 entrywise and S >> 0.
    Level-1: The form sum M_ij x_i x_j can be written as
             sum_k (nonneg polynomial)_k * (sos polynomial)_k * x_i1...x_ir
             i.e., (sum x_k) * x^T M x = x^T (inner matrix) x where inner is SOS

    Parrilo's exact formulation for Level-r:
    M is copositive <=> (sum_i x_i)^r * (x^T M x) = sum of squares + nonneg coeffs.

    For r=1: (sum_i x_i) * (x^T M x) = sum of x_i * M_{jk} * x_j * x_k
    This is a cubic form in x. It must be sum of squares. But odd-degree forms
    can't be SOS. So Parrilo actually uses:

    Level-r: (sum x_i^2)^r * (x^T M x) is SOS  (for the standard cone R^n_+)
    OR: (1^T x)^{2r} * (x^T M x) = SOS on the nonneg orthant.

    Actually for the standard simplex, x >= 0 and sum x_i = 1, so:
    x^T M x >= 0 for all x in simplex <=>
    (1^T x)^2 * (x^T M x) - which is degree 4, must be >= 0 on nonneg orthant.

    But (1^T x)^2 = 1 on the simplex, so this doesn't help directly.

    Let me use the SIMPLER Parrilo hierarchy for copositivity on R^n_+:
    Level-r: M is copositive if (e^T x)^{2r} * x^T M x = SOS (homogeneous of degree 2r+2)

    For r=1: (sum x_i)^2 * (sum M_ij x_i x_j) must be SOS as a degree-4 homogeneous polynomial.
    """
    t0 = time.time()
    A_mats = build_A_matrices(P)
    n_diags = len(A_mats)

    # Variables
    w = cp.Variable(n_diags, nonneg=True)
    c_var = cp.Variable()

    # We need (1^T x)^2 * x^T (Q-cJ) x to be SOS
    # This is a degree-4 homogeneous polynomial in P variables
    # Monomials of degree 2: basis for the Gram matrix
    basis = list(combinations_with_replacement(range(P), 2))
    D = len(basis)

    # Build exponent vectors
    exps = np.zeros((D, P), dtype=int)
    for idx, mono in enumerate(basis):
        for v in mono:
            exps[idx, v] += 1

    # Gram matrix for SOS
    G = cp.Variable((D, D), symmetric=True)

    # For each degree-4 monomial x^alpha, compute the coefficient from:
    # 1) The Gram matrix: sum G[i,j] * x^{exps[i]+exps[j]}
    # 2) (1^T x)^2 * x^T M x: expand (sum x_i)^2 = sum x_i^2 + 2*sum_{i<j} x_i x_j
    #    times sum M_{ab} x_a x_b

    # Collect degree-4 monomials
    coeff_map_gram = defaultdict(list)
    for i in range(D):
        for j in range(i, D):
            alpha = tuple(exps[i] + exps[j])
            mult = 1 if i == j else 2
            coeff_map_gram[alpha].append((i, j, mult))

    # Build coefficients of (1^T x)^2 * x^T M x
    # (1^T x)^2 has terms: for each pair (a,b), coefficient is 1 if a=b, 2 if a<b
    # x^T M x has terms: for each pair (c,d), coefficient M[c,d] (symmetrized: 2 if c<d)
    # Product: for each (a,b,c,d), coefficient is coeff_e * coeff_M * x_a x_b x_c x_d

    # Pre-compute which (a,b,c,d) contribute to each degree-4 monomial
    target_coeffs = defaultdict(list)
    for a in range(P):
        for b in range(a, P):
            e_coeff = 1 if a == b else 2  # coefficient in (1^T x)^2
            for c in range(P):
                for d in range(c, P):
                    m_coeff = 1 if c == d else 2  # coefficient in x^T M x
                    alpha = [0] * P
                    alpha[a] += 1
                    alpha[b] += 1
                    alpha[c] += 1
                    alpha[d] += 1
                    alpha = tuple(alpha)
                    target_coeffs[alpha].append((a, b, c, d, e_coeff * m_coeff))

    constraints = [cp.sum(w) == 1, G >> 0]

    # For each degree-4 monomial, Gram coefficient = target coefficient
    all_alphas = set(coeff_map_gram.keys()) | set(target_coeffs.keys())

    for alpha in all_alphas:
        # Gram side
        gram_expr = 0
        if alpha in coeff_map_gram:
            for i, j, mult in coeff_map_gram[alpha]:
                gram_expr = gram_expr + mult * G[i, j]

        # Target side: (1^T x)^2 * x^T (Q(w) - c*J) x
        if alpha in target_coeffs:
            target_expr = 0
            for a, b, c, d, coeff in target_coeffs[alpha]:
                # M[c,d] from Q(w) - c*J
                # Q(w)[c,d] = 2P * sum_k w_k * A_k[c,d]
                # (Q(w) - c*J)[c,d] = 2P * sum_k w_k * A_k[c,d] - c
                A_cd_vals = np.array([A[c, d] for A in A_mats])
                q_cd = 2 * P * (A_cd_vals @ w) - c_var  # Note: A_mats are NOT symmetric always
                # Actually A_k[c,d] = 1 if c+d=k and both in range, else 0
                # But we built them that way
                target_expr = target_expr + coeff * q_cd
            constraints.append(gram_expr == target_expr)
        else:
            constraints.append(gram_expr == 0)

    prob = cp.Problem(cp.Maximize(c_var), constraints)
    prob.solve(solver=SOLVER, verbose=False, mosek_params={
        'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,
        'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,
        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8,
    })
    elapsed = time.time() - t0
    c_val = float(c_var.value) if c_var.value is not None else None

    # Also get optimal w if available
    w_val = w.value if w.value is not None else None

    return c_val, elapsed, D, w_val


print("\n" + "=" * 72)
print("Copositivity on PRIMAL side: (1^T x)^2 * x^T (Q(w)-cJ) x = SOS")
print("Q(w) = 2P * sum w_k A_k, optimize over w and c")
print("=" * 72)

print(f"\n{'P':>3} | {'Shor':>10} | {'DNN (L0)':>10} | {'Level-1':>10} | {'Gap L1-Shor':>12} | {'D':>5} | {'Time':>6}")
print('-' * 72)

for P in range(3, 10):
    shor = 2 * P / (2 * P - 1)
    l0 = solve_cop_primal_level0(P)
    l1, t1, D, w_opt = solve_cop_primal_level1(P)
    diff = (l1 - shor) if l1 else 0
    l0_s = f"{l0:.6f}" if l0 else "FAIL"
    l1_s = f"{l1:.6f}" if l1 else "FAIL"
    print(f"{P:>3} | {shor:>10.6f} | {l0_s:>10} | {l1_s:>10} | {diff:>+12.2e} | {D:>5} | {t1:>5.1f}s")
    if w_opt is not None and l1 is not None:
        # Show non-trivial w entries
        nz = np.where(w_opt > 0.01)[0]
        if len(nz) <= 10:
            print(f"      w nonzero at k={list(nz)}, vals={np.round(w_opt[nz], 4)}")
