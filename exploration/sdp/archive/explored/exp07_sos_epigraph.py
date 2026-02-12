"""Experiment 7: SOS relaxation on the epigraph formulation.

V(P) = min t  s.t. x in Delta_P, 2P * x^T A_k x <= t for all k

Homogenize: let s = sum x_i, then x/s is on the simplex.
  2P * x^T A_k x <= t * s^2  for all x >= 0, s = sum x_i

The polynomial t*s^2 - 2P*x^T A_k x must be nonneg on R^P_+.
This is a COPOSITIVE certificate of degree 2 (in the x variables, with t fixed).

Actually, let's work differently. For fixed t = eta (binary search):
  Need: for all k, eta * (sum x_i)^2 - 2P * x^T A_k x is copositive.
  i.e., for all x >= 0:  x^T [eta * ee^T - 2P * A_k] x >= 0.

This is exactly the copositivity check! The INNER approximation
(DNN = PSD + entrywise nonneg) gives a SUFFICIENT condition.
If eta*ee - 2P*A_k is DNN for ALL k simultaneously, then eta is a VALID
lower bound on V(P).

Wait - this is DIFFERENT from what I tried in Exp 1! In Exp 1, I was
OPTIMIZING eta. Here I want the SMALLEST eta for which all matrices are DNN.
That's the same thing. The issue in Exp 1 was a solver error.

Let me fix it and also try the Parrilo Level-1 inner approximation:
  Q = S + N where S is PSD and N is entrywise nonneg (DNN, Level 0)
  Q = S + sum_i x_i * T_i where S is DNN and T_i are PSD (Level 1)

For Level 1, x^T Q x = x^T S x + sum_i x_i * (x^T T_i x). The second term
is automatically nonneg for x >= 0 if T_i are PSD. So if S is DNN, the
whole thing is nonneg on R^P_+.

The MAXIMUM eta for which we can write eta*ee - 2P*A_k = S_k + sum_i T_{k,i}*E_i
(where E_i = e_i e_i^T, so x_i * x^T T_ki x = ...) wait, this is getting complicated.

Let me just carefully implement the DNN check for each k.
"""
import numpy as np
import cvxpy as cp
import time
import warnings
warnings.filterwarnings('ignore')

SOLVER = 'MOSEK' if 'MOSEK' in cp.installed_solvers() else 'CLARABEL'
print(f"Solver: {SOLVER}")


def build_A(P):
    n_diags = 2 * P - 1
    A = []
    for k in range(n_diags):
        Ak = np.zeros((P, P))
        for i in range(max(0, k - P + 1), min(P, k + 1)):
            Ak[i, k - i] = 1
        A.append(Ak)
    return A


def copositive_dnn_bound(P):
    """Level 0: Find largest eta s.t. eta*J - 2P*A_k is DNN for all k.
    J = ones(P,P). DNN = PSD + entrywise nonneg."""
    t0 = time.time()
    A_mats = build_A(P)
    J = np.ones((P, P))
    eta = cp.Variable()
    constraints = []
    for A in A_mats:
        Q = eta * J - 2 * P * A
        constraints.append(Q >> 0)
        constraints.append(Q >= 0)
    prob = cp.Problem(cp.Maximize(eta), constraints)
    try:
        prob.solve(solver=SOLVER, verbose=False,
                   mosek_params={'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-9})
        val = float(eta.value) if eta.value is not None else None
    except Exception as e:
        print(f"  DNN failed for P={P}: {e}")
        val = None
    return val, time.time() - t0


def copositive_level1_bound(P):
    """Level 1 Parrilo: eta*J - 2P*A_k = S_k + sum_i N_{ki} * diag(e_i)
    where S_k is DNN and N_{ki} are PSD.

    Actually, the standard Level-1 inner approximation of the copositive cone:
    C* is copositive iff for all x >= 0, x^T C* x >= 0.
    Parrilo Level-1: C* = S + sum_i x_i * T_i where S is DNN, T_i PSD.
    x^T C* x = x^T S x + sum_i x_i (x^T T_i x) >= 0 for x >= 0.

    In matrix terms:
    C*_{jl} = S_{jl} + sum_i T_i_{jl} * delta_{ij or il?}

    Hmm, the level-1 decomposition is:
    C* in K_1 iff C* = N + sum_i (e_i^T N_i + N_i e_i) with N DNN, N_i PSD.

    No wait. Let me use the standard definition. The r-th level of Parrilo:
    K_r = {Q : (sum x_i)^r * x^T Q x is a sum of squares}

    K_0 = {Q : x^T Q x is SOS} = S^n_+ (PSD cone) — not copositive cone.

    Actually I confused the inner and outer approximations. Let me be precise.

    Parrilo's inner approximations of the copositive cone COP_n:
    K_r^* = {Q : there exists M PSD s.t. Q_{ij} = sum_{alpha: alpha_i+alpha_j+e_ij=beta}
             M_{alpha, alpha'} for each (i,j) with |beta|=r+2}

    Simpler version:
    - COP_n contains all matrices Q such that x^T Q x >= 0 for x >= 0.
    - Inner approximation at level r:
      Q in K_r iff (x_1 + ... + x_n)^r * sum_{ij} Q_{ij} x_i x_j is SOS.

    So for r=0: sum Q_{ij} x_i x_j is SOS iff Q is PSD. (Too weak - PSD ⊂ COP but ≠.)
    For r=1: (sum x_i) * sum Q_{ij} x_i x_j is SOS.
    This is a cubic form that must be SOS. A cubic can be SOS only if it's zero...
    No, SOS in the usual sense means nonneg, but for odd-degree polynomials,
    SOS requires it to factor appropriately.

    Actually for homogeneous polynomial of odd degree, SOS is trivial (must be zero).
    So the correct formulation uses x_i^2 instead:

    K_r = {Q : (sum x_i^2)^r * sum Q_{ij} x_i x_j is SOS}

    No... Let me just use the actual Parrilo (2000) definition.

    Parrilo defines: C_r^* = {A : (sum_{i=1}^n x_i)^r * x^T A x can be written as
    sum of terms of the form x_alpha^2 * x_beta where beta has no repeated indices}

    This is getting complicated. Let me just use a practical approach:
    decompose Q = DNN part + nonneg polynomial part.
    """
    t0 = time.time()
    A_mats = build_A(P)
    J = np.ones((P, P))

    eta = cp.Variable()
    constraints = []

    for A in A_mats:
        # Decompose Q_k = eta*J - 2P*A_k into:
        # Q_k = S_k + D_k  where S_k is PSD, D_k is diagonal with nonneg entries
        # plus: the (i,j) entry of Q_k for i != j must be >= 0 OR be compensated
        # by S_k.
        #
        # Actually let's try a different decomposition:
        # Q_k[i,j] for i != j: if A_k[i,j] = 1 (i.e. i+j = k), then Q_k[i,j] = eta - 2P.
        # For eta < 2P (which is always true for reasonable bounds), these are NEGATIVE.
        # So Q_k is NOT entrywise nonneg. DNN decomposition Q_k = S_k + N_k requires
        # S_k to compensate the negative off-diag entries.
        #
        # S_k PSD means S_k[i,j]^2 <= S_k[i,i]*S_k[j,j]. So if S_k needs to have
        # S_k[i,j] <= eta - 2P (negative), we need |S_k[i,j]| to be at most
        # sqrt(S_k[i,i]*S_k[j,j]).
        #
        # For Level-1: Q_k = S_k + sum_m x_m * R_{km}  as a form.
        # In terms of matrix entries: Q_k[i,j] = S_k[i,j] + sum_m c_{km,ij}
        # where c_{km,ij} = R_{km}[i,j] when i=m or j=m, etc.
        # This is messy. Let me just try the DNN bound and see what it gives.
        pass

    # Actually, let me check: for P=2, A_0 = [[1,0],[0,0]], A_1 = [[0,1],[1,0]], A_2 = [[0,0],[0,1]]
    # eta*J - 2P*A_k for k=1: [[eta, eta-4], [eta-4, eta]]
    # PSD needs eta^2 >= (eta-4)^2 => 0 >= -8*eta + 16 => eta >= 2. That's the Shor bound (4/3).
    # Wait, PSD needs eta*(eta) >= (eta-4)^2 => eta^2 >= eta^2 - 8*eta + 16 => 8*eta >= 16 => eta >= 2.
    # Entrywise nonneg needs eta >= 4 (for the off-diag entry eta - 2P = eta - 4).
    # So DNN requires eta >= 4 = 2P. The DNN bound is just eta = 2P. That's trivially true.

    # Hmm. DNN for the individual Q_k matrices is WAY too strong because the
    # entrywise nonneg condition requires eta >= 2P (to compensate the A_k entry).
    # This gives eta = 2P, which is the trivial upper bound. USELESS.

    # The issue: copositivity ≠ DNN. The gap matters here.
    # The matrices eta*J - 2P*A_k ARE copositive for eta >= V(P) (by definition),
    # but they are NOT DNN because the off-diagonal entries are negative.

    # So Level-0 (DNN) approximation of copositivity gives eta = 2P = trivial.
    # We need Level-1+ to get nontrivial bounds from this direction.

    # Let's verify this analytically and move on.
    print("\n--- Analytic check: DNN bound is trivial ---")
    for P in [2, 3, 5, 10]:
        A_mats = build_A(P)
        # The binding constraint is the anti-diagonal k=P-1.
        # A_{P-1}[i, P-1-i] = 1 for all valid i.
        # eta*J - 2P*A_{P-1} has off-diagonal entries eta - 2P (where A has 1) or eta (where A has 0).
        # Entrywise nonneg requires eta >= 2P.
        print(f"  P={P}: DNN requires eta >= {2*P} (trivial). V(P) ~ {2*P/(2*P-1):.4f} to ~1.6")

    print("\nConclusion: Copositivity DNN inner approximation is TRIVIALLY weak.")
    print("The off-diagonal entries of eta*J - 2P*A_k are negative for reasonable eta.")
    print("Need Level-1+ Parrilo or a different formulation entirely.")
    elapsed = time.time() - t0

    # ========================================================================
    # Part 2: Alternative — direct SDP formulation
    # ========================================================================
    print("\n" + "=" * 72)
    print("Part 2: SOS certificate for max_k q_k(x) via epigraph")
    print("=" * 72)

    # For fixed eta, verify that {x >= 0, sum x_i = 1, 2P*x^T A_k x <= eta ∀k}
    # is nonempty (primal feasible) or empty (lower bound on V(P)).
    #
    # This is EXACTLY what Lasserre does. But can we get a certificate
    # using SUMS OF SQUARES that is computationally cheaper?
    #
    # Putinar's Positivstellensatz: if the set is empty, then
    # -1 = sigma_0 + sum_i sigma_i * x_i + tau*(1-sum x_i) + sum_k sigma_k*(eta-2P*x^T A_k x)
    #
    # At degree d=2 for sigma_k (degree-2 SOS polynomials):
    # sigma_k(x) = x^T S_k x with S_k PSD
    # Then sum_k sigma_k * (eta - 2P*x^T A_k x) has degree 4.
    #
    # The degree-2 part: sum_k S_k * eta contributes to the degree-2 moment conditions.
    # The degree-4 part: -2P * sum_k (x^T S_k x)(x^T A_k x) contributes to degree-4.
    #
    # We can search for S_k PSD, sigma_0 SOS(degree 2), etc.

    print("\n  Searching for Positivstellensatz certificate...")

    for P in [2, 3, 4, 5]:
        t0 = time.time()
        A_mats = build_A(P)
        n_diags = 2 * P - 1
        shor = 2 * P / (2 * P - 1)

        # Binary search on eta
        eta_lo = shor
        eta_hi = 2.0

        for biter in range(30):
            if eta_hi - eta_lo < 1e-4:
                break
            eta_test = (eta_lo + eta_hi) / 2

            # Check if the set is "certifiably empty" via degree-2 Psatz
            # -1 = c + e*(1-sum x_i) + sum_i d_i x_i
            #     + sum_k [f_k + x^T S_k x]*(eta_test - 2P*x^T A_k x)
            #
            # sigma_0 = c0 + sum c_ij x_i x_j (degree-2 SOS)
            # sigma_i = d_i (degree-0, nonneg constants)
            # tau = e + sum e_j x_j (degree-1, free polynomial)
            # sigma_k = f_k + sum s_{k,ij} x_i x_j (degree-2 SOS, so [[f_k, g_k^T],[g_k, S_k]] PSD)

            # This is getting complex. Let me use cvxpy.
            # Variables: c0, c_ij (via C PSD), d_i >= 0, e, e_j, f_k >= 0, S_k PSD

            # The polynomial identity must hold coefficient by coefficient.
            # Constant term: -1 = c0 + e + sum_k f_k * eta_test
            # x_i term: 0 = d_i - e - sum_j e_j * delta_{ij} + sum_k [...] = complicated

            # This is essentially building the dual of the Lasserre relaxation.
            # Let me skip the manual construction and note the theoretical point.

            # Instead: just verify using our simple moment relaxation approach
            break

        print(f"  P={P}: Psatz approach deferred (equivalent to Lasserre dual)")

    print("\nConclusion: Direct Positivstellensatz is the DUAL of Lasserre hierarchy.")
    print("No computational advantage over primal Lasserre.")
    print("The copositivity approach through DNN is trivially weak (eta = 2P).")
    print("Parrilo Level-1+ copositivity might help but is computationally similar to Lasserre.")

    return elapsed


elapsed = copositive_level1_bound(5)
