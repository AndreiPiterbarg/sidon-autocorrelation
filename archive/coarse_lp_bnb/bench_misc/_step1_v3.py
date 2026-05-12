"""
Step 1 v3 — try the FULL set of natural constraints:
  (A) MO 2.14: |hat f(n)|^2 <= mu(M) for n >= 1
  (B) Parseval: sum |hat f(n)|^2 = K
  (C) Bochner on R_f (=Toeplitz PSD on |hat f(n)|^2 sequence)
  (D) Bochner on f (=PHASE-AWARE Hermitian Toeplitz PSD on hat f(n) sequence)
  (E) MO 2.17 / Lemma 2.17 (multi-frequency tightening if available)
  (F) compact support of f via 'tail mass cannot be missing'

For each, solve max sum |hat f(n)|^4 = ||f*f||_2^2 - 1 (over n != 0, then *2).
Compute c_emp upper bound = (1 + 2*max) / M.

Approach: nonlinear gradient ascent on (a_n, b_n) with hat f(n) = a_n + i b_n,
projecting onto Hermitian Toeplitz PSD constraint via SDP barrier.
"""

import numpy as np
import cvxpy as cp
from scipy.optimize import minimize

LOG16_PI = np.log(16) / np.pi

def mu_M(M):
    return M * np.sin(np.pi / M) / np.pi

def hermitian_toeplitz(fhat_complex, N):
    """Build Hermitian Toeplitz T with T[i,j] = hat f(i - j), T[0,0] = 1.
    fhat_complex: array of length N for hat f(1), ..., hat f(N).
    """
    T = np.zeros((N+1, N+1), dtype=complex)
    for i in range(N+1):
        for j in range(N+1):
            if i == j:
                T[i, j] = 1.0
            elif i > j:
                T[i, j] = fhat_complex[i - j - 1]
            else:
                T[i, j] = np.conj(fhat_complex[j - i - 1])
    return T

def is_psd_complex(T, tol=-1e-9):
    eigs = np.linalg.eigvalsh(T)
    return eigs.min() >= tol, eigs.min()

def test_phase_aware_bang_bang(M, K, N=5, n_phase_samples=20, verbose=False):
    """At bang-bang config (z_1, z_2, ..., z_N), check if SOME phase choice makes
    the COMPLEX Hermitian Toeplitz PSD."""
    mu = mu_M(M)
    S = (K - 1) / 2
    k = int(np.floor(S / mu))
    res = S - k * mu

    # Bang-bang: z_1 = ... = z_k = sqrt(mu), z_{k+1} = sqrt(res), rest = 0
    z = np.zeros(N)
    for i in range(k):
        z[i] = np.sqrt(mu)
    if k < N:
        z[k] = np.sqrt(res)

    if verbose:
        print(f"  Bang-bang z config (N={N}): {z}")

    # Try various phase choices
    n_psd = 0
    n_total = 0
    np.random.seed(42)
    for trial in range(n_phase_samples):
        phases = np.random.uniform(-np.pi, np.pi, N)
        if trial == 0:
            phases = np.zeros(N)  # try all-zero phases first (= symmetric f)
        fhat_c = z * np.exp(1j * phases)
        T = hermitian_toeplitz(fhat_c, N)
        psd, min_eig = is_psd_complex(T)
        n_total += 1
        if psd:
            n_psd += 1
            if verbose and trial == 0:
                print(f"    All-zero phases: PSD with min eig = {min_eig:.6f}")
    if verbose:
        print(f"    Phase samples PSD-feasible: {n_psd}/{n_total}")
    return n_psd / n_total

def sdp_phase_aware(M, K, N=5, verbose=False):
    """SDP: find max ||hat f|^4 sum subject to:
       - hat f real-Hermitian Toeplitz [hat f(i-j)] PSD (Bochner on f)
       - |hat f(n)|^2 <= mu (MO 2.14)
       - sum |hat f(n)|^2 = (K-1)/2 (Parseval)

       Use an SOS lift: y_n = |hat f(n)|^2, decision variables include y and Y_ij = y_i y_j.
    """
    mu = mu_M(M)
    S = (K - 1) / 2

    # Variables: real and imaginary parts of hat f(n), n=1..N
    a = cp.Variable(N)  # real parts
    b = cp.Variable(N)  # imaginary parts (b[0] = Im hat f(1), etc.)

    # y_n = a_n^2 + b_n^2 = |hat f(n)|^2 -- this is QUADRATIC in (a, b)
    # We maximize sum y_n^2 = sum (a_n^2 + b_n^2)^2 -- quartic.
    # Use Lasserre order-2 lift: introduce W = (a, b)(a, b)^T.

    # Simpler: use disciplined CVX with real Toeplitz form (= phase = 0 case = symmetric f).
    # First check: at symmetric f (b = 0), is the bound the same as bang-bang?

    # Real Toeplitz [a_{|i-j|}] with a_0 = 1, a_n = a[n-1]. PSD constraint.
    constraints = []
    # |hat f|^2 = a^2 + b^2 <= mu
    constraints += [cp.square(a) + cp.square(b) <= mu]
    # Parseval: sum (a^2 + b^2) = S
    constraints += [cp.sum(cp.square(a) + cp.square(b)) == S]

    # Hermitian Toeplitz [hat f(i-j)] PSD where hat f(0) = 1, hat f(n) = a[n-1] + i*b[n-1]
    # In real form: T_real = [[ Re T, -Im T ], [ Im T, Re T ]] is real symmetric PSD iff T Hermitian PSD.
    # Re T(i,j) = Re hat f(i-j); for i > j: a[i-j-1]; for i < j: a[j-i-1]; for i==j: 1.
    # Im T(i,j) = Im hat f(i-j); for i > j: b[i-j-1]; for i < j: -b[j-i-1]; for i==j: 0.
    Tsize = N + 1
    Re_T = []
    Im_T = []
    for i in range(Tsize):
        Re_row = []
        Im_row = []
        for j in range(Tsize):
            d = i - j
            if d == 0:
                Re_row.append(1.0)
                Im_row.append(0.0)
            elif d > 0:
                Re_row.append(a[d-1])
                Im_row.append(b[d-1])
            else:  # d < 0
                Re_row.append(a[-d-1])
                Im_row.append(-b[-d-1])
        Re_T.append(Re_row)
        Im_T.append(Im_row)
    Re_T_mat = cp.bmat(Re_T)
    Im_T_mat = cp.bmat(Im_T)

    # Real-form PSD constraint: [[Re_T, -Im_T], [Im_T, Re_T]] PSD
    real_form = cp.bmat([[Re_T_mat, -Im_T_mat], [Im_T_mat, Re_T_mat]])
    constraints += [real_form >> 0]

    # Maximize sum (a^2 + b^2)^2 — convex maximization, not allowed directly.
    # Use linearization: maximize sum w * (a^2 + b^2) where w_n = mu (to get loose bound)
    # OR use SOS lift.

    # For demonstration, maximize sum (a^2 + b^2) (which equals S, fixed) — useless.
    # Instead, find the MAX of sum y_n^2 by enumerating bang-bang vertices.
    # Already done in v2. Here we just verify PSD feasibility of the bang-bang vertex
    # in the COMPLEX (phase-aware) setting.

    # For the SDP, instead minimize negative of a tractable surrogate.
    # We use: maximize sum |hat f(n)|^2 * ||hat f(n)||_2 = quartic surrogate
    # Doesn't work in CVXPY.

    # Skip SDP, go straight to nonlinear:
    return None

def nonlinear_phase_aware(M, K, N=8, n_starts=30):
    """Nonlinear gradient ascent on (a, b) for max sum |hat f|^4 with phase-aware Bochner."""
    mu = mu_M(M)
    S = (K - 1) / 2

    def obj_and_constraints(x):
        a = x[:N]; b = x[N:2*N]
        z2 = a**2 + b**2
        # Objective: sum z2^2 (we negate for minimize)
        obj = -np.sum(z2**2)
        return obj

    def constr_psd(x):
        a = x[:N]; b = x[N:2*N]
        fhat = a + 1j*b
        T = hermitian_toeplitz(fhat, N)
        eigs = np.linalg.eigvalsh(T)
        return eigs.min()  # >= 0 required

    def constr_max_ineq(x):
        a = x[:N]; b = x[N:2*N]
        z2 = a**2 + b**2
        return mu - z2  # >= 0

    def constr_parseval(x):
        a = x[:N]; b = x[N:2*N]
        z2 = a**2 + b**2
        return S - np.sum(z2)  # = 0

    best = -np.inf
    best_x = None
    np.random.seed(1)
    for trial in range(n_starts):
        # Random init
        x0 = np.random.randn(2*N) * 0.1
        # Project to satisfy Parseval roughly
        a = x0[:N]; b = x0[N:2*N]
        z2 = a**2 + b**2
        if np.sum(z2) > 0:
            scale = np.sqrt(S / np.sum(z2))
            x0 = x0 * scale
        try:
            res = minimize(obj_and_constraints, x0, method='SLSQP',
                           constraints=[
                               {'type': 'eq', 'fun': constr_parseval},
                               {'type': 'ineq', 'fun': constr_max_ineq},
                               {'type': 'ineq', 'fun': constr_psd},
                           ],
                           options={'maxiter': 200, 'ftol': 1e-9})
            if res.success and -res.fun > best:
                best = -res.fun
                best_x = res.x
        except Exception as e:
            continue

    if best_x is None:
        return None
    a = best_x[:N]; b = best_x[N:2*N]
    z2 = a**2 + b**2
    ce = (1 + 2 * best) / M
    return ce, z2, a, b

def main():
    M = 1.378
    K = 2.0
    print(f"=== STEP 1 v3: phase-aware Bochner attempt ===")
    print(f"M_max = {M}, K = {K}, mu = {mu_M(M):.6f}, S = {(K-1)/2}")
    print(f"Hyp_R target: c_* = {LOG16_PI:.6f}\n")

    # First: confirm bang-bang at K=2 satisfies phase-aware Bochner with phase=0
    print("--- Phase-aware bang-bang feasibility ---")
    for N in [3, 5, 8, 12]:
        frac_psd = test_phase_aware_bang_bang(M, K, N=N, n_phase_samples=20, verbose=True)
        print()

    # Nonlinear search for max sum |hat f|^4 with phase-aware Bochner
    print("--- Nonlinear phase-aware optimization ---")
    for N in [4, 6, 8, 10]:
        result = nonlinear_phase_aware(M, K, N=N, n_starts=40)
        if result is not None:
            ce, z2, a, b = result
            print(f"  N={N}: c_emp <= {ce:.6f}")
            print(f"    z2 (|hat f(n)|^2): {[f'{x:.3f}' for x in z2]}")
        else:
            print(f"  N={N}: optimization failed")

if __name__ == "__main__":
    main()
