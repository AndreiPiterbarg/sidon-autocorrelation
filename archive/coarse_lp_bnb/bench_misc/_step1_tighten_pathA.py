"""
Step 1 — tighten Path A's chain at K=2 via:
  1. Bang-bang on the L1-bounded, L_inf-bounded Fourier-mass distribution
  2. Bochner / Toeplitz-PSD constraint on the {z_n^2} sequence
  3. Sweep K up from 2 (no upper bound on K, but K=2 is the binding minimum)

Setup:
  f >= 0, supp f subset [-1/4, 1/4], int f = 1, M = ||f*f||_inf
  y_n := |f_hat(n)|^2 (period-1 Fourier; supp f embedded in [-1/2, 1/2])
  y_0 = 1
  y_n in [0, mu] for n >= 1, mu := mu(M) = M sin(pi/M) / pi  (MO 2.14)
  Sum_{n>=1} y_n = (K - 1) / 2  (Parseval; K = ||f||_2^2)
  K >= 2  (Cauchy-Schwarz on supp; equality only for uniform f)
  Toeplitz [y_{|i-j|}] PSD  (Bochner: R_f >= 0)
  ||f*f||_2^2 = 1 + 2 Sum_{n>=1} y_n^2

Goal: bound c_emp(M) := ||f*f||_2^2 / M from above.
Hyp_R threshold: c_* = log(16) / pi ~= 0.88254.

We compute:
  * loose Path A: c_emp <= (1 + mu (K-1)) / M
  * bang-bang on {y_n in [0,mu], sum = S}: closed-form
  * SDP relaxation with Toeplitz-PSD: cvxpy
  * loop over K to identify the worst K
"""

import numpy as np
import cvxpy as cp

LOG16_PI = np.log(16) / np.pi  # ~0.88254

def mu_M(M):
    return M * np.sin(np.pi / M) / np.pi

def loose_path_a(M, K):
    """Path A's chain (†): c_emp <= (1 + mu (K-1)) / M."""
    return (1 + mu_M(M) * (K - 1)) / M

def bang_bang(M, K):
    """Closed-form max of sum y_n^2 with y_n in [0,mu], sum y_n = S = (K-1)/2.
    Without Toeplitz PSD."""
    mu = mu_M(M)
    S = (K - 1) / 2
    if S <= 0:
        return 1.0 / M
    k = int(np.floor(S / mu))
    res = S - k * mu
    T = k * mu**2 + res**2
    return (1 + 2 * T) / M

def sdp_relaxation(M, K, N=20, verbose=False):
    """SDP: max sum_{n=1..N} y_n^2 with
       - y_n in [0, mu]
       - sum y_n <= S (truncation: tail >= 0)
       - Toeplitz [y_{|i-j|}] (size N+1) PSD with y_0 = 1
       - lifted Y = y y^T relaxation: M_lift = [[1, y^T];[y, Y]] PSD,
         diag(Y) <= mu^2

    Tail: sum_{n>N} y_n^2 <= mu * sum_{n>N} y_n <= mu * (S - sum_{n=1..N} y_n)
    So total bound = sum diag(Y) + mu * (S - sum y).
    """
    mu = mu_M(M)
    S = (K - 1) / 2

    y = cp.Variable(N, nonneg=True)
    Y = cp.Variable((N, N), symmetric=True)

    constraints = [y <= mu]
    constraints += [cp.sum(y) <= S]              # truncation
    constraints += [cp.diag(Y) <= mu**2]
    constraints += [cp.diag(Y) >= 0]

    # Lifted PSD: [[1, y^T];[y, Y]] >> 0  =>  Y >= y y^T
    Mlift = cp.bmat([[np.array([[1.0]]), cp.reshape(y, (1, N))],
                     [cp.reshape(y, (N, 1)), Y]])
    constraints += [Mlift >> 0]

    # Toeplitz PSD on (y_0=1, y_1, ..., y_N)
    T_size = N + 1
    rows = []
    for i in range(T_size):
        row = []
        for j in range(T_size):
            idx = abs(i - j)
            if idx == 0:
                row.append(np.array([[1.0]]))
            else:
                row.append(cp.reshape(y[idx-1], (1, 1)))
        rows.append(row)
    T = cp.bmat(rows)
    constraints += [T >> 0]

    # Maximize sum diag(Y)  + tail bound mu * (S - sum y)
    obj_inner = cp.sum(cp.diag(Y))
    tail = mu * (S - cp.sum(y))
    objective = cp.Maximize(obj_inner + tail)

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=verbose, eps=1e-9, max_iters=20000)
    except Exception as e:
        print(f"  Solver error: {e}")
        return None
    if prob.status not in ("optimal", "optimal_inaccurate"):
        print(f"  Solver status: {prob.status}")
        return None

    T_val = prob.value
    return (1 + 2 * T_val) / M, y.value, np.diag(Y.value)

def main():
    M = 1.378
    print(f"Hyp_R target: c_* = log(16)/pi = {LOG16_PI:.6f}")
    print(f"Working at M = M_max = {M}")
    print(f"mu(M) = {mu_M(M):.6f}")
    print()

    # K = 2 (minimum)
    K = 2.0
    print(f"=== K = {K} (minimum, Cauchy-Schwarz on supp) ===")
    print(f"  Path A (loose):  c_emp <= {loose_path_a(M, K):.6f}")
    print(f"  Bang-bang:       c_emp <= {bang_bang(M, K):.6f}")
    for N in [3, 5, 8, 12, 16]:
        result = sdp_relaxation(M, K, N=N)
        if result is not None:
            ce, y_val, diag_Y = result
            print(f"  SDP (N={N:>2}):       c_emp <= {ce:.6f}")

    # Sweep K
    print()
    print(f"=== K sweep at M = {M} ===")
    print(f"  K     | loose  | bang-bang | SDP(N=8)")
    print(f"  ------|--------|-----------|---------")
    for K in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
        loose = loose_path_a(M, K)
        bb = bang_bang(M, K)
        sdp_res = sdp_relaxation(M, K, N=8)
        sdp_str = f"{sdp_res[0]:.4f}" if sdp_res else "FAIL"
        print(f"  {K:.1f}   | {loose:.4f} |  {bb:.4f}  |  {sdp_str}")

    print()
    print(f"Hyp_R threshold: {LOG16_PI:.6f}")
    print("If any column entry exceeds 0.88254, the chain doesn't close at that K.")

if __name__ == "__main__":
    main()
