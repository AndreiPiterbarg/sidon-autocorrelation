"""
Step 1 — tighten Path A's chain at K=2 (CORRECTED).

Key facts:
  Variables y_n := |f_hat(n)|^2, n>=1, with y_0 = 1.
  Constraints: y_n in [0, mu], mu := mu(M) = M sin(pi/M)/pi  (MO 2.14)
               sum_{n>=1} y_n = (K-1)/2 =: S  (Parseval)
               Toeplitz [y_{|i-j|}] PSD  (Bochner; R_f >= 0)
  Objective:   max sum_{n>=1} y_n^2  (this gives ||f*f||_2^2 = 1 + 2*max)

Without Toeplitz PSD: max is bang-bang -- some y_n at mu, one at residual S-k*mu, rest 0.
This gives the smallest valid upper bound on the unconstrained problem.

With Toeplitz PSD: bang-bang vertices may be INFEASIBLE.
We:
  (a) enumerate candidate bang-bang vertex configurations (n-positions)
  (b) check Toeplitz PSD for each
  (c) the max sum-of-squares over PSD-feasible vertices is the rigorous bound
  (d) if all vertices infeasible, max attained at interior of feasibility,
      use SDP / nonlinear optimizer with PSD constraint

Note: bang-bang vertices are the BEST for max sum-of-squares (objective is convex),
so the rigorous max is at a vertex of {y_n in [0,mu], sum = S, Toeplitz PSD}.
PSD feasibility may force a non-bang-bang vertex if all bang-bangs fail.
"""

import numpy as np
from scipy.optimize import minimize
from itertools import combinations

LOG16_PI = np.log(16) / np.pi  # ~0.88254

def mu_M(M):
    return M * np.sin(np.pi / M) / np.pi

def loose_path_a(M, K):
    return (1 + mu_M(M) * (K - 1)) / M

def bang_bang_naive(M, K):
    """Bang-bang on (y_n in [0,mu], sum = S), ignoring Toeplitz PSD."""
    mu = mu_M(M)
    S = (K - 1) / 2
    if S <= 0:
        return 1.0 / M, [], 0.0
    k = int(np.floor(S / mu))
    res = S - k * mu
    T = k * mu**2 + res**2
    return (1 + 2 * T) / M, [(mu, k), (res, 1)], T

def make_toeplitz(y_seq, N=None):
    """y_seq: dict {n: y_n} for n in [0, N]. Returns (N+1)x(N+1) symmetric Toeplitz."""
    if N is None:
        N = max(y_seq.keys())
    T = np.zeros((N+1, N+1))
    for i in range(N+1):
        for j in range(N+1):
            idx = abs(i - j)
            T[i, j] = y_seq.get(idx, 0.0)
    return T

def is_psd(M, tol=-1e-9):
    return np.min(np.linalg.eigvalsh(M)) >= tol

def enumerate_bang_bang_psd(M, K, N_pos=15, N_toeplitz=None):
    """Enumerate bang-bang vertices: choose k positions from {1,..,N_pos} for y=mu,
       one position for residual, rest 0. Check Toeplitz PSD on (N_toeplitz+1)x size.
       Returns: dict of best vertex info."""
    if N_toeplitz is None:
        N_toeplitz = N_pos
    mu = mu_M(M)
    S = (K - 1) / 2
    k = int(np.floor(S / mu))
    res = S - k * mu

    print(f"  Bang-bang: k={k} positions at mu={mu:.4f}, 1 residual={res:.4f}")
    print(f"  Each vertex: sum y_n^2 = {k * mu**2 + res**2:.6f}, c_emp <= {(1 + 2*(k*mu**2 + res**2))/M:.6f}")
    print(f"  Searching positions {{1,...,{N_pos}}}, Toeplitz size {N_toeplitz+1}x{N_toeplitz+1}")

    best_psd_T = -np.inf
    best_psd_y = None
    n_total = 0
    n_psd = 0

    # Enumerate: choose k positions for mu, then 1 residual position
    for mu_positions in combinations(range(1, N_pos+1), k):
        for res_pos in range(1, N_pos+1):
            if res_pos in mu_positions:
                continue  # residual must be at a different position
            n_total += 1
            y_seq = {0: 1.0}
            for p in mu_positions:
                y_seq[p] = mu
            y_seq[res_pos] = res

            T = make_toeplitz(y_seq, N=N_toeplitz)
            if is_psd(T):
                n_psd += 1
                T_val = sum(v**2 for kk, v in y_seq.items() if kk > 0)
                if T_val > best_psd_T:
                    best_psd_T = T_val
                    best_psd_y = dict(y_seq)

    print(f"  Vertices tried: {n_total}, PSD-feasible: {n_psd}")
    if best_psd_y is not None:
        ce = (1 + 2 * best_psd_T) / M
        print(f"  Best PSD-feasible bang-bang: sum y_n^2 = {best_psd_T:.6f}, c_emp <= {ce:.6f}")
        nz = sorted([(n, v) for n, v in best_psd_y.items() if n > 0 and v > 0])
        print(f"    y configuration: {nz}")
        return ce, best_psd_y, best_psd_T
    return None, None, None

def add_extra_psd_check(y_seq, N_toeplitz_max=30):
    """Check Toeplitz PSD at larger sizes — does a tail of zeros remain PSD?"""
    for N in [10, 15, 20, 25, 30]:
        T = make_toeplitz(y_seq, N=N)
        eigs = np.linalg.eigvalsh(T)
        min_eig = eigs.min()
        if min_eig < -1e-9:
            return N, min_eig
    return None, None

def main():
    M = 1.378
    print(f"=== STEP 1: tighten Path A at K=2 ===")
    print(f"M_max = {M}, mu(M) = {mu_M(M):.6f}")
    print(f"Hyp_R target: c_* = log(16)/pi = {LOG16_PI:.6f}\n")

    K = 2.0
    print(f"K = {K}")
    print(f"  Loose Path A:    c_emp <= {loose_path_a(M, K):.6f}")
    bb_ce, bb_cfg, bb_T = bang_bang_naive(M, K)
    print(f"  Bang-bang naive: c_emp <= {bb_ce:.6f}\n")

    print(f"  Checking Toeplitz PSD on bang-bang vertices...")
    for N_pos in [5, 10, 15]:
        for N_toep in [N_pos, N_pos + 5]:
            print(f"  --- N_pos={N_pos}, N_toeplitz={N_toep} ---")
            ce, y, T = enumerate_bang_bang_psd(M, K, N_pos=N_pos, N_toeplitz=N_toep)
            if y is not None:
                # Re-check at larger Toeplitz
                fail_N, fail_eig = add_extra_psd_check(y, 30)
                if fail_N is not None:
                    print(f"    Note: at Toeplitz size {fail_N+1}, min eig = {fail_eig:.3e} < 0 — vertex actually fails")

    # K sweep
    print(f"\n=== K sweep at M = {M} ===")
    print(f"  {'K':>4} | {'loose':>6} | {'bb-naive':>8} | {'best PSD bb':>11}")
    for K in [2.0, 2.5, 3.0, 3.5]:
        loose = loose_path_a(M, K)
        bb = bang_bang_naive(M, K)[0]
        ce, _, _ = enumerate_bang_bang_psd(M, K, N_pos=10, N_toeplitz=15)
        ce_str = f"{ce:.4f}" if ce is not None else "infeas"
        print(f"  {K:>4.1f} | {loose:.4f} |  {bb:.4f}  |   {ce_str}")
    print(f"\n  Hyp_R threshold = {LOG16_PI:.6f}")

if __name__ == "__main__":
    main()
