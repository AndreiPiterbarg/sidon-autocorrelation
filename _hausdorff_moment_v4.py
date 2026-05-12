"""
Hausdorff Moment SDP v4 — clean formulation using extended moment matrix.

Key insight: f >= 0 with given moments m_n = int x^n f means f is a measure
with moments m_n. By the Hausdorff theorem, m is feasible iff:
  - H[i,j] = m_{i+j} PSD (positivity)
  - L[i,j] = (1/16 - x^2) Hankel PSD (interval [-1/4, 1/4])

The convolution f*f has moments g_n = sum_k C(n,k) m_k m_{n-k},
QUADRATIC in m. To turn into SDP: lift moment matrix M = m m^T -> PSD matrix
of rank 1. PSD relaxation gives a LOWER bound on min(int p g) (the relaxation
is OUTER, so we get a LOOSE lower bound).

Cleaner view: g_n is itself a moment of g. g is the convolution of f with itself,
supported on [-1/2, 1/2]. So g_n satisfies its own Hausdorff conditions on [-1/2,1/2]:
  H_g[i,j] = g_{i+j} PSD,
  L_g[i,j] = (1/4 - x^2) Hankel of g PSD.
PLUS the constraint that g comes from CONVOLUTION of f. The convolution constraint
is what's hard.

If we DROP the convolution constraint and just consider any Hausdorff measure g
on [-1/2, 1/2] with g_0 = 1, then min ||g||_inf = 1 (uniform measure).
This gives M >= 1, weak.

ALTERNATIVE: use g_n = sum_k C(n,k) m_k m_{n-k} directly with the f's moment
matrix being PSD. Specifically:
  Let H be the (N+1) x (N+1) moment matrix of f. H[i,j] = m_{i+j}.
  Let G be the (N/2+1) x (N/2+1) localizing matrix.
  Then g_n = sum_k C(n,k) m_k m_{n-k} = sum_k C(n,k) H[k, n-k] (when valid).

Wait — that's a great observation! g_n is a SUM OF ENTRIES of H. We don't need
a separate lifted G matrix. So everything is LINEAR in H entries.

So:  g_n = sum_{k} C(n,k) H[k, n-k]  for n <= 2*(N/2) = N (using only main entries).

This makes the SDP fully linear in H + standard PSD constraints.

Now the objective:
  M >= int p(t) g(t) dt = sum_n p_n g_n      (p_n = coeffs of test poly)
                       = sum_n p_n sum_k C(n,k) H[k, n-k]
                       = sum_{k,j} (p_{k+j} C(k+j, k)) H[k, j].

LP/SDP: maximize alpha s.t. for all Hausdorff-feasible H,
          alpha <= sum_{k,j} a_{k,j} H[k,j]   where a_{k,j} = p_{k+j} C(k+j, k).

Equivalently: alpha = inf_H sum a_{k,j} H[k,j] subject to PSD constraints.

We want the BEST p (sup-norm <= 1, nonneg on [-1/2,1/2]). For now, fix p = (1-4t^2)^k.

Note: g_n only valid for n <= N (using H of size (N/2+1)) — so p must have deg <= N.
"""
import numpy as np
import cvxpy as cp
from math import comb


def make_peaked_poly(k):
    """p(t) = (1 - 4 t^2)^k.  Sup-norm 1 at t=0; vanishes at +-1/2."""
    base = np.array([1.0, 0.0, -4.0])  # 1 - 4 t^2
    p = np.array([1.0])
    for _ in range(k):
        p = np.convolve(p, base)
    return p


def integrate_poly(p, a=-0.5, b=0.5):
    s = 0.0
    for n, c in enumerate(p):
        s += c * (b**(n+1) - a**(n+1)) / (n+1)
    return s


def lb_via_test_poly(N: int, p_coeffs: np.ndarray, verbose: bool = False):
    """
    min over Hausdorff-feasible f-moments of (int p(t) g(t) dt) / int p(t) dt,
    where g_n = sum_k C(n,k) m_k m_{n-k}.

    NOTE: g_n is QUADRATIC in m. We need a moment matrix lifting.
    Let M be PSD (N+1)x(N+1) with M[i,j] = m_i m_j (rank-1 ideal; PSD relax).
    First row m_i = M[0,i].

    Hausdorff on m: H_f[i,j] = m_{i+j} PSD (size N/2+1) — NOT the same as M!
    H_f is the moment matrix of the MEASURE f (PSD by def of moments).
    M is the OUTER PRODUCT m m^T (also PSD).

    g_n = sum_k C(n,k) M[k, n-k].

    With M PSD only (no rank-1), the relaxation can blow up. To prevent unboundedness
    (negative-going), we tie M to m via M[0,i] = m[i] and add Hausdorff on m via H_f.

    But for the OBJECTIVE int p g (= sum p_n g_n), to be bounded below we need
    constraints on M[i,j] off-diag. Add: M is itself a Hausdorff-feasible moment
    matrix? NO, M = m m^T isn't a moment matrix of f, it's outer product.

    However, we can use the inequality:
      |M[i,j]| = |m_i m_j| <= ||x^i||_inf ||x^j||_inf m_0 = (1/4)^{i+j}.
    So bound M[i,j] in [-(1/4)^{i+j}, (1/4)^{i+j}].
    """
    deg_p = len(p_coeffs) - 1
    assert deg_p <= N, f"deg p={deg_p} > N={N} (need higher N)"

    N2 = N // 2
    m = cp.Variable(N + 1)
    M_mat = cp.Variable((N + 1, N + 1), symmetric=True)

    constraints = [m[0] == 1, M_mat >> 0]
    constraints.append(M_mat[0, 0] == 1)
    for i in range(N + 1):
        constraints.append(M_mat[0, i] == m[i])

    # Symmetric ansatz on m: m_{odd} = 0
    for k in range(1, N + 1, 2):
        constraints.append(m[k] == 0)

    # Hausdorff on f's moment matrix
    H_f = cp.bmat([[m[i + j] for j in range(N2 + 1)] for i in range(N2 + 1)])
    constraints.append(H_f >> 0)
    if N2 >= 1:
        L_f = cp.bmat([[(1/16) * m[i + j] - m[i + j + 2] for j in range(N2)]
                       for i in range(N2)])
        constraints.append(L_f >> 0)

    # Box constraints on M[i,j]: |M[i,j]| <= (1/4)^{i+j}
    for i in range(N + 1):
        for j in range(N + 1):
            bnd = 0.25 ** (i + j)
            constraints.append(M_mat[i, j] <= bnd)
            constraints.append(M_mat[i, j] >= -bnd)

    # g_n = sum_k C(n,k) M[k, n-k]
    g_n = []
    for n in range(deg_p + 1):
        terms = []
        for k in range(max(0, n - N), min(n, N) + 1):
            if 0 <= k <= N and 0 <= n - k <= N:
                terms.append(comb(n, k) * M_mat[k, n - k])
        g_n.append(sum(terms) if terms else 0)

    obj = sum(p_coeffs[n] * g_n[n] for n in range(deg_p + 1))
    objective = cp.Minimize(obj)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=verbose)
    except Exception:
        prob.solve()
    return prob, m, M_mat, g_n


def main():
    print("=" * 70)
    print("Hausdorff Moment SDP v4")
    print("=" * 70)
    for N in [4, 6, 8, 10, 12]:
        print(f"\n--- N = {N} ---")
        best_lb = -np.inf
        best_k = -1
        for k in range(1, N // 2 + 1):
            p = make_peaked_poly(k)
            if len(p) - 1 > N:
                break
            p_int = integrate_poly(p)
            if p_int <= 1e-12:
                continue
            prob, m, M_mat, g_n = lb_via_test_poly(N, p)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                continue
            val = float(prob.value)
            lb = val / p_int
            if lb > best_lb:
                best_lb = lb
                best_k = k
        print(f"  Best LB = {best_lb:.6f}  (best k = {best_k})")


if __name__ == "__main__":
    main()
