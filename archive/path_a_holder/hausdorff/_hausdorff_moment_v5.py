"""
Hausdorff Moment SDP v5 — work directly with g's moments.

If we DROP the convolution structure and just consider g >= 0 on [-1/2, 1/2]
with g_0 = 1 and g_n satisfying Hausdorff, then ||g||_inf can be as low as 1
(uniform on [-1/2, 1/2]).

To get a useful bound, we need EXTRA structure from convolution:
  - g(0) = ||f||_2^2 >= 1/(b-a) = 2 (Cauchy-Schwarz). But max may be elsewhere.
  - g_n is constrained by f's moments via g_n = sum C(n,k) m_k m_{n-k}.
  - g is positive-definite as a function (since g = f*f and Fourier transform
    of g = |f_hat|^2 >= 0). g_n's are correlations of m: positive in Fourier sense.

Use Cauchy-Schwarz at moment level:
  g_n^2 <= g_{2n} * g_0 = g_{2n}.
  In particular g_2 <= sqrt(g_4) which gives a relation but no LB on max.

Use Markov inequality:
  ||g||_inf >= g_n / int_{-1/2}^{1/2} t^n dt   IF g_n is positive and we can
                                                 truncate to where t^n is largest.

Specifically for n even: int_{-1/2}^{1/2} t^n dt = 2 / ((n+1) * 2^{n+1}) = 1/((n+1) * 2^n).
So ||g||_inf * (1/((n+1) * 2^n)) >= g_n => ||g||_inf >= (n+1) * 2^n * g_n.

But we need a LOWER BOUND on g_n. The min of g_n over Hausdorff-feasible f is... 0
(concentrate f at origin: g_n = 0 for n > 0). So this doesn't directly work.

KEY INSIGHT: We need a min-max SDP:
  M >= max_p inf_f int p(t) g(t) dt / int p(t) dt
where the inner inf is over Hausdorff-feasible f.

Already tried — gives ~0 because f can concentrate at origin (g concentrates at origin),
but then int p g is ~ p(0) * 1 = 1, and int p (= integral over [-1/2,1/2]) varies.

Wait: if f is delta at 0, then g = delta at 0, and int p g = p(0).
For p(t) = (1 - 4t^2)^k, p(0) = 1. int p = depends on k.
So LB = 1 / int p.
For k=1: int p = 1/3 (NO: int_{-1/2}^{1/2} (1 - 4t^2) dt = 1 - 4*(1/12) = 2/3).
LB = 1 / (2/3) = 1.5.

But the SDP gave ~0! Bug. Let me check — the SDP relaxation lets M[i,j] be anything
in box (1/4)^{i+j} subject to PSD; this DOES NOT correspond to delta-at-0.

The issue: the lifted M matrix is OVER-relaxed. For delta-at-0:
  m = (1, 0, 0, ..., 0), so m m^T = e_0 e_0^T (only [0,0] entry = 1).
  g_n = sum C(n,k) m_k m_{n-k} = C(n,0) m_0 m_n + ... + C(n,n) m_n m_0
       = 0 for n>0 (since m_n = 0 for n>0... wait NO, m_0 = 1).
       For n=0: g_0 = m_0^2 = 1. Good.
       For n=1: g_1 = 2 m_0 m_1 = 0. Good.
       For n=2: g_2 = m_0 m_2 + m_1^2 + m_2 m_0 = 0. (m_2 = 0)
  So g = delta at 0 has g_n = 0 for n > 0.

  int p g = sum p_n g_n = p_0 * 1 + 0 + ... = p_0 = 1 (for our p which has p_0 = 1).
  / int p = 2/3.  LB = 1.5.

In the SDP, with M = e_0 e_0^T, M is PSD, satisfies M[0,i] = m[i], in box.
g_n = sum C(n,k) M[k, n-k] = M[0, n] when k=0 or k=n: 2 M[0,n] - M[0,0] for n=0
     = 2 * m_n - delta_{n,0}? Let me recompute:
  n=0: k=0 only. g_0 = C(0,0) M[0,0] = 1.
  n=1: k=0 or 1. g_1 = M[0,1] + M[1,0] = 2 M[0,1] = 2 m_1.
  n=2: k=0,1,2. g_2 = M[0,2] + 2 M[1,1] + M[2,0] = 2 m_2 + 2 M[1,1].
                Hmm: 2 m_2 + 2 M[1,1] (where M[1,1] is m_1^2 ideally).

For delta-at-0: m = (1, 0, ..., 0), M[1,1] = 0, M[2,0] = 0, so g_2 = 0. Good.
For SDP relaxation: M[1,1] could be any value with M[1,1] * M[0,0] >= M[0,1]^2.
M[1,1] >= 0 (PSD diagonal). With m_1 = 0, M[0,1] = 0, so M[1,1] >= 0 freely up to box.
Box: |M[1,1]| <= (1/4)^2 = 1/16. So M[1,1] in [0, 1/16].

So g_2 in [0, 2/16] = [0, 1/8].
int p g = 1 * g_0 + (-4) * g_2 + ... = 1 - 4 g_2 (for k=1).
To MINIMIZE this in the SDP: maximize g_2 subject to PSD => set M[1,1] = 1/16.
g_2 = 2/16 = 1/8.
int p g = 1 - 4/8 = 1/2.
LB = (1/2) / (2/3) = 3/4.

But we got 0.000010. There's a sign or scaling bug.

Let me FIX the implementation and check.
"""
import numpy as np
import cvxpy as cp
from math import comb


def make_peaked_poly(k):
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
    min over (m, M) [ int p g / int p ]
      g_n = sum_k C(n,k) M[k, n-k]
    subject to:
      m_0 = 1
      M >> 0,  M[0, i] = m_i
      H_f (size N/2+1) PSD on m,  L_f (interval) PSD
      Box on |M[i,j]| <= (1/4)^{i+j}
      (optional) M Hausdorff-like as moment matrix of measure on [-1/4,1/4]:
        the matrix M is PSD, its (i,j) entry is the moment <x^i, y^j> against
        f(x) f(y); this is itself a 2D moment matrix of a measure on
        [-1/4,1/4]^2. So box bound + interval localizing on each variable
        applies.
    """
    deg_p = len(p_coeffs) - 1
    assert deg_p <= N

    N2 = N // 2
    m = cp.Variable(N + 1)
    M_mat = cp.Variable((N + 1, N + 1), symmetric=True)

    constraints = [m[0] == 1, M_mat >> 0, M_mat[0, 0] == 1]
    for i in range(N + 1):
        constraints.append(M_mat[0, i] == m[i])

    # Symmetric ansatz on m
    for k in range(1, N + 1, 2):
        constraints.append(m[k] == 0)

    # Hausdorff f's moment matrix
    H_f = cp.bmat([[m[i + j] for j in range(N2 + 1)] for i in range(N2 + 1)])
    constraints.append(H_f >> 0)
    if N2 >= 1:
        L_f = cp.bmat([[(1/16) * m[i + j] - m[i + j + 2] for j in range(N2)]
                       for i in range(N2)])
        constraints.append(L_f >> 0)

    # 2D Hausdorff on M (treat as moment matrix of f(x) f(y) on [-1/4,1/4]^2):
    # M is PSD (already)
    # Localizing in x:  (1/16 - x^2) f(x) f(y) >= 0 ->
    #   (1/16) M[i,j] - M[i+2, j] PSD as function of (i,j) ranged over (i, j).
    # We need (1/16) M[i,j] - M[i+2, j] to be the (i,j) entry of a PSD matrix.
    # In 2D moment SDP, this is M_{loc}[i,j] = (1/16) M[i,j] - M[i+2, j], for i,j up to N-2.
    # Same for y: (1/16) M[i,j] - M[i, j+2].
    if N >= 2:
        L_x = cp.bmat([[(1/16) * M_mat[i, j] - M_mat[i + 2, j]
                        for j in range(N - 1)]
                       for i in range(N - 1)])
        constraints.append(L_x >> 0)
        L_y = cp.bmat([[(1/16) * M_mat[i, j] - M_mat[i, j + 2]
                        for j in range(N - 1)]
                       for i in range(N - 1)])
        constraints.append(L_y >> 0)

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
    print("Hausdorff Moment SDP v5 — full 2D moment matrix structure")
    print("=" * 70)
    for N in [4, 6, 8, 10, 12]:
        print(f"\n--- N = {N} ---")
        best_lb = -np.inf
        best_k = -1
        best_val = None
        best_pint = None
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
                best_val = val
                best_pint = p_int
        print(f"  Best LB = {best_lb:.6f}  (k={best_k}, int p g={best_val}, int p={best_pint})")

    # Special check at N=4 with k=1: hand-computed expected ~3/4 (without 2D Hausdorff)
    # With 2D Hausdorff added, what do we get?
    print("\n--- Hand check N=6, k=2: p = (1-4t^2)^2 = 1 - 8 t^2 + 16 t^4 ---")
    p = make_peaked_poly(2)
    print(f"  p_coeffs = {p}, int p = {integrate_poly(p)}")
    prob, m, M_mat, g_n = lb_via_test_poly(6, p, verbose=False)
    print(f"  status: {prob.status}")
    print(f"  m: {[float(m.value[i]) for i in range(7)]}")
    print(f"  M[0,:]: {[float(M_mat.value[0, j]) for j in range(7)]}")
    print(f"  M[1,1] = {float(M_mat.value[1,1])}, M[2,2] = {float(M_mat.value[2,2])}")
    print(f"  g_0..g_4 = {[float(g_n[n].value) for n in range(5)]}")
    print(f"  int p g = {float(prob.value)}, LB = {float(prob.value) / integrate_poly(p)}")


if __name__ == "__main__":
    main()
