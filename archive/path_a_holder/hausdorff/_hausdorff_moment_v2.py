"""
Hausdorff moment SDP v2: Chebyshev-style bound on max(f*f) from finite moments.

Idea:
  ||g||_inf >= sup_{p polynomial}  (int p(t) g(t) dt) / (max_{t in [-1/2,1/2]} |p(t)|)
  By duality, this equals  sup_p  <c, g> / ||p||_inf
  where g_n are moments of g = f*f, c = coefficients of p, <c,g> = sum c_n g_n.

So:  M >= max over polynomials p of degree <= 2N with sup_{[-1/2,1/2]} |p| <= 1:
                sum c_n g_n.

where g_n = sum_k C(n,k) m_k m_{n-k}.

Min over feasible f-moments (Hausdorff on [-1/4, 1/4]).
Max over poly p with sup norm <= 1 on [-1/2, 1/2].

The min-max is the lower bound on M.

To make it SDP-able:
  - Outer min over f-moments (and lifted moment matrix G)
  - Lower bound int p g via:  M >= int p g (when p >= 0 and ||p||_inf=1).

For nonneg p with sup_{[-1/2,1/2]} p = 1:
  int p g <= ||g||_inf * int p dt  =>  ||g||_inf >= int p g / int p.

So pick any nonneg poly p of degree 2N on [-1/2, 1/2], compute:
  M >= (sum c_n g_n) / (int_{-1/2}^{1/2} p(t) dt).

The MAX over such p (to get the best LB) is itself an SDP via Putinar.

Implementation: parametrize p(t) = q(t)^2 * (1/4 - t^2)^? + r(t)^2 (sum of squares),
or simply enumerate Chebyshev polynomials and pick the best.

Quick version: try p(t) = (1 - 4t^2)^k  (peaked at 0, vanishes at +-1/2).
"""
import numpy as np
import cvxpy as cp
from math import comb
from scipy.special import roots_legendre

A, B = -0.25, 0.25  # support of f


def build_v2(N: int, p_coeffs: np.ndarray, p_int: float, verbose: bool = False):
    """
    Build SDP minimizing M, where M >= sum c_n g_n / p_int,
    with g_n = sum_k C(n,k) m_k m_{n-k}, and Hausdorff on f-moments.

    p_coeffs: array of length deg(p)+1, polynomial coeffs in t (deg 0 first).
    p_int: int_{-1/2}^{1/2} p(t) dt  (presumed positive).
    Need: p(t) >= 0 on [-1/2, 1/2].
    """
    deg = len(p_coeffs) - 1
    assert deg <= 2 * N, "polynomial degree exceeds 2N"

    N2 = N // 2
    m = cp.Variable(N + 1)
    constraints = [m[0] == 1]

    # Symmetric ansatz: m_{odd} = 0
    for k in range(1, N + 1, 2):
        constraints.append(m[k] == 0)

    # Hausdorff PSD
    H = cp.bmat([[m[i + j] for j in range(N2 + 1)] for i in range(N2 + 1)])
    constraints.append(H >> 0)
    if N2 >= 1:
        L = cp.bmat([[(1/16) * m[i + j] - m[i + j + 2] for j in range(N2)]
                     for i in range(N2)])
        constraints.append(L >> 0)

    # Lift to G[i,j] = m_i * m_j (PSD relaxation)
    G = cp.Variable((N + 1, N + 1), symmetric=True)
    constraints.append(G >> 0)
    constraints.append(G[0, 0] == 1)
    for i in range(N + 1):
        constraints.append(G[0, i] == m[i])
    # Also enforce G[i,j] = G[j,i] (already symmetric) and Hausdorff on G's row 1
    # Couple G to be consistent with moments via G[i,0]=m[i], G[i,j] arbitrary PSD.

    # Apply Hausdorff to G as a moment matrix on f x f? Treat G as m m^T.
    # Constraint: G is rank-1 approx — in SDP just PSD. Add Hausdorff-like bounds:
    # G[i,j] should equal m[i]*m[j], we have |m[k]| <= (1/4)^k * m[0] = (1/4)^k.
    # But we don't directly enforce this; PSD relaxation does the rest.

    # g_n = sum_k C(n,k) G[k, n-k]
    g_n = []
    for n in range(deg + 1):
        terms = []
        for k in range(max(0, n - N), min(n, N) + 1):
            if 0 <= k <= N and 0 <= n - k <= N:
                terms.append(comb(n, k) * G[k, n - k])
        g_n.append(sum(terms) if terms else 0)

    # M >= (sum c_n g_n) / p_int
    M_var = cp.Variable()
    test_expr = sum(p_coeffs[n] * g_n[n] for n in range(deg + 1))
    constraints.append(M_var * p_int >= test_expr)

    # M is at least 0 (trivial); we want to MINIMIZE M as the lower bound certificate.
    # Wait — we're proving M >= LB; the LB is the OPTIMAL value of (min M s.t. M >= int p g / int p).
    # So minimize M.

    objective = cp.Minimize(M_var)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except Exception:
        prob.solve()
    return prob, m, g_n, M_var


def make_peaked_poly(k: int, t0: float = 0.0):
    """
    p(t) = ((1/2)^2 - (t - t0)^2)^k = (1/4 - (t-t0)^2)^k.
    Wait: we want p(t) >= 0 on [-1/2, 1/2]. Use p(t) = (1/4 - t^2)^k for centered.
    For t0 != 0, use p(t) = ((1/2 - |t-t0|))^... hmm, won't be polynomial.

    Stick to t0=0: p(t) = (1/4 - t^2)^k, deg = 2k.
    """
    # (1/4 - t^2)^k
    base = np.array([0.25, 0, -1.0])  # 1/4 - t^2 (deg-0,1,2 coeffs)
    p = np.array([1.0])
    for _ in range(k):
        p = np.convolve(p, base)
    # Compute int_{-1/2}^{1/2} p(t) dt
    deg = len(p) - 1
    p_int = 0.0
    for n, c in enumerate(p):
        # int_{-1/2}^{1/2} t^n dt = ((1/2)^{n+1} - (-1/2)^{n+1})/(n+1) = 2*(1/2)^{n+1}/(n+1) if n even, 0 if odd
        if n % 2 == 0:
            p_int += c * 2 * (0.5) ** (n + 1) / (n + 1)
    return p, p_int


def main():
    print("=" * 70)
    print("Hausdorff Moment SDP v2: Chebyshev-Type Bound")
    print("=" * 70)
    for N in [4, 6, 8, 10, 12]:
        print(f"\n--- N = {N} (max poly degree 2N = {2*N}) ---")
        best_lb = -np.inf
        best_k = -1
        for k in range(1, N + 1):
            p, p_int = make_peaked_poly(k)
            if p_int <= 0:
                continue
            prob, m, g_n, M_var = build_v2(N, p, p_int, verbose=False)
            if prob.status not in ("optimal", "optimal_inaccurate"):
                continue
            lb = float(M_var.value)
            if lb > best_lb:
                best_lb = lb
                best_k = k
        print(f"  Best k = {best_k}, LB = {best_lb:.6f}")


if __name__ == "__main__":
    main()
