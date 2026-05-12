"""
Hausdorff Moment SDP v3 — correct min-max ordering.

For each fixed nonneg test polynomial p with sup_{[-1/2,1/2]} p = 1,
we have for ANY feasible f:
  ||g||_inf >= int p(t) g(t) dt = sum_n c_n g_n      where g = f*f.

So the LOWER BOUND on M is:
  LB(p) = min over (Hausdorff-feasible f-moments)  sum_n c_n g_n.

  (If LB(p) > 0, this is a valid lower bound on M.)

We want max over p (with sup-norm <= 1, and p >= 0) of LB(p).

Strategy:
  1. Fix p = (1/4 - t^2)^k / max_t (1/4 - t^2)^k.  But max is at t=0: (1/4)^k.
     So p_normalized(t) = (1 - 4t^2)^k. Sup-norm 1, nonneg on [-1/2, 1/2].
  2. Compute LB(p) via SDP: min sum c_n g_n s.t. Hausdorff PSD on f.
  3. Report best LB.

For symmetric f (CS conjecture), g = f*f is symmetric, peak at t = ?
  CS 2017 conjecture: peak at t = 1/4 not at 0. So p centered at 0 may be wasteful.
  Use also p centered at t0 = 1/4: p(t) = (1 - (4(t-1/4))^2)^k / ... but we need
  p polynomial nonneg on [-1/2,1/2]. Use Lukacs/Markov representation:
    p(t) = a(t)^2 + (1/2 - t)(1/2 + t) b(t)^2 (sum of squares with boundary).

For now: try p(t) = (1 - (4(t - t0))^2)^k for various t0, ensuring the support
of the bump is inside [-1/2,1/2].
"""
import numpy as np
import cvxpy as cp
from math import comb


def hausdorff_min_int_p_g(N: int, p_coeffs: np.ndarray, verbose: bool = False):
    """
    min sum_n c_n g_n  s.t. f-moments satisfy Hausdorff on [-1/4, 1/4],
                            G[0,i] = m_i (lifted moment matrix PSD).
    Returns (status, value, m_value, g_values).
    """
    deg = len(p_coeffs) - 1
    assert deg <= 2 * N, f"deg(p) = {deg} > 2N = {2*N}"

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

    # Lifted moment matrix G
    G = cp.Variable((N + 1, N + 1), symmetric=True)
    constraints.append(G >> 0)
    constraints.append(G[0, 0] == 1)
    for i in range(N + 1):
        constraints.append(G[0, i] == m[i])

    # Add Hausdorff to G as well: it's a moment matrix of a measure (m_i m_j conv),
    # but in PSD relax we just leave PSD.
    # Add: G satisfies same Hausdorff bounds row-wise — i.e., G[i,:] is m_i times
    # a Hausdorff-feasible vector. We don't enforce that beyond PSD.

    # g_n
    g_n = []
    for n in range(deg + 1):
        terms = []
        for k in range(max(0, n - N), min(n, N) + 1):
            if 0 <= k <= N and 0 <= n - k <= N:
                terms.append(comb(n, k) * G[k, n - k])
        g_n.append(sum(terms) if terms else 0)

    obj_expr = sum(p_coeffs[n] * g_n[n] for n in range(deg + 1))
    objective = cp.Minimize(obj_expr)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"SCS error: {e}")
        prob.solve()
    return prob, m, g_n


def make_peaked_poly_centered(k: int, t0: float = 0.0, half_width: float = 0.5):
    """
    p(t) = (1 - ((t - t0)/half_width)^2)^k.
    Nonneg on [t0 - half_width, t0 + half_width], zero outside.
    Sup-norm = 1 at t = t0.
    Need p polynomial, so we DON'T truncate to zero outside; on [-1/2,1/2] this
    polynomial may be NEGATIVE if t0 +- half_width extends beyond, OR may have
    other zero crossings. So we restrict half_width = min(t0+1/2, 1/2-t0).

    For t0 = 0, half_width = 1/2: p(t) = (1 - 4t^2)^k, vanishes at +-1/2.
    For t0 = 1/4, half_width = 1/4: p(t) = (1 - 16(t-1/4)^2)^k, vanishes at 0, 1/2.
    """
    # Polynomial in t: (1 - ((t - t0)/half_width)^2)^k
    # Build symbolically using numpy.poly1d
    from numpy.polynomial import polynomial as P
    inv_h2 = 1.0 / (half_width ** 2)
    # 1 - inv_h2 * (t - t0)^2 = 1 - inv_h2 * (t^2 - 2 t0 t + t0^2)
    # Coeffs in increasing degree:
    base = np.array([1.0 - inv_h2 * t0**2, 2 * t0 * inv_h2, -inv_h2])
    p = np.array([1.0])
    for _ in range(k):
        p = P.polymul(p, base)
    return p


def integrate_poly_on_interval(p, a, b):
    """int_a^b p(t) dt for p in increasing-degree coeffs."""
    s = 0.0
    for n, c in enumerate(p):
        s += c * (b ** (n + 1) - a ** (n + 1)) / (n + 1)
    return s


def main():
    print("=" * 70)
    print("Hausdorff Moment SDP v3 — Chebyshev-type LB")
    print("=" * 70)

    for N in [4, 6, 8, 10]:
        print(f"\n--- N = {N} ---")
        best_lb = -np.inf
        best_info = None
        # Try various (k, t0, half_width)
        for t0 in [0.0, 0.1, 0.15, 0.2, 0.25, 0.3]:
            for half_width in [0.5 - abs(t0), max(1e-3, min(0.5 - t0, 0.5 + t0))]:
                if half_width <= 0:
                    continue
                for k in range(1, N + 1):
                    p = make_peaked_poly_centered(k, t0, half_width)
                    if len(p) - 1 > 2 * N:
                        continue
                    p_int = integrate_poly_on_interval(p, -0.5, 0.5)
                    if p_int <= 1e-10:
                        continue
                    prob, m, g_n = hausdorff_min_int_p_g(N, p)
                    if prob.status not in ("optimal", "optimal_inaccurate"):
                        continue
                    val = float(prob.value)
                    lb = val / p_int
                    if lb > best_lb:
                        best_lb = lb
                        best_info = (k, t0, half_width, val, p_int)
        print(f"  Best LB = {best_lb:.6f}")
        if best_info:
            k, t0, hw, val, pint = best_info
            print(f"    k={k}, t0={t0}, half_width={hw:.4f}, "
                  f"int p g={val:.5f}, int p={pint:.5f}")


if __name__ == "__main__":
    main()
