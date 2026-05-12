"""
Simple analytic check: with small N, hand-verify the moment SDP idea.

For symmetric f on [-1/4, 1/4] with int f = 1, m_2 = int x^2 f.
Hausdorff: 0 <= m_2 <= 1/16 (since |x| <= 1/4).
Convolution g = f*f. g_0 = 1. g_2 = 2 m_2.

Test poly p(t) = 1 - 4 t^2, sup_{[-1/2,1/2]} p = 1 (at t=0), nonneg on [-1/2,1/2].
int_{-1/2}^{1/2} p dt = 2/3.
int p g dt = g_0 - 4 g_2 = 1 - 8 m_2.

Lower bound: M >= (1 - 8 m_2) / (2/3) = (3/2) (1 - 8 m_2) = 1.5 - 12 m_2.

To get a valid LB on M, take MIN over feasible m_2 in [0, 1/16] => max value 12 m_2 = 12 * (1/16) = 3/4.
LB = 1.5 - 3/4 = 0.75.

So with m_2 = 1/16 (f concentrated at endpoints), we get M >= 0.75. WEAK.

Now consider higher-degree p: p(t) = (1 - 4 t^2)^2 = 1 - 8 t^2 + 16 t^4.
int p = 1 - 8/12 + 16/80 = 1 - 2/3 + 1/5 = (15 - 10 + 3)/15 = 8/15.
int p g = g_0 - 8 g_2 + 16 g_4 = 1 - 16 m_2 + 16 (2 m_4 + 6 m_2^2/... wait)

Hmm: g_4 = sum_{k=0..4} C(4,k) m_k m_{4-k}
       = m_0 m_4 + 4 m_1 m_3 + 6 m_2^2 + 4 m_3 m_1 + m_4 m_0 (m_odd=0)
       = 2 m_4 + 6 m_2^2.

int p g = 1 - 8 (2 m_2) + 16 (2 m_4 + 6 m_2^2)
        = 1 - 16 m_2 + 32 m_4 + 96 m_2^2.

LB(p) = (1 - 16 m_2 + 32 m_4 + 96 m_2^2) / (8/15)
      = (15/8) (1 - 16 m_2 + 32 m_4 + 96 m_2^2).

Minimize over (m_2, m_4) feasible (Hausdorff [-1/4,1/4]):
  m_2 in [0, 1/16], m_4 in [0, 1/256].
  Hausdorff constraints: m_2^2 <= m_4 (Cauchy-Schwarz on H_f)
                        and (1/16) m_0 - m_2 = 1/16 - m_2 >= 0 (already)
                        and (1/16) m_2 - m_4 >= 0 (localizing)
                        i.e. m_4 <= m_2/16.

So we minimize  1 - 16 m_2 + 32 m_4 + 96 m_2^2
  over 0 <= m_2 <= 1/16, m_2^2 <= m_4 <= m_2/16.

To minimize wrt m_4: the coefficient of m_4 is +32, so set m_4 = m_2^2 (smallest).
Then: F(m_2) = 1 - 16 m_2 + 32 m_2^2 + 96 m_2^2 = 1 - 16 m_2 + 128 m_2^2.
F'(m_2) = -16 + 256 m_2 = 0 => m_2 = 1/16.
F(1/16) = 1 - 1 + 128/256 = 0.5.
F(0) = 1.
So min F = 0.5 at m_2 = 1/16.

LB = (15/8) * 0.5 = 15/16 = 0.9375.

Better than 0.75 but still <1.

Let me try (1-4t^2)^3:
"""
import numpy as np
from math import comb


def make_peaked_poly(k):
    base = np.array([1.0, 0.0, -4.0])
    p = np.array([1.0])
    for _ in range(k):
        p = np.convolve(p, base)
    return p


def integrate_poly(p, a=-0.5, b=0.5):
    s = 0.0
    for n, c in enumerate(p):
        s += c * (b**(n+1) - a**(n+1)) / (n+1)
    return s


def g_moment_quadratic_form(n, m_vec_var):
    """Returns symbolic-ish: g_n = sum_k C(n,k) m_k m_{n-k} (numpy expression)."""
    # Just return as numpy array of coefficients in pairs (k, n-k)
    pass


def evaluate_lb_for_p(k, m_vals, N):
    """Evaluate (int p g) / (int p) given m_vals for f's moments [m_0..m_N]."""
    p = make_peaked_poly(k)
    deg_p = len(p) - 1
    if deg_p > 2 * N:
        return None
    p_int = integrate_poly(p)
    # g_n = sum_l C(n,l) m_l m_{n-l}
    g_vals = []
    for n in range(deg_p + 1):
        g_n = 0.0
        for l in range(max(0, n - N), min(n, N) + 1):
            g_n += comb(n, l) * m_vals[l] * m_vals[n - l]
        g_vals.append(g_n)
    intpg = sum(p[n] * g_vals[n] for n in range(deg_p + 1))
    return intpg, p_int, intpg / p_int


def grid_search_lb(N=8, n_grid=15):
    """Grid search over Hausdorff-feasible f-moments to find minimum LB(p)."""
    print(f"\n=== Grid search N = {N} ===")
    # Symmetric: m_odd = 0; vary m_2, m_4, ... up to N
    n_even = N // 2  # number of even moments after m_0
    # Bounds: m_{2j} <= 1/16^j roughly, m_{2j} <= m_{2j-2} / 16 (localizing)
    # Build Hausdorff-feasible distributions: parameterize as mixture of point masses

    # Use mixtures of delta functions on a grid in [-1/4, 1/4]
    grid = np.linspace(-0.25, 0.25, n_grid)

    # Random mixtures + extremes
    best_lb_per_k = {}
    for trial in range(500):
        weights = np.random.dirichlet(np.ones(n_grid))
        # Compute moments
        m = np.array([sum(weights * grid**n) for n in range(N + 1)])

        for k in range(1, N + 1):
            res = evaluate_lb_for_p(k, m, N)
            if res is None:
                continue
            intpg, p_int, lb = res
            if k not in best_lb_per_k or lb < best_lb_per_k[k][0]:
                best_lb_per_k[k] = (lb, m.copy())

    # The MINIMUM over feasible f gives the LB on M (worst-case f).
    print("k | min int p g / int p over sampled f's:")
    for k in sorted(best_lb_per_k):
        lb, m = best_lb_per_k[k]
        print(f"  k={k}:  LB = {lb:.5f}  (m_0..m_{N} = {m})")

    # Best LB across k
    best = max((lb for lb, _ in best_lb_per_k.values()), default=None)
    print(f"\n  Best LB over all k:  {best:.5f}")


def deltas_at_endpoints():
    """f = 1/2 delta_{-1/4} + 1/2 delta_{1/4}.  m_n = (1/2) (-1/4)^n + (1/2)(1/4)^n
       = (1/4)^n if n even, 0 if n odd."""
    print("\n=== f = 1/2 (delta_{-1/4} + delta_{1/4}) ===")
    N = 10
    m = np.array([(0.25)**n if n % 2 == 0 else 0.0 for n in range(N + 1)])
    print(f"  m: {m}")
    # g = 1/4 delta_{-1/2} + 1/2 delta_0 + 1/4 delta_{1/2}
    # ||g||_inf = 1/2 (point mass at 0).  But we're treating as DENSITY — point masses
    # have inf "density"; the bound applies to DENSITIES, not point masses.
    # g_n = sum C(n,l) m_l m_{n-l}: for n=0 -> 1, n=2 -> 2 * (1/16) + 0 = 1/8, etc.
    for k in range(1, 6):
        res = evaluate_lb_for_p(k, m, N)
        if res:
            intpg, p_int, lb = res
            print(f"  k={k}: int p g = {intpg:.5f}, int p = {p_int:.5f}, LB = {lb:.5f}")


def uniform_f():
    """f(x) = 2 on [-1/4, 1/4], else 0.  ||f||_inf = 2.
       g = f*f = triangular on [-1/2, 1/2], peak at 0 = 2.
       So M(uniform) = 2."""
    print("\n=== f = uniform on [-1/4, 1/4] ===")
    N = 10
    m = []
    for n in range(N + 1):
        if n % 2 == 0:
            m.append(2 * (0.25)**(n+1) / (n+1) - 2 * (-0.25)**(n+1) / (n+1))
        else:
            m.append(0.0)
    m = np.array(m)
    print(f"  m: {m}")
    for k in range(1, 6):
        res = evaluate_lb_for_p(k, m, N)
        if res:
            intpg, p_int, lb = res
            print(f"  k={k}: int p g = {intpg:.5f}, int p = {p_int:.5f}, LB = {lb:.5f}")


if __name__ == "__main__":
    deltas_at_endpoints()
    uniform_f()
    grid_search_lb(N=4, n_grid=15)
    grid_search_lb(N=6, n_grid=15)
    grid_search_lb(N=8, n_grid=15)
