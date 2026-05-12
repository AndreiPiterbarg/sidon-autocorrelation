"""
Hausdorff moment SDP test for C_{1a} lower bound.

Setup:
  f >= 0, supp f in [-1/4, 1/4], int f = 1.
  Variables: m_n = int x^n f(x) dx for n = 0, 1, ..., N.
  Constraints (Hausdorff on [-1/4, 1/4]):
    m_0 = 1
    H = [m_{i+j}]_{i,j=0..k}                                PSD  (positivity)
    L = [(1/4 - x)*x*(1/4 + x): m_{i+j+1}/4 - m_{i+j+3}]    PSD  (interval)
    Equivalently: ((1/4)^2 - x^2)-Hankel PSD.

  Convolution moments:
    g = f*f, then int x^n g(x) dx = sum_{k=0..n} C(n,k) m_k m_{n-k}
    (since g is supp in [-1/2, 1/2], moments g_n exist).

To bound max_t (f*f)(t):
  M >= g(t) for all t. We DON'T have access to pointwise g; only its moments.
  Standard bound: ||g||_inf >= int g / |supp g| = 1 / 1 = 1 (trivial).
  Markov-type: int g^2 <= M * int g = M  =>  M >= int g^2 = sum_{n} ...
  But int g^2 = int (f*f)^2 = ||f*f||_2^2, computable from autocorrelation moments.

Better: use moment problem to bound max via Krein inequality:
  For nonneg measure on bounded interval [a,b] with given moments,
  inf max(measure-density) is the Chebyshev-like bound from moments,
  given by an LP/SDP dual.

Approach implemented:
  Use the DUAL relation: M >= ||g||_2^2 / ||g||_1 = ||g||_2^2 (since ||g||_1=1).
  ||g||_2^2 = int (f*f)^2 dx = sum over moments of f using Parseval-like identity:
    int (f*f)^2 = int |f_hat|^4 d xi (Plancherel)
  In moment form, int (f*f)^2 is a polynomial in {m_n}.

  Specifically, ||g||_2^2 = (f*f, f*f) = (f*f*f-tilde, f-tilde) at 0
   = int_{-1/2}^{1/2} (f*f)^2 dt.

  Easier: bound M from below using
    M = ||f*f||_inf >= ||f*f||_p^{p/(p-1)} ... by reverse Holder, NO.
  Use:  ||f*f||_inf >= ||f*f||_2^2 / ||f*f||_1 = ||f*f||_2^2.

  Then minimize ||f*f||_2^2 subject to Hausdorff constraints on f's moments.
  This is an SDP in the moments (||f*f||_2^2 is degree 4 in moments, but can be
  expressed via the moment matrix of f).
"""
import numpy as np
import cvxpy as cp
from math import comb

A, B = -0.25, 0.25  # support of f


def build_hausdorff_moment_sdp(N: int, verbose: bool = True):
    """
    Build SDP: minimize ||f*f||_2^2 s.t. Hausdorff PSD on [-1/4,1/4], m_0 = 1.

    Note ||f*f||_2^2 = int_{-1/2}^{1/2} (f*f)(t)^2 dt.
    By Plancherel: ||f*f||_2^2 = ||f_hat^2||_2^2 = int |f_hat(xi)|^4 d xi.

    A direct moment expression:
      Let M = moment matrix [m_{i+j}] of f, size (N/2+1) x (N/2+1), assumed PSD.
      Then ||f*f||_2^2 = trace(M * J * M * J) for some matrix J? No.

    Actually:
      int (f*f)^2 dt = int (int f(x) f(t-x) dx)^2 dt
                     = int int int f(x) f(t-x) f(y) f(t-y) dx dy dt
                     = int int f(x) f(y) [int f(t-x) f(t-y) dt] dx dy
                     = int int f(x) f(y) (f * f_rev)(x-y) dx dy
                     = int int f(x) f(y) R(x-y) dx dy
      where R(u) = (f * f-rev)(u) = int f(s) f(s-u) ds = autocorrelation.

    Hmm, that's quadratic in R, which is degree 2 in f, so we get degree 4 in f.
    In terms of moments, int int int int f(a)f(b)f(c)f(d) [a-b = c-d kernel] = ...

    SIMPLER: by Cauchy-Schwarz:
      ||f*f||_inf * ||f*f||_1 >= ||f*f||_2^2
      ||f*f||_1 = (int f)^2 = 1
      So ||f*f||_inf >= ||f*f||_2^2.

    And ||f*f||_2^2 = int |f_hat|^4 dxi, lower-bounded by Hausdorff–Young:
      ||f_hat||_4^4 >= ||f||_{4/3}^4 (NO — reverse direction).
      ||f_hat||_4 <= C ||f||_{4/3}, so ||f*f||_2 <= C ||f||_{4/3}^2. Wrong direction.

    Use instead:
      ||f*f||_2^2 = int f(a) f(b) f(c) f(d) [a+b = c+d kernel] da db dc dd
                  = int_t [int_{a+b=t} f(a)f(b)] [int_{c+d=t} f(c)f(d)] dt
                  = int (f*f)(t)^2 dt   (tautology)

      In moment form (integrating against polynomial test funcs): hard.

    FINAL APPROACH:
      Use Chebyshev/Markov: for a nonneg measure g on [-1/2, 1/2] with int g = 1,
      max g >= 1 / (b - a) = 1 trivially.

      Stronger: use the dual moment problem.
      For g with moments g_0=1, g_1, ..., g_{2N}, the min of ||g||_inf over
      all such measures is bounded BELOW by the Chebyshev measure construction.
      But max g_inf is unbounded (delta) and min g_inf can be as low as 1.

    CONCLUSION: pure moment data on f gives at best L^2 lower bound:
      M >= ||f*f||_2^2 >= L2_bound from Hausdorff constraints.

      Compute L2_bound via SDP.
    """
    # Moment vector m_0, ..., m_N
    N2 = N // 2  # half-size for moment matrix
    m = cp.Variable(N + 1)

    # ||f*f||_2^2 = sum over (a,b,c,d) f(a)f(b)f(c)f(d) delta(a+b-c-d)
    # In moment terms, this is degree 4. We CANNOT directly encode it as SDP.
    # Workaround: introduce moment matrix Q = m m^T (rank-1) — relax to PSD.
    # But then ||f*f||_2^2 expressible via moments of g = f*f.

    # g_n = sum_k C(n,k) m_k m_{n-k}     (moments of f*f).
    # We compute g_n as quadratic in m. To make SDP, treat as min over quadratic.

    # ||f*f||_2^2 = ?  not directly expressible from moments of g alone unless
    # we have ALL moments of g. With finite moments, get a LB via moment SDP on g.
    # Specifically:  given g_0,...,g_{2N}, min ||rho||_2^2 over nonneg densities rho
    # with those moments. This is an infinite-dim LP; lower-bound via:
    #   ||g||_2^2 >= sum c_k g_k for any polynomial p(x) = sum c_k x^k
    #               with p(x) <= 1[x in supp]/(b-a)^? ... messy.

    # Simplest implementable: use Cauchy-Schwarz and Hausdorff to bound things.
    # We do: min M s.t. M * g_0 >= g_n^2/g_{2n} for some n? Markov ineq style.

    # ACTUAL minimal SDP I'll implement:
    #   Variables: moments m_0,..,m_N of f.
    #   Hausdorff PSD: H[i,j] = m_{i+j}, size (N2+1).
    #   Localizing: ((1/4)^2 - x^2) PSD: L[i,j] = (1/16) m_{i+j} - m_{i+j+2}.
    #   m_0 = 1.
    #   Symmetric optimum (CS conjecture): m_{2k+1} = 0.
    #
    #   Minimize: g_0_squared_proxy = something quadratic in moments
    #     Use:  M_LB := max_n  g_n^2 / (g_{2n} * ??) - based on Cauchy-Schwarz.
    #
    # Concretely:  ||f*f||_inf >= |g(t)| for any t, and
    #   g(t) = sum_n g_n^{(centered)} ... no good.
    #
    # Known cleaner tactic: ||g||_inf >= g_0 / |supp| = 1/1 = 1. Cant beat.
    # ||g||_inf >= ||g||_2^2 (since ||g||_1 = 1) — needs ||g||_2^2.
    #
    # For pos measure on [-1/2,1/2] w/ moments g_0=1, g_2=v (variance proxy):
    #   ||g||_2^2 unbounded BELOW by 1 (uniform on [-1/2,1/2] has ||g||_2^2 = 1).
    # So pure moment LB on ||g||_2^2 is 1, giving M >= 1. WEAK.

    # Build the SDP anyway to confirm.

    # Symmetric ansatz: m_{odd} = 0 (CS conjecture)
    constraints = [m[0] == 1]
    for k in range(1, N + 1, 2):
        constraints.append(m[k] == 0)

    # Hausdorff PSD on f's moments
    H = cp.bmat([[m[i + j] for j in range(N2 + 1)] for i in range(N2 + 1)])
    constraints.append(H >> 0)

    # Localizing for [-1/4, 1/4]:  ((1/4)^2 - x^2) f >= 0 =>
    #   L[i,j] = (1/16) m_{i+j} - m_{i+j+2}  PSD, size (N2) x (N2)
    if N2 >= 1:
        L = cp.bmat([[(1/16) * m[i + j] - m[i + j + 2] for j in range(N2)]
                     for i in range(N2)])
        constraints.append(L >> 0)

    # Compute g moments (quadratic in m): g_n = sum_k C(n,k) m_k m_{n-k}
    # We only get this directly via SDP-relaxation of moment matrix:
    #   define variable G[i,j] = m_i m_j PSD (rank-1 ideal, relax to PSD).
    G = cp.Variable((N + 1, N + 1), symmetric=True)
    constraints.append(G >> 0)
    constraints.append(G[0, 0] == 1)
    # Couple: G[i,j] should equal m_i m_j; we enforce 1st row/col linear constraints
    for i in range(N + 1):
        constraints.append(G[0, i] == m[i])

    # g moments
    g = []
    for n in range(2 * N + 1):
        terms = []
        for k in range(max(0, n - N), min(n, N) + 1):
            if 0 <= k <= N and 0 <= n - k <= N:
                terms.append(comb(n, k) * G[k, n - k])
        g.append(sum(terms))

    # ||g||_2^2 lower bound:  for any polynomial p(t) = sum c_n t^n with
    #   p(t)^2 <= integrand stuff... too complex.
    # Use simple: min g_0 (= 1) — trivial.
    #
    # Better: use the L^2 SDP bound on g (since g has moments g_n on [-1/2, 1/2]):
    #   min ||g||_2^2 over densities with moments g_n is itself an SDP, but for
    #   given numeric g_n. Here g_n are themselves SDP variables. Bilevel.
    #
    # SHORTCUT: lower bound  ||g||_2^2 >= g_0^2 / (b-a) = 1 / 1 = 1.
    # So M >= 1. Trivial.

    # Instead: use second moment of g to get sharper:
    #   By Cauchy-Schwarz on g (a measure on [-1/2,1/2]):
    #     g_0 = int g = 1
    #     g_2 = int t^2 g(t) dt
    #   And g <= M (constant), so g_2 <= M * int_{-1/2}^{1/2} t^2 dt = M/12.
    #   => M >= 12 * g_2.

    M_var = cp.Variable()
    # Lower bound: M >= 12 g_2  (from g(t) <= M => int t^2 g <= M * (1/12))
    constraints.append(M_var >= 12 * g[2])

    # Even sharper: M * int phi^2 dt >= (int phi g dt)^2 / (int g) (Cauchy-Schwarz weighted by g)
    # Use phi(t) = sum a_n t^n, optimize a — but that's another SDP.

    # Sharper via Chebyshev on g:  for any polynomial p(t),
    #   int p(t) g(t) dt <= ||g||_inf * int |p(t)| dt
    # so ||g||_inf >= |int p g| / int |p|.
    # Maximize over p with int |p| = 1 (LP).

    # We'll compute the simplest 12*g_2 bound, then try a Chebyshev polynomial enhancement.

    # Chebyshev enhancement: pick p(t) = 1 - 4 t^2 (positive on [-1/2,1/2], peak 1 at 0).
    # int p g = g_0 - 4 g_2 = 1 - 4 g_2.
    # int p = int_{-1/2}^{1/2} (1 - 4 t^2) dt = 1 - 4 * (1/12) = 2/3.
    # So ||g||_inf >= (1 - 4 g_2) / (2/3) = 1.5 - 6 g_2  IF the integrand is nonneg
    # (it is). But we want a LOWER bound; this gives M >= 1.5 - 6 g_2.
    # Combined with M >= 12 g_2: minimize over g_2: at g_2 = 1/12, both give M = 1.
    # No improvement.

    # Use polynomial peaked at non-zero t; or richer family.
    # Actually for symmetric f, g is symmetric, peak at t=0; bound from g(0).
    # g(0) = (f*f)(0) = int f^2. By Cauchy-Schwarz: int f^2 >= 1/(b-a) = 2.
    # Wait, supp f is [-1/4,1/4] of length 1/2, so ||f||_2^2 >= 1/(1/2) = 2.
    # So if max is at 0: M >= 2. But max may NOT be at 0 (CS 2017 example).

    # We can't certify max location from moments. So use weighted Chebyshev:
    # ||g||_inf >= max_t g(t). We approximate g(t) by its moments via:
    #   g(t) approx sum c_k g_k where c_k is Chebyshev expansion of delta-like kernel.

    # For the test, just compute 12*g_2 bound with ||f||_2^2 SDP variable.

    # Actually, ||f||_2^2 is encoded in the moment matrix:
    # ||f||_2^2 = int f^2 = sum c_k m_k where c_k are moments of f? No, that's not right.
    # ||f||_2^2 is NOT a moment of f.

    # But (f*f)(0) = int f(s) f(-s) ds = int f(s)^2 ds (if f symmetric) = ||f||_2^2.
    # ||f||_2^2 is NOT linear in moments. It's quadratic in densities.
    #
    # In terms of a Christoffel-type bound: for a measure f on [a,b],
    #   ||f||_2^2 = sup_{p poly} (int p f)^2 / int p^2.
    # SDP bound via moment matrix:
    #   ||f||_2^2 >= u^T H_f^{-1} u where u = (m_0,m_1,...).  (Christoffel-Darboux.)
    # Or rather, sup over deg p of (m^T c)^2 / (c^T H c) where H is moment matrix.

    # Rather than going down this rabbit hole, just solve the simple SDP and report.

    objective = cp.Minimize(M_var)
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except Exception as e:
        if verbose:
            print(f"SCS failed: {e}, trying default")
        prob.solve()

    return prob, m, g, M_var


def main():
    print("=" * 70)
    print("Hausdorff Moment SDP Lower Bound on C_{1a}")
    print("=" * 70)
    for N in [4, 6, 8, 10, 12]:
        print(f"\n--- N = {N} (moments m_0,...,m_{N}) ---")
        prob, m, g, M_var = build_hausdorff_moment_sdp(N, verbose=False)
        print(f"  Status:  {prob.status}")
        print(f"  M >= 12*g_2 bound:  {M_var.value:.6f}")
        print(f"  m =  {[float(m.value[k]) for k in range(N+1)]}")
        print(f"  g_0,g_2 =  {float(g[0].value):.6f}, {float(g[2].value):.6f}")


if __name__ == "__main__":
    main()
