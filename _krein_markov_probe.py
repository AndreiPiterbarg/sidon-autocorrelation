"""
Krein-Markov moment-LP probe for C_{1a}.

Problem: f >= 0 on [-1/4, 1/4], int f = 1.
Let g = f * f, a probability measure on [-1/2, 1/2].
Goal: lower-bound max_t g(t).

Strategy:
1. Parametrize the moments m_j = int x^j f(x) dx for j=0..K, with constraints:
     - m_0 = 1
     - |m_j| <= (1/4)^j  (trivial; supp f subset [-1/4, 1/4])
     - Hankel matrix [m_{i+j}]_{i,j} PSD on [-1/4, 1/4]:
         (1/16 - x^2) >= 0 => corresponding localizing matrix PSD.
     - Equivalently the moment sequence must come from some prob measure on [-1/4, 1/4].
2. Compute mu_k = sum_j binom(k,j) m_j m_{k-j} for k=0..K.
3. Inner problem: given mu_0..mu_K, what is min_{h: density on [-1/2,1/2], moments match} ||h||_infty?
   By LP duality (Markov-Krein), this equals
       max_{p in P_K}  <c, mu> / int_{-1/2}^{1/2} p_+ (t) dt
   where p(t) = sum c_k t^k.
   Equivalently, primal: min L s.t. exists 0 <= h <= L on [-1/2,1/2] with moments mu_k.
   The condition "exists density 0 <= h <= L with given moments" is itself a generalized
   moment problem; in terms of nu = L*(unif) - h >= 0 with moments
       L * lambda_k - mu_k    where lambda_k = int_{-1/2}^{1/2} t^k dt
   we need nu to be a positive measure on [-1/2,1/2]. Both nu and h must have all
   moments and support in [-1/2,1/2]. PSD Hankel + localizing on (1/4 - t^2)>=0
   gives a tractable SDP relaxation (truncated K-moment problem).

Outer problem (on m): find m sequence minimizing the inner optimum (i.e., minimum-of-min).
This is hard jointly (bilinear in m via mu = quadratic in m). We'll do:

  (A) For a few canonical f (uniform, two-point, etc.), compute mu and run the inner SDP.
      Reports: the moment-implied lower bound on max g for those f.
  (B) Joint min: nonconvex in m. Linearize by treating mu as variable with PSD
      constraints saying mu can come from g = f*f with f a prob measure on [-1/4,1/4].
      The convex envelope: just impose mu satisfies the moment problem on [-1/2,1/2]
      (Hankel + localizing). This IGNORES the autocorrelation structure mu = sum binom*m*m.
      Without using the f*f structure we get only the trivial mu_0=1 bound.

So the meaningful object is: min over feasible m of the moment-LP lower bound.

Implementation: iterate via DCA / grid over admissible (m_2, m_4, m_6) triples
(odd moments don't reduce max via Krein for symmetric problem; do the symmetric case
as the canonical interesting one — by symmetrizing f -> f_s = (f + f(-.))/2 we have
||f_s * f_s||_infty <= ||f*f||_infty by Young/triangle ... actually NOT necessarily;
but the Cloninger-Steinerberger setup considers symmetric extremals.)

We use ALL moments (asymmetric) to be safe.

K = 6 (so mu_0..mu_6). The SDPs are tiny (4x4 Hankel, etc.).
"""

import numpy as np
import cvxpy as cp
from itertools import product

K = 6  # moment order on g
A = 0.25  # half-width of supp f
B = 0.5  # half-width of supp g

def f_moments_uniform():
    """f = 2 on [-1/4, 1/4]; m_j = int_{-1/4}^{1/4} x^j * 2 dx."""
    m = np.zeros(K + 1)
    for j in range(K + 1):
        if j % 2 == 0:
            m[j] = 2.0 / (j + 1) * (A ** (j + 1))
        else:
            m[j] = 0.0
    return m

def f_moments_two_atoms(p, a):
    """f = p delta_{-a} + (1-p) delta_{a}; m_j = p (-a)^j + (1-p) a^j."""
    m = np.zeros(K + 1)
    for j in range(K + 1):
        m[j] = p * (-a) ** j + (1 - p) * a ** j
    return m

def g_moments_from_f(m):
    """mu_k(g) = sum_{j=0}^k binom(k,j) m_j m_{k-j}."""
    mu = np.zeros(K + 1)
    from math import comb
    for k in range(K + 1):
        s = 0.0
        for j in range(k + 1):
            s += comb(k, j) * m[j] * m[k - j]
        mu[k] = s
    return mu

def lambda_moments():
    """lam_k = int_{-1/2}^{1/2} t^k dt."""
    lam = np.zeros(K + 1)
    for k in range(K + 1):
        if k % 2 == 0:
            lam[k] = 2.0 / (k + 1) * (B ** (k + 1))
        else:
            lam[k] = 0.0
    return lam

def hankel(seq, n):
    """n x n Hankel from seq of length 2n-1."""
    M = cp.bmat([[seq[i + j] for j in range(n)] for i in range(n)])
    return M

def hankel_np(seq, n):
    return np.array([[seq[i + j] for j in range(n)] for i in range(n)])

def truncated_moment_feasible(mu, n, support_half_width, name=""):
    """
    Test whether moment sequence mu (length 2n-1 at minimum) can come from
    a positive measure on [-h, h] (h = support_half_width).

    Necessary conditions:
      M_n = [mu_{i+j}]_{i,j=0..n-1}  PSD
      L_n = [(h^2 mu_{i+j} - mu_{i+j+2})]  PSD  (localizing for h^2 - t^2 >= 0)
    """
    h = support_half_width
    Mn = hankel_np(mu, n)
    # need mu of length 2n-1
    Ln_seq = [h**2 * mu[k] - mu[k + 2] for k in range(2 * (n - 1) + 1) if k + 2 <= len(mu) - 1]
    # ok this is fiddly; just check Mn PSD numerically
    eig_M = np.linalg.eigvalsh(Mn)
    return eig_M.min(), Mn

def inner_min_max_density(mu, K_use=None, n_h=None, n_nu=None, verbose=False):
    """
    SDP relaxation of:
      min L
      s.t. exists density h on [-1/2,1/2] with 0 <= h <= L, moments_k(h) = mu_k for k=0..K_use.

    We use truncated moment / Lasserre relaxation:
      - Let m_h[k] = moments of h for k=0..2N-2 (extending mu beyond K_use).
      - Let m_nu[k] = moments of nu = L*unif_measure - h (so nu has mass L - mu_0 = L - 1).
        Specifically m_nu[k] = L * lam[k] - m_h[k].
      - PSD: Hankel(m_h, N) >> 0, Hankel(m_nu, N) >> 0.
      - Localizing for support [-1/2, 1/2]: (1/4 - t^2) >= 0.
        For h: [(1/4) m_h[i+j] - m_h[i+j+2]] >> 0 for (N-1)x(N-1) matrix.
        Same for nu.
      - Match: m_h[k] = mu[k] for k=0..K_use.

    Minimize L.
    """
    if K_use is None:
        K_use = len(mu) - 1
    if n_h is None:
        n_h = (K_use // 2) + 2  # enough to require extension

    # Total moments needed: up to 2(n_h - 1) for Hankel, plus 2 more for localizing.
    # Total moment count = 2 n_h - 1 for Hankel; localizing needs up to (n_h-1)x(n_h-1)
    # i.e. moments up to 2(n_h-2) + 2 = 2 n_h - 2. So 2 n_h - 1 total moments ok.
    Nmom = 2 * n_h - 1

    L = cp.Variable(nonneg=True)
    m_h = cp.Variable(Nmom)

    lam = np.zeros(Nmom)
    for k in range(Nmom):
        if k % 2 == 0:
            lam[k] = 2.0 / (k + 1) * (B ** (k + 1))

    m_nu = L * lam - m_h

    # Match given moments
    constraints = []
    for k in range(K_use + 1):
        constraints.append(m_h[k] == mu[k])

    # Hankel PSD
    M_h = cp.bmat([[m_h[i + j] for j in range(n_h)] for i in range(n_h)])
    M_nu = cp.bmat([[m_nu[i + j] for j in range(n_h)] for i in range(n_h)])
    constraints.append(M_h >> 0)
    constraints.append(M_nu >> 0)

    # Localizing for (B^2 - t^2): size (n_h - 1) x (n_h - 1), needs moments up to 2(n_h-2)+2 = 2 n_h - 2
    nL = n_h - 1
    if nL >= 1:
        Lh = cp.bmat([[B**2 * m_h[i + j] - m_h[i + j + 2] for j in range(nL)] for i in range(nL)])
        Lnu = cp.bmat([[B**2 * m_nu[i + j] - m_nu[i + j + 2] for j in range(nL)] for i in range(nL)])
        constraints.append(Lh >> 0)
        constraints.append(Lnu >> 0)

    prob = cp.Problem(cp.Minimize(L), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    return prob.value, L.value


def joint_min_over_f(K_use=6, n_outer=4, n_inner=4, verbose=True):
    """
    Joint:
      Variables: m[0..2*n_outer-2]  (moments of f on [-1/4, 1/4])
                 mu[0..K_use]       (moments of g = f*f) -- linked by mu_k = sum binom * m * m
                 L                  (target ||h||_infty bound)
                 m_h[0..2*n_inner-2]  (moments of h, an admissible density on [-1/2,1/2])
      Constraints:
        m[0] = 1
        Hankel + localizing for f on [-1/4,1/4]    (PSD on m sequence)
        mu_k = sum_{j=0..k} binom(k,j) m[j] m[k-j]   <-- BILINEAR
        m_h[k] = mu[k] for k=0..K_use
        Hankel + localizing for h on [-1/2,1/2]
        Hankel + localizing for nu = L*unif - h
      Minimize L.

    The mu = m * m link is bilinear -> nonconvex. We linearize with a VARIABLE Mij = m_i * m_j
    and impose [[1, m^T]; [m, M]] PSD (Shor relaxation). This is a relaxation: the value
    is a LOWER BOUND on the true min L. So if the relaxed LP gives some L*, the true
    moment-constrained min ||g||_infty is >= L*.

    But we want an OBJECTIVE = "min over f of (min L compatible with mu(f))" — and we want
    a LOWER BOUND on this min, so a relaxation FROM ABOVE on the inner ||g||_infty;
    wait, we want LOWER bound on max g, so we minimize L = upper bound on max g.
    So min over f of (min L) gives min_f max g IF the inner is exact.

    Lower bound on C_{1a}: we need inf_f max g >= what we compute IF inner is upper bound on max g.
    The inner SDP gives a LOWER bound on max g (it's a moment relaxation). So min over f of inner
    is a LOWER bound on inf_f max g = C_{1a}. Good.

    The Shor relaxation on the bilinear mu = m m makes m larger feasible, so makes mu range
    larger, possibly INCLUDING mu sequences NOT realizable as f*f. This makes the inner
    max g smaller possibly. So Shor on the bilinear is a relaxation in the wrong direction:
    it gives upper bound on the relaxed problem and LOWER bound on min over f, so still
    a lower bound on C_{1a}. Good.
    """
    Nmom_f = 2 * n_outer - 1
    Nmom_h = 2 * n_inner - 1

    m = cp.Variable(Nmom_f)
    M = cp.Variable((Nmom_f, Nmom_f), symmetric=True)  # M[i,j] = m_i * m_j (Shor)

    L = cp.Variable(nonneg=True)
    m_h = cp.Variable(Nmom_h)

    lam = np.zeros(Nmom_h)
    for k in range(Nmom_h):
        if k % 2 == 0:
            lam[k] = 2.0 / (k + 1) * (B ** (k + 1))
    m_nu = L * lam - m_h

    constraints = [m[0] == 1, M[0, 0] == 1]

    # Shor: [[1, m^T], [m, M]] PSD
    block = cp.bmat([[np.array([[1.0]]), cp.reshape(m, (1, Nmom_f), order='C')],
                     [cp.reshape(m, (Nmom_f, 1), order='C'), M]])
    constraints.append(block >> 0)

    # Diagonal of M = squares
    for j in range(Nmom_f):
        # M[j,j] >= m[j]^2 (SOC)
        constraints.append(cp.SOC(1 + M[j, j], cp.hstack([1 - M[j, j], 2 * m[j]])))
    # ^ that gives M[j,j] >= m[j]^2; combined with PSD block we get M[j,j] = m[j]^2 in some sense

    # Linkage: M[i,j] consistency through PSD block doesn't pin it but PSD + diagonal SOC keeps tight

    # f-moment feasibility: Hankel and localizing for [-A, A] = [-1/4, 1/4]
    M_f = cp.bmat([[m[i + j] for j in range(n_outer)] for i in range(n_outer)])
    constraints.append(M_f >> 0)
    nL_f = n_outer - 1
    if nL_f >= 1:
        L_f = cp.bmat([[A**2 * m[i + j] - m[i + j + 2] for j in range(nL_f)] for i in range(nL_f)])
        constraints.append(L_f >> 0)

    # mu_k = sum binom(k,j) M[j, k-j]   (linear in M!)
    from math import comb
    mu = []
    for k in range(K_use + 1):
        s = sum(comb(k, j) * M[j, k - j] for j in range(k + 1) if k - j < Nmom_f)
        mu.append(s)

    # h matches mu
    for k in range(K_use + 1):
        constraints.append(m_h[k] == mu[k])

    # h on [-1/2, 1/2]
    Mh = cp.bmat([[m_h[i + j] for j in range(n_inner)] for i in range(n_inner)])
    Mnu = cp.bmat([[m_nu[i + j] for j in range(n_inner)] for i in range(n_inner)])
    constraints.append(Mh >> 0)
    constraints.append(Mnu >> 0)
    nL_h = n_inner - 1
    if nL_h >= 1:
        Lh = cp.bmat([[B**2 * m_h[i + j] - m_h[i + j + 2] for j in range(nL_h)] for i in range(nL_h)])
        Lnu = cp.bmat([[B**2 * m_nu[i + j] - m_nu[i + j + 2] for j in range(nL_h)] for i in range(nL_h)])
        constraints.append(Lh >> 0)
        constraints.append(Lnu >> 0)

    prob = cp.Problem(cp.Minimize(L), constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=20000)
    except Exception as e:
        print(f"Solve failed: {e}")
        return None, None, None
    return prob.value, m.value, M.value


if __name__ == "__main__":
    print("="*70)
    print("PART A: Inner SDP for canonical f")
    print("="*70)

    # Uniform f
    m_uni = f_moments_uniform()
    mu_uni = g_moments_from_f(m_uni)
    print(f"\nUniform f (g is triangular tent on [-1/2,1/2]; max = 2 = 2.0):")
    print(f"  m = {m_uni}")
    print(f"  mu(g) = {mu_uni}")
    val, Lstar = inner_min_max_density(mu_uni, K_use=6, n_h=5, n_nu=5)
    print(f"  Inner SDP relaxation lower bound on max g: {val:.6f}")
    print(f"  (true max for tent is 2.0)")

    # Two equal atoms at +- a — gives g concentrated at 3 points, max = inf
    # Skip: not a density.

    # Try 1/(2a) on [-a, a] with a < 1/4 (more concentrated)
    print("\nUniform f on [-a, a] for a in 1/4 -> a smaller (more concentrated f -> tent peak grows):")
    for a in [0.25, 0.20, 0.15, 0.10]:
        m = np.zeros(K + 1)
        for j in range(K + 1):
            if j % 2 == 0:
                m[j] = 2.0 / (j + 1) * (a ** (j + 1)) / (2 * a)  # density 1/(2a)
        mu = g_moments_from_f(m)
        val, _ = inner_min_max_density(mu, K_use=6, n_h=5)
        true_max = 1.0 / (2 * a)
        print(f"  a={a:.2f}: true tent peak = {true_max:.4f}; SDP inner LB = {val:.4f}")

    print("\n")
    print("="*70)
    print("PART B: Joint min over f via Shor relaxation")
    print("="*70)

    val, mvec, Mmat = joint_min_over_f(K_use=6, n_outer=4, n_inner=4)
    print(f"\nJoint min L (Shor-relaxed lower bound on C_{{1a}}): {val}")
    if mvec is not None:
        print(f"  m = {mvec}")
        print(f"  M diag = {np.diag(Mmat) if Mmat is not None else None}")
        print(f"  m_2 = {mvec[2]:.6f}, M[1,1] = {Mmat[1,1]:.6f} (should be ~m_1^2 = {mvec[1]**2:.6f})")

    # Try larger relaxation orders
    print("\nIncreasing inner order:")
    for n_in in [3, 4, 5]:
        val, _, _ = joint_min_over_f(K_use=6, n_outer=4, n_inner=n_in)
        print(f"  n_inner={n_in}: L* = {val}")

    print("\nIncreasing outer order (more f moments + larger K):")
    for K_use, n_out, n_in in [(4, 3, 3), (6, 4, 4), (8, 5, 5)]:
        val, _, _ = joint_min_over_f(K_use=K_use, n_outer=n_out, n_inner=n_in)
        print(f"  K={K_use}, n_outer={n_out}, n_inner={n_in}: L* = {val}")
