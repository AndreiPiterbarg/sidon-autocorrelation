"""
Krein-Markov v2: focus on the FUNDAMENTAL question.

Q: For g a probability measure on [-1/2, 1/2] with second moment mu_2,
   what is the LP lower bound on max_t g(t)?

Classical Markov: if g has variance >= sigma^2 and is a density on [-1/2,1/2],
the maximum density satisfies max g >= ??? Well the simple bound:
  1 = int g <= max g * (length of interval) = max g * 1, so max g >= 1.

For a tighter bound: knowing higher moments gives Hankel structure.

KEY POINT: the moments of g = f*f are constrained:
  mu_0 = 1
  mu_2 = 2 * m_2 (where m_2 = int x^2 f, for f symmetric; in general mu_2 = 2 m_0 m_2 + m_1^2 = 2 m_2 for normalized symmetric f)
  mu_2 = 2 m_2 + 2 m_1^2 ... wait let me redo:
    mu_2 = sum_{j=0}^2 binom(2,j) m_j m_{2-j} = m_0 m_2 + 2 m_1 m_1 + m_2 m_0 = 2 m_2 + 2 m_1^2
  Note: m_1 free in [-1/4, 1/4]; m_2 in [0, 1/16] (achieved by atom at +-1/4).

  So mu_2 = 2 m_2 + 2 m_1^2 <= 2 * (1/16) + 2 * (1/16) = 1/4 (if m_1, m_2 maxed).
  And mu_2 >= 0; minimum mu_2 = 2 * 0 + 0 = 0 if f = delta_0, but then m_2 = 0 means f concentrated at 0
  which isn't a density but a measure. The infimum.

What's the Markov-type lower bound on max g given mu_0=1 and mu_2?

For h on [-1/2, 1/2] density, mu_0=1, mu_2 fixed:
  By rearrangement: replacing h by h*(t) (decreasing rearrangement around 0) keeps int h and bounds mu_2.
  For symmetric h: minimum max h subject to mu_2 fixed is achieved by h = const on [-r, r] for some r.
  Then mu_0 = 2r * (1/(2r)) ... wait c * 2r = 1 means c = 1/(2r), and mu_2 = (1/(2r)) * (2r^3/3) = r^2/3.
  So r = sqrt(3 mu_2), and c = max h = 1/(2 sqrt(3 mu_2)) = 1/sqrt(12 mu_2).

  For mu_2 = 1/4: max h = 1/sqrt(3) = 0.5774. (Tent's mu_2 = 1/24, gives 1/sqrt(12/24) = sqrt(2)=1.414 < 2.)

  For min mu_2: this lower bound -> infinity as mu_2 -> 0.
  But mu_2 = 2 m_2 + 2 m_1^2, and ... hmm m_2 >= m_1^2 by Cauchy-Schwarz (variance >= 0).
  So mu_2 >= 2 m_2 >= 2 m_1^2, but lower bound on mu_2 alone — can m_1 = m_2 = 0? Yes, take
  f = unit mass on [-1/4, 1/4] symmetric (e.g., uniform): m_1 = 0, m_2 > 0. Cannot make m_2 = 0
  with f a density (would need delta at 0).

  The TRUE infimum of mu_2 over densities f >= 0, supp [-1/4,1/4], int = 1:
    inf mu_2 = inf 2 m_2 + 2 m_1^2 >= 2 * inf (m_2 + m_1^2) = 2 * inf int x^2 f if WLOG m_1 = 0 (can shift)
    No wait f's support fixed at [-1/4,1/4], can't shift. Take f concentrated near origin: e.g.,
    f = (1/h) 1_{[-h/2, h/2]} for small h: m_2 = h^2/12, mu_2 ~ h^2/6 -> 0 as h -> 0. So inf mu_2 = 0.
    But for that f, max(f*f) = 1/h * 1 = 1/h -> infinity. So the bound max g >= 1/sqrt(12 mu_2) doesn't apply
    because mu_2 -> 0 doesn't realize the symmetric uniform extremal.

The MIN mu_2 OVER ADMISSIBLE f (with constraint that f is supported in [-1/4, 1/4] AND there's no further
constraint on max) is 0 (achievable in the limit). So the Markov bound mu_2 -> 0 gives max g -> infinity,
which is consistent with concentration but doesn't give a UNIVERSAL lower bound on inf max g.

To bound inf_f max g, we need: WHENEVER mu_2 is small, max g must already be large; we want a LOWER bound
on max g that holds FOR ALL admissible (m_1, m_2, ...).

The min over admissible m of (Markov LB on max g via mu_2(m)):
  bound(m) = 1 / sqrt(12 mu_2(m)) = 1 / sqrt(12 * (2 m_2 + 2 m_1^2)) = 1 / sqrt(24 (m_2 + m_1^2))
  So minimize bound = maximize m_2 + m_1^2. Max m_2 = 1/16 (atom), max m_1 = 1/4 (atom). Cannot achieve both:
  the moment constraints (m PSD/Hankel) say m_2 >= m_1^2.
  Maximum of m_2 + m_1^2 over densities on [-1/4, 1/4] with int = 1:
    f = atom at +1/4: m_1 = 1/4, m_2 = 1/16. So m_2 + m_1^2 = 1/16 + 1/16 = 1/8.
    bound = 1/sqrt(24/8) = 1/sqrt(3) = 0.5774.
  Hmm but f = atom isn't a density.

OK so the Markov bound from mu_2 alone is HOPELESS (gives << 1.28).

Try mu_4: same analysis, the Markov-type bound from mu_4 for symmetric box density:
  h = c on [-r, r], int = 2cr = 1, mu_4 = c * 2r^5/5 = r^4/5 = (1/(2c))^4 / 5
  So mu_4 = 1/(80 c^4), c = (80 mu_4)^{-1/4}
  Minimize this bound over admissible (m): max mu_4 = m * (combo).

For HIGHER K, the SDP lower bound on max h gets better, BUT min over admissible f of that lower bound
hits the trivial answer 1 (because there exist admissible f's making the moment-LB equal to ~1).

The mu_0 = 1 lower bound on max h is just max h >= 1.
The conditioning on mu_2 only HELPS if we know mu_2 is small (forcing concentration).

INSIGHT: the moment-LB on max g is equivalent to LB on max h over ALL h with given moments,
INCLUDING h's that are NOT autocorrelations. So min over m (admissible) of (LB on max h | mu(m))
just finds the m that admits a "flat" h with the given moments. For the uniform f, mu = tent moments,
LB ~ 1.53 (we saw). For the WORST f (admissible), the LB drops to 1 trivially.

CONCLUSION (theoretical): without using the f * f structure beyond moments, the Krein-Markov approach
cannot break 1.28. The constraint "g = f*f for f a density on [-1/4,1/4]" is FAR stronger than the
moment constraints; need to encode the autocorrelation directly (Bochner).

But let's CONFIRM with a stronger inner SDP that the moment LB does drop to 1 (confirming dead end).
"""

import numpy as np
import cvxpy as cp
from math import comb

K = 8
A = 0.25
B = 0.5

def lambda_moments(N):
    lam = np.zeros(N)
    for k in range(N):
        if k % 2 == 0:
            lam[k] = 2.0 / (k + 1) * (B ** (k + 1))
    return lam

def f_moments_uniform(N):
    m = np.zeros(N)
    for j in range(N):
        if j % 2 == 0:
            m[j] = 2.0 / (j + 1) * (A ** (j + 1))  # 2 = density on [-1/4, 1/4]
    return m

def g_mu_from_f(m, K_g):
    mu = np.zeros(K_g + 1)
    for k in range(K_g + 1):
        for j in range(k + 1):
            if j < len(m) and (k - j) < len(m):
                mu[k] += comb(k, j) * m[j] * m[k - j]
    return mu

def inner_LB_max_g(mu, K_use, n_h=6):
    """SDP LB on max h over densities on [-1/2, 1/2] with given moments mu_0..mu_{K_use}."""
    Nmom = max(2 * n_h - 1, K_use + 3)
    # Use a SLAB formulation: min L s.t. h on [-1/2,1/2] with 0 <= h <= L, moments match
    L = cp.Variable(nonneg=True)
    m_h = cp.Variable(Nmom)
    lam = lambda_moments(Nmom)
    m_nu = L * lam - m_h
    cons = []
    for k in range(K_use + 1):
        cons.append(m_h[k] == mu[k])
    Mh = cp.bmat([[m_h[i + j] for j in range(n_h)] for i in range(n_h)])
    Mnu = cp.bmat([[m_nu[i + j] for j in range(n_h)] for i in range(n_h)])
    cons += [Mh >> 0, Mnu >> 0]
    nL = n_h - 1
    if nL >= 1 and 2*(nL-1)+2 < Nmom:
        Lh = cp.bmat([[B**2 * m_h[i + j] - m_h[i + j + 2] for j in range(nL)] for i in range(nL)])
        Lnu = cp.bmat([[B**2 * m_nu[i + j] - m_nu[i + j + 2] for j in range(nL)] for i in range(nL)])
        cons += [Lh >> 0, Lnu >> 0]
    prob = cp.Problem(cp.Minimize(L), cons)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=50000, eps=1e-8)
    return prob.value

def joint_LB(K_use, n_out, n_in):
    """Min over admissible f-moments of inner LB. Use separate variables m and explicit
    bilinear via Shor on M[i,j] = m_i * m_j."""
    Nm = max(2 * n_out - 1, K_use + 1)
    Nh = max(2 * n_in - 1, K_use + 3)

    m = cp.Variable(Nm)
    M = cp.Variable((Nm, Nm), symmetric=True)

    L = cp.Variable(nonneg=True)
    m_h = cp.Variable(Nh)
    lam = lambda_moments(Nh)
    m_nu = L * lam - m_h

    cons = [m[0] == 1, M[0, 0] == 1]
    # Shor PSD lift
    block = cp.bmat([
        [np.array([[1.0]]), cp.reshape(m, (1, Nm), order='C')],
        [cp.reshape(m, (Nm, 1), order='C'), M]
    ])
    cons.append(block >> 0)
    # Diagonal pinning by SOC: M[j,j] >= m[j]^2 (combined with the above gives equality at optimum often)
    for j in range(Nm):
        cons.append(M[j, j] >= 0)
        # rotated SOC: m[j]^2 <= M[j,j] * 1
        cons.append(cp.SOC(M[j, j] + 1, cp.hstack([M[j, j] - 1, 2 * m[j]])))

    # f Hankel + localizing on [-A, A]
    Mf = cp.bmat([[m[i + j] for j in range(n_out)] for i in range(n_out)])
    cons.append(Mf >> 0)
    if n_out >= 2:
        nL = n_out - 1
        Lf = cp.bmat([[A**2 * m[i + j] - m[i + j + 2] for j in range(nL)] for i in range(nL)])
        cons.append(Lf >> 0)

    # Same for the M moment matrix? The Mf above uses linear m, and that's a necessary
    # condition. We could ALSO impose moment matrix on M, but that's higher order.

    # mu_k = sum binom(k,j) M[j, k-j]
    mu = []
    for k in range(K_use + 1):
        s = 0
        for j in range(k + 1):
            if j < Nm and (k - j) < Nm:
                s = s + comb(k, j) * M[j, k - j]
        mu.append(s)

    # match h
    for k in range(K_use + 1):
        cons.append(m_h[k] == mu[k])
    # h on [-1/2, 1/2]
    Mh = cp.bmat([[m_h[i + j] for j in range(n_in)] for i in range(n_in)])
    Mnu = cp.bmat([[m_nu[i + j] for j in range(n_in)] for i in range(n_in)])
    cons += [Mh >> 0, Mnu >> 0]
    nL = n_in - 1
    if nL >= 1:
        Lh = cp.bmat([[B**2 * m_h[i + j] - m_h[i + j + 2] for j in range(nL)] for i in range(nL)])
        Lnu = cp.bmat([[B**2 * m_nu[i + j] - m_nu[i + j + 2] for j in range(nL)] for i in range(nL)])
        cons += [Lh >> 0, Lnu >> 0]

    prob = cp.Problem(cp.Minimize(L), cons)
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=50000, eps=1e-7)
    except Exception as e:
        print(f"  (solve error: {e})")
        return None, None
    return prob.value, m.value


if __name__ == "__main__":
    print("PART A: Inner LB on max g for known f.")
    print("-" * 60)

    for n_h in [4, 6, 8, 10]:
        m_uni = f_moments_uniform(2 * n_h)
        mu = g_mu_from_f(m_uni, 2 * n_h - 2)
        K_use = min(2 * n_h - 2, 6)
        val = inner_LB_max_g(mu, K_use=K_use, n_h=n_h)
        print(f"  Uniform f: K_use={K_use}, n_h={n_h}: LB on max g = {val:.4f} (true = 2.0)")

    # Try mu from a "tent-like" f with smaller support
    print("\n  More concentrated f (uniform on [-a, a]):")
    for a in [0.25, 0.20, 0.15, 0.125]:
        m = np.zeros(20)
        for j in range(20):
            if j % 2 == 0:
                m[j] = 2.0 / (j + 1) * (a ** (j + 1)) / (2 * a)
        for K_use in [4, 6, 8]:
            mu = g_mu_from_f(m, K_use)
            val = inner_LB_max_g(mu, K_use=K_use, n_h=K_use)
            print(f"    a={a}, K={K_use}: LB={val:.3f} (true peak={1/(2*a):.3f})")

    print()
    print("PART B: Joint min over f (Shor relaxation -> LB on C_{1a}).")
    print("-" * 60)

    for K_use, n_out, n_in in [(2, 3, 3), (4, 4, 4), (6, 5, 5), (8, 6, 6)]:
        val, mvec = joint_LB(K_use, n_out, n_in)
        if val is None:
            continue
        print(f"  K_use={K_use}, n_out={n_out}, n_in={n_in}: L* = {val:.4f}")
        if mvec is not None:
            print(f"    m[0..6] = {mvec[:7]}")

    print()
    print("PART C: Sanity check — fix mu_2 small and see what LB on max g is.")
    print("-" * 60)
    print("  (h on [-1/2, 1/2], int h = 1, int t^2 h = mu_2.")
    print("   Symmetric box optimum: max h = 1/sqrt(12 mu_2).)")
    for mu_2 in [1/24, 1/12, 1/8, 1/6, 1/4]:
        mu = np.zeros(7)
        mu[0] = 1
        mu[2] = mu_2
        # Set higher moments to box-density values for consistency
        # For box [-r,r] density 1/(2r), r = sqrt(3 mu_2). mu_2k = r^(2k)/(2k+1).
        r = np.sqrt(3 * mu_2)
        for k in [4, 6]:
            mu[k] = r**k / (k + 1)
        val = inner_LB_max_g(mu, K_use=6, n_h=5)
        true_box = 1 / (2 * r)
        print(f"    mu_2={mu_2:.4f} -> LB max h = {val:.4f}, true symmetric box = {true_box:.4f}")
