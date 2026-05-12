"""
Smooth-weight generalization of Theorem 1 (proof/coarse_cascade_method.md).

Generalized inequality:
    max(f*f) * \int g  >=  \int g(t) (f*f)(t) dt
                        =  \iint g(x+y) f(x) f(y) dx dy
                        >=  sum_{i,j} K_{i,j}(g) mu_i mu_j           (*)
where for nonneg g supported on [-1/2,1/2],
    K_{i,j}(g) := (1/area(B_i x B_j)) * \iint_{B_i x B_j} g(x+y) dx dy
    mu_i := \int_{B_i} f >= 0, sum mu_i = 1.

Wait: keep K_{i,j} = \iint_{B_i x B_j} g(x+y) dx dy directly (no normalization).
Then since f >= 0 and \int_{B_i} f = mu_i is only the *mass*, not enough to
control the double integral pointwise. The correct lower bound is

    \iint g(x+y) f(x) f(y) dx dy  >=  sum_{i,j} ( min_{(x,y) in B_i x B_j} g(x+y) ) mu_i mu_j

So define
    K_{i,j}(g) := h^{-2} * inf_{(x,y) in B_i x B_j} g(x+y)        (per unit mass^2)
                = inf_{u in [(i+j)h, (i+j+2)h] - 1/2}  g(u)
where bins are B_k = [k h - 1/4, (k+1)h - 1/4], h = 1/(2d), so x+y ranges over
[(i+j)h - 1/2, (i+j+2)h - 1/2].

Then
    max(f*f) * \int g  >=  sum_{i,j} K_{i,j}(g) * mu_i mu_j   ... where K_{i,j}
is computed WITHOUT the h^{-2} factor (we keep the integral form, replacing
g pointwise on B_i x B_j by its infimum):

    max(f*f) * \int g  >=  sum_{i,j} [ h^2 * inf_{B_i+B_j} g ] * mu_i mu_j

Equivalent reformulation: divide both sides by \int g, define normalized weights
    w_{i,j}(g) := h^2 * inf_{B_i+B_j} g(t) / \int g
Then
    max(f*f)  >=  Q_g(mu) := sum w_{i,j}(g) mu_i mu_j

The CS-cascade Theorem 1 is the special case g = 1_W:
    \int g = ell h,  inf_{B_i+B_j} 1_W = 1 iff [i+j, i+j+2] subset of W's support index range, 0 else.
So w_{i,j}(1_W) = h^2 / (ell h) = h/ell when (i,j) is in the "fully-contained" set,
yielding TV_W(mu) / (2d/ell) ... matches.

QUESTIONS THIS SCRIPT ANSWERS:
1) For arcsine extremizer, how does Q_g(mu_arcsine) compare to indicator bound?
2) For d=8, what's the worst-case mu in Delta_d for the *best* admissible g?
3) Is there a g such that  min_{mu in Delta_d}  max_g  Q_g(mu)  >= 1.28 ?
"""

import numpy as np
from scipy import integrate, optimize


# ------------------------------------------------------------------
# Bin geometry: B_k = [k*h - 1/4, (k+1)*h - 1/4], h = 1/(2d), k=0..d-1.
# B_i + B_j = [(i+j)*h - 1/2, (i+j+2)*h - 1/2].
# ------------------------------------------------------------------

def bin_sum_interval(i, j, d):
    h = 1.0 / (2 * d)
    return ((i + j) * h - 0.5, (i + j + 2) * h - 0.5)


def K_from_g(g, d, mode="inf"):
    """
    Bin masses: mu_i = \int_{B_i} f. Then
        \int g(t) (f*f)(t) dt = \iint g(x+y) f(x) f(y) dx dy
                              >= sum_{i,j} ( inf_{B_i+B_j} g ) * mu_i mu_j     (rigorous, mode="inf")
    so K[i,j] := inf_{B_i+B_j} g(u)   (NO h^2 factor; mu_i ALREADY contains the dx integration).

    For "exact" we set K[i,j] = inf-of-average; this is NOT a valid lower bound unless we use
    a worst-case f within each bin -- so we DON'T use "exact" for rigor. Keep only "inf".
    """
    K = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            a, b = bin_sum_interval(i, j, d)
            if mode == "inf":
                xs = np.linspace(a, b, 101)
                K[i, j] = float(np.min(g(xs)))
            elif mode == "midpoint":  # NOT rigorous, for tightness check only
                K[i, j] = float(g(0.5 * (a + b)))
            else:
                raise ValueError(mode)
    return K


def integral_g(g, lo=-0.5, hi=0.5):
    val, _ = integrate.quad(g, lo, hi, limit=200)
    return val


# ------------------------------------------------------------------
# Candidate weights g (nonneg, supported in [-1/2, 1/2])
# ------------------------------------------------------------------

def g_indicator(s, ell, d):
    """Indicator of W = [s/(2d), (s+ell)/(2d)] (NOT centered; matches CS Thm 1 W)."""
    h = 1.0 / (2 * d)
    a, b = s * h, (s + ell) * h          # this is in absolute t units (target on R)
    # But t = x+y ranges over [-1/2, 1/2]; W in CS proof is shifted: see proof step 3.
    # Actually CS uses W = [s/(2d), (s+ell)/(2d)] -- positive side. We want centered.
    # For experiment, let g be a window centered at 0.
    a = -ell * h / 2
    b = +ell * h / 2

    def g(u):
        u = np.asarray(u, dtype=float)
        return ((u >= a) & (u <= b)).astype(float)
    return g, (b - a)


def g_gaussian(sigma):
    """Truncated Gaussian on [-1/2, 1/2]."""
    def g(u):
        u = np.asarray(u, dtype=float)
        out = np.exp(-0.5 * (u / sigma) ** 2)
        out = np.where(np.abs(u) <= 0.5, out, 0.0)
        return out
    return g


def g_raised_cosine(width):
    """g(u) = (1+cos(pi u/width))/2 for |u|<=width, else 0. Smooth, nonneg, compact."""
    def g(u):
        u = np.asarray(u, dtype=float)
        out = 0.5 * (1.0 + np.cos(np.pi * u / width))
        out = np.where(np.abs(u) <= width, out, 0.0)
        return out
    return g


def g_triangle(width):
    """Triangle (Fejer-like): autocorr of indicator(width/2)."""
    def g(u):
        u = np.asarray(u, dtype=float)
        out = np.maximum(1.0 - np.abs(u) / width, 0.0)
        return out
    return g


# ------------------------------------------------------------------
# Bound Q_g(mu) and the min-mu problem
# ------------------------------------------------------------------

def Q_g(mu, K, intg):
    """  max(f*f) >= Q_g(mu) := mu^T K mu / int(g)   """
    return float(mu @ K @ mu) / intg


def min_quad_form_on_simplex(K, intg, n_starts=40, seed=0):
    """ Minimize mu^T K mu / intg over the simplex Delta_d (nonneg, sum=1).
        Returns (val, mu_argmin). Uses SLSQP with random restarts. """
    d = K.shape[0]
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    A = K + K.T
    for _ in range(n_starts):
        x0 = rng.dirichlet(np.ones(d))
        cons = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0,
                 "jac": lambda x: np.ones_like(x)},)
        bnds = [(0.0, 1.0)] * d
        try:
            res = optimize.minimize(
                lambda x: x @ K @ x / intg,
                x0,
                jac=lambda x: (A @ x) / intg,
                method="SLSQP",
                bounds=bnds,
                constraints=cons,
                options={"ftol": 1e-12, "maxiter": 300},
            )
        except Exception:
            continue
        if res.success and res.fun < best[0]:
            best = (float(res.fun), res.x.copy())
    return best


# ------------------------------------------------------------------
# Arcsine extremizer test
# ------------------------------------------------------------------

def arcsine_masses(d, support=(-0.25, 0.25)):
    """Bin-mass vector of Beta(1/2,1/2) supported on [-1/4,1/4] (so f*f peaks ~1.5029)."""
    a, b = support
    h = (b - a) / d
    # CDF of Beta(1/2,1/2) on [a,b]: F(t) = (2/pi) arcsin( sqrt((t-a)/(b-a)) )
    edges = a + h * np.arange(d + 1)
    s = (edges - a) / (b - a)
    F = (2.0 / np.pi) * np.arcsin(np.sqrt(np.clip(s, 0, 1)))
    mu = np.diff(F)
    mu /= mu.sum()
    return mu


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

if __name__ == "__main__":
    d = 8
    h = 1.0 / (2 * d)
    mu_as = arcsine_masses(d)

    print(f"d = {d}, h = {h}")
    print(f"arcsine mu = {np.round(mu_as, 4)}")
    print()

    # --- Catalogue of g ---
    cands = []
    for ell in [2, 3, 4, 5, 6]:
        gI, lenI = g_indicator(0, ell, d)
        cands.append((f"indicator(ell={ell})", gI))
    for sigma in [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]:
        cands.append((f"gaussian(sigma={sigma})", g_gaussian(sigma)))
    for w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        cands.append((f"raised_cos(w={w})", g_raised_cosine(w)))
    for w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        cands.append((f"triangle(w={w})", g_triangle(w)))

    print(f"{'name':30s} {'int g':>9s} {'Q_g(arcsine)':>14s} "
          f"{'min_mu Q_g':>13s} {'argmin entropy':>16s}")
    print("-" * 90)

    rows = []
    for name, g in cands:
        intg = integral_g(g)
        if intg <= 0:
            continue
        K = K_from_g(g, d, mode="inf")
        q_as = Q_g(mu_as, K, intg)
        q_min, x_min = min_quad_form_on_simplex(K, intg, n_starts=30, seed=1)
        ent = -np.sum(np.where(x_min > 1e-12, x_min * np.log(x_min), 0.0))
        rows.append((name, intg, q_as, q_min, ent))
        print(f"{name:30s} {intg:9.4f} {q_as:14.4f} {q_min:13.4f} {ent:16.4f}")

    # --- Best g per metric ---
    print()
    best_q_as  = max(rows, key=lambda r: r[2])
    best_q_min = max(rows, key=lambda r: r[3])
    print(f"Best Q_g at arcsine (closer to 1.5029 = better tightness):")
    print(f"  {best_q_as[0]}: Q_g(mu_arcsine) = {best_q_as[2]:.4f}")
    print(f"Best lower bound from a single g (max over g of min over mu):")
    print(f"  {best_q_min[0]}: min_mu Q_g(mu) = {best_q_min[3]:.4f}")

    # --- Convex combination over g (mixture LP) ---
    # If we mix g's: g_mix = sum a_k g_k, the bound averages.
    # The inequality  max(f*f) >= mu^T K_k mu / intg_k   for each k  =>
    #   max(f*f) >= max_k mu^T K_k mu / intg_k.
    # Worst-case: min_mu max_k Q_{g_k}(mu).
    # Compute by maximizing the min over a coarse grid of mu? Just evaluate
    # max_k Q at the arcsine mu and at the worst per-kernel argmin.
    # The TRUE worst-case is min_{mu in Delta_d} max_k Q_{g_k}(mu) -- a hard
    # nonconvex bilinear problem. We give a heuristic upper bound by random search.
    Ks    = [(r[0], K_from_g(g, d, mode="inf"), integral_g(g)) for r, (_, g) in zip(rows, cands)]
    rng = np.random.default_rng(123)
    best_minmax = (np.inf, None, None)
    for trial in range(2000):
        x = rng.dirichlet(np.ones(d))
        vals = [(name, x @ K @ x / I) for (name, K, I) in Ks]
        winner = max(vals, key=lambda v: v[1])
        if winner[1] < best_minmax[0]:
            best_minmax = (winner[1], winner[0], x)
    print()
    print(f"Random-search upper bound on  min_mu max_g Q_g(mu)  : {best_minmax[0]:.4f}")
    print(f"  attained by g = {best_minmax[1]}, mu = {np.round(best_minmax[2],3)}")

    # --- Compare to single best indicator (CS Theorem 1) ---
    ind_rows = [r for r in rows if r[0].startswith("indicator")]
    print()
    print("CS Theorem 1 (indicator-only) recap:")
    for r in ind_rows:
        print(f"  {r[0]:25s} min_mu = {r[3]:.4f}   Q(arcsine) = {r[2]:.4f}")
