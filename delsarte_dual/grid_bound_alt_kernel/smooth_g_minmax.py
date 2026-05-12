"""
Min-mu Max-g experiment: bypass the CS cascade?

For each candidate g (translated and scaled), define
   Q_g(mu) = mu^T K(g) mu / int(g),    K(g)[i,j] = inf_{u in B_i+B_j} g(u).
Lower bound rule:
   max(f*f) >= sup_g Q_g(mu).
Question:
   What is L*(d) := inf_{mu in Delta_d} sup_{g in family F} Q_g(mu)?
If L*(d) >= 1.28 at d=8, the cascade is BYPASSED at d=8.

Family F includes:
  - All translated indicators 1_{[a, a+ell*h]} (this is exactly CS Theorem 1's family).
  - Translated raised-cosines and Gaussians of various widths.
  - Selberg majorants (BoxCar*BoxCar = Triangle, smoothed).
  - PSWF-like (here approximated by Hann-window prolate proxy).
"""

import numpy as np
from scipy import integrate, optimize


def bin_sum_interval(i, j, d):
    h = 1.0 / (2 * d)
    return ((i + j) * h - 0.5, (i + j + 2) * h - 0.5)


def K_inf(g, d):
    K = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            a, b = bin_sum_interval(i, j, d)
            xs = np.linspace(a, b, 81)
            K[i, j] = float(np.min(g(xs)))
    return K


def integral_g(g, lo=-0.5, hi=0.5):
    val, _ = integrate.quad(g, lo, hi, limit=200)
    return val


# Kernel makers ----------------------------------------------------

def shift(g, c):
    return lambda u: g(np.asarray(u) - c)

def k_indicator(width):
    half = 0.5 * width
    def g(u):
        u = np.asarray(u, dtype=float)
        return ((u >= -half) & (u <= half)).astype(float)
    return g

def k_gaussian(sigma, trunc=0.5):
    def g(u):
        u = np.asarray(u, dtype=float)
        out = np.exp(-0.5 * (u / sigma) ** 2)
        out = np.where(np.abs(u) <= trunc, out, 0.0)
        return out
    return g

def k_raised_cosine(width):
    def g(u):
        u = np.asarray(u, dtype=float)
        out = 0.5 * (1.0 + np.cos(np.pi * u / width))
        return np.where(np.abs(u) <= width, out, 0.0)
    return g

def k_triangle(width):
    def g(u):
        u = np.asarray(u, dtype=float)
        return np.maximum(1.0 - np.abs(u) / width, 0.0)
    return g

def k_selberg_majorant(width):
    """
    Selberg-style majorant of indicator: smooth nonneg envelope.
    Use a Beurling-Selberg-like cosine-tapered indicator.
    """
    return k_raised_cosine(width)


def build_family(d):
    """All admissible (translated, scaled) g -- this is the OUTER LP variable family."""
    h = 1.0 / (2 * d)
    fam = []
    # Translated indicators of width ell*h, for ell=1..2d, centers c on a grid.
    centers = np.arange(-0.5, 0.5 + 1e-9, h / 2)  # half-bin grid
    for ell in range(1, 2 * d + 1):
        w = ell * h
        for c in centers:
            base = k_indicator(w)
            g = shift(base, c)
            # Skip if support outside [-1/2, 1/2]
            if c - w / 2 >= -0.5 - 1e-12 and c + w / 2 <= 0.5 + 1e-12:
                fam.append((f"ind(ell={ell},c={c:.3f})", g))
    # Translated Gaussians
    for sigma in [0.04, 0.06, 0.08, 0.10, 0.13, 0.16]:
        for c in np.arange(-0.4, 0.4 + 1e-9, h):
            g = shift(k_gaussian(sigma, trunc=0.5), c)
            fam.append((f"gauss(s={sigma},c={c:.3f})", g))
    # Translated raised cosines
    for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for c in np.arange(-0.4, 0.4 + 1e-9, h):
            base = k_raised_cosine(w)
            g = shift(base, c)
            if c - w >= -0.5 - 1e-12 and c + w <= 0.5 + 1e-12:
                fam.append((f"rcos(w={w},c={c:.3f})", g))
    # Translated triangles
    for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for c in np.arange(-0.4, 0.4 + 1e-9, h):
            base = k_triangle(w)
            g = shift(base, c)
            if c - w >= -0.5 - 1e-12 and c + w <= 0.5 + 1e-12:
                fam.append((f"tri(w={w},c={c:.3f})", g))
    return fam


def precompute(family, d):
    cache = []
    for name, g in family:
        intg = integral_g(g)
        if intg <= 1e-9:
            continue
        K = K_inf(g, d)
        cache.append((name, K, intg))
    return cache


def value_at_mu(mu, cache):
    """sup_g Q_g(mu)."""
    best = -np.inf
    best_name = None
    for name, K, I in cache:
        v = float(mu @ K @ mu) / I
        if v > best:
            best, best_name = v, name
    return best, best_name


def project_simplex(v):
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.where(u - cssv / np.arange(1, n + 1) > 0)[0].max()
    theta = cssv[rho] / (rho + 1)
    return np.maximum(v - theta, 0.0)


def min_over_simplex(cache, d, n_starts=200, max_iter=500, lr=0.05, seed=0):
    """Project-grad descent on  F(mu) = max_g Q_g(mu)  (subgradient = gradient at active g)."""
    rng = np.random.default_rng(seed)
    best = (np.inf, None, None)
    for s in range(n_starts):
        mu = rng.dirichlet(np.ones(d))
        prev = np.inf
        active = None
        for it in range(max_iter):
            # find active g
            best_v = -np.inf
            for name, K, I in cache:
                v = float(mu @ K @ mu) / I
                if v > best_v:
                    best_v, active = v, (name, K, I)
            name, K, I = active
            grad = (K + K.T) @ mu / I
            mu_new = project_simplex(mu - lr * grad)
            if abs(best_v - prev) < 1e-9:
                break
            prev = best_v
            mu = mu_new
        if best_v < best[0]:
            best = (best_v, mu.copy(), name)
    return best


# ------------------------------------------------------------------

if __name__ == "__main__":
    d = 8
    family = build_family(d)
    print(f"Family size: {len(family)}")
    cache = precompute(family, d)
    print(f"Active (int g > 0): {len(cache)}")

    # arcsine
    a, b = -0.25, 0.25
    h_bin = (b - a) / d
    edges = a + h_bin * np.arange(d + 1)
    s = (edges - a) / (b - a)
    F = (2.0 / np.pi) * np.arcsin(np.sqrt(np.clip(s, 0, 1)))
    mu_arcsine = np.diff(F)
    mu_arcsine /= mu_arcsine.sum()

    # tightness ratio at arcsine
    v_as, name_as = value_at_mu(mu_arcsine, cache)
    print(f"\nAt arcsine mu (truth = 1.5029):")
    print(f"  sup_g Q_g(mu_arcsine) = {v_as:.4f}    via {name_as}")
    print(f"  tightness ratio = {v_as/1.5029:.3f}")

    # Indicator-only sub-family (CS Theorem 1)
    cache_ind = [c for c in cache if c[0].startswith("ind(")]
    v_ind_as, name_ind_as = value_at_mu(mu_arcsine, cache_ind)
    print(f"  Indicator-only (CS Thm1):  {v_ind_as:.4f}  via {name_ind_as}")
    print(f"  smooth gain over indicator = {v_as - v_ind_as:+.4f}")

    # min over simplex
    print("\nMinimizing  F(mu) = sup_g Q_g(mu)  over Delta_d ...")
    best_full = min_over_simplex(cache, d, n_starts=80, max_iter=400, lr=0.03, seed=1)
    print(f"  Full family: F* ~= {best_full[0]:.4f}  at mu = {np.round(best_full[1],3)}")

    best_ind = min_over_simplex(cache_ind, d, n_starts=80, max_iter=400, lr=0.03, seed=1)
    print(f"  Indicators only (CS Thm1): F*_ind ~= {best_ind[0]:.4f}  at mu = {np.round(best_ind[1],3)}")
    print(f"  smooth gain in worst-case: {best_full[0] - best_ind[0]:+.4f}")

    print("\nTarget c = 1.28.")
    print(f"  CS cascade (d=8 indicator) gives:    {best_ind[0]:.4f}  -> below 1.28? {best_ind[0] < 1.28}")
    print(f"  Smooth-g family at d=8 gives:        {best_full[0]:.4f}  -> below 1.28? {best_full[0] < 1.28}")
