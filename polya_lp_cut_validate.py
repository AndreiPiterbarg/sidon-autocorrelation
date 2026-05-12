"""Verify directly: what is the actual min_{mu in Delta} mu^T M_K mu, and
what's phys_sup at the minimizing mu? If LP says >= 1.67 but the actual
phys_sup at the min is, say, 1.4, then the cut is invalid."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.optimize import minimize, linprog
from scipy.integrate import quad

from lasserre.polya_lp.build import build_window_matrices
from lasserre.polya_lp.symmetry import (
    project_window_set_to_z2_rescaled, z2_dim, project_M_to_z2,
    rescale_for_standard_simplex,
)
from lasserre.polya_lp.sidon_cuts import (
    krein_poisson_kernel, krein_poisson_family,
    discretize_kernel_exact, discretize_kernel_naive,
    bin_centers,
)


def compute_phys_sup_at_mu(mu, d, n_grid=10000):
    """Compute sup over t in [-1/2, 1/2] of (f*f)(t) numerically.
    f(s) = sum mu_i * (2d) * 1[bin_i].
    """
    x = bin_centers(d)
    h = 1.0 / (4 * d)  # half bin width

    # Evaluate f*f on a fine grid
    t_grid = np.linspace(-0.5, 0.5, n_grid)

    # f*f(t) = sum_{i,j} mu_i mu_j (2d)^2 * |bin_i intersect (t - bin_j)|
    # = sum_{i,j} mu_i mu_j (2d)^2 * triangle(t - (x_i + x_j); 1/(2d))
    # where triangle(s; w) = max(0, w - |s|).

    ff = np.zeros_like(t_grid)
    for i in range(d):
        for j in range(d):
            if mu[i] * mu[j] == 0:
                continue
            c = x[i] + x[j]
            tri = np.maximum(0, 1/(2*d) - np.abs(t_grid - c))
            ff += mu[i] * mu[j] * (2*d)**2 * tri
    return ff.max(), ff


def main():
    d = 4

    # The cut: Krein s=0.95 t0=0
    s, t0 = 0.95, 0.0
    K = krein_poisson_kernel(s, t0)
    M_orig = discretize_kernel_exact(K, d, n_quad=256) / 1.0  # ∫K = 1

    # Project to Z/2 + rescale
    M_proj = rescale_for_standard_simplex(project_M_to_z2(M_orig), d)
    d_eff = z2_dim(d)

    print(f"M_orig shape={M_orig.shape}, max={M_orig.max():.4f}", flush=True)
    print(f"M_proj shape={M_proj.shape} (d_eff={d_eff})", flush=True)
    print(f"M_proj =\n{M_proj}", flush=True)

    # Compute min over nu in standard simplex Delta_{d_eff} of nu^T M_proj nu
    # via brute force on a fine grid (d_eff = 2, so this is 1D)
    print("\n--- min over nu in Delta_2 of nu^T M_proj nu ---", flush=True)
    n_grid = 10001
    if d_eff == 2:
        ts = np.linspace(0, 1, n_grid)
        vals = np.array([np.array([t, 1-t]) @ M_proj @ np.array([t, 1-t]) for t in ts])
        i_min = np.argmin(vals)
        nu_min = np.array([ts[i_min], 1 - ts[i_min]])
        val_min = vals[i_min]
        print(f"  min = {val_min:.6f} at nu* = {nu_min}", flush=True)

    # Convert nu* back to mu_full
    # For d=4: orbit_id = [0, 1, 1, 0]. mu_i = nu_{orbit_id[i]} / orbit_size = nu/2.
    # So mu_full = (nu_0/2, nu_1/2, nu_1/2, nu_0/2). Sum = nu_0 + nu_1 = 1. Wait but we want sum mu = 1.
    # Original mu has sum 1; after Z/2 + rescale, nu' = D mu_orbit, where mu_i = nu'_{orbit(i)}/D[orbit].
    # So mu_full[i] = nu*[orbit(i)] / 2 (for d=4 even, all orbit_size=2).
    mu_full = np.array([nu_min[0]/2, nu_min[1]/2, nu_min[1]/2, nu_min[0]/2])
    print(f"  mu_full = {mu_full} (sum = {mu_full.sum():.4f})", flush=True)

    # Compute mu_full^T M_orig mu_full (should match the projected value)
    val_full = mu_full @ M_orig @ mu_full
    print(f"  mu_full^T M_orig mu_full = {val_full:.6f} (should match {val_min:.6f})", flush=True)

    # Now compute the ACTUAL phys_sup at this mu_full
    phys_sup, ff = compute_phys_sup_at_mu(mu_full, d)
    print(f"\n  phys_sup at mu_full = {phys_sup:.6f}", flush=True)

    # If cut is valid: mu^T M_K mu <= phys_sup
    print(f"\n  Check: mu^T M_K mu = {val_full:.6f} vs phys_sup = {phys_sup:.6f}", flush=True)
    print(f"         valid: {val_full <= phys_sup + 1e-6}", flush=True)

    # Also compute the integral directly via scipy
    f_func = lambda t: K(np.array([t]))[0]
    int_K, _ = quad(f_func, -0.5, 0.5, limit=200)
    print(f"\n  ∫_{{-1/2}}^{{1/2}} K dt (scipy) = {int_K:.6f} (expected 1.0)", flush=True)

    # Also: ∫(f*f) K via scipy
    def ff_K(t):
        # ff(t) at single t for mu_full
        result = 0.0
        for i in range(d):
            for j in range(d):
                if mu_full[i] * mu_full[j] == 0:
                    continue
                c = bin_centers(d)[i] + bin_centers(d)[j]
                tri = max(0, 1/(2*d) - abs(t - c))
                result += mu_full[i] * mu_full[j] * (2*d)**2 * tri
        return result * K(np.array([t]))[0]

    integ, _ = quad(ff_K, -0.5, 0.5, limit=500)
    print(f"  ∫(f*f) K dt (scipy, exact) = {integ:.6f}", flush=True)
    print(f"  Discrete: sum mu_i mu_j M_K_unnorm[i,j] should equal this", flush=True)
    M_unnorm = discretize_kernel_exact(K, d, n_quad=256) * 1.0  # M_K_unnorm = M_K * ∫K
    val_disc = mu_full @ M_unnorm @ mu_full
    print(f"  Discrete (n_quad=256): {val_disc:.6f}", flush=True)

    # Compare with naive
    M_naive = discretize_kernel_naive(K, d)
    val_naive = mu_full @ M_naive @ mu_full
    print(f"  Naive (K(x_i+x_j)): {val_naive:.6f}", flush=True)


if __name__ == "__main__":
    main()
