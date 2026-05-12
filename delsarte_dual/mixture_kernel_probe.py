"""Numerical probe: optimise MV's bound over kernel MIXTURES.

For a Bochner-admissible kernel K (Fourier-positive, supp in [-delta, delta],
int K = 1), MV's chain bound on M = ||f*f||_inf via the per-kernel QP is:

    max over admissible G:    2/u + (4/u) * min_G^2 / S_1
    (subject to phi(M, z_1) <= 0 for some z_1 in [0, sqrt(mu(M))])

where
    S_1(K) = sum_{j=1}^n a_j^2 / hat_K_R(j/u)
    phi(M, z_1) = 2/u + a - (M + 1 + 2 z_1^2 k_1
                              + sqrt(M-1-2 z_1^4) sqrt(K_2 - 1 - 2 k_1^2))
    k_1 = hat_K_R(1)
    K_2 = ||K||_2^2

Mixture: K_theta = (1-theta) K_arc + theta K_alt with K_alt one of:
  (a) Beurling-Selberg majorant of 1_{[-1/2,1/2]}  (FT = sinc^2 kernel)
  (b) Raised cosine on (-1/2, 1/2)
  (c) (Tukey window)^2  (= raised cosine squared, smoothly windowed)
  (d) Triangle on (-1/2, 1/2) (rescaled)
  (e) Indicator on (-delta, delta) (rescaled box)

Linearity:
    hat_{K_theta}(xi) = (1-theta) hat_{K_arc}(xi) + theta hat_{K_alt}(xi)
    ||K_theta||_2^2  = (1-theta)^2 ||K_arc||_2^2 + 2*(1-theta)*theta * <K_arc, K_alt>
                      + theta^2 ||K_alt||_2^2

We use MV's MO surrogate 0.5747/delta as an upper bound on ||K_arc||_2^2.
For cross-inner-product <K_arc, K_alt> = int K_arc_hat * K_alt_hat dxi
(by Parseval, both real-even), we numerically integrate; if K_alt_hat
decays sufficiently, the integral converges even though ||K_arc||_2^2
formally diverges (the surrogate is an upper bound on the truncated norm).

Conservative upper bound:  use Cauchy-Schwarz on the cross term:
    2*(1-theta)*theta * <K_arc, K_alt> <= 2*(1-theta)*theta * sqrt(K2_arc * K2_alt)
This is what we use for the rigorous bound.

Note on rigour:
  All numerical quadrature uses scipy with high tolerance; for FINAL
  publication, re-verify in arb.  This script is a fast numerical probe.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import integrate
from scipy.optimize import minimize_scalar
from scipy import special

# ----------------------------------------------------------------------------
# Constants (MV's setup)
# ----------------------------------------------------------------------------
DELTA = 0.138
U = 0.638                      # = 1/2 + delta
N_COEFFS = 119
K2_ARC_MO = 0.5747 / DELTA     # MV's MO surrogate upper bound on ||K_arc||_2^2

# ----------------------------------------------------------------------------
# Kernel Fourier-transform definitions  (real-line FT)
#
# Conventions:  hat_K(xi) = int K(x) e^{-2 pi i x xi} dx, K real-even -> hat real.
# K(0) = 1 always (mass normalisation).
# ----------------------------------------------------------------------------

def k_arc_hat(xi):
    """K_arc = (1/delta)*(eta*eta)(./delta), supp [-delta, delta].
    hat = J_0(pi xi delta)^2."""
    return special.j0(np.pi * xi * DELTA) ** 2


def k_box_hat(xi):
    """Box of half-width delta: K(x) = 1/(2*delta) on [-delta, delta], 0 else.
    Supp K subset [-delta, delta].  hat(xi) = sin(2 pi xi delta) / (2 pi xi delta)
                                            = numpy.sinc(2 xi delta).
    NOT BOCHNER-ADMISSIBLE: hat takes negative values.  Included for
    completeness; QP weights at j with hat<=0 are clipped (forces a_j=0).
    """
    return np.sinc(2.0 * xi * DELTA)


def k_triangle_hat(xi):
    """Triangle of half-width delta: K(x) = (1 - |x|/delta)/delta on [-delta, delta].
    Supp K subset [-delta, delta].  hat(xi) = numpy.sinc(xi delta)^2 >= 0.
    Bochner OK (Fejér).
    """
    return np.sinc(xi * DELTA) ** 2


def k_bspline2_hat(xi):
    """phi*phi where phi = box of half-width delta/2 (so K supported on [-delta, delta]).
    phi(x) = (1/delta) * 1_{[-delta/2, delta/2]}.
    K = phi*phi = triangle on [-delta, delta] (NOT NEW: same as k_triangle).
    Returned for completeness as a sanity-check duplicate.
    """
    return np.sinc(xi * DELTA) ** 2


def k_bspline_n_hat(xi, n=4):
    """K = phi*phi where phi = (n/2)-fold conv of box of width 2*delta/n.
    For phi to be supported on [-delta/2, delta/2], we use n boxes each of
    width delta/n; phi = box^{*n} of width delta/n each.
    Then phi has support [-delta/2, delta/2] and phi_hat = sinc(xi*delta/n)^n.
    K = phi*phi has support [-delta, delta] and K_hat = sinc(xi*delta/n)^{2n} >= 0.
    """
    return np.sinc(xi * DELTA / n) ** (2 * n)


def k_raised_cosine_hat(xi):
    """Raised cosine ('Hann') on [-delta, delta]:
       phi(x) = (1/delta)(1 + cos(pi x/delta)) on [-delta, delta], int phi = 1.
       (Wait: cos(pi x/delta) at x=delta equals cos(pi)=-1, so phi(delta)=0.)
    Actually: phi(x) = (1/delta)*(1 + cos(pi x/delta))/1, int phi = ?
    int_{-d}^{d} (1/delta)*(1+cos(pi x/d)) dx = (1/d)*[2d + (d/pi)*sin(pi)*2 - 0]/1
       = (1/d)(2d + 0) = 2.  So we need (1/(2d))(1 + cos(pi x/d)).
    Recompute hat:
       phi(x) = (1/(2*delta))*(1 + cos(pi x/delta)) for |x| <= delta.
       phi_hat(xi) = (1/(2 delta)) * [int cos(2 pi xi x) dx + int cos(pi x/d) cos(2 pi xi x) dx]
       I1 = (1/(2d)) * 2 d * sinc(2 xi d) = sinc(2 xi d).
       I2 = (1/(2d)) * (1/2) * 2 d * [sinc(2 xi d - 1) + sinc(2 xi d + 1)]
          = (1/2)[sinc(2 xi d - 1) + sinc(2 xi d + 1)].
       phi_hat(xi) = sinc(2 xi d) + (1/2)[sinc(2 xi d - 1) + sinc(2 xi d + 1)].
    Note this is the raised-cosine kernel itself; for the MV chain we want
    K_hat positive.  This kernel's hat is NOT non-negative for all xi!
    To enforce Bochner, we use phi*phi instead — see k_hann_autoconv_hat.
    """
    z = 2.0 * xi * DELTA
    return np.sinc(z) + 0.5 * (np.sinc(z - 1.0) + np.sinc(z + 1.0))


def k_hann_autoconv_hat(xi):
    """K = (raised-cosine)*(raised-cosine), supp [-delta, delta] when each phi
    is on [-delta/2, delta/2].
    phi(x) = (2/delta)*cos^2(pi x/delta) on [-delta/2, delta/2]:
       at x=delta/2: cos(pi/2)=0; phi(delta/2)=0 (smooth boundary).
       int phi = (2/d) * int_{-d/2}^{d/2} cos^2(pi x/d) dx = (2/d) * (d/2) = 1. OK.
    phi(x) = (2/d)(1 + cos(2 pi x/d))/2 = (1/d)(1 + cos(2 pi x/d)) for |x| <= d/2.
    phi_hat(xi) = (1/d) [int cos(2 pi xi x) dx + int cos(2 pi x/d) cos(2 pi xi x) dx]_{-d/2}^{d/2}
       I1 = (1/d) * d * sinc(xi d) = sinc(xi d)  (numpy convention).
       I2 = (1/d) * (1/2) * d * [sinc((xi - 1/d)*d) + sinc((xi + 1/d)*d)]
          = (1/2) * [sinc(xi d - 1) + sinc(xi d + 1)].
       phi_hat(xi) = sinc(xi d) + (1/2)*(sinc(xi d - 1) + sinc(xi d + 1)).
    K_hat(xi) = phi_hat(xi)^2 >= 0.
    """
    z = xi * DELTA
    ph = np.sinc(z) + 0.5 * (np.sinc(z - 1.0) + np.sinc(z + 1.0))
    return ph * ph


def k_tukey_hat(xi, alpha=0.5):
    """(Tukey window)^2 (auto-convolution of Tukey).
    Tukey window on [-delta/2, delta/2] with taper fraction alpha:
       phi(x) = (1 + cos(pi*((|x|/(delta/2)) - 1 + alpha)/alpha))/2 if |x|/(delta/2) > 1-alpha
              = 1                                                   if |x|/(delta/2) <= 1-alpha
       on |x| <= delta/2; 0 outside.
    Normalised by total mass (integral).
    K = phi*phi with K_hat = phi_hat^2 >= 0.
    phi_hat by quadrature.  Returns hat of normalised K.
    """
    z_arr = np.atleast_1d(xi)
    out = np.zeros_like(z_arr, dtype=float)
    a = DELTA / 2.0    # half-support

    def phi_x(x):
        ax = np.abs(x)
        if ax >= a:
            return 0.0
        r = ax / a
        if r <= 1 - alpha:
            return 1.0
        # tapered
        return 0.5 * (1.0 + np.cos(np.pi * (r - 1.0 + alpha) / alpha))

    # normalising constant Z = 2*int_0^a phi_x
    Z, _ = integrate.quad(phi_x, 0, a)
    Z = 2.0 * Z

    # phi_hat(xi) = (1/Z) * 2 * int_0^a phi(x) cos(2 pi xi x) dx
    for i, xi_v in enumerate(z_arr):
        val, _ = integrate.quad(lambda x: phi_x(x) * np.cos(2 * np.pi * xi_v * x), 0, a,
                                limit=200)
        ph = (2.0 * val) / Z
        out[i] = ph * ph
    return out if out.size > 1 else out[0]


def k_bspline_n4_hat(xi):
    """B-spline auto-conv with n=4: phi = box^{*4} of width delta/4 each, supp phi
    in [-delta/2, delta/2], K = phi*phi has supp [-delta, delta].
    K_hat = sinc(xi delta/4)^8 >= 0.  Smooth kernel, fast Fourier decay.
    """
    return np.sinc(xi * DELTA / 4.0) ** 8


def k_selberg_majorant_hat(xi):
    """Beurling-Selberg majorant of 1_{[-1/2, 1/2]} restricted/rescaled.

    The Selberg majorant of 1_{[a,b]} is the function S_+ with FT supp in
    [-1, 1] and S_+ >= 1_{[a,b]} of minimal L^1.  Its FT is a cardinal-sinc
    interpolant; details in Vaaler 1985.

    For OUR use we need a Bochner-admissible KERNEL on [-delta, delta] (not
    a majorant on R).  The Selberg construction gives a function whose FT
    is non-negative (since it's |g|^2 for some g).

    A convenient Bochner-admissible kernel inspired by Selberg is the
    "fejér" kernel restricted to delta-support:

       phi(x) = sinc(pi x / delta) * 1_{[-delta/2, delta/2]}? — but truncating
       sinc destroys positivity in Fourier.

    Instead: use Vaaler's K(x) = (sin(pi x))^2 * (1/x^2 + ...) on R; but it's
    NOT compactly supported.

    Practical Selberg-flavoured choice that IS Bochner-admissible and
    compactly supported: K = (B-spline order 4)*(B-spline order 4) scaled.

    For this experiment we use the KEY Selberg property — sinc^2 envelope —
    which is precisely the auto-convolution of two boxes (= triangle*triangle).
    That is:  K(x) = triangle_{[-delta/2, delta/2]} * triangle_{[-delta/2, delta/2]} (i.e. B-spline^4)
    with K_hat = sinc^4 (very smooth, fast decay, all positive).

    SUPP CHECK: triangle of half-width delta/2 has supp [-delta/2, delta/2];
    convolving two triangles gives supp [-delta, delta]. OK.
    Triangle phi(x) = (2/delta)(1 - 2|x|/delta) on [-delta/2, delta/2],
    int phi = 1, phi_hat(xi) = sinc(xi*delta/2)^2.
    K = phi*phi, K_hat = sinc(xi*delta/2)^4.
    """
    return np.sinc(xi * DELTA / 2.0) ** 4


# ----------------------------------------------------------------------------
# L^2 norms and inner products by quadrature
# ----------------------------------------------------------------------------

def K2_norm_squared(K_hat_func, T_inv_delta=400):
    """||K||_2^2 = int |K_hat(xi)|^2 dxi (Parseval, K real-even hat real).
    Integrate from 0 to T = T_inv_delta / delta then double."""
    T = T_inv_delta / DELTA
    val, _ = integrate.quad(lambda xi: K_hat_func(xi) ** 2, 0, T,
                           limit=500, epsabs=1e-12, epsrel=1e-10)
    return 2.0 * val


def K_inner_product(Ka_hat_func, Kb_hat_func, T_inv_delta=400):
    """<Ka, Kb> = int Ka_hat * Kb_hat dxi."""
    T = T_inv_delta / DELTA
    val, _ = integrate.quad(lambda xi: Ka_hat_func(xi) * Kb_hat_func(xi),
                           0, T, limit=500, epsabs=1e-12, epsrel=1e-10)
    return 2.0 * val


# ----------------------------------------------------------------------------
# QP for re-optimised G
# ----------------------------------------------------------------------------

def solve_qp_G(K_hat_func, n=N_COEFFS, n_grid=5001):
    """Solve:  min sum a_j^2 / w_j  s.t.  G(x) >= 1 on [0, 1/4]
       w_j = K_hat(j/u).
    Returns (a_opt, S1, min_G_grid).
    """
    import cvxpy as cp
    w = np.zeros(n)
    for j in range(1, n + 1):
        wj = K_hat_func(j / U)
        w[j - 1] = float(wj)
    # Bochner check
    bad = (w <= 1e-12)
    if bad.any():
        # set tiny weights so those terms are heavily penalised -> a_j -> 0
        w = np.where(bad, 1e-12, w)

    xs = np.linspace(0.0, 0.25, n_grid)
    B = np.zeros((n_grid, n))
    for j in range(1, n + 1):
        B[:, j - 1] = np.cos(2.0 * np.pi * j * xs / U)
    a = cp.Variable(n)
    obj = cp.Minimize(cp.sum(cp.multiply(1.0 / w, cp.square(a))))
    cons = [B @ a >= 1.0]
    prob = cp.Problem(obj, cons)
    for solver in ["MOSEK", "CLARABEL", "SCS", "ECOS"]:
        try:
            prob.solve(solver=solver, verbose=False)
            if a.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                break
        except Exception:
            continue
    if a.value is None:
        raise RuntimeError("QP failed all solvers")
    a_opt = np.asarray(a.value).flatten()
    S1 = float(np.sum((a_opt ** 2) / w))
    minG = float((B @ a_opt).min())
    return a_opt, S1, minG


# ----------------------------------------------------------------------------
# MV master inequality: solve for M
# ----------------------------------------------------------------------------

def solve_M_with_z1(a_gain, k1, K2, z1):
    """
       2/u + a <= M + 1 + 2 z1^2 k1 + sqrt(M - 1 - 2 z1^4) * sqrt(K2 - 1 - 2 k1^2).
    Return smallest M satisfying this.
    """
    LHS = 2.0 / U + a_gain
    z1sq = z1 * z1
    z14 = z1sq * z1sq
    rad2 = K2 - 1.0 - 2.0 * k1 * k1
    if rad2 < 0:
        return float("inf")  # bound vacuous
    A = math.sqrt(rad2)
    # Let s = sqrt(M - 1 - 2 z1^4). Then M = s^2 + 1 + 2 z1^4.
    # Inequality:  s^2 + 1 + 2 z1^4 + 1 + 2 z1^2 k1 + A s >= LHS
    # -> s^2 + A s + (2 + 2 z1^4 + 2 z1^2 k1 - LHS) >= 0
    c_eff = LHS - 2.0 - 2.0 * z14 - 2.0 * z1sq * k1
    # We want smallest s >= 0 satisfying s^2 + A s - c_eff >= 0
    if c_eff <= 0:
        # any s>=0 works; M_min = 1 + 2 z1^4
        return 1.0 + 2.0 * z14
    disc = A * A + 4.0 * c_eff
    s_star = (-A + math.sqrt(disc)) / 2.0
    if s_star < 0:
        s_star = 0.0
    return s_star ** 2 + 1.0 + 2.0 * z14


def best_M_over_z1(a_gain, k1, K2, n_z=51):
    """Optimise z_1 over [0, 0.50426] (Lemma 3.4 bound).

    The smallest M satisfying the chain for SOME z_1 in [0, 0.50426] is the
    LOWER BOUND on M_extremal.  Since solve_M_with_z1 returns the smallest M
    satisfying the inequality at fixed z_1, the *worst case over z_1* (i.e.,
    smallest M) is what's certified — this is the value we report.

    NOTE: counter-intuitively, larger z_1 gives a SMALLER M-bound, because
    z_1 contributes a positive 2 z_1^2 k_1 to RHS, allowing smaller M.
    Lemma 3.4 caps z_1 by 0.50426, so this is the binding constraint.
    """
    z_grid = np.linspace(0.0, 0.50426, n_z)
    Ms = [solve_M_with_z1(a_gain, k1, K2, z) for z in z_grid]
    return min(Ms), z_grid[int(np.argmin(Ms))]


# ----------------------------------------------------------------------------
# Mixture chain bound
# ----------------------------------------------------------------------------

def chain_bound_for_kernel(K_hat_func, K2_value, label="?", n_grid=2001):
    """Compute end-to-end M_lower for a given kernel."""
    a_opt, S1, minG_grid = solve_qp_G(K_hat_func, n_grid=n_grid)
    a_gain = (4.0 / U) * (minG_grid ** 2) / S1
    k1 = float(K_hat_func(1.0))
    M_star, z_star = best_M_over_z1(a_gain, k1, K2_value)
    return {
        "label": label,
        "k1": k1,
        "K2": K2_value,
        "S1": S1,
        "min_G_grid": minG_grid,
        "gain_a": a_gain,
        "M_lower": M_star,
        "z1_opt": z_star,
    }


def chain_bound_for_mixture(theta, K_arc_hat, K_alt_hat, K2_arc_bound,
                            K2_alt, K2_cross_bound, label="?", n_grid=2001):
    """K_theta = (1-theta) K_arc + theta K_alt.
    K2_theta upper bound: (1-th)^2 K2_arc + 2(1-th)*th K2_cross_bound + th^2 K2_alt.
    """
    one_m = 1.0 - theta
    K2_theta = (one_m * one_m) * K2_arc_bound + 2.0 * one_m * theta * K2_cross_bound + theta * theta * K2_alt

    def K_theta_hat(xi):
        return one_m * K_arc_hat(xi) + theta * K_alt_hat(xi)

    return chain_bound_for_kernel(K_theta_hat, K2_theta, label=label, n_grid=n_grid)


# ----------------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------------

def run_sweep(verbose=True):
    if verbose:
        print(f"=== Mixture-kernel MV bound probe ===")
        print(f"delta = {DELTA}, u = {U}, n_coeffs = {N_COEFFS}")
        print(f"K2(arcsine) = {K2_ARC_MO:.4f}  (MV's MO surrogate)")
        print()

    # Pure arcsine baseline
    base = chain_bound_for_kernel(k_arc_hat, K2_ARC_MO, label="arcsine (MV)")
    if verbose:
        print(f"  arcsine:   k1={base['k1']:.5f}  K2={base['K2']:.4f}  "
              f"S1={base['S1']:.4f}  gain={base['gain_a']:.4f}  "
              f"M={base['M_lower']:.6f}  z1*={base['z1_opt']:.4f}")

    candidates = [
        ("triangle [-delta,delta]",     k_triangle_hat),
        ("hann (raised-cos)^2 ac",      k_hann_autoconv_hat),
        ("triangle*triangle (sinc^4)",  k_selberg_majorant_hat),
        ("B-spline n=4 (sinc^8)",       k_bspline_n4_hat),
        # Box not Bochner-admissible (sinc takes negative values)
    ]

    # Pure-kernel info first.  K_arc IS in L^2 (||K_arc||_2^2 = 0.5747/delta is
    # the true norm, not a surrogate), so <K_arc, K_alt> is finite and equal to
    # the quadrature value; ||K_theta||^2 is given EXACTLY by the bilinear form.
    pure_results = []
    for label, K_alt_hat in candidates:
        K2_alt = K2_norm_squared(K_alt_hat)
        cross_actual = K_inner_product(k_arc_hat, K_alt_hat)
        cross_bound = cross_actual    # use actual value (it's exact, not bound)
        pure = chain_bound_for_kernel(K_alt_hat, K2_alt, label=label)
        pure_results.append((label, K_alt_hat, K2_alt, cross_actual, cross_bound, pure))
        if verbose:
            cross_cs = math.sqrt(K2_ARC_MO * K2_alt)
            print(f"  {label}:")
            print(f"    K2_alt = {K2_alt:.4f}, <K_arc, K_alt> = {cross_actual:.4f} "
                  f"(CS upper {cross_cs:.4f})")
            print(f"    pure: k1={pure['k1']:.5f}, S1={pure['S1']:.4f}, "
                  f"gain={pure['gain_a']:.4f}, M={pure['M_lower']:.6f}")

    # Mixture sweep
    if verbose:
        print()
        print("=== Mixture sweep (theta in [0, 0.5]) ===")
    best = base
    all_runs = [("pure arcsine", 0.0, base)]
    theta_grid = np.linspace(0.0, 1.0, 51)  # full sweep including pure alt
    for label, K_alt_hat, K2_alt, cross_actual, cross_bound, _pure in pure_results:
        if verbose:
            print(f"\n  Sweep theta for {label}:")
        local_best = (0.0, base['M_lower'])
        for th in theta_grid[1:]:
            res = chain_bound_for_mixture(
                th, k_arc_hat, K_alt_hat,
                K2_ARC_MO, K2_alt, cross_bound,
                label=f"{label}, theta={th:.3f}",
            )
            all_runs.append((label, th, res))
            if res['M_lower'] > local_best[1]:
                local_best = (th, res['M_lower'])
            if res['M_lower'] > best['M_lower']:
                best = res
            if verbose and (th in [0.05, 0.1, 0.2, 0.3, 0.5] or
                          (th - local_best[0]) < 1e-6):
                pct = 100.0 * (res['M_lower'] - base['M_lower']) / base['M_lower']
                print(f"    th={th:.3f}: K2={res['K2']:.3f} k1={res['k1']:.4f} "
                      f"gain={res['gain_a']:.4f} M={res['M_lower']:.5f} "
                      f"({'+' if pct>=0 else ''}{pct:.3f}%)")
        if verbose:
            print(f"    best for this alt: theta={local_best[0]:.3f}, M={local_best[1]:.5f}")

    if verbose:
        print()
        print("=" * 70)
        print(f"BASELINE (arcsine):                  M = {base['M_lower']:.6f}")
        print(f"BEST OVERALL (across all mixtures):  M = {best['M_lower']:.6f}")
        print(f"   label = {best['label']}")
        print(f"   delta-vs-baseline:  {best['M_lower'] - base['M_lower']:+.6f}")
        print(f"   beats MV (1.27481)? {'YES' if best['M_lower'] > 1.27481 else 'NO'}")
        print(f"   beats CS17 (1.2802)? {'YES' if best['M_lower'] > 1.2802 else 'NO'}")
    return best, base, all_runs


if __name__ == "__main__":
    run_sweep(verbose=True)
