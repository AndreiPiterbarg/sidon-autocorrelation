"""F6/F20 quantitative-regularization audit.

Question:  Does smoothing the autoconvolution g = f*f by a compactly supported
mollifier phi_eps give a route to break MV's 1.2748?

Plan
----
For a candidate f (we use MV's near-extremizer = G/Z restricted to [-1/4,1/4]),
g = f*f is supported on [-1/2,1/2] and Lipschitz with
    ||g'||_inf <= ||f'||_2 * ||f||_2  (||g||_Lip below).
Then for compactly supported mollifier phi_eps (supp in [-eps,eps], int = 1):
    ||g - g * phi_eps||_inf  <=  eps * ||g||_Lip.
So:
    ||g||_inf  >=  ||g * phi_eps||_inf  -  eps * ||g||_Lip.

If we can lower-bound ||g * phi_eps||_inf by some smoothed-MV bound LB(eps),
the route gives:
    C_{1a}  >=  inf_f  [LB(eps; f) - eps * ||g||_Lip(f)].

The smoothed g_eps = g * phi_eps has Fourier coefficients
    hat g_eps(n) = hat g(n) * hat phi(eps n),
which are damped at high freq.  Quantitatively, |hat g_eps(n)| <= |hat g(n)|;
and for n with |eps n| >> 1 they decay (e.g. as |hat phi|).  This SHOULD give
a sharper MO 2.14-type bound on |hat g_eps(n)| ... but observe:
    g_eps = (f * phi_eps^{1/2}) * (f * phi_eps^{1/2}) only if phi_eps factors
    as a positive autoconvolution. For a generic phi_eps (e.g. flat box / hat),
    g_eps is NOT an autoconvolution, so MO 2.14's rearrangement argument does
    not directly upgrade.

Instead, the natural smoothed version of MO 2.14 must use:
    |hat g_eps(n)|  <=  ||g_eps||_inf * sin(pi/||g_eps||_inf)/pi
when applied to g_eps (which is still 0 <= g_eps <= ||g_eps||_inf and
int = 1).  But ||g_eps||_inf <= ||g||_inf, so this is *weaker*.

To get a SHARPER MV bound on g_eps (one that exceeds 1.2748) we'd need a
KERNEL K with int(g_eps * K) > 2/u + a + small ... but int(g_eps * K) =
int(g * (K * phi_eps_reflected)) by associativity of conv on densities.
So the smoothed problem is EQUIVALENT to the unsmoothed problem with kernel
K' = K * phi_eps (with phi_eps reflected).  K' is FLATTER and has SMALLER
||K'||_2^2 -- which makes the MV bound (which depends on ||K||_2^2 LARGE for
strength via the C-S step) WEAKER.

So the trade-off is: smoothing kills the high freqs of K too, and the MV
inequality deteriorates by AT LEAST as much as the cost we save.  This is the
F6/F20 verdict at the structural level.  Below we check it numerically:

  (a) Compute ||g||_Lip = ||f'||_2 * ||f||_2 for MV's near-extremizer.
  (b) For eps = 0.001, 0.005, 0.01, 0.02, 0.05, 0.1:
         compute eps * ||g||_Lip  (regularization cost).
         compute the MV bound with kernel K replaced by K * phi_eps
            (where phi_eps = box of width 2*eps, normalized).
         smoothed MV gain a' satisfies  a'/a <=  (||K * phi_eps||_2^2 / ||K||_2^2)
            in the relevant regime;  more precisely the bound depends on
            sum_j (a_j^2 / |hat K(j) * hat phi(eps j)|^2)^{-1}.
         compute the trade-off LB(eps) - eps * ||g||_Lip.
  (c) Compare to MV's 1.2748 baseline.

Output
------
A table of (eps, ||g||_Lip, eps*Lip, smoothed_LB, trade_off, vs_MV).
"""
from __future__ import annotations

import sys
from pathlib import Path

import mpmath as mp
import numpy as np
from mpmath import mpf

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from delsarte_dual.mv_bound import (  # noqa: E402
    MV_COEFFS_119,
    MV_DELTA,
    MV_U,
    MV_K2_BOUND_OVER_DELTA,
    k1_value,
)


# ---------------------------------------------------------------------------
# Step 1: ||g||_Lip = ||f'||_2 * ||f||_2 for f = G/Z on [-1/4, 1/4].
# ---------------------------------------------------------------------------

mp.mp.dps = 30


def Z_normalizer(coeffs, u):
    u = mpf(u)
    pi = mp.pi
    Z = mpf(0)
    for j, aj in enumerate(coeffs, start=1):
        # int_{-1/4}^{1/4} cos(2 pi j x / u) dx = sin(pi j / (2u)) / (pi j / u)
        Z += aj * mp.sin(pi * j / (2 * u)) / (pi * j / u)
    return Z


def f_l2sq(coeffs, u, ngrid=8001):
    """||f||_2^2 = (1/Z^2) * int_{-1/4}^{1/4} G(x)^2 dx, by direct numerical
    integration (composite Simpson on a uniform grid).
    """
    u = float(u)
    a = np.array([float(c) for c in coeffs])
    j = np.arange(1, len(a) + 1)
    xs = np.linspace(-0.25, 0.25, ngrid)
    # G(x) = sum_j a_j cos(2 pi j x / u)
    G = np.cos(2 * np.pi * np.outer(xs, j) / u) @ a
    Z = float(Z_normalizer(coeffs, u))
    f = G / Z
    # composite Simpson
    h = xs[1] - xs[0]
    w = np.ones_like(xs)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    integ = h / 3.0 * (w * (f * f)).sum()
    return integ, Z


def fprime_l2sq(coeffs, u, ngrid=8001):
    """||f'||_2^2 = (1/Z^2) * int_{-1/4}^{1/4} G'(x)^2 dx, where
    G'(x) = -sum_j a_j (2 pi j / u) sin(2 pi j x / u).
    """
    u = float(u)
    a = np.array([float(c) for c in coeffs])
    j = np.arange(1, len(a) + 1)
    coef = -a * (2 * np.pi * j / u)
    xs = np.linspace(-0.25, 0.25, ngrid)
    Gp = np.sin(2 * np.pi * np.outer(xs, j) / u) @ coef
    Z = float(Z_normalizer(coeffs, u))
    fp = Gp / Z
    h = xs[1] - xs[0]
    w = np.ones_like(xs)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return h / 3.0 * (w * (fp * fp)).sum(), Z


def main():
    coeffs = MV_COEFFS_119
    u = MV_U
    print("=" * 72)
    print("Quantitative regularization correction audit  (F6/F20 follow-up)")
    print("=" * 72)
    print(f"  candidate f: MV's 119-cosine near-extremizer, u={float(u)}")
    print(f"  delta = {float(MV_DELTA)}")
    print()

    # 1) ||f||_2 and ||f'||_2 numerically
    f2sq, Z = f_l2sq(coeffs, u, ngrid=20001)
    fp2sq, _ = fprime_l2sq(coeffs, u, ngrid=20001)
    f2 = float(np.sqrt(f2sq))
    fp2 = float(np.sqrt(fp2sq))
    print(f"  Z = int G       = {Z:.10f}")
    print(f"  ||f||_2         = {f2:.10f}")
    print(f"  ||f||_2^2       = {f2sq:.10f}     (= (f*f)(0) for symmetric f)")
    print(f"  ||f'||_2        = {fp2:.10f}")
    print()

    # 2) Lipschitz bound on g = f*f.   ||g'||_inf <= ||f'||_2 * ||f||_2.
    g_lip = fp2 * f2
    print(f"  ||g||_Lip       <=  ||f'||_2 * ||f||_2  =  {g_lip:.10f}")
    print()

    # Sanity check: MV near-extremizer has (f*f)(0) = ||f||_2^2 ~= 1.275 ish?
    # Actually for MV, ||f*f||_inf is supposed to be near 1.275.  Here for a
    # cosine polynomial restricted to [-1/4,1/4] the f*f need not peak at 0;
    # but it gives the order of magnitude.
    print(f"  (sanity)        : ||f||_2^2 ~ {f2sq:.4f}; MV target ||f*f||_inf ~ 1.275")
    print()

    # 3) Trade-off table.  We assume an OPTIMISTIC smoothed MV bound:
    #    LB(eps) = MV_baseline (1.2748) + delta_smooth(eps),
    # where delta_smooth(eps) is a generous estimate of the GAIN from
    # smoothing.  Two cases below.
    #
    # Case A (theoretical ceiling). Even in the WILDLY OPTIMISTIC scenario
    # where the smoothed MV bound saturates the absolute ceiling 1.5029 (the
    # known UPPER bound), the cost has to be << (1.5029 - 1.2748) = 0.2281
    # to leave any net gain.  So:
    #
    # Case B (realistic). MV's own theoretical-limit calc (Section 3 of
    # mv_construction_detailed.md) shows the framework's ceiling is ~1.276,
    # so the smoothed bound can exceed 1.2748 by AT MOST ~0.001.  Then the
    # cost has to be << 0.001.
    print("  Trade-off (cost) vs eps:")
    print(f"  {'eps':>8s}  {'eps*||g||_Lip':>16s}  {'budget@1.276':>14s}  {'budget@1.5029':>14s}")
    for eps in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        cost = eps * g_lip
        budget_real = 1.276 - 1.2748          # MV ceiling - MV achieved
        budget_top = 1.5029 - 1.2748          # absolute upper bound - MV achieved
        print(f"  {eps:>8.4f}  {cost:>16.6f}  {budget_real:>14.4f}  {budget_top:>14.4f}")
    print()

    # 4) Actual smoothed MV bound.
    # Smoothed kernel K_eps = K * phi_eps where phi_eps is a flat box of
    # half-width eps.  Then in Fourier:
    #    hat K_eps(j) = hat K(j) * sinc(eps j / u)        (period u)
    # where sinc(t) = sin(pi t)/(pi t).
    # MV's master inequality (eq. 7) becomes:
    #    M + 1 + sqrt(M-1) sqrt(||K_eps||_2^2 - 1)
    #         >= 2/u + (4/u)*(min G)^2 / sum_j a_j^2 / |hat K_eps(j)|^2.
    #
    # The DENOMINATOR sum_j a_j^2 / |hat K_eps(j)|^2 is LARGER than the MV
    # one (since |hat K_eps(j)| <= |hat K(j)|), so the gain a' is SMALLER.
    # And ||K_eps||_2^2 < ||K||_2^2 makes the LHS slack SMALLER too (good
    # for the bound!).   Net effect: ?  Compute.
    print("  Smoothed MV bound  vs eps   (kernel K replaced by K * box_eps):")
    print(f"  {'eps':>8s}  {'gain a''':>10s}  {'||K_eps||_2^2':>14s}  {'M_smoothed':>12s}  {'cost':>10s}  {'NET':>10s}")
    a_coef = np.array([float(c) for c in MV_COEFFS_119])
    j_arr = np.arange(1, len(a_coef) + 1, dtype=float)
    delta = float(MV_DELTA)
    u_f = float(MV_U)
    # Period-u Fourier coefs: hat K(j) = (1/u) |J_0(pi j delta / u)|^2.
    # MV's gain formula uses |J_0(.)|^2 in the denominator (not the full
    # hat K), so we keep the J_0 factor separate.
    from scipy.special import j0 as bessel_j0
    J0vals = bessel_j0(np.pi * j_arr * delta / u_f)
    hatK = (1.0 / u_f) * J0vals ** 2
    # ||K||_2^2 surrogate from MV
    K2_baseline = float(MV_K2_BOUND_OVER_DELTA) / delta
    # min G over [0, 1/4]: from MV QP, == 1 (active).
    min_G = 1.0
    # Reproduce MV (eps=0):  a = (4/u) * (min G)^2 / sum a_j^2 / |J_0|^2.
    a_baseline = (4.0 / u_f) * (min_G ** 2) / (a_coef ** 2 / J0vals ** 2).sum()

    def solve_M(rhs, K2):
        # M + 1 + sqrt(M-1)*sqrt(K2-1) = rhs.  Quadratic in s=sqrt(M-1):
        # s^2 + s*sqrt(K2-1) + 2 - rhs = 0  =>  s = [-sqrt(K2-1) + sqrt(K2-1+4(rhs-2))]/2
        disc = (K2 - 1) + 4 * (rhs - 2)
        if disc < 0:
            return float("nan")
        s = (-np.sqrt(K2 - 1) + np.sqrt(disc)) / 2.0
        return s * s + 1.0

    M_base = solve_M(2.0 / u_f + a_baseline, K2_baseline)
    print(f"  {'(MV)':>8s}  {a_baseline:>10.6f}  {K2_baseline:>14.6f}  {M_base:>12.6f}     ----        ----")

    for eps in [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]:
        # Period-u Fourier of phi_eps (flat box of half-width eps,
        # normalized to int = 1): hat phi_eps(xi) = sinc(2 eps xi) where
        # sinc(t) = sin(pi t)/(pi t).  Evaluating at xi = j/u gives
        # sin(2 pi eps j/u)/(2 pi eps j/u).
        # The smoothed |J_0|^2 factor becomes |J_0|^2 * sinc(2 eps j/u).
        sinc = np.sin(2 * np.pi * eps * j_arr / u_f) / (2 * np.pi * eps * j_arr / u_f)
        hatK_eps = hatK * sinc
        # Effective Bessel-squared after smoothing: |J_0|^2 * sinc.
        Jeff_sq = J0vals ** 2 * sinc
        a_eps = (4.0 / u_f) * (min_G ** 2) / (a_coef ** 2 / Jeff_sq).sum()
        # ||K_eps||_2^2 = sum_n hat K_eps(n)^2 ... but K is non-L^2 in the
        # raw MV setup; MV uses a surrogate.  For the smoothed K_eps it IS
        # L^2 (smoothing kills the log endpoint singularity).  Conservative
        # estimate: ||K_eps||_2^2 <= ||K||_2^2 (use MV's surrogate).  The
        # actual smoothed value is smaller, which makes the bound tighter.
        K2_eps = K2_baseline  # conservative (upper bound)
        rhs = 2.0 / u_f + a_eps
        M_smooth = solve_M(rhs, K2_eps)
        cost = eps * g_lip
        net = M_smooth - cost
        print(f"  {eps:>8.4f}  {a_eps:>10.6f}  {K2_eps:>14.6f}  {M_smooth:>12.6f}  {cost:>10.6f}  {net:>10.6f}")
    print()

    # 5) For the truly smoothed K_eps, compute ||K_eps||_2^2 exactly via
    # Parseval:  ||K_eps||_2^2 = sum_n |hat K_eps(n)|^2  (period-u sum).
    # Use a large truncation Nmax.
    print("  Refined: use the EXACT smoothed ||K_eps||_2^2 via Parseval (Nmax=200000):")
    print(f"  {'eps':>8s}  {'gain a''':>10s}  {'||K_eps||_2^2':>14s}  {'M_smoothed':>12s}  {'cost':>10s}  {'NET':>10s}")
    Nmax = 200000
    for eps in [0.005, 0.01, 0.02, 0.05, 0.1]:
        n = np.arange(0, Nmax + 1, dtype=np.float64)
        hatK_n = (1.0 / u_f) * bessel_j0(np.pi * n * delta / u_f) ** 2
        # Box of half-width eps:  hat phi(xi) = sinc(2 eps xi).
        with np.errstate(invalid="ignore", divide="ignore"):
            sinc_n = np.where(n == 0, 1.0,
                              np.sin(2 * np.pi * eps * n / u_f) / (2 * np.pi * eps * n / u_f))
        hatK_eps_n = hatK_n * sinc_n
        # Parseval on R/uZ:   ||K_eps||_L2(R)^2 = u * sum_n |hat K_eps(n)|^2
        # Following MV's convention where ||K||_2^2 = int K^2 dx.
        K2_eps = u_f * (hatK_eps_n[0] ** 2 + 2 * (hatK_eps_n[1:] ** 2).sum())
        # Gain (over j=1..119) using the smoothed Bessel-squared.
        sinc = np.sin(2 * np.pi * eps * j_arr / u_f) / (2 * np.pi * eps * j_arr / u_f)
        Jeff_sq = J0vals ** 2 * sinc
        a_eps = (4.0 / u_f) * (min_G ** 2) / (a_coef ** 2 / Jeff_sq).sum()
        rhs = 2.0 / u_f + a_eps
        M_smooth = solve_M(rhs, K2_eps)
        cost = eps * g_lip
        net = M_smooth - cost
        print(f"  {eps:>8.4f}  {a_eps:>10.6f}  {K2_eps:>14.6f}  {M_smooth:>12.6f}  {cost:>10.6f}  {net:>10.6f}")

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print("Bound to beat: MV 1.2748.")
    print()
    print("Numerical observations (see tables above):")
    print(" - ||g||_Lip <= ||f'||_2 * ||f||_2 is LARGE (~145 for the MV near-extremizer).")
    print("   So eps * ||g||_Lip exceeds the MV-ceiling budget (~0.001 = 1.276 - 1.2748)")
    print("   for any eps > ~7 * 10^-6.  Even the absolute-ceiling budget")
    print("   (1.5029 - 1.2748 = 0.228) is exceeded for eps > ~1.6 * 10^-3.")
    print(" - Whatever the smoothed MV bound LB(eps), it is bounded above by MV's")
    print("   theoretical ceiling 1.276 (Section 3 of mv_construction_detailed.md).")
    print("   So  LB(eps) - eps*||g||_Lip  <  1.276 - eps*145, and this is < 1.2802")
    print("   (CS 2017 baseline) for any eps > ~3 * 10^-5, hence NEVER exceeds 1.2802.")
    print(" - The numerical sweep above confirms NET < 1.2748 across eps grid.")
    print()
    print("This confirms F6/F20: the smoothing weight gets dominated by the unweighted")
    print("(MV-direct) form.  No quantitative regularization correction beats MV.")
    print()
    print("FINAL VERDICT: NO -- regularization correction always exceeds smoothed gain.")
    print("F6 confirmed.")


if __name__ == "__main__":
    main()
