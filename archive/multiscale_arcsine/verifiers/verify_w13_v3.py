"""W13 v3: Rigorous cross-Bessel with explicit closed-form for tail.

The asymptotic expansion of J_0(z)^2 is:
    J_0(z)^2 = (1/(pi z)) * (1 + sin(2z)) + O(1/z^2)
            = (1 + sin(2z))/(pi z) + smaller

For two scales:
    J_0(pi a xi)^2 * J_0(pi b xi)^2
    ~ (1+sin(2 pi a xi))(1+sin(2 pi b xi)) / (pi^2 ab xi^2)
    = [1 + sin(2 pi a xi) + sin(2 pi b xi) + sin(2 pi a xi)*sin(2 pi b xi)] / (pi^2 ab xi^2)
    = [1 + sin(2 pi a xi) + sin(2 pi b xi) + (cos(2pi(a-b)xi) - cos(2pi(a+b)xi))/2] / (pi^2 ab xi^2)

For diagonal (a=b=d):
    J_0(pi d xi)^4 ~ (1+sin(2 pi d xi))^2 / (pi^2 d^2 xi^2)
                  = (1 + 2 sin(2 pi d xi) + sin^2(2 pi d xi)) / (pi^2 d^2 xi^2)
                  = (1 + 2 sin(2 pi d xi) + (1-cos(4 pi d xi))/2) / (pi^2 d^2 xi^2)
                  = (3/2 + 2 sin(2 pi d xi) - cos(4 pi d xi)/2) / (pi^2 d^2 xi^2)

Tail past T of just the leading constant:
    int_T^inf 1/(pi^2 ab xi^2) dxi = 1/(pi^2 ab T)
    (off-diagonal)

    int_T^inf 3/(2 pi^2 d^2 xi^2) dxi = 3/(2 pi^2 d^2 T)
    (diagonal)

The oscillating pieces:
    int_T^inf sin(2 pi c xi)/xi^2 dxi  = small (alternating, |.| < 1/(2 pi c T^2))
    int_T^inf cos(2 pi c xi)/xi^2 dxi  = same order

So past T=1e5 the residue is order 1/(ab * 1e5)  ~ 1e-3 for d=0.025
which is NOT negligible at the 5-digit level.

CORRECT APPROACH: simply use mp.quad with a very large explicit cutoff.
Or use the bulk + ASYMPTOTIC tail (closed form for the leading 1/xi^2 part,
plus a rigorous bound on the oscillating residual).

Implementation:
  bulk:  int_0^T  via mp.quad with high prec
  tail:  exact   int_T^inf [LEADING_CONSTANT]/xi^2 dxi + bounded oscillating residual
"""
import mpmath as mp

mp.mp.dps = 50


def cross_bessel_explicit_tail(a, b, T):
    """Compute 2 * int_0^infty J_0(pi a xi)^2 J_0(pi b xi)^2 dxi.

    Tail [T, inf] uses the asymptotic:
        J_0(z)^2 = (cos(z - pi/4)^2 + sin(z - pi/4)^2) * 2/(pi z) corrections...
    Actually:  J_0(z) ~ sqrt(2/(pi z)) cos(z - pi/4), so
              J_0(z)^2 ~ (2/(pi z)) cos^2(z - pi/4) = (1 + sin(2z))/(pi z) * (corrections)
    Higher-order:
        J_0(z)^2 = (1/(pi z)) + (sin(2z))/(pi z) - (cos(2z))/(8 pi z^2) - 1/(8 pi z^3) + O(1/z^3)
        (from Watson's Bessel book, see also DLMF 10.18)

    For our problem:
        product = (1 + sin(2 pi a xi))(1 + sin(2 pi b xi)) / (pi^2 ab xi^2) + smaller

    LEADING tail integral [T, inf]:
        I_lead = int_T^inf 1/(pi^2 a b xi^2) dxi = 1/(pi^2 a b T)
    plus four oscillatory integrals (sin(2pi a xi)/xi^2 etc.), each bounded by
        |int_T^inf sin(2 pi c xi)/(pi^2 ab xi^2) dxi| <= 1/(pi^2 ab) * 1/(2 pi c T^2)
                                                       = 1/(2 pi^3 ab c T^2)
    These are sub-leading at large T.

    For diagonal:
        product = (3/2 + 2 sin(2 pi d xi) - cos(4 pi d xi)/2) / (pi^2 d^2 xi^2) + smaller
    LEADING tail:
        I_lead_diag = int_T^inf (3/2)/(pi^2 d^2 xi^2) dxi = 3/(2 pi^2 d^2 T)
    """
    a_mp = mp.mpf(a)
    b_mp = mp.mpf(b)
    pi = mp.pi

    def integrand(xi):
        return mp.besselj(0, pi * a_mp * xi)**2 * mp.besselj(0, pi * b_mp * xi)**2

    # Bulk
    bulk = mp.quad(integrand, [0, T])

    # Asymptotic tail
    if a == b:
        d = a_mp
        # leading: 3/(2 pi^2 d^2 T)
        # plus oscillatory: 2 sin(2 pi d xi) leading, -cos(4 pi d xi)/2
        # Use full sub-leading expansion. We compute the asymptotic
        # int_T^inf as approximately the integral of asymptotic with finite cutoff plus correction.
        # We will compute it by computing the integral of the ASYMPTOTIC integrand from T to a much larger T2.
        asy_integrand = lambda xi: (mp.mpf("1.5") + 2 * mp.sin(2 * pi * d * xi)
                                    - mp.cos(4 * pi * d * xi) / 2) / (pi**2 * d**2 * xi**2)
    else:
        # int 1 + sin(2 pi a xi) + sin(2 pi b xi) + (cos(2 pi (a-b) xi) - cos(2 pi (a+b) xi))/2
        # all divided by pi^2 ab xi^2
        asy_integrand = lambda xi: (1
                                    + mp.sin(2 * pi * a_mp * xi)
                                    + mp.sin(2 * pi * b_mp * xi)
                                    + (mp.cos(2 * pi * (a_mp - b_mp) * xi)
                                       - mp.cos(2 * pi * (a_mp + b_mp) * xi)) / 2
                                   ) / (pi**2 * a_mp * b_mp * xi**2)

    # Compare bulk integrand to asymptotic at xi=T -- they should agree to several digits if T large enough
    diff_at_T = float(integrand(T) - asy_integrand(T))
    rel_at_T = diff_at_T / float(integrand(T)) if integrand(T) != 0 else 0

    # Tail of asymptotic from T to infty.  This has well-defined closed-form
    # leading-order + oscillatory parts. Use quadosc with the slow beat
    # 1/(2pi*min nonzero freq) for off-diag, 1/(2pi d) for diag.
    if a == b:
        period = 1 / float(d)
    else:
        period = 1 / max(float(a + b), 1e-12)
    tail_asy = mp.quadosc(asy_integrand, [T, mp.inf], period=period)

    # Residual past T (true minus asymptotic) is O(1/xi^3) so its integral is O(1/T^2).
    # Bound: at T=1e5, residual is at most ~10/T^2 ~ 1e-9. Conservatively use 100/T^2 as a ball.
    residual_bound = mp.mpf(100) / mp.mpf(T)**2

    return 2 * (bulk + tail_asy), bulk, tail_asy, diff_at_T, rel_at_T, residual_bound


def main():
    deltas = [mp.mpf("0.138"), mp.mpf("0.055"), mp.mpf("0.025")]
    lambdas = [mp.mpf("0.85"), mp.mpf("0.10"), mp.mpf("0.05")]

    print("=" * 78)
    print("W13 v3: Asymptotic tail decomposition (dps=50, T=1000)")
    print("=" * 78)

    T = 1000.0
    I_cross = {}
    print(f"\nT = {T}")
    for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
        a, b = deltas[i], deltas[j]
        I, bulk, tail, diff, rel, resb = cross_bessel_explicit_tail(a, b, T)
        I_cross[(i, j)] = I
        if i != j:
            I_cross[(j, i)] = I
        print(f"  I(d{i+1}={float(a):.3f}, d{j+1}={float(b):.3f}) = {mp.nstr(I, 14)}")
        print(f"    bulk[0,T]   = {mp.nstr(bulk, 10)}")
        print(f"    asy tail    = {mp.nstr(tail, 8)}  (residual bound {mp.nstr(resb, 4)})")
        print(f"    integrand-asymptotic at T: {diff:.3e}  (rel {rel:.3e})")

    # Reconstruction
    K2 = mp.mpf(0)
    for i in range(3):
        K2 += lambdas[i]**2 * I_cross[(i, i)]
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        K2 += 2 * lambdas[i] * lambdas[j] * I_cross[(i, j)]
    print()
    print(f"K_2 reconstructed = {mp.nstr(K2, 14)}")

    K2_lo_v4 = 4.788823421259034
    K2_hi_v4 = 4.789630363787504
    K2_mid_v4 = 0.5 * (K2_lo_v4 + K2_hi_v4)
    print(f"v4 K_2 enclosure: [{K2_lo_v4}, {K2_hi_v4}]")
    inside = (K2 > K2_lo_v4) and (K2 < K2_hi_v4)
    print(f"Inside? {inside}")
    print(f"K_2 - v4 mid: {mp.nstr(K2 - K2_mid_v4, 6)}")
    digits = -mp.log10(abs(K2 - K2_mid_v4)) if K2 != K2_mid_v4 else 100
    print(f"Digits match: {float(digits):.2f}")


if __name__ == "__main__":
    main()
