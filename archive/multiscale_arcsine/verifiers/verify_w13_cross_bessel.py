"""W13 final: Cross-Bessel verification with robust mpmath.

Use mpmath's quadosc which handles slowly-decaying oscillatory integrals
directly from a finite start to infinity by Euler-Maclaurin /Levin-type method.
We test sensitivity to the cutoff T1 by checking that results stabilize.
"""
import mpmath as mp

mp.mp.dps = 30


def cross_bessel_two_sided(a, b, T1):
    """2 * int_0^infty J_0(pi a xi)^2 J_0(pi b xi)^2 dxi.

    For i != j, the integrand decays like O(1/xi^2) asymptotically;
    for i = j (diagonal), it decays like O(1/xi^2) as well but with sign
    constant.
    """
    a_mp = mp.mpf(a)
    b_mp = mp.mpf(b)
    pi = mp.pi

    def integrand(xi):
        if xi == 0:
            return mp.mpf(1)
        return mp.besselj(0, pi * a_mp * xi)**2 * mp.besselj(0, pi * b_mp * xi)**2

    # Bulk [0, T1]
    bulk = mp.quad(integrand, [0, T1])
    # Tail [T1, inf) via quadosc. Use the asymptotic period structure.
    # J_0(pi a xi)^2 has dominant oscillation ~ cos(2 pi a xi - pi/2) / (pi a xi)
    # so its period in xi is 1/a. Product of two has fast and slow beats:
    # fast: 1/(a+b),  slow: 1/|a-b|.  quadosc needs the SHORTEST period for accuracy.
    if a == b:
        period = 1.0 / float(a)
    else:
        period = 1.0 / (float(a) + float(b))  # use FAST oscillation (shortest period)
    tail = mp.quadosc(integrand, [T1, mp.inf], period=period)
    return 2 * (bulk + tail), bulk, tail


def main():
    deltas = [mp.mpf("0.138"), mp.mpf("0.055"), mp.mpf("0.025")]
    lambdas = [mp.mpf("0.85"), mp.mpf("0.10"), mp.mpf("0.05")]

    print("=" * 78)
    print("W13 final: Cross-Bessel verification (mpmath dps=30)")
    print("=" * 78)
    print(f"deltas  = {[float(d) for d in deltas]}")
    print(f"lambdas = {[float(l) for l in lambdas]}")
    print()

    # Run with two different T1 values to see if results are stable
    for T1 in [1000.0, 5000.0]:
        print(f"--- T1 = {T1} ---")
        I_cross = {}
        for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
            a, b = deltas[i], deltas[j]
            I, bulk, tail = cross_bessel_two_sided(a, b, T1)
            I_cross[(i, j)] = I
            if i != j:
                I_cross[(j, i)] = I
            print(f"  I(d{i+1}={float(a):.3f}, d{j+1}={float(b):.3f}) = {mp.nstr(I, 12)}  "
                  f"(bulk={mp.nstr(bulk, 8)}, tail={mp.nstr(tail, 6)})")

        # Reconstruction
        K2 = mp.mpf(0)
        for i in range(3):
            K2 += lambdas[i]**2 * I_cross[(i, i)]
        for (i, j) in [(0, 1), (0, 2), (1, 2)]:
            K2 += 2 * lambdas[i] * lambdas[j] * I_cross[(i, j)]
        print(f"  K_2 reconstructed = {mp.nstr(K2, 14)}")
        print()

    # Now look at the consistency with v4
    K2_lo_v4 = 4.788823421259034
    K2_hi_v4 = 4.789630363787504
    K2_mid_v4 = 0.5 * (K2_lo_v4 + K2_hi_v4)
    print(f"v4 K_2 enclosure: [{K2_lo_v4}, {K2_hi_v4}]")
    print(f"v4 K_2 midpoint:  {K2_mid_v4}")


if __name__ == "__main__":
    main()
