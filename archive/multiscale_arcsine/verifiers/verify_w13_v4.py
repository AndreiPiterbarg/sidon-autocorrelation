"""W13 v4: Brute-force long-range mp.quad with adaptive panel subdivision.

Avoid quadosc; instead integrate over many short panels covering [0, T_huge]
plus a rigorous tail bound.  Use the explicit asymptotic
    J_0(z)^2 = 1/(pi z) + sin(2z)/(pi z) + O(1/z^2)
so for off-diagonal:
    J_0(pi a xi)^2 J_0(pi b xi)^2 ~ 1/(pi^4 a b xi^2) (1+sin(2 pi a xi))(1+sin(2 pi b xi)) + O(1/xi^3)
For diagonal: 1/(pi^4 d^2 xi^2) (1+sin(2 pi d xi))^2.

Past T: rigorous tail bound = 1/(pi^4 a b T)  (using (1+sin)(1+sin) <= 4)
                            = 4/(pi^4 a b T).
"""
import mpmath as mp

mp.mp.dps = 30

def cross_bessel_brute_force(a, b, T=2000.0, panels=None):
    """2 * int_0^T  via mp.quad subdivided panels."""
    a_mp = mp.mpf(a)
    b_mp = mp.mpf(b)
    pi = mp.pi

    def integrand(xi):
        return mp.besselj(0, pi * a_mp * xi)**2 * mp.besselj(0, pi * b_mp * xi)**2

    # Subdivide [0, T] into panels of length L such that L * max(a, b) ~ 100 oscillations.
    # period in xi is 1/(a+b) for the dominant frequency; use L = 50/(a+b).
    if panels is None:
        L = 50.0 / float(a + b) if a + b > 0 else T / 100
        n_panels = max(int(T / L), 200)
    else:
        n_panels = panels

    edges = [mp.mpf(i) * mp.mpf(T) / n_panels for i in range(n_panels + 1)]
    total = mp.mpf(0)
    for k in range(n_panels):
        total += mp.quad(integrand, [edges[k], edges[k+1]])

    # Tail bound: |J_0(z)^2| <= 1 always, and the asymptotic 1/(pi^2 a xi) for large xi.
    # Conservative tail past T: use the rigorous Watson bound
    #   |J_0(z)^2| <= 2/(pi z)   for z >= some z0 ~ 1
    # so product <= 4/(pi^2 a b xi^2)
    # tail [T, inf] <= 4 / (pi^2 a b T)
    tail_upper = mp.mpf(4) / (pi**2 * a_mp * b_mp * mp.mpf(T))
    # The integrand is positive, so tail is in [0, tail_upper]
    return 2 * total, 2 * total, tail_upper, n_panels


def main():
    deltas = [mp.mpf("0.138"), mp.mpf("0.055"), mp.mpf("0.025")]
    lambdas = [mp.mpf("0.85"), mp.mpf("0.10"), mp.mpf("0.05")]

    print("=" * 78)
    print("W13 v4: Brute-force mp.quad panels (dps=30)")
    print("=" * 78)

    # Use T=2000 first for sanity, then T=10000 for higher accuracy
    for T in [2000.0, 10000.0]:
        print(f"\n--- T = {T} ---")
        I_cross = {}
        I_lower = {}
        I_upper = {}
        for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
            a, b = deltas[i], deltas[j]
            I_main, _, tail_up, np_pan = cross_bessel_brute_force(a, b, T)
            # I in [I_main, I_main + 2 * tail_upper]   (factor 2 for two-sided)
            I_lo = I_main
            I_hi = I_main + 2 * tail_up
            I_mid = (I_lo + I_hi) / 2
            I_cross[(i, j)] = I_mid
            I_lower[(i, j)] = I_lo
            I_upper[(i, j)] = I_hi
            if i != j:
                I_cross[(j, i)] = I_mid
                I_lower[(j, i)] = I_lo
                I_upper[(j, i)] = I_hi
            print(f"  I(d{i+1}={float(a):.3f}, d{j+1}={float(b):.3f}) in "
                  f"[{mp.nstr(I_lo, 12)}, {mp.nstr(I_hi, 12)}]  "
                  f"(np={np_pan}, tail_up={mp.nstr(tail_up, 6)})")

        # Reconstruction with interval arithmetic (lower bound K2)
        K2_lo = mp.mpf(0)
        K2_hi = mp.mpf(0)
        for i in range(3):
            K2_lo += lambdas[i]**2 * I_lower[(i, i)]
            K2_hi += lambdas[i]**2 * I_upper[(i, i)]
        for (i, j) in [(0, 1), (0, 2), (1, 2)]:
            K2_lo += 2 * lambdas[i] * lambdas[j] * I_lower[(i, j)]
            K2_hi += 2 * lambdas[i] * lambdas[j] * I_upper[(i, j)]
        K2_mid = (K2_lo + K2_hi) / 2

        print(f"\n  K_2 in [{mp.nstr(K2_lo, 14)}, {mp.nstr(K2_hi, 14)}]")
        print(f"  K_2 mid = {mp.nstr(K2_mid, 14)}")

    print()
    K2_lo_v4 = 4.788823421259034
    K2_hi_v4 = 4.789630363787504
    K2_mid_v4 = 0.5 * (K2_lo_v4 + K2_hi_v4)
    print(f"v4 K_2 enclosure: [{K2_lo_v4}, {K2_hi_v4}]")
    print(f"v4 K_2 midpoint:  {K2_mid_v4}")

    # Compare the LAST T's K2 to v4
    overlap_lo = max(K2_lo, K2_lo_v4)
    overlap_hi = min(K2_hi, K2_hi_v4)
    overlap = overlap_hi > overlap_lo
    print(f"\nv4 enclosure overlaps our enclosure? {overlap}")
    print(f"Overlap: [{overlap_lo}, {overlap_hi}]")


if __name__ == "__main__":
    main()
