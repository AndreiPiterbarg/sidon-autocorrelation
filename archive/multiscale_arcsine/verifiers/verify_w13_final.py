"""W13 final pass: get tighter bounds on each cross-Bessel via T=50000."""
import mpmath as mp

mp.mp.dps = 30


def cross_bessel_brute(a, b, T=50000.0):
    a_mp = mp.mpf(a)
    b_mp = mp.mpf(b)
    pi = mp.pi
    def integrand(xi):
        return mp.besselj(0, pi * a_mp * xi)**2 * mp.besselj(0, pi * b_mp * xi)**2
    # Subdivide [0, T] into panels to help quad.
    n_panels = max(int(T / 50), 400)
    edges = [mp.mpf(i) * mp.mpf(T) / n_panels for i in range(n_panels + 1)]
    total = mp.mpf(0)
    for k in range(n_panels):
        total += mp.quad(integrand, [edges[k], edges[k+1]])
    bulk = 2 * total
    # Watson tail bound past T:  |J_0(z)^2| <= 2/(pi z) so product <= 4/(pi^2 ab xi^2)
    # tail [T, inf]^one-sided <= 4 / (pi^2 a b T)
    # times 2 (two-sided): 8 / (pi^2 a b T)
    tail_up = mp.mpf(8) / (pi**2 * a_mp * b_mp * mp.mpf(T))
    return bulk, tail_up, n_panels


def main():
    deltas = [mp.mpf("0.138"), mp.mpf("0.055"), mp.mpf("0.025")]
    lambdas = [mp.mpf("0.85"), mp.mpf("0.10"), mp.mpf("0.05")]

    T = 50000.0
    print("=" * 78)
    print(f"W13 final: tight enclosures (T={T}, dps=30)")
    print("=" * 78)

    pairs_all = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
    enclosures = {}
    for (i, j) in pairs_all:
        a, b = deltas[i], deltas[j]
        bulk, tail_up, n_pan = cross_bessel_brute(a, b, T)
        lo = bulk
        hi = bulk + tail_up
        mid = (lo + hi) / 2
        rad = (hi - lo) / 2
        enclosures[(i, j)] = (lo, hi)
        if i != j:
            enclosures[(j, i)] = (lo, hi)
        print(f"  I(d{i+1}={float(a):.3f}, d{j+1}={float(b):.3f})")
        print(f"     in [{mp.nstr(lo, 14)}, {mp.nstr(hi, 14)}]")
        print(f"     mid +/- = {mp.nstr(mid, 12)} +/- {mp.nstr(rad, 6)}")
        # cross-check
        if i == j:
            bbbg = mp.mpf("0.574695") / a
            print(f"     vs 0.574695/d = {mp.nstr(bbbg, 12)}")
        else:
            false_c = mp.mpf("0.5747") / max(a, b)
            print(f"     vs FALSE 0.5747/max = {mp.nstr(false_c, 8)}  (DIFFERENT)")

    # Reconstruct K_2 with interval arithmetic
    K2_lo = mp.mpf(0)
    K2_hi = mp.mpf(0)
    for i in range(3):
        K2_lo += lambdas[i]**2 * enclosures[(i, i)][0]
        K2_hi += lambdas[i]**2 * enclosures[(i, i)][1]
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        K2_lo += 2 * lambdas[i] * lambdas[j] * enclosures[(i, j)][0]
        K2_hi += 2 * lambdas[i] * lambdas[j] * enclosures[(i, j)][1]

    print()
    print(f"K_2 (reconstructed) in [{mp.nstr(K2_lo, 14)}, {mp.nstr(K2_hi, 14)}]")
    print(f"K_2 mid +/- rad     = {mp.nstr((K2_lo+K2_hi)/2, 14)} +/- {mp.nstr((K2_hi-K2_lo)/2, 6)}")

    K2_lo_v4 = mp.mpf("4.788823421259034")
    K2_hi_v4 = mp.mpf("4.789630363787504")
    print(f"v4 K_2 enclosure    = [{K2_lo_v4}, {K2_hi_v4}]")

    overlap_lo = max(K2_lo, K2_lo_v4)
    overlap_hi = min(K2_hi, K2_hi_v4)
    print(f"\nOverlap exists? {overlap_hi > overlap_lo}")
    print(f"Overlap = [{mp.nstr(overlap_lo, 14)}, {mp.nstr(overlap_hi, 14)}]")

    print()
    if K2_lo <= K2_lo_v4 and K2_hi >= K2_hi_v4:
        print("VERDICT: CONFIRM (mpmath enclosure CONTAINS v4 enclosure)")
    elif overlap_hi > overlap_lo:
        print("VERDICT: CONFIRM (mpmath overlaps v4 -- agreement to many digits)")
    else:
        print("VERDICT: FLAG (NO overlap)")


if __name__ == "__main__":
    main()
