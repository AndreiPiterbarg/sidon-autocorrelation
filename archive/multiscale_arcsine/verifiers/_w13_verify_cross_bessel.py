"""W13: Verify the three cross-Bessel pairs for Cascade-130 3-scale config.

3-scale config (from _cohn_elkies_128_v4_results.json, threescale_N200_best):
    deltas = [0.138, 0.055, 0.025]
    lambdas = [0.85, 0.10, 0.05]
    v4 K_2 enclosure: [4.788823421259034, 4.789630363787504]

We need to verify the three cross-Bessel pairs:
    I(0.138, 0.055), I(0.138, 0.025), I(0.055, 0.025)
where I(a, b) = 2 * int_0^inf J_0(pi a xi)^2 J_0(pi b xi)^2 d xi.

Also the diagonals I(d_i, d_i) which by Bailey-Borwein-Broadhurst-Glasser 2008
should equal 0.574695 / d_i (here the "2 *" two-sided convention).

Reconstruction:
    K_2 = sum_i lam_i^2 * I(d_i,d_i)
        + 2 * sum_{i<j} lam_i * lam_j * I(d_i,d_j)
"""
import mpmath as mp

mp.mp.dps = 30

def cross_bessel_two_sided(a, b, T1=500, T2=100000):
    """Compute  2 * int_0^infty J_0(pi a xi)^2 J_0(pi b xi)^2 d xi.

    Strategy:
      - bulk on [0, T1] via quad (J_0 is oscillating but well-behaved)
      - tail [T1, infty) via quadosc with period set by larger of a,b
      - explicit tail bound past T2 reported separately for sanity:
            |J_0(pi a xi)| <= sqrt(2/(pi^2 a xi))  for xi large,
        so the integrand is bounded by (2/(pi^2 a xi)) * (2/(pi^2 b xi))
                                     = 4 / (pi^4 a b xi^2)
        and tail past T2 is at most 4 / (pi^4 a b T2).
    """
    a_mp = mp.mpf(a)
    b_mp = mp.mpf(b)
    pi = mp.pi

    def integrand(xi):
        return mp.besselj(0, pi * a_mp * xi)**2 * mp.besselj(0, pi * b_mp * xi)**2

    # Bulk [0, T1] -- non-oscillatory enough, use quad
    bulk = mp.quad(integrand, [0, T1])
    # Tail [T1, infty) via quadosc with the dominant oscillation period.
    # J_0(z)^2 has dominant period pi in z, so J_0(pi a xi)^2 has period 1/a in xi.
    # Product of two has beats at 1/|a-b| and 1/(a+b); the slower beat dominates.
    period = 1.0 / max(abs(float(a) - float(b)), 1e-12)
    tail_inf = mp.quadosc(integrand, [T1, mp.inf], period=period)
    # Far tail (informational) bound past T2 (not used in numeric value -- quadosc handles it).
    far_tail = mp.mpf(4) / (mp.pi**4 * a_mp * b_mp * T2)
    # one-sided integral
    one_sided = bulk + tail_inf
    # Two-sided
    two_sided = 2 * one_sided
    return two_sided, far_tail, bulk, tail_inf

def main():
    deltas = [mp.mpf("0.138"), mp.mpf("0.055"), mp.mpf("0.025")]
    lambdas = [mp.mpf("0.85"), mp.mpf("0.10"), mp.mpf("0.05")]

    print("=" * 78)
    print("W13: Cross-Bessel verification for Cascade-130 3-scale")
    print("=" * 78)
    print(f"deltas  = {[float(d) for d in deltas]}")
    print(f"lambdas = {[float(l) for l in lambdas]}")
    print()

    # Diagonal I(d_i, d_i) -- closed form 0.574695 / d_i (two-sided)
    # We will verify each diagonal numerically too.
    print("Computing CROSS pairs (oscillatory mpmath integration):")
    print("-" * 78)

    pairs = [
        (0, 1),  # (0.138, 0.055)
        (0, 2),  # (0.138, 0.025)
        (1, 2),  # (0.055, 0.025)
    ]

    I_cross = {}
    for (i, j) in pairs:
        a, b = deltas[i], deltas[j]
        print(f"\nI({float(a):.3f}, {float(b):.3f}) :")
        I, tail_bd, bulk, tail_inf = cross_bessel_two_sided(a, b)
        I_cross[(i, j)] = I
        I_cross[(j, i)] = I
        print(f"  bulk[0,500]    = {mp.nstr(bulk, 12)}")
        print(f"  tail[500,inf]  = {mp.nstr(tail_inf, 12)}")
        print(f"  T=1e5 tail bd  = {mp.nstr(tail_bd, 6)}  (informational)")
        print(f"  I (two-sided)  = {mp.nstr(I, 12)}")
        # Compare to the FALSE conjecture
        false_conj = mp.mpf("0.5747") / max(a, b)
        diff = I - false_conj
        print(f"  FALSE conj 0.5747/max(a,b) = {mp.nstr(false_conj, 8)}  (diff = {mp.nstr(diff, 6)})")

    print()
    print("Computing DIAGONALS via direct integration (no surrogate):")
    print("-" * 78)
    I_diag = {}
    for i in range(3):
        d = deltas[i]
        # Two-sided diagonal:  2 * int_0^inf J_0(pi d xi)^4 dxi
        # Closed form: 0.574695 / d  approximately, BBBG 2008 gives exact rational pi rep.
        # For diagonal we cannot use the |a-b|->0 quadosc; use direct closed form
        # 2 * int_0^inf J_0(pi d xi)^4 dxi = (Gamma(1/4)^2 / (4 pi^{3/2} d))
        # Numerically Gamma(1/4)^2 / (4 pi^{3/2}) = 0.574695...
        # We will trust the closed form here (it's BBBG 2008, rigorously known).
        # But we also verify numerically via a slightly different route:
        # Compute 2 * (bulk[0,T1] + tail[T1, T2] explicitly).
        a_mp = d
        def integrand(xi, am=a_mp):
            return mp.besselj(0, mp.pi * am * xi)**4
        bulk = mp.quad(integrand, [0, 500])
        # For diagonal, quadosc with period 1/d (the J_0^2 period in xi)
        period = 1.0 / float(d)
        tail = mp.quadosc(integrand, [500, mp.inf], period=period)
        I = 2 * (bulk + tail)
        I_diag[i] = I
        # Exact closed form via Gamma function
        bbbg_closed = mp.gamma(mp.mpf("0.25"))**2 / (4 * mp.pi**mp.mpf("1.5") * d)
        bbbg_approx = mp.mpf("0.574695") / d
        print(f"I({float(d):.3f}, {float(d):.3f}):")
        print(f"  I (computed)        = {mp.nstr(I, 12)}")
        print(f"  BBBG closed form    = {mp.nstr(bbbg_closed, 12)}")
        print(f"  0.574695 / d        = {mp.nstr(bbbg_approx, 12)}")
        print(f"  diff (I - BBBG)     = {mp.nstr(I - bbbg_closed, 6)}")
        I_cross[(i, i)] = I

    # Reconstruct K_2
    print()
    print("Reconstruction of K_2:")
    print("-" * 78)
    K2 = mp.mpf(0)
    # diagonals
    for i in range(3):
        term = lambdas[i]**2 * I_cross[(i, i)]
        print(f"  lam_{i+1}^2 * I(d_{i+1},d_{i+1}) = {float(lambdas[i])}^2 * {mp.nstr(I_cross[(i,i)],10)} = {mp.nstr(term, 10)}")
        K2 += term
    # cross terms
    for (i, j) in [(0,1), (0,2), (1,2)]:
        term = 2 * lambdas[i] * lambdas[j] * I_cross[(i, j)]
        print(f"  2 lam_{i+1} lam_{j+1} I(d_{i+1},d_{j+1}) = 2 * {float(lambdas[i])} * {float(lambdas[j])} * {mp.nstr(I_cross[(i,j)],10)} = {mp.nstr(term, 10)}")
        K2 += term

    print()
    print(f"  K_2 (reconstructed) = {mp.nstr(K2, 14)}")
    print()

    # Compare to v4 enclosure
    K2_lo_v4 = 4.788823421259034
    K2_hi_v4 = 4.789630363787504
    K2_mid_v4 = 0.5 * (K2_lo_v4 + K2_hi_v4)
    print(f"  v4 K_2 enclosure = [{K2_lo_v4}, {K2_hi_v4}]  (mid {K2_mid_v4})")
    print(f"  v4 K_2 width     = {K2_hi_v4 - K2_lo_v4:.6e}")
    print(f"  K_2 - v4_mid     = {mp.nstr(K2 - K2_mid_v4, 6)}")
    inside = (K2 > K2_lo_v4 - 1e-6) and (K2 < K2_hi_v4 + 1e-6)
    print(f"  Inside v4 enclosure? {inside}")

    # Pass criterion: 5-digit match
    digits_match = -mp.log10(abs(K2 - K2_mid_v4)) if abs(K2 - K2_mid_v4) > 0 else 100
    print(f"  Matching digits with v4 mid: {float(digits_match):.2f}")
    print()
    print("=" * 78)
    if inside and digits_match >= 5:
        print("VERDICT: CONFIRM")
    else:
        print("VERDICT: FLAG (does not match v4 enclosure)")
    print("=" * 78)

if __name__ == "__main__":
    main()
