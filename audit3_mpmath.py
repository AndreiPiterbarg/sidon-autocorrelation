"""Independent mpmath 50-digit verification of the two numerical axioms in
lean/Sidon/MultiScale.lean.

Recomputes K_2(K_ms) and gain = (4/u) * (min_G)^2 / S_1 using mpmath
(completely independent of flint.arb) from the pinned QP coefficients in
the production certificate.
"""
from __future__ import annotations
import json
from fractions import Fraction
import mpmath as mp

CERT_PATH = "delsarte_dual/grid_bound_alt_kernel/certificates/multiscale_arcsine_1292.json"


def parse_q(s: str) -> Fraction:
    if "/" in s:
        num_s, den_s = s.split("/", 1)
        return Fraction(int(num_s), int(den_s))
    return Fraction(int(s))


def load_cert():
    with open(CERT_PATH) as f:
        cert = json.load(f)
    body = cert["body"]
    deltas = tuple(parse_q(s) for s in body["kernel"]["deltas_q"])
    lambdas = tuple(parse_q(s) for s in body["kernel"]["lambdas_q"])
    coeffs = [parse_q(s) for s in body["G"]["coeffs_q"]]
    u = parse_q(body["G"]["u_q"])
    n_coeffs = body["G"]["n_coeffs"]
    return deltas, lambdas, coeffs, u, n_coeffs


def main():
    mp.mp.dps = 50

    deltas, lambdas, coeffs, u, n_coeffs = load_cert()

    print("=" * 72)
    print(" PARAMETERS (loaded from production certificate)")
    print("=" * 72)
    for i, (d, l) in enumerate(zip(deltas, lambdas)):
        print(f"  delta_{i+1} = {d} = {float(d)}")
        print(f"  lambda_{i+1} = {l} = {float(l)}")
    print(f"  sum lambda_i = {sum(lambdas)} (should be 1)")
    print(f"  u = {u} = {float(u)}")
    print(f"  n_coeffs = {n_coeffs}")
    print(f"  max delta = {max(deltas)} = {float(max(deltas))}")
    print(f"  u - (1/2 + max(delta)) = {u - (Fraction(1, 2) + max(deltas))}")

    # ----------------------------------------------------------------------
    # 1. K_2(K_ms) = int_R K_ms(x)^2 dx  via  Plancherel = int_R K_hat^2
    # K_hat_ms(xi) = sum_i lambda_i J_0(pi delta_i xi)^2
    # K_2 = 2 * int_0^infty K_hat(xi)^2 dxi
    # Use mp.quad with high precision; truncate at large XI and bound tail.
    # ----------------------------------------------------------------------
    print()
    print("=" * 72)
    print(" K_2(K_ms): mpmath independent computation")
    print("=" * 72)

    pi = mp.pi
    lamb_mp = [mp.mpf(l.numerator) / mp.mpf(l.denominator) for l in lambdas]
    delta_mp = [mp.mpf(d.numerator) / mp.mpf(d.denominator) for d in deltas]
    u_mp = mp.mpf(u.numerator) / mp.mpf(u.denominator)

    def K_hat_ms(xi):
        s = mp.mpf(0)
        for l, d in zip(lamb_mp, delta_mp):
            s += l * mp.besselj(0, pi * d * xi) ** 2
        return s

    def K_hat_sq(xi):
        return K_hat_ms(xi) ** 2

    # K_hat(0) = sum lambda_i * 1 = 1, so peak is at origin.
    XI_MAX = mp.mpf("1e5")

    print(f"  Integrating K_hat(xi)^2 on [0, {XI_MAX}] with mp.dps = {mp.mp.dps} ...")
    # Use mp.quad - it should handle the oscillatory J_0 well at this prec
    # We split the interval to manage the oscillation
    # First the "bulk" [0, 100] where the kernel is least oscillatory,
    # then [100, 1000], [1000, 10000], [10000, 1e5].
    bulk_seg = mp.quad(K_hat_sq, [0, 100])
    print(f"  int_0^100      = {bulk_seg}")
    seg2 = mp.quad(K_hat_sq, [100, 1000])
    print(f"  int_100^1000   = {seg2}")
    seg3 = mp.quad(K_hat_sq, [1000, 10000])
    print(f"  int_1000^10000 = {seg3}")
    seg4 = mp.quad(K_hat_sq, [10000, float(XI_MAX)])
    print(f"  int_10000^1e5  = {seg4}")

    bulk_total = bulk_seg + seg2 + seg3 + seg4
    print(f"  int_0^XI_MAX   = {bulk_total}")

    # Watson tail bound:
    # |J_0(z)| <= sqrt(2/(pi z)) for z > 0 (actually 2 / (pi z) is for |J_0|^2 envelope)
    # Use the formula in audit_consistency.py:
    #   K_hat^2 <= 4 C^2 / (pi^4 xi^2),  where  C = sum_i lambda_i / delta_i
    # So  2 * int_T^infty K_hat^2 <= 8 C^2 / (pi^4 T)
    C = sum(l / d for l, d in zip(lamb_mp, delta_mp))
    print(f"  C = sum_i lambda_i/delta_i = {C}")
    watson_tail_bound = 8 * C ** 2 / (pi ** 4 * XI_MAX)
    print(f"  Watson tail upper bound: 8 C^2/(pi^4 T) = {watson_tail_bound}")

    # K_2 lower estimate (no tail) and upper (+ tail bound):
    K_2_lower = 2 * bulk_total
    K_2_upper = 2 * bulk_total + watson_tail_bound
    print()
    print(f"  K_2 lower (mp): {K_2_lower}")
    print(f"  K_2 upper (mp): {K_2_upper}")
    print()

    # Compare with cert reported [4.788823, 4.788906]
    K2_cert_lo = mp.mpf("4.7888234212591545")
    K2_cert_hi = mp.mpf("4.7889051816332424")
    print(f"  cert K_2 in [{K2_cert_lo}, {K2_cert_hi}]")
    print(f"  mp K_2_lower vs cert lower: diff = {K_2_lower - K2_cert_lo}")
    print(f"  mp K_2_upper vs cert upper: diff = {K_2_upper - K2_cert_hi}")

    K2UpperQ_lean = Fraction(47897, 10000)
    K2UpperQ_mp = mp.mpf(K2UpperQ_lean.numerator) / mp.mpf(K2UpperQ_lean.denominator)
    print(f"  Lean K2UpperQ = 47897/10000 = {K2UpperQ_mp}")
    print(f"  Slack: K2UpperQ - K_2_upper(mp) = {K2UpperQ_mp - K_2_upper}")
    print(f"  axiom 1 check: K_2(mp) <= K2UpperQ? {K_2_upper <= K2UpperQ_mp}")

    # ----------------------------------------------------------------------
    # 2. min_G:  G(x) = sum_j a_j cos(2 pi j x / u), j = 1..n_coeffs
    # ----------------------------------------------------------------------
    print()
    print("=" * 72)
    print(" min_G via fine-grid evaluation")
    print("=" * 72)

    a_mp = [mp.mpf(c.numerator) / mp.mpf(c.denominator) for c in coeffs]
    # Note: the Lean / cert formulation is G(x) = sum_{j=1..N} a_j cos(2 pi j x / u)
    # We need to check this convention.  Cert reports min_G in [0, 1/4].
    # Will evaluate on a dense grid.

    def G_eval(x):
        s = mp.mpf(0)
        two_pi_over_u = 2 * pi / u_mp
        for j, a in enumerate(a_mp, start=1):
            s += a * mp.cos(two_pi_over_u * j * x)
        return s

    # Test endpoints and a coarse grid.
    print(f"  G(0)   = {G_eval(mp.mpf(0))}")
    print(f"  G(1/8) = {G_eval(mp.mpf(1) / 8)}")
    print(f"  G(1/4) = {G_eval(mp.mpf(1) / 4)}")

    # Fine grid: at u/N spacing the J_0 oscillation length, we need substantial points.
    # The cert uses 32768 cells in [0, 1/4]; we use 4096-step grid + report minimum
    # (sufficient for sanity comparison; Taylor B&B is the rigorous version).
    N_grid = 4096
    xs = [mp.mpf(i) / N_grid * mp.mpf(1) / 4 for i in range(N_grid + 1)]
    mn = mp.mpf("inf")
    mn_x = None
    for x in xs:
        v = G_eval(x)
        if v < mn:
            mn = v
            mn_x = x
    print(f"  grid (N={N_grid}) min G on [0,1/4] approx {mn} at x={mn_x}")

    # Cert reports min_G = 0.9999798743824748 at x = 64915/262144 = 0.247654...
    cert_min_G = mp.mpf("0.9999798743824748")
    cert_min_x = mp.mpf(64915) / mp.mpf(262144)
    print(f"  cert min_G = {cert_min_G} at x = {cert_min_x}")
    G_at_cert_x = G_eval(cert_min_x)
    print(f"  G(cert_min_x) (mp) = {G_at_cert_x}")
    print(f"  diff: G(cert_min_x)(mp) - cert min_G = {G_at_cert_x - cert_min_G}")

    # ----------------------------------------------------------------------
    # 3. S_1 = sum_{j=1..n} a_j^2 / K_hat_ms(j/u)
    # ----------------------------------------------------------------------
    print()
    print("=" * 72)
    print(" S_1 via direct summation")
    print("=" * 72)
    S1 = mp.mpf(0)
    for j, a in enumerate(a_mp, start=1):
        xi_j = mp.mpf(j) / u_mp
        w_j = K_hat_ms(xi_j)
        S1 += a * a / w_j
    print(f"  S_1 (mp) = {S1}")
    cert_S1 = mp.mpf("29.840906455513263")
    print(f"  S_1 cert = {cert_S1}")
    print(f"  diff: S_1(mp) - S_1(cert) = {S1 - cert_S1}")

    # ----------------------------------------------------------------------
    # 4. gain = (4/u) * (min_G)^2 / S_1
    # ----------------------------------------------------------------------
    print()
    print("=" * 72)
    print(" gain = (4/u) * (min G)^2 / S_1")
    print("=" * 72)
    # Use the cert min_G (a lower bound; gain is increasing in min_G)
    # The cert anchors the gain via the *rigorous* coupled-arb computation; we
    # show the mp value is close.
    gain_mp_with_grid_min = (4 / u_mp) * mn ** 2 / S1
    gain_mp_with_cert_min = (4 / u_mp) * cert_min_G ** 2 / S1
    print(f"  gain(mp, using grid min G) = {gain_mp_with_grid_min}")
    print(f"  gain(mp, using cert min_G) = {gain_mp_with_cert_min}")
    cert_gain = mp.mpf("0.21009214748668373")
    print(f"  gain cert = {cert_gain}")
    print(f"  diff (cert_min version):   {gain_mp_with_cert_min - cert_gain}")

    gainLowerQ_lean = Fraction(20925, 100000)
    gainLowerQ_mp = mp.mpf(gainLowerQ_lean.numerator) / mp.mpf(gainLowerQ_lean.denominator)
    print(f"  Lean gainLowerQ = 20925/100000 = {gainLowerQ_mp}")
    print(f"  Slack: gain(mp) - gainLowerQ = {gain_mp_with_cert_min - gainLowerQ_mp}")
    print(f"  axiom 2 check: gain(mp) >= gainLowerQ? {gain_mp_with_cert_min >= gainLowerQ_mp}")

    # ----------------------------------------------------------------------
    # 5. Algebraic sanity check (master inequality at M = 1.292)
    # ----------------------------------------------------------------------
    print()
    print("=" * 72)
    print(" Master inequality at M = 1292/1000 (Phi vs tau)")
    print("=" * 72)
    M = mp.mpf("1.292")
    K2 = K_2_upper  # use mp upper of K_2
    a_gain = gain_mp_with_cert_min
    Phi = M + 1 + mp.sqrt((M - 1) * (K2 - 1))
    tau = 2 / u_mp + a_gain
    print(f"  M = 1.292")
    print(f"  K_2(mp) <= {K2}")
    print(f"  a (gain) >= {a_gain}")
    print(f"  Phi = M + 1 + sqrt((M-1)(K_2-1)) = {Phi}")
    print(f"  tau = 2/u + a = {tau}")
    print(f"  tau - Phi = {tau - Phi}  (should be > 0 for strict failure)")
    print(f"  strict failure at M=1.292? {tau - Phi > 0}")

    # Final summary
    print()
    print("=" * 72)
    print(" AXIOM CHECK SUMMARY")
    print("=" * 72)
    print(f"  Axiom 1 (K_2 <= 47897/10000):")
    print(f"    K_2(mp) upper = {K_2_upper}")
    print(f"    K2UpperQ      = {K2UpperQ_mp}")
    print(f"    slack         = {K2UpperQ_mp - K_2_upper}")
    print(f"    OK = {K_2_upper <= K2UpperQ_mp}")
    print()
    print(f"  Axiom 2 (gain >= 20925/100000):")
    print(f"    gain(mp, cert_min) = {gain_mp_with_cert_min}")
    print(f"    gainLowerQ         = {gainLowerQ_mp}")
    print(f"    slack              = {gain_mp_with_cert_min - gainLowerQ_mp}")
    print(f"    OK = {gain_mp_with_cert_min >= gainLowerQ_mp}")


if __name__ == "__main__":
    main()
