"""Agent V2: Independent mpmath recompute of K_2 = ||K_multi||_2^2.

Cert claims K_2 in arb enclosure [4.758752819398309, 4.762081244807873]
with mid 4.760417 and width 3.328e-3.

K_multi(xi) on R: hat_K(xi) = 0.85*J_0(pi*0.138*xi)^2 + 0.15*J_0(pi*0.045*xi)^2.
K_2 = int_{-inf}^{inf} hat_K(xi)^2 dxi = 2 * int_0^inf hat_K(xi)^2 dxi.

The cert uses MO surrogate 0.5747/(2*delta) for diagonal terms.  The CORRECT
diagonal integral via mpmath is what we will compute.  We expect:
  - HONEST mpmath K_2: somewhat different from cert mid (since cert uses
    surrogate diag).  But the question asks: does mpmath K_2 lie within the
    arb enclosure?

We also compute I(0.138, 0.045) = int_R J_0(pi*0.138*xi)^2 J_0(pi*0.045*xi)^2 dxi.

NOTE on factor-of-2 conventions: The agent-N1 / memory note "5.736" appears
to be the 2-sided integral (int over R).  The mpmath 1-sided is half that.
Memory says I(0.138, 0.045) ~ 5.736 — clarify by computing both 1- and 2-sided.
"""
import mpmath
from mpmath import mp, mpf, mpc, quad, quadosc, besselj, pi

mp.dps = 50

LAM1 = mpf("0.85")
LAM2 = mpf("0.15")
D1 = mpf("0.138")
D2 = mpf("0.045")


def j0sq(x):
    j = besselj(0, x)
    return j * j


def _quad_chunks(f, lo, hi, chunk=200.0):
    nodes = [lo]
    x = lo
    while x < hi:
        x = min(hi, x + chunk)
        nodes.append(x)
    return quad(f, nodes)


def integrand_K2(xi):
    """(hat_K(xi))^2 with hat_K = lam1*J0(pi*d1*xi)^2 + lam2*J0(pi*d2*xi)^2."""
    a = LAM1 * j0sq(pi * D1 * xi)
    b = LAM2 * j0sq(pi * D2 * xi)
    s = a + b
    return s * s


def cross_integrand(xi):
    """J_0(pi*d1*xi)^2 * J_0(pi*d2*xi)^2."""
    return j0sq(pi * D1 * xi) * j0sq(pi * D2 * xi)


def diag_integrand(xi, d):
    j = besselj(0, pi * d * xi)
    return j * j * j * j  # J_0^4


# ----------------------------------------------------------------------------
# Honest mpmath K_2 via quadosc (preferred for J_0 oscillations)
# ----------------------------------------------------------------------------
print("=" * 76)
print("Agent V2: mpmath recompute of K_2 = ||K_multi||_2^2 at dps =", mp.dps)
print("=" * 76)
print(f"  delta_1 = {D1}, lambda_1 = {LAM1}")
print(f"  delta_2 = {D2}, lambda_2 = {LAM2}")
print()

# Strategy: 2 * integral over [0, T] + rigorous tail bound.
# Use quadosc with period = 2 (J_0(z)^2 oscillates with period pi -> in xi
# the smaller scale is pi*d1*xi with z stepping by pi every ~ 1/d1 in xi).
# Just use plain quad on [0, T] splitting at moderate points.

print("Phase 1: compute 2 * int_0^T (hat_K)^2 dxi for T = 1000 and T = 10000")
print()


def K2_finite(T):
    # Split into [0, 50] (transient) and [50, T] (oscillatory tail).
    # Subdivide [50, T] into chunks of length ~ period to help quad.
    val1 = quad(integrand_K2, [0, 50])
    period = float(mpf(2) / D1)  # ~14.49 in xi -- largest J_0^2 oscillation
    # Use coarser subdivision: chunks of length 100 are fine; quad handles it.
    nodes = [50]
    x = 50
    chunk = 200.0
    while x < T:
        x = min(T, x + chunk)
        nodes.append(x)
    val2 = quad(integrand_K2, nodes)
    return 2 * (val1 + val2)


for T in [1000, 10000]:
    val = K2_finite(T)
    # Tail bound past T:
    # |J_0(z)|^2 <= 2/(pi z) for z >= z_0 ~ 1, so
    #   hat_K(xi) <= lam1 * 2/(pi^2 d1 xi) + lam2 * 2/(pi^2 d2 xi)
    #             = (2/(pi^2 xi)) * (lam1/d1 + lam2/d2)
    # tail past T (2-sided, so doubled):
    #   2 * int_T^inf [(2/(pi^2 xi))*(lam1/d1 + lam2/d2)]^2 dxi
    # = 2 * (2/(pi^2))^2 * (lam1/d1 + lam2/d2)^2 * int_T^inf dxi/xi^2
    # = 2 * 4/pi^4 * (lam1/d1 + lam2/d2)^2 * (1/T)
    c = LAM1 / D1 + LAM2 / D2
    tail_bound = mpf(8) / (pi ** 4) * c * c / mpf(T)
    print(f"  T = {T:6d}:  2*int_0^T = {val}")
    print(f"             tail bound <= {tail_bound}  ({float(tail_bound):.3e})")
    print(f"             K_2 in     [{val}, {val + tail_bound}]")
    print()

# ----------------------------------------------------------------------------
# Cross-integral I(0.138, 0.045) = int_R J0^2 J0^2 dxi (= 2 * 1-sided)
# ----------------------------------------------------------------------------
print("=" * 76)
print("Phase 2: Cross-integral I(d1, d2) (2-sided over R)")
print("=" * 76)
val1 = quad(cross_integrand, [0, 50])
period = mpf(2) / D1
val2_T = _quad_chunks(cross_integrand, 50, 10000)
I12_one_sided = val1 + val2_T
# tail past 10000:
T = mpf(10000)
tail = mpf(4) / (pi ** 2 * D1 * D2 * T)
print(f"  I_12 (1-sided, T=10000)  = {I12_one_sided}")
print(f"   + tail bound <= {tail} ({float(tail):.3e})")
print(f"  I_12 (2-sided over R)    = {2 * I12_one_sided}")
print(f"  cert's expected ~5.736 (2-sided)? {float(2*I12_one_sided):.4f}")
print()

# ----------------------------------------------------------------------------
# Diagonal integrals I_11 = int_0^inf J_0(pi*d1*xi)^4 dxi
# ----------------------------------------------------------------------------
print("=" * 76)
print("Phase 3: Diagonal integrals I_ii (1-sided)")
print("=" * 76)
for label, d in [("I_11", D1), ("I_22", D2)]:
    v1 = quad(lambda x: diag_integrand(x, d), [0, 50])
    per = mpf(2) / d
    v2 = _quad_chunks(lambda x: diag_integrand(x, d), 50, 10000)
    Ii = v1 + v2
    # surrogate (MO):  0.5747 / (2*d)
    surrogate = mpf("0.5747") / (2 * d)
    print(f"  {label} (1-sided)  = {Ii}")
    print(f"    2-sided          = {2 * Ii}")
    print(f"    surrogate 0.5747/(2*delta) = {surrogate}")
    print(f"    surrogate * 2 (2-sided)    = {2 * surrogate}")
    print(f"    relative diff (1-sided): {float((Ii - surrogate)/Ii):.4e}")
    print()

# ----------------------------------------------------------------------------
# K_2 decomposition cross-check (1-sided convention)
# K_2 = 2 * (lam1^2 I_11 + 2 lam1 lam2 I_12 + lam2^2 I_22)
# ----------------------------------------------------------------------------
print("=" * 76)
print("Phase 4: Decomposition K_2 = 2*(lam1^2 I_11 + 2 lam1 lam2 I_12 + lam2^2 I_22)")
print("=" * 76)
v1 = quad(lambda x: diag_integrand(x, D1), [0, 50]) + _quad_chunks(
    lambda x: diag_integrand(x, D1), 50, 10000
)
v2 = quad(lambda x: diag_integrand(x, D2), [0, 50]) + _quad_chunks(
    lambda x: diag_integrand(x, D2), 50, 10000
)
v12 = quad(cross_integrand, [0, 50]) + _quad_chunks(
    cross_integrand, 50, 10000
)
print(f"  I_11 (1-sided) = {v1}     [memory says ~4.164]")
print(f"  I_22 (1-sided) = {v2}     [memory says ~12.771]")
print(f"  I_12 (1-sided) = {v12}    [memory says 5.736 if 2-sided]")
print()
K2_decomp = 2 * (LAM1 ** 2 * v1 + 2 * LAM1 * LAM2 * v12 + LAM2 ** 2 * v2)
print(f"  K_2 (decomposition, 1-sided diag) = {K2_decomp}")
print()

# Cert claims K_2 in [4.758753, 4.762082], width 3.33e-3.
LOWER = mpf("4.758752819398309")
UPPER = mpf("4.762081244807873")
print("=" * 76)
print("Phase 5: Verification vs cert enclosure")
print("=" * 76)
print(f"  Cert enclosure:        [{LOWER}, {UPPER}]")
print(f"  Cert mid:              {(LOWER + UPPER) / 2}")
print(f"  Cert width:            {UPPER - LOWER}")
print()
K2_honest = K2_finite(10000)
print(f"  mpmath honest K_2 (T=10000): {K2_honest}")
print(f"  In enclosure? {LOWER <= K2_honest <= UPPER}")
print(f"  Decomp K_2:                  {K2_decomp}")
print(f"  Decomp in enclosure?         {LOWER <= K2_decomp <= UPPER}")
print()
# Reconstruct cert K_2 with MO surrogate for diag (what the cert actually claims)
surrogate1 = mpf("0.5747") / (2 * D1)  # I_11 surrogate
surrogate2 = mpf("0.5747") / (2 * D2)  # I_22 surrogate
K2_cert = 2 * (LAM1 ** 2 * surrogate1 + 2 * LAM1 * LAM2 * v12 + LAM2 ** 2 * surrogate2)
print(f"  cert-style K_2 (MO surrogate diag): {K2_cert}")
print(f"  In enclosure? {LOWER <= K2_cert <= UPPER}")
