"""
Route C: Numerical analysis of White-2022's f_c family for C_{1a} transfer.

White's near-optimizer (sign-allowed L^2 problem):
    f_c(x) = alpha_c * (1/4 - x^2)^{-c}    on [-1/2, 1/2]
    alpha_c = Gamma(2-2c) / Gamma(1-c)^2
    c = 0.4942
gives ||f_c * f_c||_2^2 = 0.5746482 within 4.5e-6 of mu_2^2.

Question: how does f_c (or a rescaled version) behave under L^infty / nonneg
analogues relevant to C_{1a}?

C_{1a} normalization: support [-1/4, 1/4], int f = 1, minimize ||f*f||_infty.

For f_c we have f_c >= 0 on the OPEN interval (-1/2, 1/2) (since c > 0 and the
exponent is non-integer; alpha_c > 0 for c in (0, 1/2)... we'll check), and
int_{-1/2}^{1/2} f_c = 1 by definition (Beta-function identity).

Rescale to [-1/4, 1/4]: F(x) = 2 f_c(2x).
"""

import numpy as np
from math import gamma, pi
from scipy.integrate import quad

C = 0.4942

def alpha(c):
    return gamma(2 - 2*c) / gamma(1 - c)**2

def fc(x, c=C):
    """f_c on [-1/2, 1/2]."""
    return alpha(c) * (0.25 - x*x)**(-c)

# Verify integral = 1
I, err = quad(fc, -0.5, 0.5)
print(f"alpha({C}) = {alpha(C):.6f}")
print(f"int_{{-1/2}}^{{1/2}} f_c = {I:.10f}  (should be 1; err {err:.2e})")

# ||f_c||_p for p in {1, 4/3, 3/2, 2, 4, infinity-effective}
def Lp_norm(f, p, a=-0.5, b=0.5, limit=200):
    if np.isinf(p):
        # f_c is unbounded at +/-1/2; sup is +infty
        return float('inf')
    I, _ = quad(lambda x: abs(f(x))**p, a, b, limit=limit)
    return I**(1.0/p)

print()
for p in [1.0, 4/3, 3/2, 2.0, 3.0, 4.0]:
    n = Lp_norm(fc, p)
    print(f"||f_c||_{p}    = {n:.8f}")

print(f"||f_c||_infty = +infty  (singular at x=+/-1/2; c={C} > 0)")

# Autoconvolution f_c * f_c via Fourier
# Use grid + FFT.  Support of f*f is [-1, 1].
N = 200000
x = np.linspace(-0.5, 0.5, N+1)
dx = x[1] - x[0]
# Cap singularity safely; integrate analytically near endpoints would be better,
# but for diagnostic-level numerics we cap at distance dx/2 from +/-0.5.
f_vals = fc(x)
# Cap edge: replace inf with finite (won't be exact but the bulk integral dominates)
mask = ~np.isfinite(f_vals)
# Use a milder approach: shrink x range slightly
eps = 1e-6
xx = np.linspace(-0.5+eps, 0.5-eps, N+1)
dxx = xx[1] - xx[0]
ff = fc(xx)

# Convolution via FFT, padded
def fft_autoconv(f, dx):
    n = len(f)
    pad = 2*n
    F = np.fft.fft(f, pad)
    g = np.fft.ifft(F*F).real * dx
    # output on grid [-(b-a), (b-a)] step dx, length 2n-1
    return g[:2*n-1]

g = fft_autoconv(ff, dxx)
t = np.arange(2*len(ff)-1) * dxx - (len(ff)-1)*dxx
# So t in [-(1-2eps), (1-2eps)].

print()
print("f_c * f_c diagnostics (numerical, on [-1+, 1-], N={}, eps={}):".format(N, eps))
print(f"  max (f_c * f_c)      = {g.max():.8f}")
print(f"  at t = {t[g.argmax()]:.6f}")
print(f"  int (f_c * f_c)      = {(g*dxx).sum():.8f}  (should be 1)")
gg2 = (g**2 * dxx).sum()
print(f"  ||f_c * f_c||_2^2    = {gg2:.8f}  (White says 0.5746482)")
gg_p = lambda p: ((np.abs(g)**p * dxx).sum())**(1.0/p)
print(f"  ||f_c * f_c||_{{4/3}} = {gg_p(4/3):.8f}")
print(f"  ||f_c * f_c||_{{3/2}} = {gg_p(3/2):.8f}")
print(f"  ||f_c * f_c||_4     = {gg_p(4.0):.8f}")
print(f"  ||f_c * f_c||_infty = {g.max():.8f}")

# Now: rescale to C_{1a} support [-1/4, 1/4].
# If g is on [-1/2,1/2] with int g = 1, define F(x) = 2 g(2x) on [-1/4,1/4].
# Then int F = 1 and (F*F)(t) = 2 (g*g)(2t), supported on [-1/2,1/2].
# So ||F*F||_infty = 2 * ||g*g||_infty.
print()
print("Rescaled to C_{1a} normalization (support [-1/4,1/4]):")
print(f"  ||F*F||_infty = 2 * max(f_c*f_c) = {2*g.max():.6f}")
print()
print("CURRENT BOUNDS:  MV >= 1.2748, CS 2017 (invalidated) was 1.2802, UB 1.5029")
print(f"f_c gives only an UPPER bound on C_{{1a}}: {2*g.max():.6f}")
print("That is LARGER than the known UB 1.5029, so f_c is not even competitive as an UB.")
print()

# Lower-bound transfer:  Holder ||f*f||_infty * ||1||_q >= ||f*f||_p  for 1/p+1/q=1
# Equivalently ||f*f||_infty >= ||f*f||_p / |supp(f*f)|^{1/q}.
# supp(f*f) on [-1/2,1/2] domain has length 2.  On rescaled domain length 1.
# We want a LOWER bound on inf_f ||f*f||_infty over nonneg pdf.

print("Holder relations on [-1/2,1/2] (support of f*f has length 2):")
for p in [4/3, 3/2, 2.0]:
    q = 1.0/(1.0 - 1.0/p)
    sup_len = 2.0
    lb = gg_p(p) / sup_len**(1.0/q)
    print(f"  ||f*f||_p={gg_p(p):.5f}, q={q:.3f}: lb on ||f*f||_inf = {lb:.6f}")
print()
print("After rescaling to [-1/4,1/4] (support of F*F has length 1):")
for p in [4/3, 3/2, 2.0]:
    q = 1.0/(1.0 - 1.0/p)
    sup_len_resc = 1.0
    # ||F*F||_p = 2^{1 - 1/p} * ||g*g||_p  (from F*F(t) = 2 g*g(2t))
    # int |F*F|^p dt = int |2 g*g(2t)|^p dt = 2^p * (1/2) int |g*g|^p ds = 2^{p-1} ||g*g||_p^p
    # so ||F*F||_p = 2^{(p-1)/p} ||g*g||_p = 2^{1 - 1/p} ||g*g||_p
    FFp = 2**(1.0 - 1.0/p) * gg_p(p)
    lb = FFp / sup_len_resc**(1.0/q)
    print(f"  ||F*F||_p={FFp:.5f}, q={q:.3f}: lb on ||F*F||_inf = {lb:.6f}")

print()
print("="*72)
print("Interpretation:")
print("- These are LOWER bounds on ||F*F||_inf evaluated AT f_c (the candidate).")
print("- They do not by themselves give a lower bound on C_{1a} = inf_F ||F*F||_inf.")
print("- For a transferable LB on C_{1a}, we would need a UNIVERSAL lower bound on")
print("  ||F*F||_{4/3} valid for ALL nonneg pdf F on [-1/4,1/4] with int F = 1.")
print()
print("Universal LB on ||F*F||_p for general nonneg pdf F on supp length L:")
print("- ||F*F||_1 = 1 (always, since (int F)^2 = 1).")
print("- By Holder, ||F*F||_1 <= |supp|^{(p-1)/p} ||F*F||_p, so ||F*F||_p >= L^{-(p-1)/p}.")
print("- For L=1, p=4/3:  ||F*F||_{4/3} >= 1.  Then LB on ||F*F||_inf via Holder:")
print("  ||F*F||_inf >= ||F*F||_{4/3} / 1^{1/4} = 1.  Trivial (worse than MV 1.2748).")
print("- For L=1, p=2:  ||F*F||_2 >= 1 (Cauchy-Schwarz reversed: ||F*F||_1=1 and supp len=1).")
print("  Then ||F*F||_inf >= ||F*F||_2^2 / ||F*F||_1 = ||F*F||_2^2 >= 1.  Trivial.")
print()
print("White's mu_2^2 ~ 0.5746 is over sign-allowed f; for nonneg f the relevant")
print("L^2 lower bound is actually >= 1 (since (int F)^2 = 1 = ||F*F||_1 and")
print("|supp(F*F)| = 1 on rescaled domain).  But that already gives ||F*F||_inf >= 1,")
print("which is MUCH weaker than 1.2748.")
