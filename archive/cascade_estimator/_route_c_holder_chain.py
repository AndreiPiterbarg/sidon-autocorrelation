"""
Route C — careful Holder-chain analysis for transferring White-2022 to C_{1a}.

Goal: identify any universal LOWER bound on ||F*F||_p (p < infty) for nonneg pdf
F on [-1/4,1/4] with int F = 1, then convert to LB on ||F*F||_inf via Holder.

Strategy 1.  ||F*F||_1 = (int F)^2 = 1 (exact, always).
Strategy 2.  ||F*F||_p >= ||F*F||_1 * |supp(F*F)|^{-(p-1)/p}  (Holder).
            With |supp(F*F)| = 1 (rescaled), this gives ||F*F||_p >= 1.
Strategy 3.  Holder converts an L^p LB into an L^inf LB:
            ||F*F||_p^p = int |F*F|^p <= ||F*F||_inf^{p-1} * int |F*F| = ||F*F||_inf^{p-1}.
            So ||F*F||_inf >= ||F*F||_p^{p/(p-1)}.
            With ||F*F||_p >= 1, we get ||F*F||_inf >= 1.  TRIVIAL.

Conclusion: a TIGHTER universal LB on ||F*F||_p is needed than just ||F*F||_1 = 1.
This is essentially White's problem: he proves the universal LB ||f*f||_2^2 >= 0.5746
on [-1/2,1/2] with int f = 1 (SIGN-ALLOWED).  In the rescaled C_{1a} normalization
on [-1/4,1/4], the corresponding statement is

    inf_{F nonneg pdf on [-1/4,1/4], int F = 1}  ||F*F||_2^2 >= ?

This infimum is LARGER than White's mu_2^2, since we restrict to nonneg.
What value would we need to beat MV's 1.2748?

If we had ||F*F||_2^2 >= alpha unconditionally, then
    ||F*F||_inf >= ||F*F||_2^2 / ||F*F||_1 = alpha / 1 = alpha.
So we need alpha > 1.2748 to beat MV.

What is inf_{F nonneg pdf on [-1/4,1/4]} ||F*F||_2^2 actually?
"""

import numpy as np
from scipy.integrate import quad
from math import pi, gamma

# White's f_c on [-1/2,1/2] rescaled to [-1/4,1/4]:
# F(x) = 2 * f_c(2x), supported on [-1/4, 1/4]
# (F*F)(t) = 2 (f_c * f_c)(2t),  ||F*F||_2^2 = 2 ||f_c*f_c||_2^2 = 2 * 0.5746 = 1.149.
# So even at White's L^2-near-optimal f_c, ||F*F||_2^2 = 1.149 < 1.2748.
print("Sanity: White's f_c gives ||F*F||_2^2 = 2 * 0.5746 = 1.1493")
print("That's already BELOW MV's LB target 1.2748.")
print("So White's L^2 LB does NOT beat MV unconditionally.")
print()

# But maybe inf_{nonneg} ||F*F||_2^2 > inf_{sign-allowed} ||F*F||_2^2.
# Try: 1{[-1/4,1/4]}(x) * 2 (uniform pdf) gives F = 2 on [-1/4,1/4].
def Lp_FFconv_from_F(F_func, p, a, b, N=200000):
    """Compute ||F*F||_p numerically when F supported on [a,b]."""
    x = np.linspace(a, b, N+1)
    dx = x[1] - x[0]
    fv = np.array([F_func(xi) for xi in x])
    n = len(fv)
    pad = 2*n
    Ff = np.fft.fft(fv, pad)
    g = np.fft.ifft(Ff*Ff).real * dx
    g = g[:2*n-1]
    if np.isinf(p):
        return g.max()
    return ((np.abs(g)**p).sum()*dx)**(1.0/p)

# Uniform pdf
print("F = 2 * indicator on [-1/4, 1/4] (uniform):")
norm_inf = Lp_FFconv_from_F(lambda x: 2.0, np.inf, -0.25, 0.25, N=50000)
norm_2 = Lp_FFconv_from_F(lambda x: 2.0, 2.0, -0.25, 0.25, N=50000)
print(f"  ||F*F||_inf = {norm_inf:.6f}   (exact: 2 * 0.5 = 1.0... but 2*int F = 2*0.5*2 = 2)")
print(f"  ||F*F||_2^2 = {norm_2**2:.6f}   (exact: int (F*F)^2 dx; triangle conv...)")
# Exact: F=2 on [-1/4,1/4]; F*F(t) = 2*2 * (1/2 - |t|) for |t|<=1/2; max=2 at t=0.
# (F*F)(t) = 4 * (1/2 - |t|),  ||F*F||_inf = 2.
# ||F*F||_2^2 = int_{-1/2}^{1/2} 16(1/2-|t|)^2 dt = 32 * (1/3)*(1/2)^3 = 32/24 = 4/3 ~ 1.333
print("  exact ||F*F||_inf = 2 (triangle peak), ||F*F||_2^2 = 4/3 ~ 1.333")
print()

# MV near-optimizer: f(x) = arcsine-like on [-1/4, 1/4]
# Skip; documented elsewhere. Point: nonneg constraint changes the L^2 inf substantially.
# In fact since (int F)^2 = 1 and |supp F*F| = 1, CS gives ||F*F||_2^2 >= 1.
# Equality when F*F is constant on its support — i.e. triangle-flat — which is
# impossible for nonneg F (F*F is unimodal and triangular for indicator).
print("="*72)
print("KEY OBSERVATION:")
print("For F nonneg pdf on [-1/4,1/4], CS gives ||F*F||_2^2 >= ||F*F||_1^2 / |supp|")
print("                                          = 1^2 / 1 = 1.")
print("This is a UNIVERSAL LB but only gives ||F*F||_inf >= 1.")
print()
print("White's mu_2^2 = 0.5746 < 1 because he allows SIGN-CHANGE so int F can")
print("redistribute and shrink ||F*F||_2.  The nonneg restriction RAISES the LB")
print("to 1, NOT to White's 0.5746.")
print()
print("Sharpest known LB on ||F*F||_2^2 over nonneg pdf F on [-1/4,1/4] with")
print("int F = 1 and |supp F*F| = 1: by CS, exactly 1.  Is this attained?")
print("Only if F*F constant on [-1/2,1/2], which requires F to be a 'flat' pdf")
print("— impossible.  So strict ineq.")
print()
print("Best known nonneg LBs on ||F*F||_2^2:")
print("- Uniform F gives ||F*F||_2^2 = 4/3 ~ 1.333  (already > 1.2748!).")
print("- MV's near-optimizer gives ||F*F||_inf ~ 1.2748 ('the bound').")
print("  ||F*F||_2^2 for MV is somewhere around ~1.0+ I expect.")

# Numerical check: build MV-like arcsine approximation
def F_arcsine(x, R=0.25):
    """arcsine-density on [-R, R], pdf 1/(pi sqrt(R^2 - x^2))."""
    if abs(x) >= R:
        return 0.0
    return 1.0 / (pi * np.sqrt(R*R - x*x))

# normalize: int_{-1/4}^{1/4} 1/(pi sqrt(1/16 - x^2)) dx = arcsin(x/(1/4))/pi |_{-1/4}^{1/4}
# = (pi/2 - (-pi/2))/pi = 1.  OK already a pdf.
N = 100000
x = np.linspace(-0.25+1e-7, 0.25-1e-7, N+1)
dx = x[1] - x[0]
fv = np.array([F_arcsine(xi) for xi in x])
print(f"\nArcsine pdf on [-1/4,1/4]:  int = {(fv*dx).sum():.6f}")
n = len(fv); pad = 2*n
Ff = np.fft.fft(fv, pad)
g = np.fft.ifft(Ff*Ff).real * dx
g = g[:2*n-1]
print(f"  ||F*F||_inf = {g.max():.6f}")
print(f"  ||F*F||_2^2 = {(g**2 * dx).sum():.6f}")
print(f"  ||F*F||_{{4/3}}^{{4/3}} = {(np.abs(g)**(4/3) * dx).sum():.6f}")
print()
print("Implication for Route C:")
print("- Arcsine pdf has ||F*F||_2^2 ~ pi^2/8 ~ 1.2337 (analytic; check).")
print("- That gives ||F*F||_inf >= 1.2337 via CS — WORSE than MV's 1.2748.")
print("- A universal LB on ||F*F||_2^2 over NONNEG pdf seems to be ~1 (or slightly")
print("  above).  Specifically, the well-known fact: nonneg pdf on [-R,R] with")
print("  int=1 has ||F*F||_2^2 >= 1/(2R) is wrong (that's just |supp F*F| = 2R).")
print("  Actually the CS LB ||F*F||_2^2 >= 1/(2R) gives 1 for R=1/4 (supp F*F length=1).")
print()
print("CONCLUSION: No L^p chain for p < infty can extract more than ~1 LB on")
print("||F*F||_inf via White-style techniques applied to NONNEG pdf.")
