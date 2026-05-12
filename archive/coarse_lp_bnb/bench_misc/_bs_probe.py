"""Beurling-Selberg / Carneiro-Vaaler probe for C_{1a}.

Test the chain:
   M * int_{|t|<=1/2} |hat L(t)| dt  >=  int_{|xi|<=1/2} |hat f(xi)|^2 dxi
where L is the Selberg majorant of 1_{|xi|<=1/2} of exp type pi.

Notation: hat g(t) = int g(xi) e^{-2 pi i t xi} dxi.
Selberg's identity: int (L - 1_{|xi|<=1/2}) dxi = 1/delta = 2.
So int L = 3, hence hat L(0) = 3.
By Paley-Wiener / type pi, hat L is supported in [-1/2, 1/2].

We don't actually need the messy entire-function formula for L;
all we need are the Selberg identity (int L = 3) and the Paley-Wiener
support fact.  Lower bound int|hat L| dt >= |hat L(0)| = 3 trivially.
"""
import numpy as np
from scipy.integrate import quad

# ------------------------------------------------------------
# Tail demonstration: nonneg f with supp [-1/4,1/4], int f = 1
# can have tail(A) := int_{|xi|>A} |hat f|^2 unbounded.
# ------------------------------------------------------------
def sincsq(x):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-14
    out[nz] = (np.sin(np.pi*x[nz]) / (np.pi*x[nz]))**2
    return out

# Family f_k = k * 1_{[-1/(2k), 1/(2k)]} for k >= 2 (so support inside [-1/4,1/4]).
# hat f_k(xi) = sinc(xi/k);  R0 = ||f_k||_2^2 = k.   max f_k * f_k = k.
# tail(1/2) = int_{|xi|>1/2} sinc^2(xi/k) dxi -> infinity as k -> infinity
# (since sinc^2(xi/k) -> 1 pointwise).
print("Tail growth for f_k = k * 1_{[-1/(2k),1/(2k)]}:")
print(" k   R0     inner    tail(1/2)   M=max(f*f)=k")
for k in [2, 4, 8, 16, 32]:
    R0, _ = quad(lambda xi: sincsq(xi/k), -2000, 2000, limit=2000, epsabs=1e-6)
    inner, _ = quad(lambda xi: sincsq(xi/k), -0.5, 0.5)
    tail = R0 - inner
    print(f" {k:2d}  {R0:6.3f}  {inner:6.4f}   {tail:7.3f}     {k}")

# Conclusion: tail(1/2) is unbounded above as f varies in our class.

# ------------------------------------------------------------
# Worst-case analysis of the proposed inequality
# ------------------------------------------------------------
# M * int|hat L| >= R(0) - tail
# int|hat L| >= |hat L(0)| = 3 (Selberg).  Best case: int|hat L| = 3 exactly.
# So:        M >= (R(0) - tail) / 3.
# But R(0) - tail = int_{|xi|<=1/2} |hat f|^2 dxi.
#
# For a SCALE-INVARIANT lower bound we need
#    inf_f  int_{|xi|<=1/2} |hat f|^2 dxi  / 1   subject to f>=0, supp[-1/4,1/4], int f = 1.
# But we CAN make int_{|xi|<=1/2} |hat f|^2 -> 0:
#   take f = uniform on a tiny interval [-1/(2k), 1/(2k)] (still inside [-1/4,1/4]),
#   then hat f(xi) = sinc(xi/k), and int_{|xi|<=1/2} sinc^2(xi/k) -> 0 as k grows.
print()
print("inf of int_{|xi|<=1/2} |hat f|^2 over our class:")
for k in [2, 4, 16, 64, 256]:
    inner, _ = quad(lambda xi: sincsq(xi/k), -0.5, 0.5)
    print(f"  k={k:3d}: inner = {inner:.4f}")

# So inf is 0 (achieved in limit).  Hence
#    inf_f  M  / [bound from chain]  is  inf_f M >= 0/3 = 0,  empty.

print()
print("="*64)
print("ANALYTIC SUMMARY")
print("="*64)
print("""
L formula: Selberg majorant L of 1_{|xi|<=1/2} of exp type pi (delta=1/2).
   L(xi) = (1/2)[ B(xi/2 + 1/4) + B(1/4 - xi/2) ]
   where B is Beurling's majorant of sgn (Vaaler 1985 BAMS, eq. 1.5),
       B(z) = ((sin pi z)/pi)^2 [ 2/z + sum_{n!=0} sgn(n)/(z-n)^2 ].
   Selberg identity:  int (L - 1_{|xi|<=1/2}) dxi = 1/delta = 2,
   so int L = 3,  hat L(0) = 3,  supp hat L in [-1/2, 1/2].

The chain  M * int|hat L| >= int_{|xi|<=1/2} |hat f|^2  is correct,
but useless because:

  (i)   int |hat L|(t) dt >= |hat L(0)| = 3     (cannot be smaller).
  (ii)  inf over our f-class of int_{|xi|<=1/2} |hat f|^2  = 0.
        (witness: f = k * 1_{[-1/(2k),1/(2k)]}, sinc-mass leaks to |xi|>>1/2.)

Therefore the chain gives  M >= 0 / 3 = 0,  no lower bound.

Where it's loose:
 (1) Replacing R(t) by M (trivial bound) loses everything except R's L^infty
     mass; but R has fixed L^1 mass = 1 and varying support inside [-1/2,1/2].
 (2) |hat L| has no sign cancellation, so int|hat L| = int hat L = hat L(0) = 3
     in the best case.
 (3) The "tail bound via Plancherel-Polya" cannot exist: f >= 0 supp [-1/4,1/4]
     int f = 1 places NO constraint on int_{|xi|>A} |hat f|^2 for any finite A.
     (Plancherel-Polya is for a fixed entire fn, controlling supremum vs L2 etc.,
      not the high-frequency mass of a class of functions.)

VERDICT: NO  (rigorous LB from this approach: 0 << 1.2802).
""")

print("="*64)
print("WHY MV'S 1.2748 SUCCEEDS WHERE THIS FAILS")
print("="*64)
print("""
MV use a Bochner-positive g (g = h * \\bar h, hat g = |hat h|^2 >= 0).
Then  M * int g = M * |hat g(0)|... no, the right setup is:
   int (f*f)(t) g(t) dt = int |hat f|^2(xi) hat g(xi) dxi
With g >= 0 and hat g >= 0 (Bochner), and hat g (xi) = 0 for |xi| > a:
   M * int g >= int (f*f) g >= positive sum at any point of the support.
The arcsine kernel realizes the optimal MV constant 1.2748.

To beat 1.2748, one needs a kernel that's NOT Bochner-positive but
benefits from extra structure of nonneg f -- e.g. higher-moment
constraints (Mat. & Vinuesa 2010, Cohn-Elkies-style LP).  Beurling-
Selberg majorants are sign-indefinite and do not exploit f >= 0; they
exploit pointwise majoration of an indicator.  In our setup the
indicator 1_{|xi|<=1/2} has no preferred role -- |hat f|^2 has no
mass-concentration property at low frequencies for non-symmetric f.
""")
