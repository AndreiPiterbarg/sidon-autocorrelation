"""
L^p Lyapunov chain audit for C_{1a}.

PROBLEM: f >= 0, supp f subset [-1/4, 1/4], int f = 1.
        C_{1a} = inf_f ||f*f||_inf.  Currently 1.2802 <= C_{1a} <= 1.5029.

CHAIN 1 (Hölder, the one in the prompt):
   ||f*f||_inf >= ||f*f||_p^{p/(p-1)}   (since ||f*f||_1 = 1)

CHAIN 2 (Lyapunov interp between p=2 and p=inf, generally tighter):
   ||f*f||_inf >= (||f*f||_p^p / ||f*f||_2^2)^{1/(p-2)},  p > 2.

We need rigorous LOWER bounds on inf_f ||f*f||_p.  What we know:

(A) Trivial:  ||f*f||_p >= 1   (Holder + supp length 1).
(B) p = 2:  inf ||f*f||_2^2 >= 2 mu_2^2 = 2 * 0.574635728 = 1.14927  (White 2022,
            real-valued infimum on [-1/2,1/2], rescaled to [-1/4,1/4]; nonneg
            infimum is >= real infimum).
(C) p in (1, 2): no improvement over trivial known unconditionally.
(D) p > 2: by Lyapunov (h = f*f, ||h||_1 = 1):
       ||h||_p >= ||h||_2^{2(p-1)/p}        (Lyapunov(1,2,p), see derivation)
            -- since ||h||_1 = 1, log||h||_p is convex in 1/p, so
               (1/p) log||h||_p >= (1/2)(1/p) ... actually let's just do it
               from Holder: 1 = ||h||_1 = int h.h^0 with weights;
               ||h||_p^p >= ||h||_2^{2 . ?}.

   Concretely Lyapunov: ||h||_2 <= ||h||_1^{1-theta} ||h||_p^{theta} where
      1/2 = (1-theta) + theta/p  =>  theta = (1/2)/((p-1)/p) = p/(2(p-1)).
   With ||h||_1 = 1, ||h||_2 <= ||h||_p^{p/(2(p-1))}, so
      ||h||_p >= ||h||_2^{2(p-1)/p}.                                     (LP-bound)

(E) p > 2: Holder upper bound on ||h||_p (going OTHER way for inf):
      ||h||_p <= ||h||_2^{2/p} ||h||_inf^{1 - 2/p}
   so   ||h||_inf >= (||h||_p / ||h||_2^{2/p})^{p/(p-2)}.
   At inf side we use the LB for ||h||_p (LP-bound) and the LB ||h||_2^2 >= 1.149:
      ||h||_p >= 1.149^{(p-1)/p}.
   But we'd need an UPPER bound on ||h||_2 to use Lyapunov-inverse, which
   the inf-problem does NOT give us.  So the Lyapunov route (Chain 2) needs
   an a-priori UPPER bound on ||f*f||_2 (e.g. from the BL upper bound 1.5029
   on C_inf and the trivial ||f*f||_2^2 <= ||f*f||_inf), but that destroys
   the inf direction.

The CLEAN unconditional chain is therefore Chain 1 (Holder), with the LB
on ||f*f||_p from (LP-bound) seeded by White's 1.149:

   ||f*f||_inf >= ||f*f||_p^{p/(p-1)} >= [ 1.149^{(p-1)/p} ]^{p/(p-1)} = 1.149.   (!)

   ==>  Chain 1 + Lyapunov LB on ||h||_p collapses to White's p=2 bound, 1.149,
        for ALL p >= 2.

   This is the F8 finding the prompt mentions: every p in the chain just
   gives back 1.149.

UPSHOT: To beat 1.149 via this chain, we need a STRICTLY BETTER LB on
||h||_p (= inf_f ||f*f||_p) than what White's L^2 bound + Lyapunov gives.

For p > 2 there is NO published rigorous lower bound on inf ||f*f||_p that
exceeds the Lyapunov-from-L^2 value 1.149^{(p-1)/p} in the nonneg class;
the standard sup bound c <= 1 (Martin-O'Bryant 2009; recent c >= 0.94136 of
arXiv:2508.02803) goes the WRONG way for our inf chain (it lower-bounds
||h||_inf in terms of ||h||_2^2 with constant 1/c <= 1/.94136 = 1.0623).

We can however COMPUTE inf-style numerical bounds (NOT rigorous) on
||f*f||_p for f a known good candidate (CS extremiser, MV step function,
arcsine kernel) and see what the resulting Chain 1 bound would be.

This script:
 1. Defines several candidate f's.
 2. Computes ||f*f||_p numerically for p in {2, 2.5, 3, 4, 5, 6, 8, 10, 20, inf}.
 3. For each, reports the Chain-1 bound: ||f*f||_inf >= ||f*f||_p^{p/(p-1)}.
 4. Reports also the Lyapunov-from-L^2 best LB ||f*f||_p >= 1.149^{(p-1)/p}
    and the resulting (collapsing-to-1.149) chain bound.
 5. Prints the optimal p for each candidate.
"""

import numpy as np
from scipy.signal import fftconvolve

# ---- Discretization ----
N = 2**15  # samples on [-1/4, 1/4]
h = 0.5 / N  # step
x = -0.25 + h * (0.5 + np.arange(N))  # midpoints

def normalize(f):
    return f / (f.sum() * h)

def conv_norms(f, ps):
    """Return ||f*f||_p for p in ps, plus ||f*f||_inf."""
    g = fftconvolve(f, f) * h          # h-step quadrature
    # supp f*f subset [-1/2, 1/2], length 1.0, on grid of 2N-1 pts spaced h.
    g = np.maximum(g, 0)
    out = {}
    out['inf'] = float(g.max())
    out['1']   = float((g * h).sum())   # should be 1
    for p in ps:
        out[f'{p}'] = float(((g**p).sum() * h) ** (1.0 / p))
    return out, g


# ---- Candidate functions on [-1/4, 1/4] with int f = 1 ----

# (1) Uniform on [-1/4, 1/4]
f_uniform = normalize(np.ones_like(x))

# (2) Triangular peaked at 0
f_triangle = normalize(np.maximum(0, 0.25 - np.abs(x)))

# (3) MV-style arcsine: f(x) = (2/pi) (1/4 - 4 x^2)^{-1/2} on (-1/4, 1/4) ?
# Actually MV's beta(x) = (2/pi)(1 - 4 x^2)^{-1/2} 1_{(-1/2, 1/2)}.
# Rescale to [-1/4, 1/4]: f(x) = 2 beta(2 x) = (4/pi)(1 - 16 x^2)^{-1/2}.
denom_arc = np.maximum(1 - 16 * x * x, 1e-9)
f_arcsine = normalize((4.0 / np.pi) * denom_arc ** (-0.5))

# (4) White's near-optimiser, c = 0.4942, on [-1/2, 1/2] then rescaled.
# f_c(y) = alpha_c (1/4 - y^2)^{-c} on [-1/2, 1/2], int = 1.
# Map to [-1/4, 1/4]: f(x) = 2 f_c(2x).
# alpha_c = Gamma(2-2c)/Gamma(1-c)^2.
from scipy.special import gamma as Gamma
c_white = 0.4942
alpha_c = Gamma(2 - 2*c_white) / Gamma(1 - c_white) ** 2
y_for_c = 2 * x  # in (-1/2, 1/2)
denom_c = np.maximum(0.25 - y_for_c**2, 1e-9)
f_white = normalize(2 * alpha_c * denom_c ** (-c_white))

# (5) Cloninger-Steinerberger known LB extremiser proxy: just a smooth bump
# (we don't have the actual extremiser; use parabolic profile)
f_parabola = normalize(np.maximum(0, 1 - (x / 0.25) ** 2))

candidates = {
    "uniform":   f_uniform,
    "triangle":  f_triangle,
    "arcsine":   f_arcsine,
    "white_c=0.4942": f_white,
    "parabola":  f_parabola,
}

ps = [2, 2.5, 3, 4, 5, 6, 8, 10, 20]

print("=" * 88)
print("L^p chain audit:  C_{1a} = inf_f ||f*f||_inf, f >= 0, supp [-1/4,1/4], int f = 1.")
print("Chain 1 (Holder):  ||f*f||_inf >= ||f*f||_p^{p/(p-1)}  (since ||f*f||_1 = 1).")
print("=" * 88)

print()
print(f"{'candidate':<20}{'||h||_1':>10}{'||h||_inf':>12}{'||h||_2':>10}{'||h||_4':>10}")
print("-" * 62)
for name, f in candidates.items():
    out, _ = conv_norms(f, ps)
    print(f"{name:<20}{out['1']:>10.5f}{out['inf']:>12.5f}{out['2']:>10.5f}{out['4']:>10.5f}")

print()
print("Chain-1 bound  ||f*f||_inf >= (||f*f||_p)^{p/(p-1)}  evaluated at each candidate:")
print("(this is an UPPER bound on what Chain 1 can give us, since we evaluate at one f).")
print()
header = f"{'candidate':<20}" + "".join(f"p={p:<5}" for p in ps) + f"{'||h||_inf':>12}"
print(header)
print("-" * len(header))
for name, f in candidates.items():
    out, _ = conv_norms(f, ps)
    row = f"{name:<20}"
    for p in ps:
        ub = out[f'{p}'] ** (p / (p - 1))
        row += f"{ub:<7.4f}"
    row += f"{out['inf']:>12.5f}"
    print(row)

print()
print("=" * 88)
print("Now apply Chain 1 to the BEST KNOWN RIGOROUS LOWER BOUND on inf ||f*f||_p:")
print("Lyapunov (with ||h||_1 = 1, ||h||_2^2 >= 2 mu_2^2 = 1.14927 from White 2022,")
print("nonneg class) gives  inf_f ||h||_p >= 1.14927^{(p-1)/p}.  Chain 1 then gives")
print("  ||h||_inf >= [1.14927^{(p-1)/p}]^{p/(p-1)} = 1.14927   for ALL p >= 2.")
print("=" * 88)

mu2sq_real = 0.574635728  # White 2022 lower bound, real-valued, on [-1/2, 1/2]
mu2sq_nonneg_quarter = 2 * mu2sq_real  # rescaling: ||f*f||_2^2 on [-1/4,1/4] = 2 . ||tilde f * tilde f||_2^2

print(f"\nWhite 2022:  inf_real ||tilde f * tilde f||_2^2 >= {mu2sq_real:.9f}")
print(f"Rescaled:    inf_real ||f * f||_2^2 (supp [-1/4,1/4], int=1) >= {mu2sq_nonneg_quarter:.9f}")
print(f"NONNEG class: same lower bound applies (nonneg infimum >= real infimum).")
print()
print(f"{'p':>6}{'inf ||h||_p (rigor.)':>22}{'Chain-1 bound on C_{1a}':>30}")
print("-" * 60)
for p in ps:
    lb_p = mu2sq_nonneg_quarter ** ((p - 1) / p)
    chain1 = lb_p ** (p / (p - 1))
    print(f"{p:>6}{lb_p:>22.6f}{chain1:>30.6f}")

print()
print("As predicted, Chain 1 collapses to 1.14927 (= 2 * mu_2^2) for every p.")
print(f"Compare with current LB on C_{{1a}}: 1.2802 (Cloninger-Steinerberger).")
print(f"Gap: 1.2802 - 1.14927 = {1.2802 - mu2sq_nonneg_quarter:.5f}.")

print()
print("=" * 88)
print("CONCLUSION")
print("=" * 88)
print("""
The Lyapunov / Holder chain ||f*f||_inf >= ||f*f||_p^{p/(p-1)} cannot beat
the L^2 bound 1.14927 unless we have a STRICTLY BETTER lower bound on
inf_f ||f*f||_p than what L^2 + Lyapunov gives.  No such bound is known
in the literature for the nonneg class (the only published improvements
are arXiv:2508.02803's c >= 0.94136 for ||h||_2^2 / (||h||_inf ||h||_1),
which is the WRONG direction for our chain).

VERDICT: NO.  The L^p Lyapunov chain alone cannot push C_{1a} above 1.2802;
in fact it is stuck at 1.14927 < 1.2802 for every p in (2, infty).
""")
