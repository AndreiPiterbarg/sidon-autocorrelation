"""
Phase-rotation gap probe v2.

Sharper investigation. Key observation:

    K - M = R_f(0) - max_t (f*f)(t)
         <= R_f(0) - Re(f*f)(0)             (M >= Re(f*f)(0) since (f*f)(0) = ||f||_2^2 = K is real and >=0... wait)

Actually (f*f)(0) = int f(x) f(-x) dx = int f(x) f(-x) dx. NOT K = int f(x)^2 dx.

So we should NOT use t=0. Instead:
   M = max_t (f*f)(t),  achieved at some t*.
   K = R_f(0) = (f * f_tilde)(0) = int f^2.

The difference R_f(t) - (f*f)(t) at the SAME t equals
   R_f(t) - (f*f)(t) = int f(x)[f(x-t) - f(t-x)] dx
                    = int f(x)[f(x-t) - f(t-x)] dx.

For symmetric f (f(-x)=f(x)): f(t-x) = f(x-t), so this is 0 for all t. Then M = K.
For asymmetric: nonzero.

And:
   max_t (f*f)(t) >= (f*f)(t*)
   max_t R_f(t)   = R_f(0) = K.
   K - M = R_f(0) - (f*f)(t*) where t* = argmax (f*f)
        <= R_f(0) - (f*f)(0)  (NO: depends on direction; (f*f)(0) might be small)

Better:  K - M <= sup_t |R_f(t) - (f*f)(t)|  +  (R_f(0) - max_t R_f(t)) [=0]
                = sup_t |R_f(t) - (f*f)(t)|.
But that's not quite right either: K - M = R_f(0) - max_t (f*f)(t)
       R_f(0) - max_t (f*f)(t) <= R_f(0) - (f*f)(t*)
                                = [R_f(t*) - (f*f)(t*)] + [R_f(0) - R_f(t*)]
                                <= sup_t |R_f - f*f|  +  (R_f(0) - max_t R_f(t))   no, not max
                                <= sup_t |R_f(t) - (f*f)(t)| + (K - R_f(t*))
                                <= sup_t |R_f - f*f| + 2 K  ... too loose

The correct decomposition is:
       K - M = R_f(0) - max_t (f*f)(t)
             >= R_f(0) - (f*f)(t)  for any t
             so for any t:  R_f(0) - (f*f)(t) >= K - M.

       Take t = 0:  R_f(0) - (f*f)(0) = K - (f*f)(0) >= K - M.

That gives M >= (f*f)(0). And (f*f)(0) = int f(x) f(-x) dx (a kind of "self-overlap").
For symmetric f: (f*f)(0) = K. So M >= K is tight there.
For very asymmetric f: (f*f)(0) can be small.

The right object: D(f) := sup_t [R_f(t) - (f*f)(t)] (with sign).
Then K - M  = max_t R_f(t) - max_t (f*f)(t).
            <= max_t [R_f(t) - (f*f)(t)] = D(f).
(since max(a) - max(b) <= max(a-b) when a,b achieve max at possibly different points, ALL valid)

Wait is that right?  max(a_i) - max(b_i) ?<= max_i (a_i - b_i)?
Let i* = argmax a, j* = argmax b. Then max a - max b = a(i*) - b(j*).
And max(a-b) >= a(i*) - b(i*).  We have a(i*) - b(j*) ?<= a(i*) - b(i*)?
That requires b(i*) <= b(j*) which is true since j*=argmax b. So:
       max a - max b <= a(i*) - b(i*) <= max(a-b).  YES. So K - M <= D(f).

D(f) = sup_t [R_f(t) - (f*f)(t)] >= 0 (=0 iff symmetric).
D(f) = sup_t [(f * f_tilde)(t) - (f * f)(t)] = sup_t (f * (f_tilde - f))(t).

So we have a CLEAN BOUND:    K - M <= D(f) := ||f * (f_tilde - f)||_infty.

And by Young's inequality:   D(f) <= ||f||_p ||f_tilde - f||_q where 1/p + 1/q = 1.
Take p=q=2:                   D(f) <= ||f||_2 * ||f_tilde - f||_2 = sqrt(K) * sqrt(2K - 2 (f*f_tilde)(... ))
                                    NO: ||f - f_tilde||_2^2 = 2 K - 2 (f, f_tilde) = 2K - 2 (f*f)(0).
                              So D(f) <= sqrt(K) * sqrt(2(K - (f*f)(0))).

Let A := (f*f)(0) = int f(x) f(-x) dx (the "self-flip overlap").
Then K - M <= sqrt(K) * sqrt(2(K - A)) = sqrt(2K(K-A)).

If we further knew K and A in some constrained way, we could bound M >= K - sqrt(2K(K-A)).

Test this numerically.
"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq

L = 2.0
N = 1 << 15
dx = L / N
xs = (np.arange(N) - N // 2) * dx
mask_q = (xs >= -0.25 - 1e-12) & (xs <= 0.25 + 1e-12)
mask_half = (xs >= -0.5 - 1e-12) & (xs <= 0.5 + 1e-12)

def normalize(f):
    f = np.maximum(f, 0.0)
    f = np.where(mask_q, f, 0.0)
    s = f.sum() * dx
    if s <= 0: return None
    return f / s

def conv(f, g):
    F = fft(np.fft.ifftshift(f))
    G = fft(np.fft.ifftshift(g))
    return np.fft.fftshift(np.real(ifft(F * G))) * dx

# parametric library
def f_indicator():
    return normalize(np.where(mask_q, 1.0, 0.0))

def f_skewed_bump(a, b):
    f = np.zeros_like(xs)
    inL = (xs >= -a) & (xs <= 0); inR = (xs >= 0) & (xs <= b)
    if a > 0: f[inL] = 1.0 - (-xs[inL])/a
    if b > 0: f[inR] = 1.0 - xs[inR]/b
    return normalize(f)

def f_two_bump(c1, c2, w1, w2, h1, h2):
    f = h1*np.exp(-((xs-c1)/w1)**2) + h2*np.exp(-((xs-c2)/w2)**2)
    return normalize(f)

def f_skew_normal(loc, scale, alpha):
    z = (xs - loc)/scale
    phi = np.exp(-0.5*z**2)
    Phi = 0.5 * (1 + np.tanh(alpha * z / 1.5))
    return normalize(phi * Phi)

def f_three_atoms(p, q, r, x1, x2, x3, w):
    f = (p*np.exp(-((xs-x1)/w)**2) + q*np.exp(-((xs-x2)/w)**2) + r*np.exp(-((xs-x3)/w)**2))
    return normalize(f)

rng = np.random.default_rng(11)

def f_random_smooth():
    K = rng.integers(3, 12)
    centers = rng.uniform(-0.24, 0.24, size=K)
    widths = rng.uniform(0.005, 0.08, size=K)
    weights = rng.exponential(1.0, size=K)
    f = np.zeros_like(xs)
    for c, ww, w in zip(centers, widths, weights):
        f += w * np.exp(-((xs-c)/ww)**2)
    return normalize(f)

def all_quantities(f):
    f_tilde = f[::-1]
    Rf = conv(f, f_tilde)
    ff = conv(f, f)
    K = float(np.sum(f**2) * dx)
    M = float(np.max(ff[mask_half]))
    A = float((conv(f, f)[N//2]))     # (f*f)(0)
    Asym = float(conv(f, f)[N//2])    # also = (f*f)(0)
    # D(f) = sup_t (R_f - f*f)(t)
    diff = Rf - ff
    Dpos = float(np.max(diff[mask_half]))
    Dabs = float(np.max(np.abs(diff[mask_half])))
    # tightness check
    gap = K - M
    # bound1 = sqrt(2 K (K - A))
    bd_young = float(np.sqrt(max(0.0, 2.0 * K * (K - A))))
    # also exact L1 norm of (f * (f_tilde - f)) approximation = ||f||_1 * ||f_tilde - f||_1
    bd_L1 = float(np.sum(np.abs(f - f[::-1])) * dx) * 1.0  # ||f - f_tilde||_1 since ||f||_1 = 1
    # third candidate: ||f - f_tilde||_2 * ||f||_2  (Young 2,2)
    delta2 = float(np.sum((f - f[::-1])**2) * dx)   # = 2K - 2A
    return dict(K=K, M=M, A=A, gap=gap, Dpos=Dpos, Dabs=Dabs,
                bd_young=bd_young, bd_L1=bd_L1, delta2=delta2)

cases = []
cases.append(("symm_indicator", f_indicator()))
cases.append(("symm_triangle", f_skewed_bump(0.25, 0.25)))
cases.append(("symm_gauss", f_two_bump(0.0, 0.0, 0.08, 0.08, 1.0, 0.0)))
for r in [0.0, 0.02, 0.05, 0.08, 0.12, 0.18, 0.24]:
    cases.append((f"tri_skew{r:.2f}", f_skewed_bump(max(0.005, 0.25-r), 0.25)))
for d in [0.0, 0.05, 0.10, 0.15, 0.20]:
    for hr in [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        cases.append((f"twob_d{d:.2f}_h{hr:.1f}", f_two_bump(-d, d, 0.03, 0.03, 1.0, hr)))
for a in [-3, -1, 0, 1, 2, 4, 8]:
    cases.append((f"sN_a{a}", f_skew_normal(0.0, 0.07, a)))
for trip in [(0.5, 0.3, 0.2, -0.18, 0.0, 0.20),
             (0.4, 0.4, 0.4, -0.20, -0.05, 0.18),
             (0.3, 0.5, 0.2, -0.22, 0.10, 0.22),
             (0.2, 0.4, 0.4, -0.22, 0.05, 0.20)]:
    cases.append((f"3atom_{trip[3]:+.2f}{trip[4]:+.2f}{trip[5]:+.2f}",
                  f_three_atoms(*trip[:3], *trip[3:6], 0.015)))
for k in range(15):
    cases.append((f"rand{k:02d}", f_random_smooth()))
# extreme: single shifted atom (K and M can both be large)
for c in [-0.20, -0.10, 0.0, 0.10, 0.20]:
    cases.append((f"atom_c{c:+.2f}", f_two_bump(c, c, 0.03, 0.03, 1.0, 0.0)))

results = []
for name, f in cases:
    if f is None: continue
    r = all_quantities(f)
    r["name"] = name
    results.append(r)

print(f"# v2: gap = K - M, bound: K - M <= D(f) := sup (R_f - f*f), and Young: D(f) <= sqrt(2K(K-A))")
print(f"# A = (f*f)(0) = int f(x)f(-x)dx,  K = ||f||_2^2")
print()
print(f"{'name':28s} {'K':>8s} {'M':>8s} {'gap':>8s} {'A':>8s} {'D+':>8s} {'sqrt2K(K-A)':>12s} {'gap/D+':>8s} {'D+/Yng':>8s}")
for r in results:
    if r['Dpos'] > 1e-12:
        rg = r['gap'] / r['Dpos']
    else:
        rg = float('nan')
    if r['bd_young'] > 1e-12:
        rdy = r['Dpos'] / r['bd_young']
    else:
        rdy = float('nan')
    print(f"{r['name']:28s} {r['K']:8.3f} {r['M']:8.3f} {r['gap']:8.4f} {r['A']:8.3f} "
          f"{r['Dpos']:8.4f} {r['bd_young']:12.4f} {rg:8.4f} {rdy:8.4f}")

# stats
print()
gaps = np.array([r['gap'] for r in results])
Dposes = np.array([r['Dpos'] for r in results])
yngs = np.array([r['bd_young'] for r in results])

# 1) Verify K - M <= D(f) holds (theorem!)
viol = (gaps - Dposes > 1e-9)
print(f"# Bound 1: K - M <= D(f).  Max excess = {(gaps - Dposes).max():.3e}, violations = {viol.sum()}/{len(results)}")

# 2) Verify D(f) <= Young bound
viol2 = (Dposes - yngs > 1e-9)
print(f"# Bound 2: D(f) <= sqrt(2K(K-A)).  Max excess = {(Dposes - yngs).max():.3e}, violations = {viol2.sum()}/{len(results)}")

# 3) Combined: K - M <= sqrt(2K(K-A))
viol3 = (gaps - yngs > 1e-9)
print(f"# Bound 3: K - M <= sqrt(2K(K-A)).  Max excess = {(gaps - yngs).max():.3e}, violations = {viol3.sum()}/{len(results)}")

# tightness ratios in asym regime
mask = gaps > 1e-3
print(f"# Tightness in asym regime ({mask.sum()} cases):")
print(f"#   gap/D+:  min={(gaps[mask]/Dposes[mask]).min():.3f}, max={(gaps[mask]/Dposes[mask]).max():.3f}, "
      f"mean={(gaps[mask]/Dposes[mask]).mean():.3f}")
print(f"#   gap/Yng: min={(gaps[mask]/yngs[mask]).min():.3f}, max={(gaps[mask]/yngs[mask]).max():.3f}, "
      f"mean={(gaps[mask]/yngs[mask]).mean():.3f}")
