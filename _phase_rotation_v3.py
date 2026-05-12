"""
Phase-rotation v3:
We have K - M <= sqrt(2 K (K - A)) where A = (f*f)(0) = int f(x) f(-x) dx.

To push C_{1a} above 1.2802, we want to PROVE  M >= 1.2802 + epsilon  for all admissible f.
The chain gives:
       M >= K - sqrt(2 K (K - A)).
For symmetric f (A = K): M >= K, which is the well-known M >= ||f||_2^2 / 1 (Plancherel-ish).
But ||f||_2^2 has the lower bound  ||f||_2^2 >= ||f||_1^2 / |supp f|  = 1 / (1/2) = 2 (for our normalization).
Wait, supp = [-1/4, 1/4], length 1/2. So ||f||_2^2 >= ||f||_1^2 / |supp| = 1/(1/2) = 2.

So in the symmetric case: M >= K >= 2.  But the actual C_{1a} >= 1.2802, much LESS than 2.
The reason: nothing forces f symmetric.  Asymmetric f's can have M as small as ~1.28.

To push C_{1a} up, we'd need: for ALL f (asym OK), M >= 1.2802 + delta.

Using K - M <= sqrt(2K(K-A)):
   M >= K - sqrt(2K(K-A))
For this lower bound to exceed 1.2802, we need K - sqrt(2K(K-A)) > 1.2802 over all f.
But what's the worst (smallest) value of K - sqrt(2K(K-A)) over valid f?

f arbitrary nonneg, supp [-1/4,1/4], int f = 1.
Let z := K - A = ||f - f_tilde||_2^2 / 2.   (This is the "asymmetry energy" in L2.)
So bound: M >= K - sqrt(2K z) = K * (1 - sqrt(2z/K)).

We want the minimum of K * (1 - sqrt(2z/K)) = K - sqrt(2 K z).
For fixed K, the bound is maximized by z=0 (symmetric) giving M>=K.
For fixed K, bound is MINIMIZED by max possible z. What's the max?
z = K - A. A = (f * f)(0) = int f(x) f(-x) dx >= 0.
So max z = K (achieved when A = 0, i.e. f and f_tilde have disjoint supports).
That gives M >= K - sqrt(2K * K) = K(1 - sqrt(2)) < 0, useless.

So this bound is USELESS in the worst case. The bound K - M <= sqrt(2K(K-A)) cannot beat 1.2802 alone.

BUT: it might be useful combined with OTHER constraints. E.g., if we also know that
A is bounded BELOW in terms of K (which would happen if we have a separate Hölder-type constraint).

Let's compute the BEST possible bound from this chain, assuming we run M and z optimization:
   For each f, the bound predicts M >= K - sqrt(2K(K-A)).
   The conjectured C_{1a} >= 1.2802.  How tight is M_pred to the actual M?

Plot the (M_actual - M_pred) vs gap, and look at C_{1a} candidate from this bound:
  C_{1a}_pred(via this chain) = inf over admissible f of  K - sqrt(2K(K-A)).
For f shifted-atom (f = delta_x narrowed): K -> infty but K-A also -> K, so K - sqrt(2K(K-A)) ~ K - K sqrt(2) -> -infty.

So this bound alone doesn't give a positive C_{1a}. Need additional ingredients.

THE INSIGHT: the bound K - M <= sqrt(2K(K-A)) is ALWAYS true but doesn't directly cap C_{1a}.
It gives us a HANDLE to convert symmetric-only bounds into asymmetric bounds, IF we
can simultaneously control K and (K-A).

Test: under the OPTIMAL near-extremizer for the actual C_{1a} (MV's extremizer with M ~ 1.2748),
what are K and A?  Does the bound K - M <= sqrt(2K(K-A)) become TIGHT or LOOSE there?
"""

import numpy as np
from numpy.fft import fft, ifft, fftfreq

L = 2.0; N = 1 << 16; dx = L / N
xs = (np.arange(N) - N // 2) * dx
mask_q = (xs >= -0.25 - 1e-12) & (xs <= 0.25 + 1e-12)
mask_half = (xs >= -0.5 - 1e-12) & (xs <= 0.5 + 1e-12)

def normalize(f):
    f = np.maximum(f, 0.0); f = np.where(mask_q, f, 0.0)
    s = f.sum() * dx
    return f / s if s > 0 else None

def conv(f, g):
    return np.fft.fftshift(np.real(ifft(fft(np.fft.ifftshift(f)) * fft(np.fft.ifftshift(g))))) * dx

# MV's near-extremizer in cosine series.
# Look for it in the repo
import os
for path in ["delsarte_dual/restricted_holder/coeffMV.txt",
             "delsarte_dual/restricted_holder/coeffMV_119.txt",
             "delsarte_dual/coeffMV.txt"]:
    if os.path.exists(path):
        print(f"# Found MV coefs at {path}")
        break

# fallback: build a Boyer-Li-style witness from the simple parametric forms.
# Actually let's load any coefficients available.
import glob
coef_files = glob.glob("delsarte_dual/**/coeff*.txt", recursive=True)
print(f"# Coefficient files in repo: {coef_files}")

# Build an MV-like density: f(x) = c * (1 + sum a_k cos(2 pi k x / (1/2)))_+
# normalized to integrate to 1 on [-1/4, 1/4]. The cosine series uses period 1/2.
def f_from_cos_coefs(coefs):
    """coefs[k] for k=0,1,...; period of cosine = 1/2 (matches [-1/4, 1/4] support)."""
    f = coefs[0] * np.ones_like(xs)
    for k, ak in enumerate(coefs[1:], start=1):
        f = f + ak * np.cos(2 * np.pi * k * xs / 0.5)  # 4 pi k x
    f = np.where(mask_q, np.maximum(f, 0.0), 0.0)
    s = f.sum() * dx
    return f / s if s > 0 else None

def quants(f):
    f_t = f[::-1]
    Rf = conv(f, f_t); ff = conv(f, f)
    K = float(np.sum(f**2) * dx); M = float(np.max(ff[mask_half]))
    A = float(ff[N//2])
    return K, M, A

# 1) Symmetric MV-style — flat trial
print("\n# === MV-like SYMMETRIC ===")
for trial in [
    [1.0],                        # uniform
    [1.0, 0.5],                   # 1+0.5 cos(4 pi x)
    [1.0, 0.7, 0.3],
    [1.0, 0.4, 0.2, 0.1],
    [1.0, 0.6, 0.3, 0.15, 0.07]
]:
    f = f_from_cos_coefs(trial)
    K, M, A = quants(f)
    z = K - A
    bd = K - np.sqrt(2*K*max(0, z))
    print(f"  coefs={trial}: K={K:.4f}, M={M:.4f}, A={A:.4f}, K-A={z:.4e}, "
          f"M_lb=K-sqrt(2K(K-A))={bd:.4f}")

# 2) load Boyer-Li if present
print("\n# === Try loading BL (Boyer-Li 2025) ===")
for p in ["delsarte_dual/restricted_holder/coeffBL.txt",
          "delsarte_dual/coeffBL.txt"]:
    if os.path.exists(p):
        print(f"# Loading {p}")
        with open(p) as fh:
            txt = fh.read()
        # try to parse
        import re
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
        nums = [float(x) for x in nums]
        print(f"#   parsed {len(nums)} numbers")
        if len(nums) > 5:
            f = f_from_cos_coefs(nums)
            if f is not None:
                K, M, A = quants(f)
                z = K - A
                bd = K - np.sqrt(2*K*max(0, z))
                print(f"  BL: K={K:.4f}, M={M:.4f}, A={A:.4f}, K-M = {K-M:.4e}, "
                      f"sqrt(2K(K-A))={np.sqrt(2*K*z):.4e}, M_lb={bd:.4f}")
        break

# 3) Now: scan in a neighborhood of the symmetric-MV extremizer and see how M and the bound
#    behave when we add asymmetric perturbation.
print("\n# === Asymmetric perturbations of symm. extremizer-like ===")
sym_coefs = [1.0, 0.6, 0.3, 0.15, 0.07]
f_sym = f_from_cos_coefs(sym_coefs)
K0, M0, A0 = quants(f_sym)
print(f"# Base (symm): K={K0:.4f}, M={M0:.4f}")

rng = np.random.default_rng(42)
best_C_lb = float("inf")
for trial in range(80):
    # add antisymmetric perturbation: c * sum b_k sin(2 pi k x / 0.5)
    bs = 0.05 * rng.normal(size=8)
    perturb = sum(bk * np.sin(2 * np.pi * k * xs / 0.5) for k, bk in enumerate(bs, start=1))
    f = np.maximum(f_sym * (1.0 + 0.2 * perturb), 0.0)
    f = np.where(mask_q, f, 0.0)
    s = f.sum() * dx
    if s <= 0: continue
    f = f / s
    K, M, A = quants(f)
    z = K - A
    bd = K - np.sqrt(2*K*max(0, z))
    if M < 1e3:
        if M < best_C_lb: best_C_lb = M
        if trial < 8 or bd > 1.0:
            print(f"  trial {trial:02d}: K={K:.4f}, M={M:.4f}, K-A={z:.4e}, "
                  f"M_lb={bd:.4f}")

print(f"\n# Best (smallest) actual M observed in scan: {best_C_lb:.4f}")
print(f"# If C_{{1a}} ~ 1.2802 then M cannot drop below this; checking we don't observe smaller M")
print(f"# (We only sample low-d cosines, so M stays high; doesn't disprove anything.)")

# 4) Final crucial test: SHIFTED-CONJUGATE structure.
#    For an extremizer-candidate of the actual C_{1a}, what does our bound predict?
#    We can use White's optimum: f close to a 1-bump narrow Gaussian (gives M ~ 1.2802 + ?)
#    Try explicit: f(x) = c * exp(-(x/sigma)^2) constrained.
print("\n# === Single Gaussian sweep ===")
for sigma in [0.05, 0.07, 0.10, 0.12, 0.15]:
    f = normalize(np.exp(-((xs)/sigma)**2))
    K, M, A = quants(f)
    z = K - A
    bd = K - np.sqrt(2*K*max(0, z))
    print(f"  sigma={sigma}: K={K:.4f}, M={M:.4f}, K-A={z:.4e}, M_lb={bd:.4f}")
