"""Probe eps_margin behaviour — find a c where the kernel's `int(thr+eps)`
truncation could differ from Lean's real-valued strict `>`.

The kernel's threshold:
  thr_real = 2d·c·m² + 2W + n_bins
  thr_kernel = int(thr_real + eps_margin)         (eps_margin = 1e-9·m²)
  kernel fires ⟺ conv[q] > thr_kernel

Lean fires ⟺ pointeval_value > c_target + correction
            ⟺ conv[q] > thr_real    (as integer/real comparison)

Question: can `int(thr_real + eps)` undershoot floor(thr_real)?  That
would let `conv[q] > thr_kernel` fire while `conv[q] ≤ thr_real`, which
is unsound vs Lean.

Since eps > 0, int(thr_real + eps) ≥ int(thr_real) ≥ floor(thr_real).
So thr_kernel ≥ floor(thr_real).  Therefore:
  kernel fires ⟹ conv[q] > thr_kernel ≥ floor(thr_real) ≥ ...

For integer conv[q]:  conv[q] > floor(thr_real)  iff  conv[q] > thr_real
  (when thr_real is not integer; if integer, equality of floors).

If thr_real is exactly integer N:
  thr_kernel = int(N + eps) = N (since eps tiny).
  kernel fires iff conv[q] > N.
  Lean fires iff conv[q] > N.
  AGREE.

If thr_real = N + δ, 0 < δ < 1:
  thr_kernel = int(N + δ + eps) = N (assuming δ + eps < 1)
  kernel fires iff conv[q] > N, i.e. conv[q] ≥ N+1.
  Lean fires iff conv[q] > N + δ, i.e. conv[q] ≥ N+1 (since conv is int).
  AGREE.

But what if eps PUSHES δ + eps ≥ 1?  i.e., 1 - δ ≤ eps?
  Then thr_kernel = N+1.  kernel fires iff conv[q] > N+1.
  Lean fires iff conv[q] ≥ N+1.
  KERNEL UNDERPRUNES (kernel = stricter).
  ⟹ no soundness violation; just a missed prune at the boundary.

So eps_margin can only HURT the kernel (cause it to miss a near-boundary
prune), never make it unsound.

Probe: construct a c where δ = 1 - eps - 1e-15 (just barely under 1).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _M1_bench import prune_P


def show(c, n_half, m, c_target):
    d = 2 * n_half
    arr = np.zeros((1, d), dtype=np.int32)
    arr[0, :] = c
    survived = bool(prune_P(arr, n_half, m, c_target)[0])
    pruned = not survived
    # Compute thr_real and conv[q] for each q.
    print(f"  c={c}, n={n_half}, m={m}, ct={c_target}")
    conv = np.zeros(2 * d - 1, dtype=np.int64)
    for i in range(d):
        ci = int(c[i])
        if ci != 0:
            conv[2 * i] += ci * ci
            for j in range(i + 1, d):
                cj = int(c[j])
                if cj != 0:
                    conv[i + j] += 2 * ci * cj
    prefix = np.zeros(d + 1, dtype=np.int64)
    for i in range(d):
        prefix[i + 1] = prefix[i] + int(c[i])
    for q in range(2 * d - 1):
        i_lo = (q - (d - 1)) if q >= d - 1 else 0
        i_hi = q if q < d else d - 1
        W = int(prefix[i_hi + 1] - prefix[i_lo])
        nb = i_hi - i_lo + 1
        thr_real = 2 * d * c_target * m * m + 2 * W + nb
        eps = 1e-9 * m * m
        thr_kernel = int(thr_real + eps)
        # Lean: real-valued, strict >.  conv[q] > thr_real.
        kernel_q = int(conv[q]) > thr_kernel
        lean_q = int(conv[q]) > thr_real
        flag = "  "
        if kernel_q != lean_q:
            flag = "**"
        print(f"   {flag}q={q}: conv={conv[q]:>6d}, thr_real={thr_real:.6f}, "
              f"thr_kernel={thr_kernel}, kern_fires={kernel_q}, lean_fires={lean_q}")
    print(f"  kernel pruned: {pruned}")


print("=== EPS-MARGIN PROBE ===")
print()
print("Probe 1: pick c, ct such that thr_real ≈ N + 0.999...")
# thr_real = 2d·c·m² + 2W + n_bins.  Pick c=[a, b] with d=2 (n_half=1).
# At q=0: i_lo=0, i_hi=0, W=a, n_bins=1.
#   thr_real = 4·c_target·m² + 2a + 1.
# At q=1: i_lo=0, i_hi=1, W=a+b, n_bins=2.
#   thr_real = 4·c_target·m² + 2(a+b) + 2.
# At q=2: i_lo=1, i_hi=1, W=b, n_bins=1.
#   thr_real = 4·c_target·m² + 2b + 1.
# Use n_half=1, m=10.  4·c_target·100 = 400·c_target.
# Want thr_real fractional, e.g. with c_target = 1.20 + 0.001 = 1.201:
#   400·1.201 = 480.4
# thr_real(q=0) = 480.4 + 2a + 1.  For c=(a,b) with a+b=4nm=40, conv[0]=a².
# Pick a=22 ⟹ conv[0]=484. thr_real = 480.4 + 44 + 1 = 525.4. conv[0]=484 < 525.4 → no fire.
# Pick a=23 ⟹ conv[0]=529. thr_real = 525.4. fires (529>525.4 = real). kernel: int(525.4 + 1e-7) = 525. 529>525 fires. AGREE.
# Now try thr_real that ends in .9999..something.
# c_target = 1.31249975 (4·m²·ct = 524.9999), thr_real(q=0) for c=(a,b), a=...
# Actually, we want thr_real = N+δ with δ close to 1 (approaching N+1).
# thr_real = 525 - eps for some small eps.  Then thr_kernel = 524 (kernel sees N=524).
# But + eps_margin = 1e-9·m² = 1e-7.  524.999... + 1e-7 = 525.0... if 1-δ < 1e-7.
# That requires δ > 1 - 1e-7 = 0.9999999.
# Hard to hit with c_target chosen by formula.  Let's try.
# We want 4·c_target·m² + 2W + 1 = N - 1e-8 for some integer N.
# Choose m=10, c=(20,20) — W(q=0)=20, conv[0]=400.
# thr_real(0) = 400·c_target + 41.  Want = some integer N - 1e-8.
# Pick N=441: c_target = (441 - 41 - 1e-8)/400 = 400/400 - 1e-8/400 = 1.0 - 2.5e-11.
# That's representable as a double: 1.0 - 2.5e-11.
# Then 4·m²·ct = 400·(1 - 2.5e-11) = 400 - 1e-8 = 399.99999999.
# thr_real(0) = 399.99999999 + 41 = 440.99999999.
# eps_margin = 1e-9·100 = 1e-7.
# thr_real + eps = 440.99999999 + 1e-7 ≈ 441.00000... wait, 1e-7 > 1e-8, so it crosses.
# int(441.000...) = 441.
# kernel: conv[0]=400 > 441? No.
# Lean: conv[0]=400 > 440.999...? No.
# Both no fire.  Need conv[0] = 441 to test the boundary.
# Try c=(21,19) — conv[0]=441.
#   With ct = 1.0 - 2.5e-11 (so 4·m²·ct = 399.99999999):
#   thr_real(0) = 399.99999999 + 2·21 + 1 = 399.99999999 + 43 = 442.99999999.
#   conv[0]=441 < 442.999. No fire either way.
# OK let's think differently.  We need thr_real just below conv[q].
# Pick c = [21, 19], n=1, m=10.  conv = [441, 798, 361].  S=40.
# At q=0: W=21, nb=1, thr_real = 400·ct + 43.  Want thr_real just under 441.
#   Want thr_real ∈ (440, 441).  400·ct = thr_real - 43 ∈ (397, 398).
#   ct ∈ (0.9925, 0.995).
# Pick ct = 0.99499999975  →  400·ct = 397.999999900.  thr_real = 440.999999900.
# eps = 1e-7.  thr_real + eps = 441.000000000... 0   → int = 441 (since eps just over 1e-7 may push to 441 or stay at 440 due to float rounding).
# Hmm, 1e-7 vs 1e-10 of the underflow → eps wins.  So thr_kernel = 441.
# Lean: conv=441 > 440.99999990 = TRUE → fires.
# Kernel: 441 > 441 = FALSE → does NOT fire.
# That's a divergence: KERNEL UNDERPRUNES.  Sound but missed prune.
show([21, 19], 1, 10, 0.99499999975)

print()
print("Probe 2: same c, ct such that thr_real = exact integer 441.0")
# 400·ct + 43 = 441  →  ct = 398/400 = 0.995.  Exact.
show([21, 19], 1, 10, 0.995)

print()
print("Probe 3: slightly above 0.995.  Both should NOT fire.")
show([21, 19], 1, 10, 0.99500001)

print()
print("Probe 4: way above for fire.")
show([21, 19], 1, 10, 0.99)
