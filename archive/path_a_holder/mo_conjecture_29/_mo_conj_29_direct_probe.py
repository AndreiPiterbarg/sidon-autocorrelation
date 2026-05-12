"""Probe c_*(f) := ||f*f||_2^2 / (||f*f||_inf * ||f*f||_1)
   for several parametric families on [-1/4, 1/4].

We test:
 (A) uniform indicator (M = 2)
 (B) "tent" peak-decreasing f(x) = c*(1 - 4|x|) on [-1/4,1/4]
 (C) arcsine-like f_c(x) = N * (1/4 - x^2)^{-c} clipped (White-style)
 (D) two-bump centered f(x) = a*(1_{[-1/4,-1/4+w]} + 1_{[1/4-w,1/4]})
 (E) MV's 119-cosine extremizer (from delsarte_dual.restricted_holder)
 (F) Indicator on [-r, r] for r in (0, 1/4]
 (G) Half-support 2*1_{[-1/4, 0]} (asymmetric -> we compute || f*f ||_inf)
 (H) Bumped symmetric peaks of varying separation
 (I) Boyer-Li 575-step witness (rescaled to [-1/4,1/4]) for sanity

For each: ||f*f||_1 = (int f)^2 = 1 (we normalize), M = sup f*f, L2 = ||f*f||_2^2.

c_*(f) < log 16 / pi = 0.88254... => f is consistent with MO 2.9.
c_*(f) > 0.88254 => Boyer-Li-type witness.
"""
from __future__ import annotations
import math
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np

C_STAR = math.log(16) / math.pi  # 0.88254240...

def autoconv_grid(f_vals, dx):
    """Discrete approximation of (f*f). f_vals on uniform grid step dx."""
    g = np.convolve(f_vals, f_vals) * dx
    return g

def quantities(f_vals, dx):
    g = autoconv_grid(f_vals, dx)
    ff_inf = float(np.max(g))
    ff_1 = float(np.sum(g) * dx)        # = (sum(f) * dx)^2
    ff_l22 = float(np.sum(g * g) * dx)
    return ff_inf, ff_1, ff_l22, g

def report(name, f_vals, dx, normalize=True):
    if normalize:
        s = float(np.sum(f_vals) * dx)
        f_vals = f_vals / s
    ff_inf, ff_1, ff_l22, g = quantities(f_vals, dx)
    c_emp = ff_l22 / (ff_inf * ff_1)
    margin = c_emp - C_STAR
    flag = " *VIOL*" if margin > 0 else ""
    print(f"  {name:42s}  M={ff_inf:.6f}  L2={ff_l22:.6f}  c_*={c_emp:.6f}  d={margin:+.4f}{flag}")
    return dict(name=name, M=ff_inf, L1=ff_1, L2sq=ff_l22, c_star=c_emp, margin=margin)

def main():
    print("=" * 92)
    print(f"  MO Conj 2.9 probe:  c_*(f) := ||f*f||_2^2 / (||f*f||_inf * ||f*f||_1)")
    print(f"  Threshold c_* = log(16)/pi = {C_STAR:.10f}")
    print("=" * 92)

    # Grid:
    N = 8000
    x = np.linspace(-0.25, 0.25, N+1)
    dx = x[1] - x[0]

    results = []
    print("\n(A) Uniform indicator on [-1/4,1/4] (M=2):")
    f = np.ones_like(x) * 2.0   # 2 on [-1/4,1/4], int = 1
    results.append(report("uniform_indicator", f, dx))

    print("\n(B) Tent / triangular f(x) = 4*(1 - 4|x|):")
    f = 4.0 * np.clip(1 - 4*np.abs(x), 0, None)
    results.append(report("triangle_peak_0", f, dx))

    print("\n(C) arcsine-like f_c(x) ∝ (1/4 - x^2)^{-c} clipped, c in {0.3,0.4,0.45,0.4942,0.55,0.6,0.7}:")
    eps = 1e-3
    for c in [0.30, 0.40, 0.45, 0.4942, 0.55, 0.60, 0.70, 0.80]:
        f = ((0.25 + eps)**2 - x**2 + 0.0)
        f = np.where(np.abs(x) < 0.25, np.maximum(f, 1e-12)**(-c), 0.0)
        results.append(report(f"arcsine_c={c:.4f}", f, dx))

    print("\n(D) Two-bump symmetric, width w in {0.02,0.05,0.10,0.125}:")
    for w in [0.02, 0.05, 0.10, 0.125]:
        f = np.where((np.abs(x) >= 0.25 - w) & (np.abs(x) <= 0.25), 1.0, 0.0)
        results.append(report(f"two_bump_w={w}", f, dx))

    print("\n(E) Indicator on [-r,r] for r in {0.1,0.125,0.15,0.20,0.25}:")
    for r in [0.05, 0.10, 0.125, 0.15, 0.20, 0.25]:
        f = np.where(np.abs(x) <= r, 1.0, 0.0)
        results.append(report(f"indicator_r={r}", f, dx))

    print("\n(F) Asymmetric half-support 2*1_{[-1/4,0]}:")
    f = np.where((x >= -0.25) & (x <= 0.0), 2.0, 0.0)
    results.append(report("half_supp_left", f, dx))

    print("\n(G) Spike trio (asymmetric Sidon style), 3 narrow bumps:")
    for centers in [(-0.20, -0.05, 0.10), (-0.22, -0.10, 0.20), (-0.24, 0.00, 0.24)]:
        f = np.zeros_like(x)
        w = 0.02
        for c0 in centers:
            f = f + np.where(np.abs(x - c0) < w/2, 1.0, 0.0)
        results.append(report(f"3_spike_{centers}", f, dx))

    print("\n(H) Cosine bump f(x) = (1 + cos(4*pi*x))/something:")
    f = (1 + np.cos(4*math.pi*x)) / 0.5  # normalize so int = 1 over [-1/4,1/4]
    results.append(report("cos_bump_1cos", f, dx))

    print("\n(I) Boyer-Li-mimic narrow asymmetric step (concentrated near edge):")
    # Construct a heavily concentrated f near right edge: produces large M.
    f = np.where((x >= 0.24) & (x <= 0.25), 100.0, 0.0)  # tiny support => M is huge
    results.append(report("edge_spike", f, dx))

    # Summary
    print("\n" + "=" * 92)
    print("Sorted by c_* (ascending):")
    print("=" * 92)
    for r in sorted(results, key=lambda r: r['c_star']):
        flag = " *VIOL*" if r['margin'] > 0 else ""
        print(f"  c_*={r['c_star']:.5f}  M={r['M']:8.4f}  {r['name']}{flag}")
    print(f"\nThreshold c_*=log(16)/pi = {C_STAR:.10f}")
    print(f"# violators (c_*(f) > c_*) : {sum(1 for r in results if r['margin'] > 0)}/{len(results)}")
    n_lowM = [r for r in results if r['M'] <= 1.51]
    print(f"\nRESTRICTED (M <= 1.51): {len(n_lowM)} families, max c_* = "
          f"{max((r['c_star'] for r in n_lowM), default=float('nan')):.6f}")
    n_targ = [r for r in results if r['M'] <= 1.378]
    print(f"RESTRICTED (M <= 1.378): {len(n_targ)} families, max c_* = "
          f"{max((r['c_star'] for r in n_targ), default=float('nan')):.6f}")

if __name__ == "__main__":
    main()
