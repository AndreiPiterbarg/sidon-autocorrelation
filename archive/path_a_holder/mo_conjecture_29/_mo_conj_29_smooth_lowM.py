"""Push to low M via smoothed densities and explicit MV-style construction.
Goal: find c_*(f) for f in restricted class M <= 1.378 (and M <= 1.51).
"""
from __future__ import annotations
import math, sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np

C_STAR = math.log(16) / math.pi
N = 16000
x = np.linspace(-0.25, 0.25, N+1)
dx = x[1] - x[0]

def qrr(f):
    s = float(np.sum(f) * dx)
    if s <= 0: return None
    f = f / s
    g = np.convolve(f, f) * dx
    Minf = float(np.max(g))
    L1 = float(np.sum(g) * dx)
    L2 = float(np.sum(g*g) * dx)
    return Minf, L1, L2, L2 / (Minf * L1)

def main():
    print(f"Threshold = log(16)/pi = {C_STAR:.6f}")

    # (1) Indicator on [-1/4,1/4] -> M=2.  But what about asymmetric / off-center?
    # Indicators on subintervals: always c_* = 2/3.

    # (2) tent + indicator mixtures
    print("\n--- Symmetric f(x) = a + b*(1/4 - |x|) on [-1/4, 1/4] ---")
    print("  (Linear-on-positive density: peak at 0)")
    for a, b in [(0,4),(0.5,3),(1.0,2),(1.5,1),(2,0),(3,-2)]:
        f = a + b*(0.25 - np.abs(x))
        if np.any(f < 0): continue
        out = qrr(f)
        if out:
            Minf,L1,L2,c = out
            print(f"  a={a:.1f}, b={b:+.1f}:  M={Minf:.5f}  c_*={c:.6f}  margin={c-C_STAR:+.4f}")

    # (3) shifted symmetric f^*: f(x) = c*(1/4 - x^2)^p (smooth Wigner-like)
    print("\n--- f(x) = c*(1/4 - x^2)^p / norm  for p in {0.1, 0.25, 0.5, 1, 2} ---")
    for p in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
        f = np.maximum(0.0625 - x*x, 0)**p
        out = qrr(f)
        if out:
            Minf,L1,L2,c = out
            print(f"  p={p:.2f}:  M={Minf:.5f}  c_*={c:.6f}  margin={c-C_STAR:+.4f}")

    # (4) Centered Gaussian truncated to [-1/4,1/4]
    print("\n--- Gaussian f(x) = exp(-x^2/(2s^2)) truncated to [-1/4,1/4] ---")
    for s in [0.05, 0.08, 0.10, 0.125, 0.15, 0.20]:
        f = np.exp(-x*x / (2*s*s))
        out = qrr(f)
        if out:
            Minf,L1,L2,c = out
            print(f"  s={s:.3f}:  M={Minf:.5f}  c_*={c:.6f}  margin={c-C_STAR:+.4f}")

    # (5) MV-style arcsine kernel: f_arcsine_delta(x) = (1/delta) * eta(x/delta)
    # eta(t) = (2/pi) / sqrt(1-4t^2), supp [-1/2, 1/2]. Embed into [-1/4,1/4]
    # by rescaling delta=0.5: eta(2x)*2 on [-1/4,1/4]; check M.
    print("\n--- MV arcsine kernel eta(2x)*2 on [-1/4,1/4] ---")
    f = np.where(np.abs(2*x) < 1, (2/math.pi)/np.sqrt(np.maximum(1 - 4*x*x, 1e-12)), 0.0)
    out = qrr(f)
    if out:
        Minf,L1,L2,c = out
        print(f"  arcsine:  M={Minf:.5f}  c_*={c:.6f}  margin={c-C_STAR:+.4f}")

    # (6) MV's 119-cosine extremizer (loaded from existing code if available)
    print("\n--- MV's 119-cosine extremizer (from delsarte_dual.restricted_holder) ---")
    try:
        from delsarte_dual.restricted_holder.sidon_extremizer_ratio import holder_ratio
        res = holder_ratio(Nmax=2000, dps=30)
        c = float(res['c_emp']); ffinf = float(res['ff_inf']); ffl2 = float(res['ff_l22'])
        print(f"  c_emp={c:.6f}  ffinf={ffinf:.5f}  ffl22={ffl2:.5f}  margin={c-C_STAR:+.4f}")
    except Exception as e:
        print(f"  Could not load: {e}")

    # (7) Two-peak symmetric (cos(2pi k x) bumps near boundary)
    print("\n--- Two-peak boundary-concentrated (asymmetric loading) ---")
    for d, w in [(0.20, 0.04), (0.22, 0.025), (0.235, 0.012)]:
        f = np.exp(-((x - d)**2)/(2*w*w)) + np.exp(-((x + d)**2)/(2*w*w))
        out = qrr(f)
        if out:
            Minf,L1,L2,c = out
            print(f"  d={d:.3f},w={w:.3f}: M={Minf:.5f}  c_*={c:.6f}  margin={c-C_STAR:+.4f}")

if __name__ == "__main__":
    main()
