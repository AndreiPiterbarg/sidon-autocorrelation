"""Dense (M, c) scan for f's near the Sidon-extremizer regime.

Strategy: build interpolations between known low-M f's (BL witness, MV
near-extremizer rescaled to a true pdf, uniform pdf, scaled triangle) and
record the (M, c) trace.  Also: try to drive M *below* C_{1a} (impossible,
since C_{1a} = 1.2802 is a rigorous lower bound) -- if the descent finds
anything below this, it's a numerical artifact, but the trajectory tells us
about the c-behavior at the extremum.
"""
from __future__ import annotations
import numpy as np
import re
from numpy.fft import rfft, irfft

D = 1024
dx = 0.5 / D
xs = -0.25 + dx * (np.arange(D) + 0.5)
PAD = 2 * D - 1
PAD2 = 1 << (PAD - 1).bit_length()


def autoconv(p):
    f = p / dx
    F = rfft(f, n=PAD2)
    g = irfft(F * F, n=PAD2)[:PAD] * dx
    return g


def Mc(p):
    g = autoconv(p)
    M = g.max()
    return M, (g * g).sum() * dx / M


def make_BL():
    with open("delsarte_dual/restricted_holder/coeffBL.txt") as fh:
        nums = [int(x) for x in re.findall(r"\d+", fh.read())]
    v = np.array(nums[:575], dtype=np.float64)
    # Map BL's 575 cells onto our D-bin grid: bin i (centered at xs[i]) gets
    # the linear interpolation of the BL piecewise-constant function.
    # bin index in BL grid:
    BL_x = -0.25 + (np.arange(575) + 0.5) / 1150.0  # centers of BL bins
    BL_v = v / v.sum() * 575.0  # density (so sum = 575)
    # piecewise-constant interpolation:
    p = np.zeros(D)
    for i, x in enumerate(xs):
        idx = int((x + 0.25) * 1150)
        idx = max(0, min(574, idx))
        p[i] = BL_v[idx]
    p /= p.sum()
    return p


def make_uniform():
    return np.ones(D) / D


def make_triangle():
    """Symmetric tent on [-0.25, 0.25]."""
    p = (1 - np.abs(xs / 0.25))
    p = np.maximum(p, 0)
    return p / p.sum()


def make_MV():
    """MV's G(x) = sum a_j cos(2 pi j x / u) on [-1/4, 1/4], normalized."""
    coeffs = [
        +2.16620392, -1.87775750, +1.05828868, -0.729790538,
        +0.428008515, +0.217832838, -0.270415201, +0.0272834790,
        -0.191721888, +0.0551862060, +0.321662512, -0.164478392,
        +0.0395478603, -0.205402785, -0.0133758316, +0.231873221,
        -0.0437967118, +0.0612456374, -0.157361919, -0.0778036253,
        +0.138714392, -0.000145201483, +0.0916539824, -0.0834020840,
    ]
    u = 0.638
    f = np.zeros_like(xs)
    for j, a in enumerate(coeffs, start=1):
        f += a * np.cos(2 * np.pi * j * xs / u)
    f = np.maximum(f, 0)
    return f / f.sum()


def main():
    print(f"D = {D}, dx = {dx:.5e}")
    p_BL = make_BL()
    p_U = make_uniform()
    p_T = make_triangle()
    p_MV = make_MV()
    print(f"\nReference points:")
    print(f"  BL   : M = {Mc(p_BL)[0]:.4f}  c = {Mc(p_BL)[1]:.4f}")
    print(f"  Unif : M = {Mc(p_U)[0]:.4f}   c = {Mc(p_U)[1]:.4f}")
    print(f"  Tri  : M = {Mc(p_T)[0]:.4f}   c = {Mc(p_T)[1]:.4f}")
    print(f"  MV24 : M = {Mc(p_MV)[0]:.4f}   c = {Mc(p_MV)[1]:.4f}")

    # ---- Homotopy through pairs ----
    print("\n--- Homotopies through reference pairs ---")
    pairs = [("Unif", p_U, "BL", p_BL),
             ("Tri", p_T, "BL", p_BL),
             ("MV24", p_MV, "BL", p_BL),
             ("Unif", p_U, "MV24", p_MV)]
    for n1, p1, n2, p2 in pairs:
        print(f"\n{n1} -> {n2}:")
        print(f"{'alpha':>5s}  {'M':>7s}  {'c':>7s}  {'flag':>5s}")
        best_M, best_c = 999, 0
        for alpha in np.linspace(0, 1, 41):
            p = (1 - alpha) * p1 + alpha * p2
            p /= p.sum()
            M, c = Mc(p)
            flag = "*" if M < 1.378 else (" " if M < 1.65 else "")
            if M < best_M:
                best_M, best_c = M, c
            if alpha in (0.0, 0.25, 0.5, 0.75, 1.0) or M < 1.378:
                print(f"{alpha:>5.2f}  {M:>7.4f}  {c:>7.4f}  {flag:>5s}")
        print(f"  min M along path = {best_M:.4f} at c = {best_c:.4f}")

    # ---- 2D simplex of (Unif, BL, MV24) ----
    print("\n--- Convex combinations a*Unif + b*BL + c*MV24 ---")
    print(f"{'a':>5s} {'b':>5s} {'c':>5s}  {'M':>7s}  {'c':>7s}  {'note'}")
    sub_M = []
    for a in np.arange(0, 1.01, 0.1):
        for b in np.arange(0, 1.01 - a, 0.1):
            ccoef = 1 - a - b
            p = a * p_U + b * p_BL + ccoef * p_MV
            if p.min() < 0:
                continue
            p /= p.sum()
            M, c = Mc(p)
            if M < 1.40:
                sub_M.append((a, b, ccoef, M, c))
            if M < 1.50 or (a, b) in [(0, 0), (0, 1), (1, 0)]:
                print(f"{a:>5.2f} {b:>5.2f} {ccoef:>5.2f}  {M:>7.4f}  {c:>7.4f}")
    print(f"Convex combos with M < 1.40: {len(sub_M)}")
    for a, b, c_, M, cc in sub_M:
        print(f"  a={a} b={b} c={c_:.2f}  M={M:.4f}  c={cc:.4f}")

    # ---- Asymmetric BL variant: drop one half ----
    print("\n--- Asymmetric BL variants ---")
    print(f"{'modification':<30s}  {'M':>8s}  {'c':>8s}")
    for fac in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        p = p_BL.copy()
        p[D//2:] *= fac
        if p.sum() == 0:
            continue
        p /= p.sum()
        M, c = Mc(p)
        print(f"BL[right]*{fac:.1f}                     {M:>8.4f}  {c:>8.4f}")

    # ---- Symmetric vs asymmetric BL fold ----
    p_BL_sym = (p_BL + p_BL[::-1]) / 2
    M_s, c_s = Mc(p_BL_sym)
    print(f"\n(BL + BL_reversed)/2 (symmetrized):  M = {M_s:.4f}, c = {c_s:.4f}")
    print(f"BL itself (verify):                  M = {Mc(p_BL)[0]:.4f}, "
          f"c = {Mc(p_BL)[1]:.4f}")

    # ---- BL stretched/compressed (still on [-1/4, 1/4]) ----
    print("\n--- BL stretched: f_a(x) = (1/a) f(x/a)*indicator ---")
    print(f"{'stretch':>8s}  {'M':>8s}  {'c':>8s}")
    for a in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        # Re-sample BL with effective bin width changed.
        # Instead, just take p_BL on a sub-window.
        if a < 1:
            # compress: set p outside [-a/4, a/4] to 0
            p = p_BL.copy()
            mask = np.abs(xs) > 0.25 * a
            p[mask] = 0
            if p.sum() == 0:
                continue
            p /= p.sum()
        else:
            p = p_BL.copy() / p_BL.sum()
        M, c = Mc(p)
        print(f"{a:>8.2f}  {M:>8.4f}  {c:>8.4f}")

    # ---- Very specific question: what's the (M, c) graph as we move along
    # the line BL_sym -> uniform?  This is the symmetric counterpart that
    # the conditional theorem most directly applies to.
    print("\n--- (M, c) along BL_sym -> Uniform ---")
    print(f"{'alpha':>5s}  {'M':>7s}  {'c':>7s}  {'flag'}")
    for alpha in np.linspace(0, 1, 41):
        p = alpha * p_U + (1 - alpha) * p_BL_sym
        M, c = Mc(p)
        flag = "M<1.378" if M < 1.378 else ""
        print(f"{alpha:>5.2f}  {M:>7.4f}  {c:>7.4f}  {flag}")


if __name__ == "__main__":
    main()
