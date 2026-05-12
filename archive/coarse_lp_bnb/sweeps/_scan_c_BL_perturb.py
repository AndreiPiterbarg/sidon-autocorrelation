"""Perturb the Boyer-Li step witness (which has c~0.901, M~1.652) and trace
the (M, c) trajectory as we drive M down toward 1.378 / 1.28.

We use M-descent (smoothed-max gradient + simplex projection) starting from
the BL step function and several variants. We also try mass-redistribution
perturbations: swap a small bit of mass between bins, re-normalize, recompute.

Goal: find an asymmetric f with M < 1.378 (i.e., inside the conditional
bound's restricted class) and c > 0.8825 (which would falsify Hyp_R restricted
to that class). If no such f appears, that's strong numerical evidence that
Hyp_R holds in the restricted class.
"""
from __future__ import annotations
import re
import numpy as np
from numpy.fft import rfft, irfft

D = 575                       # match BL's native bin count
dx = 0.5 / D
PAD = 2 * D - 1
PAD2 = 1 << (PAD - 1).bit_length()
xs = -0.25 + dx * (np.arange(D) + 0.5)


def autoconv(p):
    f = p / dx
    F = rfft(f, n=PAD2)
    g = irfft(F * F, n=PAD2)[:PAD] * dx
    return g


def Mc(p):
    g = autoconv(p)
    M = g.max()
    return M, (g * g).sum() * dx / M


def project_simplex(v):
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u + (1 - css) / np.arange(1, n + 1) > 0)[0][-1]
    theta = (css[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def softmax(x, T):
    a = x / T
    m = a.max()
    e = np.exp(a - m)
    return e / e.sum()


def grad_smax(p, T):
    """Gradient of smooth_max(g; T) w.r.t. p where g = autoconv(p)."""
    f = p / dx
    F = rfft(f, n=PAD2)
    G = F * F
    g = irfft(G, n=PAD2)[:PAD] * dx
    w = softmax(g, T)
    W = rfft(w, n=PAD2)
    Fc = F.conjugate()
    corr = irfft(Fc * W, n=PAD2)[:D] * dx
    return 2.0 * corr / dx, g


def descend_M_traj(p_init, n_iters=1500, T_start=0.05, T_end=0.001, lr=0.03):
    p = project_simplex(p_init.copy())
    Ms, cs = [], []
    for it in range(n_iters):
        T = T_start * (T_end / T_start) ** (it / max(1, n_iters - 1))
        gradp, _ = grad_smax(p, T)
        p = project_simplex(p - lr * gradp)
        if it % 50 == 0 or it == n_iters - 1:
            M, c = Mc(p)
            Ms.append(M); cs.append(c)
    M, c = Mc(p)
    return p, Ms, cs


def load_BL():
    """Load the Boyer-Li witness as a 575-bin probability vector on [-1/4,1/4]."""
    with open("delsarte_dual/restricted_holder/coeffBL.txt") as fh:
        txt = fh.read()
    nums = [int(x) for x in re.findall(r"\d+", txt)]
    v = np.array(nums[:575], dtype=np.float64)
    p = v / v.sum()
    return p


def main():
    print(f"D={D}, dx={dx:.5e}")
    p_BL = load_BL()
    M, c = Mc(p_BL)
    print(f"\nBL witness raw:        M = {M:.6f}, c = {c:.6f}")

    # ---- 1. Pure descent on M starting from BL witness ----
    print("\n--- Trajectory: descent on smoothed-M from BL initialization ---")
    p_f, Ms, cs = descend_M_traj(p_BL, n_iters=2000, T_start=0.03,
                                   T_end=0.0005, lr=0.02)
    print(f"{'iter*50':>8s}  {'M':>8s}  {'c':>8s}")
    for i, (M_i, c_i) in enumerate(zip(Ms, cs)):
        if i % 4 == 0 or i == len(Ms) - 1:
            print(f"{i*50:>8d}  {M_i:>8.4f}  {c_i:>8.4f}")
    M_f, c_f = Mc(p_f)
    print(f"FINAL: M = {M_f:.6f}, c = {c_f:.6f}")
    print(f"Symmetric? max|p[i]-p[D-1-i]| = {np.max(np.abs(p_f-p_f[::-1])):.3e}")

    # ---- 2. Perturb BL by small asymmetric noise + descent ----
    print("\n--- Trajectory: BL + asymmetric noise, then M-descent ---")
    rng = np.random.default_rng(0)
    Best = []
    for trial in range(8):
        noise_amp = 0.5 * rng.uniform()
        noise = noise_amp * rng.normal(size=D)
        p0 = np.maximum(p_BL + noise * p_BL.mean(), 0)
        p0 /= p0.sum()
        p_f, _, _ = descend_M_traj(p0, n_iters=1200, T_start=0.03,
                                     T_end=0.001, lr=0.02)
        M_f, c_f = Mc(p_f)
        sym = np.max(np.abs(p_f - p_f[::-1]))
        Best.append((trial, M_f, c_f, sym))
        print(f"trial {trial}: M={M_f:.4f}  c={c_f:.4f}  asym={sym:.3e}")

    # ---- 3. ASYMMETRIC SEEDS that might find different basins ----
    print("\n--- Asymmetric seeds (left-heavy, right-heavy, skewed) ---")
    seeds = []
    # Left-heavy
    p = np.zeros(D); p[:D//2] = np.linspace(2, 0.5, D//2)
    p[D//2:] = np.linspace(0.5, 0.1, D-D//2); p /= p.sum()
    seeds.append(("left_heavy_linear", p))
    # Right-heavy
    seeds.append(("right_heavy_linear", p[::-1].copy()))
    # Two off-center bumps
    for c1, c2, w in [(-0.18, 0.10, 0.02), (-0.10, 0.20, 0.04),
                       (-0.05, 0.22, 0.03)]:
        p = np.exp(-((xs - c1) / w) ** 2) + 0.7 * np.exp(-((xs - c2) / w) ** 2)
        p /= p.sum()
        seeds.append((f"asym_2g({c1},{c2},{w})", p))
    # BL-tail-cropped (asymmetric truncation)
    for crop in [50, 100, 150]:
        p = p_BL.copy()
        p[:crop] *= 0.1
        p /= p.sum()
        seeds.append((f"BL_left_crop_{crop}", p))
    # BL right-cropped
    for crop in [50, 100, 150]:
        p = p_BL.copy()
        p[-crop:] *= 0.1
        p /= p.sum()
        seeds.append((f"BL_right_crop_{crop}", p))

    print(f"{'seed':<32s}  {'M0':>8s}  {'c0':>8s}  {'M_fin':>8s}  {'c_fin':>8s}  {'asym':>10s}")
    for name, p0 in seeds:
        M0, c0 = Mc(p0)
        p_f, _, _ = descend_M_traj(p0, n_iters=1200, T_start=0.03,
                                     T_end=0.001, lr=0.02)
        M_f, c_f = Mc(p_f)
        asym = np.max(np.abs(p_f - p_f[::-1]))
        print(f"{name:<32s}  {M0:>8.4f}  {c0:>8.4f}  {M_f:>8.4f}  {c_f:>8.4f}  {asym:>10.3e}")

    # ---- 4. CRITICAL: measure c on f's that lie ON the M=1.378 boundary
    # constructed by INTERPOLATION between known low-M and BL points.
    print("\n--- (M, c) along uniform-->BL homotopy ---")
    p_unif = np.ones(D) / D
    print(f"{'alpha':>6s}  {'M':>8s}  {'c':>8s}")
    for alpha in np.linspace(0, 1, 21):
        p = (1 - alpha) * p_unif + alpha * p_BL
        p /= p.sum()
        M, c = Mc(p)
        flag = "*" if M < 1.378 else " "
        print(f"{alpha:>6.3f}  {M:>8.4f}  {c:>8.4f} {flag}")

    # ---- 5. Same homotopy but with asymmetric perturbations.
    print("\n--- (M, c) along uniform-->ASYMMETRIC-BL homotopy ---")
    # Asymmetric BL = BL with the right half scaled down by 0.6
    p_BL_asym = p_BL.copy()
    p_BL_asym[D//2:] *= 0.6
    p_BL_asym /= p_BL_asym.sum()
    M_a, c_a = Mc(p_BL_asym)
    print(f"BL_asym (right scaled 0.6): M={M_a:.4f}, c={c_a:.4f}")
    print(f"{'alpha':>6s}  {'M':>8s}  {'c':>8s}")
    for alpha in np.linspace(0, 1, 21):
        p = (1 - alpha) * p_unif + alpha * p_BL_asym
        p /= p.sum()
        M, c = Mc(p)
        flag = "*" if M < 1.378 else " "
        print(f"{alpha:>6.3f}  {M:>8.4f}  {c:>8.4f} {flag}")

    # ---- 6. Aggressive search: many random asymmetric perturbations of BL
    # constrained to keep M small. Look for high c at small M.
    print("\n--- Random asymmetric perturbations of BL: cap M < 1.5 ---")
    rng = np.random.default_rng(1)
    hits = []
    for trial in range(2000):
        amp = rng.uniform(0.0, 0.6)
        # smooth random asymmetric perturbation
        n_modes = rng.integers(1, 8)
        delta = np.zeros(D)
        for _ in range(n_modes):
            k = rng.uniform(1, 12)
            phi = rng.uniform(0, 2 * np.pi)
            delta += rng.normal() * np.sin(2 * np.pi * k * np.arange(D) / D + phi)
        p = np.maximum(p_BL + amp * delta * p_BL.mean(), 0)
        if p.sum() == 0:
            continue
        p /= p.sum()
        M, c = Mc(p)
        if M < 1.5:
            hits.append((M, c, amp, n_modes, trial))
    hits.sort()
    print(f"Found {len(hits)} perturbed f with M < 1.5")
    print(f"{'M':>8s}  {'c':>8s}  {'amp':>6s}  {'modes':>6s}")
    # Show 10 lowest M and 10 highest c
    print("Lowest M:")
    for M, c, amp, modes, _ in hits[:10]:
        print(f"  M={M:.4f}  c={c:.4f}  amp={amp:.3f}  modes={modes}")
    hits.sort(key=lambda r: -r[1])
    print("Highest c:")
    for M, c, amp, modes, _ in hits[:10]:
        print(f"  M={M:.4f}  c={c:.4f}  amp={amp:.3f}  modes={modes}")
    # Maximum c subject to M < 1.378
    sub = [r for r in hits if r[0] < 1.378]
    print(f"\nWith M < 1.378: {len(sub)} samples")
    for M, c, amp, modes, _ in sub[:10]:
        print(f"  M={M:.4f}  c={c:.4f}  amp={amp:.3f}  modes={modes}")


if __name__ == "__main__":
    main()
