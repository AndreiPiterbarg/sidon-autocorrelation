"""Find admissible f with LOW M = ||f*f||_inf via gradient descent on the sup,
then read off c(f) = ||f*f||_2^2 / M.

Methodology:
  - Parametrize f as a piecewise-constant nonneg function on [-1/4, 1/4]
    over D bins, normalized to sum = 1.
  - Objective: M = ||f*f||_inf via FFT, smoothed by log-sum-exp at temperature T.
  - Project to simplex after each step.
  - Run from many random/asymmetric initializations.
  - Track the trajectory (M, c) and look for asymmetric f's with small M and
    interesting c-values.
"""
from __future__ import annotations
import numpy as np
from numpy.fft import rfft, irfft

D = 256                            # number of bins on [-1/4, 1/4]
dx = 0.5 / D
PAD = 2 * D - 1
PAD2 = 1 << (PAD - 1).bit_length()


def autoconv(p):
    """p is a probability vector on D bins. Treat f = p/dx (so int f = 1)."""
    f = p / dx
    F = rfft(f, n=PAD2)
    g = irfft(F * F, n=PAD2)[:PAD] * dx
    return g


def Mc(p):
    g = autoconv(p)
    M = g.max()
    ff_l22 = (g * g).sum() * dx
    return M, ff_l22 / M


def project_simplex(v):
    """Project v onto the unit simplex {x>=0, sum x = 1}."""
    n = len(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u + (1 - css) / np.arange(1, n + 1) > 0)[0][-1]
    theta = (css[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def smooth_max_grad(g, T):
    """Compute log-sum-exp(g/T)*T as smoothed max + its gradient w.r.t. g."""
    a = g / T
    m = a.max()
    e = np.exp(a - m)
    Z = e.sum()
    smax = T * (np.log(Z) + m)
    grad = e / Z   # softmax weights, sum = 1
    return smax, grad


def descend_M(p_init, n_iters=400, T_start=0.05, T_end=0.005, lr=0.05,
              record_traj=False):
    """Gradient descent on smoothed max(f*f) over the simplex.

    Returns (p_final, M_final, c_final, traj) where traj = list of (M, c).
    """
    p = p_init.copy()
    p = project_simplex(p)
    traj = []
    for it in range(n_iters):
        T = T_start * (T_end / T_start) ** (it / max(1, n_iters - 1))
        f = p / dx
        F = rfft(f, n=PAD2)
        G = F * F
        g = irfft(G, n=PAD2)[:PAD] * dx
        # Smoothed max and softmax weights w.
        _, w = smooth_max_grad(g, T)  # len = PAD; sum = 1
        # Gradient of smooth-max(g) w.r.t. f is given by autocorrelation:
        # g[k] = sum_{i,j: i+j=k} f[i] f[j] dx ; deriv g[k]/df[i] = 2 f[i+k-?] dx
        # Easier: dM_smooth/df = 2 * (f * w)  via correlation theorem.
        # Specifically g = f*f -> dg/df = 2 f (in continuous sense).
        # smax = sum w[k] g[k]; d smax / df[i] = sum w[k] dg[k]/df[i]
        # But d g[k]/df[i] = 2 f[k-i] dx  (sum_j f[j] f[k-j] dx, deriv in f[i])
        # So d smax /df = 2 dx (w * f_reversed?). FFT trick:
        W = rfft(w, n=PAD2)
        # convolve w with f (reversed): cross-correlation
        # Actually: d smax / df[i] = 2 dx * sum_k w[k] f[k-i] = 2 dx (corr(f, w))[i]
        # corr(f, w)[i] = sum_k w[k] f[k-i] = sum_j w[i+j] f[j] = (w shifted by i)·f
        # In terms of FFT: corr(f,w) = ifft( conj(F) * W ); since both real, this works.
        Fconj = F.conjugate()
        corr = irfft(Fconj * W, n=PAD2)[:D] * dx
        # gradient w.r.t. f, then convert back to p: f = p/dx so df/dp = 1/dx
        # but we'll just gradient-descend p directly and reproject.
        grad_p = 2.0 * corr / dx   # (since f = p/dx)
        # Descent + projection.
        p = p - lr * grad_p
        p = project_simplex(p)
        if record_traj and (it % 20 == 0 or it == n_iters - 1):
            M, c = Mc(p)
            traj.append((it, M, c))
    M, c = Mc(p)
    return p, M, c, traj


def main():
    print(f"D={D} bins, dx={dx:.4f}, PAD={PAD}, PAD2={PAD2}")
    # ---- Family of initializations -------------------------------------
    rng = np.random.default_rng(42)
    inits = []

    # (a) uniform
    p = np.ones(D) / D
    inits.append(("uniform", p))

    # (b) symmetric two-bump
    for off in [0.10, 0.15, 0.20, 0.22]:
        for sigma in [0.02, 0.05, 0.08]:
            x = -0.25 + dx * np.arange(D)
            v = np.exp(-0.5 * ((x - off) / sigma) ** 2) + np.exp(-0.5 * ((x + off) / sigma) ** 2)
            v = v / v.sum()
            inits.append((f"sym_2bump_off{off:.2f}_s{sigma:.2f}", v))

    # (c) ASYMMETRIC two-bump (off1 != -off2)
    for off1, off2 in [(-0.20, 0.05), (-0.15, 0.10), (-0.22, 0.18),
                        (-0.10, 0.22), (-0.05, 0.20), (-0.18, 0.22),
                        (-0.05, 0.18), (-0.22, 0.05)]:
        for sigma in [0.02, 0.04, 0.06]:
            x = -0.25 + dx * np.arange(D)
            v = (np.exp(-0.5 * ((x - off1) / sigma) ** 2)
                  + np.exp(-0.5 * ((x - off2) / sigma) ** 2))
            v = v / v.sum()
            inits.append((f"asym_2bump_({off1:.2f},{off2:.2f})_s{sigma:.2f}", v))

    # (d) ASYMMETRIC unequal heights
    for off1, off2 in [(-0.20, 0.20), (-0.15, 0.15), (-0.22, 0.22)]:
        for h1 in [0.3, 0.5, 0.7, 1.5, 2.0, 3.0]:
            for sigma in [0.02, 0.05]:
                x = -0.25 + dx * np.arange(D)
                v = (h1 * np.exp(-0.5 * ((x - off1) / sigma) ** 2)
                      + 1.0 * np.exp(-0.5 * ((x - off2) / sigma) ** 2))
                v = v / v.sum()
                inits.append((f"asym_h{h1}_({off1:.2f},{off2:.2f})_s{sigma:.2f}", v))

    # (e) Random asymmetric multi-bump
    for seed in range(15):
        rng_local = np.random.default_rng(seed)
        nb = int(rng_local.integers(3, 9))
        x = -0.25 + dx * np.arange(D)
        v = np.zeros_like(x)
        for _ in range(nb):
            mu = rng_local.uniform(-0.22, 0.22)
            sigma = rng_local.uniform(0.01, 0.05)
            h = rng_local.uniform(0.2, 1.0)
            v += h * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        v = v / v.sum()
        inits.append((f"rand_seed{seed}_nb{nb}", v))

    # ---- Initial values -----------------------------------------------
    print(f"\n{'-'*92}")
    print("INITIAL c, M for each init (BEFORE descent on M):")
    print(f"{'-'*92}")
    print(f"{'init':<48s}  {'M_init':>8s}  {'c_init':>8s}")
    init_data = []
    for name, p in inits:
        M, c = Mc(p)
        init_data.append((name, p, M, c))
        # print(f"{name:<48s}  {M:>8.4f}  {c:>8.4f}")

    # Show only those with small M and asymmetric ones with small M
    init_data.sort(key=lambda r: r[2])
    print("Lowest-M initializations:")
    for name, p, M, c in init_data[:15]:
        print(f"  {name:<46s}  M={M:.4f}  c={c:.4f}")

    # ---- Descent ------------------------------------------------------
    print(f"\n{'='*92}")
    print("After M-descent (350 iters, T 0.05->0.005):")
    print(f"{'='*92}")
    print(f"{'init':<48s}  {'M_init':>8s}  {'c_init':>8s} -> "
          f"{'M_fin':>8s}  {'c_fin':>8s}")
    final_data = []
    for name, p in inits:
        M0, c0 = Mc(p)
        p_f, M_f, c_f, _ = descend_M(p, n_iters=350,
                                      T_start=0.05, T_end=0.005, lr=0.05)
        final_data.append((name, p_f, M_f, c_f, M0, c0))

    # Sort by final M and show top results.
    final_data.sort(key=lambda r: r[2])
    print("\nLowest-M after descent:")
    for name, p, M, c, M0, c0 in final_data[:20]:
        print(f"  {name:<46s}  M0={M0:.4f} c0={c0:.4f} -> M={M:.4f} c={c:.4f}")

    # Find asymmetric f's that ended at low M.
    print("\nALL post-descent results sorted by final M:")
    for name, p, M, c, M0, c0 in final_data:
        sym_flag = "sym" if "sym" in name else ("asym" if "asym" in name or "rand" in name or "h" in name else "?")
        print(f"  M={M:.4f}  c={c:.4f}  ({sym_flag})  init={name}  (init M0={M0:.4f})")

    # ---- Specific question: among final f's, what's the c distribution
    #      for those with M < 1.40, M < 1.378, M < 1.30 ?
    print(f"\n{'-'*92}")
    print("c distribution by final-M bracket:")
    print(f"{'-'*92}")
    for M_cap in [1.40, 1.378, 1.30, 1.28]:
        sub = [r for r in final_data if r[2] < M_cap]
        if sub:
            cs = [r[3] for r in sub]
            print(f"  M < {M_cap}: n={len(sub)}, c range [{min(cs):.4f}, {max(cs):.4f}], "
                  f"median {np.median(cs):.4f}")
            n_viol_BL = sum(1 for c in cs if c > 0.8825)
            n_viol_1 = sum(1 for c in cs if c > 1.0)
            print(f"     #(c > 0.8825) = {n_viol_BL},  #(c > 1.0) = {n_viol_1}")
        else:
            print(f"  M < {M_cap}: 0 samples reached this bracket")


if __name__ == "__main__":
    main()
