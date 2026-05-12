"""Numerical search for asymmetric near-extremizers of C_{1a}.

Parametrize f as a sum of K Gaussians or piecewise-constant bumps in
[-1/4, 1/4], minimize ||f*f||_inf subject to f >= 0, supp in [-1/4,1/4],
int f = 1.

CS 2017 says C_{1a} >= 1.2802. MV near-extremizer ~ 1.2748 (asymm).
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time, json, sys


# Common evaluation grid for f and its convolution
NX = 4001            # grid for f on [-1/4, 1/4]
NX_HALF = (NX - 1) // 2
DX = 0.5 / (NX - 1)  # spacing on [-1/4, 1/4]
xs = np.linspace(-0.25, 0.25, NX)
# Convolution of two functions on [-1/4,1/4] lives on [-1/2, 1/2].
# f*f at sample point t in [-1/2, 1/2] = sum_i f[i] f[j] dx where x_i + x_j = t.
NCONV = 2 * NX - 1   # 8001 sample points on [-1/2, 1/2]


def _normalise(f, dx=DX):
    """Trapezoidal-rule normalisation of f to integral 1, with clipping at 0."""
    f = np.maximum(f, 0.0)
    # trapezoidal weights
    s = (f[0] + f[-1]) * 0.5 + f[1:-1].sum()
    s *= dx
    if s <= 0:
        return f
    return f / s


def conv_inf(f, dx=DX):
    """||f*f||_inf via discrete convolution.  f sampled at NX equispaced pts."""
    # discrete conv length 2N-1; multiply by dx for the integral
    c = np.convolve(f, f) * dx
    return c.max(), c


def gaussian_f(params, K):
    """Build f from K Gaussians: params = [c_1,...,c_K, w_1,...,w_K, a_1,...,a_K].

    centers c_i in (-1/4, 1/4), widths w_i > 0, amplitudes a_i > 0.
    Truncate to [-1/4, 1/4] and renormalize to integral 1.
    """
    c = params[0:K]
    w = np.exp(params[K:2*K])      # ensure positivity
    a = np.exp(params[2*K:3*K])    # ensure positivity
    f = np.zeros_like(xs)
    for k in range(K):
        f += a[k] * np.exp(-0.5 * ((xs - c[k]) / w[k]) ** 2)
    return _normalise(f)


def bump_f(params, K, dx=DX):
    """Piecewise-constant bumps: params = [centers..., halfwidths..., heights...]."""
    c = params[0:K]
    w = np.exp(params[K:2*K]) * 0.05  # halfwidth scale
    h = np.exp(params[2*K:3*K])
    f = np.zeros_like(xs)
    for k in range(K):
        mask = (xs >= c[k] - w[k]) & (xs <= c[k] + w[k])
        f[mask] += h[k]
    return _normalise(f)


def objective(params, K, builder=gaussian_f):
    f = builder(params, K)
    M, _ = conv_inf(f)
    return M


def random_init(K, seed=None, asym_bias=True):
    rng = np.random.default_rng(seed)
    # centers spread in (-0.22, 0.22)
    c = rng.uniform(-0.22, 0.22, size=K)
    if asym_bias:
        # bias one center off-zero to break symmetry
        c[0] = rng.uniform(-0.22, -0.05) if rng.random() < 0.5 else rng.uniform(0.05, 0.22)
    # log-widths
    lw = np.log(rng.uniform(0.02, 0.08, size=K))
    # log-amplitudes
    la = np.log(rng.uniform(0.5, 2.0, size=K))
    return np.concatenate([c, lw, la])


def smooth_obj(params, K, builder=gaussian_f, p=80):
    """Smoothed objective: (sum c^p)^(1/p) approximates max(c)."""
    f = builder(params, K)
    c = np.convolve(f, f) * DX
    cmax = c.max()
    # log-sum-exp form for numerical stability
    return cmax * np.power(np.mean(np.power(c / cmax, p)), 1.0 / p)


def search_K(K, n_random=8, builder=gaussian_f, seed0=0, use_de=True,
             de_maxiter=200, label="gauss"):
    """Run multistart L-BFGS plus optional DE for parametrization with K bumps."""
    best_M = np.inf
    best_x = None
    t0 = time.time()
    # Box bounds on params
    bounds_c = [(-0.245, 0.245)] * K
    bounds_lw = [(np.log(0.005), np.log(0.20))] * K
    bounds_la = [(-3.0, 3.0)] * K
    bounds = bounds_c + bounds_lw + bounds_la

    # 1) Multistart L-BFGS-B on smoothed objective then exact objective
    for s in range(n_random):
        x0 = random_init(K, seed=seed0 + s)
        # Smoothed pre-solve
        try:
            res = minimize(smooth_obj, x0, args=(K, builder, 80), method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": 200, "ftol": 1e-9})
            res2 = minimize(objective, res.x, args=(K, builder), method="L-BFGS-B",
                            bounds=bounds, options={"maxiter": 200, "ftol": 1e-10})
            M = res2.fun
            if M < best_M:
                best_M = M
                best_x = res2.x
        except Exception as e:
            print(f"  [{label} K={K} seed={seed0+s}] failed: {e}", flush=True)

    # 2) Differential evolution (global) — optional
    if use_de:
        try:
            res = differential_evolution(
                smooth_obj,
                bounds=bounds,
                args=(K, builder, 80),
                maxiter=de_maxiter,
                popsize=12,
                tol=1e-7,
                seed=seed0 + 1000,
                workers=1,
                polish=False,
                init="sobol",
            )
            res2 = minimize(objective, res.x, args=(K, builder), method="L-BFGS-B",
                            bounds=bounds, options={"maxiter": 300, "ftol": 1e-11})
            if res2.fun < best_M:
                best_M = res2.fun
                best_x = res2.x
        except Exception as e:
            print(f"  [{label} K={K}] DE failed: {e}", flush=True)

    elapsed = time.time() - t0
    return best_M, best_x, elapsed


def describe(f):
    """Return dict with diagnostics: support, peaks, K=||f||_2^2, asymmetry, M."""
    M, c = conv_inf(f)
    K2 = (f ** 2).sum() * DX
    integ = f.sum() * DX  # roughly 1 by construction
    # asymmetry: ||f - f(-x)||_1 / ||f||_1
    fr = f[::-1]
    asym = np.abs(f - fr).sum() * DX
    # support measure (where f > 1e-3 * max)
    thr = 0.001 * f.max() if f.max() > 0 else 0
    support_mass = (f > thr).sum() * DX
    # peak count via local maxima
    peaks = []
    for i in range(1, len(f) - 1):
        if f[i] > f[i - 1] and f[i] >= f[i + 1] and f[i] > 0.05 * f.max():
            peaks.append((float(xs[i]), float(f[i])))
    return {"M": float(M), "K_l2sq": float(K2), "integral": float(integ),
            "asymmetry": float(asym), "support_mass": float(support_mass),
            "n_peaks": len(peaks), "peaks": peaks[:10]}


def main():
    print("=" * 70)
    print("Numerical search for asymmetric near-extremizers of C_{1a}")
    print(f"Grid NX={NX}, dx={DX:.5e}")
    print("=" * 70, flush=True)
    results = {}

    # Sanity check: known box-uniform f gives M = 2 on [-1/4, 1/4].
    f_box = np.ones_like(xs)
    f_box = _normalise(f_box)
    M_box, _ = conv_inf(f_box)
    print(f"Sanity: uniform on [-1/4,1/4] -> M = {M_box:.6f} (analytic 2.000)")

    K_list = [3, 5, 10]
    for K in K_list:
        for builder, label in [(gaussian_f, "gauss"), (bump_f, "bump")]:
            print(f"\n--- K={K} {label} ---", flush=True)
            n_rand = 12 if K <= 5 else 8
            de_iter = 200 if K <= 5 else 150
            M, x, t = search_K(K, n_random=n_rand, builder=builder, seed0=42,
                               use_de=True, de_maxiter=de_iter, label=label)
            f = builder(x, K)
            d = describe(f)
            print(f"  best M = {M:.7f} (search M = {d['M']:.7f})  time {t:.1f}s")
            print(f"  K=||f||_2^2 = {d['K_l2sq']:.5f}  asym = {d['asymmetry']:.4f}")
            print(f"  n_peaks = {d['n_peaks']}  peaks = {d['peaks'][:5]}")
            results[f"{label}_K{K}"] = {
                "best_M": float(M),
                "params": x.tolist() if x is not None else None,
                "describe": d,
                "time_s": float(t),
            }

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'config':<14} {'best M':>10} {'K = ||f||_2^2':>14} {'asym':>8} "
          f"{'M - 1.2802':>10}")
    for k, v in results.items():
        d = v["describe"]
        print(f"{k:<14} {v['best_M']:>10.6f} {d['K_l2sq']:>14.5f} "
              f"{d['asymmetry']:>8.4f} {v['best_M'] - 1.2802:>+10.6f}")
    bestM = min(v["best_M"] for v in results.values())
    print(f"\nLOWEST M overall = {bestM:.7f}")
    print(f"CS 2017 bound      = 1.2802000")
    print(f"MV near-ext        = 1.2748 (asymm, theoretical)")
    print(f"Diff to CS bound   = {bestM - 1.2802:+.6f}")

    with open("_asym_search_results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("\nSaved _asym_search_results.json")


if __name__ == "__main__":
    main()
