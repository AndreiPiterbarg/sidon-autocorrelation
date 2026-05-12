"""V2: tighter, anneal-smoothed Gaussian-mixture search for asymmetric near-extremizers of C_{1a}.

Improvements over _asym_search.py:
  * Centers constrained to [-0.235, 0.235] so that >= 4-sigma of each Gaussian
    fits inside [-1/4, 1/4] (reduces truncation artefacts).
  * Widths in [0.005, 0.06]; amplitudes via softmax to enforce mass split exactly.
  * Smoothed objective with annealed sharpness p (10 -> 200) inside L-BFGS.
  * Multistart from MV-style asymmetric init + uniform-random init.
  * Larger evaluation grid (NX=8001) for verification at the very end.
"""
import json, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax


NX = 4001
DX = 0.5 / (NX - 1)
xs = np.linspace(-0.25, 0.25, NX)


def trapz_int(g, dx=DX):
    return (g[0] + g[-1]) * 0.5 * dx + g[1:-1].sum() * dx


def build_f(params, K):
    """params = [c_1..c_K, lw_1..lw_K, alpha_1..alpha_K] with softmax over alpha."""
    c = params[0:K]
    w = np.exp(params[K:2*K])           # widths > 0
    alpha = params[2*K:3*K]             # raw logits, will softmax
    a = softmax(alpha)                  # mass weights summing to 1

    f = np.zeros_like(xs)
    for k in range(K):
        # un-truncated Gaussian, integrate to 1 (over R) approximately;
        # we'll renormalize at the end.
        gk = np.exp(-0.5 * ((xs - c[k]) / w[k]) ** 2) / (np.sqrt(2 * np.pi) * w[k])
        f += a[k] * gk
    # Final renorm against trapezoid integral on the box.
    s = trapz_int(f)
    if s > 0:
        f /= s
    return f


def conv_max(f, dx=DX):
    return float((np.convolve(f, f) * dx).max())


def smooth_obj(params, K, p):
    f = build_f(params, K)
    c = np.convolve(f, f) * DX
    cmax = c.max()
    # log-sum-exp stable
    return cmax * (np.mean((c / cmax) ** p)) ** (1.0 / p)


def hard_obj(params, K):
    f = build_f(params, K)
    return conv_max(f)


def random_init(K, seed=0, asym=True):
    rng = np.random.default_rng(seed)
    c = rng.uniform(-0.20, 0.20, size=K)
    if asym:
        # bias the heaviest atom away from 0
        c[0] = rng.uniform(-0.22, -0.05) if rng.random() < 0.5 else rng.uniform(0.05, 0.22)
    lw = np.log(rng.uniform(0.012, 0.05, size=K))
    alpha = rng.normal(0, 1.0, size=K)
    return np.concatenate([c, lw, alpha])


def mv_style_init(K, seed=0):
    """Initialize centers spaced asymmetrically in [-0.22, 0.22], weights skewed."""
    rng = np.random.default_rng(seed)
    # asymmetric center spacing: cluster more on left
    cs_neg = np.linspace(-0.22, -0.02, num=(K + 1) // 2)
    cs_pos = np.linspace(0.04, 0.22, num=K - len(cs_neg))
    c = np.concatenate([cs_neg, cs_pos]) + rng.normal(0, 0.005, size=K)
    lw = np.full(K, np.log(0.018)) + rng.normal(0, 0.1, size=K)
    alpha = rng.normal(0, 0.5, size=K)
    # boost the leftmost/heaviest atom
    alpha[0] += 1.0
    return np.concatenate([c, lw, alpha])


def search_K(K, n_random=24, label="gauss"):
    bounds_c = [(-0.235, 0.235)] * K
    bounds_lw = [(np.log(0.005), np.log(0.06))] * K
    bounds_alpha = [(-3.0, 3.0)] * K
    bounds = bounds_c + bounds_lw + bounds_alpha

    best_M = np.inf
    best_x = None
    t0 = time.time()

    for s in range(n_random):
        if s < n_random // 2:
            x0 = random_init(K, seed=100 + s, asym=True)
        else:
            x0 = mv_style_init(K, seed=200 + s)
        x = x0
        # Anneal: solve smooth_obj at p=10 -> 40 -> 100 -> 200, then exact
        for p in [10, 40, 100, 200]:
            try:
                res = minimize(smooth_obj, x, args=(K, p), method="L-BFGS-B",
                               bounds=bounds, options={"maxiter": 200, "ftol": 1e-9})
                x = res.x
            except Exception as e:
                print(f"  [K={K} seed={s} p={p}] failed: {e}", flush=True)
        try:
            res = minimize(hard_obj, x, args=(K,), method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": 300, "ftol": 1e-12})
            M = res.fun
            if M < best_M:
                best_M = M
                best_x = res.x
                print(f"  [K={K} seed={s}] new best M = {M:.7f}", flush=True)
        except Exception as e:
            print(f"  [K={K} seed={s}] hard failed: {e}", flush=True)

    elapsed = time.time() - t0
    return best_M, best_x, elapsed


def describe(f):
    M = conv_max(f)
    K2 = (f ** 2).sum() * DX
    integ = trapz_int(f)
    fr = f[::-1]
    asym = np.abs(f - fr).sum() * DX
    thr = 0.001 * f.max() if f.max() > 0 else 0
    support_mass = (f > thr).sum() * DX
    peaks = []
    for i in range(1, len(f) - 1):
        if f[i] > f[i - 1] and f[i] >= f[i + 1] and f[i] > 0.05 * f.max():
            peaks.append((float(xs[i]), float(f[i])))
    return {"M": float(M), "K_l2sq": float(K2), "integral": float(integ),
            "asymmetry": float(asym), "support_mass": float(support_mass),
            "n_peaks": len(peaks), "peaks": peaks[:10]}


def verify_on_fine_grid(params, K, NX_fine=16001):
    """Recompute M on a finer grid as a verification of M reported by search."""
    dx_fine = 0.5 / (NX_fine - 1)
    xs_fine = np.linspace(-0.25, 0.25, NX_fine)
    c = params[0:K]
    w = np.exp(params[K:2*K])
    alpha = params[2*K:3*K]
    a = softmax(alpha)
    f = np.zeros_like(xs_fine)
    for k in range(K):
        gk = np.exp(-0.5 * ((xs_fine - c[k]) / w[k]) ** 2) / (np.sqrt(2 * np.pi) * w[k])
        f += a[k] * gk
    # renorm
    s = (f[0] + f[-1]) * 0.5 * dx_fine + f[1:-1].sum() * dx_fine
    f /= s
    c2 = np.convolve(f, f) * dx_fine
    M_fine = float(c2.max())
    K2_fine = float((f ** 2).sum() * dx_fine)
    integ = float((f[0] + f[-1]) * 0.5 * dx_fine + f[1:-1].sum() * dx_fine)
    return {"M_fine": M_fine, "K_fine": K2_fine, "integral_fine": integ, "NX_fine": NX_fine}


def main():
    print("=" * 70)
    print("V2 asymmetric near-extremizer search (Gaussian-mixture, anneal)")
    print(f"NX={NX}  dx={DX:.4e}")
    print("=" * 70, flush=True)
    f_box = np.ones_like(xs)
    f_box /= trapz_int(f_box)
    print(f"Sanity uniform: M = {conv_max(f_box):.6f} (analytic 2)")

    results = {}
    for K in [3, 5, 10]:
        print(f"\n--- K={K} (anneal) ---", flush=True)
        n_r = 24 if K <= 5 else 16
        M, x, t = search_K(K, n_random=n_r, label=f"K{K}")
        f = build_f(x, K)
        d = describe(f)
        v = verify_on_fine_grid(x, K)
        print(f"  best (NX=4001)  M = {M:.7f}")
        print(f"  verify NX={v['NX_fine']}: M = {v['M_fine']:.7f}, K = {v['K_fine']:.5f}, integ = {v['integral_fine']:.6f}")
        print(f"  asym = {d['asymmetry']:.4f}, n_peaks = {d['n_peaks']}, time = {t:.1f}s")
        results[f"K{K}"] = {
            "best_M": float(M),
            "verify": v,
            "params": x.tolist(),
            "describe": d,
            "time_s": float(t),
        }

    print("\n" + "=" * 70)
    print("SUMMARY (V2)")
    print("=" * 70)
    print(f"{'K':>4} {'M (4001)':>12} {'M (16001)':>12} {'K_l2sq':>10} "
          f"{'asym':>8} {'M-1.2802':>10}")
    for k, v in results.items():
        d = v["describe"]
        print(f"{k:>4} {v['best_M']:>12.7f} {v['verify']['M_fine']:>12.7f} "
              f"{d['K_l2sq']:>10.5f} {d['asymmetry']:>8.4f} "
              f"{v['best_M'] - 1.2802:>+10.6f}")
    bestM = min(v["best_M"] for v in results.values())
    print(f"\nLOWEST M overall = {bestM:.7f}")
    print(f"CS 2017 bound      = 1.2802000")
    print(f"MV near-ext theory = ~1.2748")

    with open("_asym_search_v2_results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("\nSaved _asym_search_v2_results.json")


if __name__ == "__main__":
    main()
