"""V3: FFT-based, single-process, robust asym near-ext search.

Designed to complete K=3, 5, 10 within a few minutes total. Uses scipy.signal
fftconvolve for ~30x speedup over numpy.convolve at NX=4001.

Critical fixes vs V1/V2:
  * fftconvolve for the inner loop;
  * fewer maxiter per stage so each seed completes in seconds;
  * eager flush after every seed;
  * top-level try/except so DE/L-BFGS errors don't kill the run;
  * try-from-best chain: K=5 init from K=3 result (split heaviest gaussian),
    K=10 init from K=5 result.
"""
import json, time, sys, traceback
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import softmax


NX = 4001
DX = 0.5 / (NX - 1)
xs = np.linspace(-0.25, 0.25, NX)


def trapz_int(g, dx=DX):
    return (g[0] + g[-1]) * 0.5 * dx + g[1:-1].sum() * dx


def build_f(params, K):
    c = params[0:K]
    w = np.exp(params[K:2*K])
    alpha = params[2*K:3*K]
    a = softmax(alpha)
    diff = (xs[:, None] - c[None, :]) / w[None, :]
    g = np.exp(-0.5 * diff * diff) / (np.sqrt(2 * np.pi) * w[None, :])
    f = (g * a[None, :]).sum(axis=1)
    s = trapz_int(f)
    if s > 0:
        f /= s
    return f


def conv_max(f, dx=DX):
    return float(fftconvolve(f, f).max() * dx)


def smooth_obj(params, K, p):
    f = build_f(params, K)
    c = fftconvolve(f, f) * DX
    cmax = c.max()
    return cmax * (np.mean((c / cmax) ** p)) ** (1.0 / p)


def hard_obj(params, K):
    return conv_max(build_f(params, K))


def random_init(K, seed=0):
    rng = np.random.default_rng(seed)
    c = rng.uniform(-0.20, 0.20, size=K)
    c[0] = rng.uniform(-0.22, -0.05) if rng.random() < 0.5 else rng.uniform(0.05, 0.22)
    lw = np.log(rng.uniform(0.012, 0.05, size=K))
    alpha = rng.normal(0, 1.0, size=K)
    return np.concatenate([c, lw, alpha])


def split_heaviest(params, K_old, K_new):
    """Take K_old gaussians, split the heaviest into 2 nearby ones to make K_new."""
    assert K_new == K_old + 1 or K_new > K_old
    c = list(params[0:K_old])
    lw = list(params[K_old:2*K_old])
    alpha = list(params[2*K_old:3*K_old])
    a = softmax(np.array(alpha))
    while len(c) < K_new:
        # split heaviest
        idx = int(np.argmax(a))
        eps = 0.01
        c1, c2 = c[idx] - eps, c[idx] + eps
        c[idx] = c1
        c.append(c2)
        lw_new = lw[idx] - 0.1
        lw.append(lw_new)
        # alpha: equal split (each new alpha has half the mass)
        a_idx = a[idx]
        new_alpha_each = np.log(a_idx / 2 + 1e-12)
        alpha[idx] = new_alpha_each
        alpha.append(new_alpha_each)
        a = softmax(np.array(alpha))
    return np.concatenate([np.array(c), np.array(lw), np.array(alpha)])


def make_bounds(K):
    bounds_c = [(-0.235, 0.235)] * K
    bounds_lw = [(np.log(0.005), np.log(0.06))] * K
    bounds_alpha = [(-3.0, 3.0)] * K
    return bounds_c + bounds_lw + bounds_alpha


def search_K(K, seeds, init_x_list=None, p_anneal=(20, 80, 200), maxiter=120):
    bounds = make_bounds(K)
    best_M = np.inf
    best_x = None
    init_x_list = init_x_list or []
    n_done = 0
    for s in seeds:
        try:
            x = random_init(K, seed=s)
            for p in p_anneal:
                res = minimize(smooth_obj, x, args=(K, p), method="L-BFGS-B",
                               bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-9})
                x = res.x
            res = minimize(hard_obj, x, args=(K,), method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": 200, "ftol": 1e-12})
            M = res.fun
            n_done += 1
            if M < best_M:
                best_M = M
                best_x = res.x
                print(f"    K={K} seed={s}  new best M = {M:.7f}", flush=True)
        except Exception as e:
            print(f"    K={K} seed={s}  failed: {e}", flush=True)

    # also try the warm-start inits
    for ix, (label, x0) in enumerate(init_x_list):
        try:
            x = x0
            for p in p_anneal:
                res = minimize(smooth_obj, x, args=(K, p), method="L-BFGS-B",
                               bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-9})
                x = res.x
            res = minimize(hard_obj, x, args=(K,), method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": 300, "ftol": 1e-12})
            M = res.fun
            n_done += 1
            if M < best_M:
                best_M = M
                best_x = res.x
                print(f"    K={K} warm-start={label}  new best M = {M:.7f}", flush=True)
            else:
                print(f"    K={K} warm-start={label}  M = {M:.7f}", flush=True)
        except Exception as e:
            print(f"    K={K} warm={label}  failed: {e}", flush=True)
    print(f"  -> K={K} best M = {best_M:.7f} after {n_done} seeds/warmstarts", flush=True)
    return best_M, best_x


def describe(f):
    M = conv_max(f)
    K2 = (f ** 2).sum() * DX
    integ = trapz_int(f)
    fr = f[::-1]
    asym = np.abs(f - fr).sum() * DX
    peaks = []
    for i in range(1, len(f) - 1):
        if f[i] > f[i - 1] and f[i] >= f[i + 1] and f[i] > 0.05 * f.max():
            peaks.append((float(xs[i]), float(f[i])))
    return {"M": float(M), "K_l2sq": float(K2), "integral": float(integ),
            "asymmetry": float(asym), "n_peaks": len(peaks),
            "peaks": peaks[:12]}


def verify_fine(params, K, NX_fine=16001):
    dx = 0.5 / (NX_fine - 1)
    xs_f = np.linspace(-0.25, 0.25, NX_fine)
    c = params[0:K]
    w = np.exp(params[K:2*K])
    alpha = params[2*K:3*K]
    a = softmax(alpha)
    diff = (xs_f[:, None] - c[None, :]) / w[None, :]
    g = np.exp(-0.5 * diff * diff) / (np.sqrt(2 * np.pi) * w[None, :])
    f = (g * a[None, :]).sum(axis=1)
    s = (f[0] + f[-1]) * 0.5 * dx + f[1:-1].sum() * dx
    f /= s
    M = float(fftconvolve(f, f).max() * dx)
    K2 = float((f ** 2).sum() * dx)
    return {"M_fine": M, "K_fine": K2, "NX_fine": NX_fine, "integral_fine": (f.sum() * dx).item()}


def main():
    t0_total = time.time()
    print("=" * 70)
    print("V3 (FFT) asymmetric near-ext search")
    print(f"NX={NX} dx={DX:.4e}", flush=True)
    print("=" * 70)
    f_box = np.ones_like(xs)
    f_box /= trapz_int(f_box)
    print(f"Sanity uniform M = {conv_max(f_box):.6f} (analytic 2.000)", flush=True)

    results = {}

    # ------ K = 3 ------
    print("\n--- K = 3 ---", flush=True)
    t = time.time()
    seeds = list(range(20))
    M3, x3 = search_K(3, seeds=seeds, init_x_list=[], p_anneal=(20, 80, 200), maxiter=150)
    f3 = build_f(x3, 3)
    d3 = describe(f3)
    v3 = verify_fine(x3, 3)
    print(f"  K=3 done: M={M3:.7f}  verify NX=16001 M={v3['M_fine']:.7f}  K={v3['K_fine']:.5f}  time {time.time()-t:.1f}s", flush=True)
    results["K3"] = {"best_M": float(M3), "params": x3.tolist(), "describe": d3, "verify": v3}

    # ------ K = 5 ------
    print("\n--- K = 5 ---", flush=True)
    t = time.time()
    seeds = list(range(20))
    init_5 = [("from_K3_split", split_heaviest(x3, 3, 5))]
    M5, x5 = search_K(5, seeds=seeds, init_x_list=init_5, p_anneal=(20, 80, 200), maxiter=150)
    f5 = build_f(x5, 5)
    d5 = describe(f5)
    v5 = verify_fine(x5, 5)
    print(f"  K=5 done: M={M5:.7f}  verify NX=16001 M={v5['M_fine']:.7f}  K={v5['K_fine']:.5f}  time {time.time()-t:.1f}s", flush=True)
    results["K5"] = {"best_M": float(M5), "params": x5.tolist(), "describe": d5, "verify": v5}

    # ------ K = 10 ------
    print("\n--- K = 10 ---", flush=True)
    t = time.time()
    seeds = list(range(20))
    init_10 = [("from_K5_split", split_heaviest(x5, 5, 10))]
    M10, x10 = search_K(10, seeds=seeds, init_x_list=init_10, p_anneal=(20, 80, 200), maxiter=120)
    f10 = build_f(x10, 10)
    d10 = describe(f10)
    v10 = verify_fine(x10, 10)
    print(f"  K=10 done: M={M10:.7f}  verify NX=16001 M={v10['M_fine']:.7f}  K={v10['K_fine']:.5f}  time {time.time()-t:.1f}s", flush=True)
    results["K10"] = {"best_M": float(M10), "params": x10.tolist(), "describe": d10, "verify": v10}

    # SUMMARY
    print("\n" + "=" * 70)
    print("SUMMARY (V3 FFT)")
    print("=" * 70)
    print(f"{'K':>4} {'M (4001)':>12} {'M (16001)':>12} {'K_l2sq':>10} {'asym':>8} {'M-1.2802':>10}")
    for k, v in results.items():
        d = v["describe"]
        print(f"{k:>4} {v['best_M']:>12.7f} {v['verify']['M_fine']:>12.7f} "
              f"{d['K_l2sq']:>10.5f} {d['asymmetry']:>8.4f} "
              f"{v['best_M'] - 1.2802:>+10.6f}")
    bestM = min(v["best_M"] for v in results.values())
    print(f"\nLOWEST M overall = {bestM:.7f}", flush=True)
    print(f"CS 2017 bound      = 1.2802000")
    print(f"MV near-ext theory = ~1.2748")
    print(f"Total time = {time.time()-t0_total:.1f}s", flush=True)

    # Best peaks structure
    print("\nBest near-min description:")
    bestK = min(results.items(), key=lambda kv: kv[1]["best_M"])[0]
    bv = results[bestK]
    print(f"  Best: {bestK}, M={bv['best_M']:.7f}, K=||f||_2^2={bv['describe']['K_l2sq']:.5f}")
    print(f"  asymmetry={bv['describe']['asymmetry']:.4f}")
    print(f"  n_peaks={bv['describe']['n_peaks']}")
    for x_p, h_p in bv["describe"]["peaks"][:12]:
        print(f"    peak at x = {x_p:+.4f}  height f(x) = {h_p:.4f}")

    with open("_asym_search_v3_results.json", "w") as fp:
        json.dump(results, fp, indent=2)
    print("\nSaved _asym_search_v3_results.json", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
