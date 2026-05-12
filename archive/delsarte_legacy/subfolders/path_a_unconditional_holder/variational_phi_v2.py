"""
Faster v2 of variational ascent on Phi(M).
Uses N=512, FFT-based conv, fewer L-BFGS iters per restart, more restarts.
Output is line-buffered (flush=True) for live monitoring.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import time
import json
import sys

print = lambda *a, **k: __builtins__.print(*a, **k, flush=True)

N = 512
DX = 0.5 / N
xs = np.linspace(-0.25 + DX/2, 0.25 - DX/2, N)


def conv_self(f):
    g = fftconvolve(f, f, mode='full') * DX  # length 2N-1
    return g


def metrics(f):
    g = conv_self(f)
    Linf = float(np.max(g))
    L2sq = float(np.sum(g * g) * DX)
    L1 = float(np.sum(g) * DX)
    return Linf, L2sq, L1, L2sq / Linf if Linf > 0 else 0


def normalize_f(f):
    f = np.maximum(f, 0.0)
    s = float(np.sum(f) * DX)
    if s <= 1e-12:
        return None
    return f / s


# === Parameterizations ===
def f_from_gaussians(params, K):
    lw = params[:K]
    cs = np.tanh(params[K:2*K]) * 0.24  # bound centers in [-0.24, 0.24]
    lws = params[2*K:3*K]
    w = np.exp(lw)
    sigmas = np.exp(lws)
    f = np.zeros_like(xs)
    for i in range(K):
        f += w[i] * np.exp(-0.5 * ((xs - cs[i])/sigmas[i])**2) / (sigmas[i] * np.sqrt(2*np.pi))
    return normalize_f(f)


def f_from_steps(params, K):
    w = np.exp(params)
    block = N // K
    f = np.zeros_like(xs)
    for i in range(K):
        s = i * block
        e = (i+1) * block if i < K-1 else N
        f[s:e] = w[i]
    return normalize_f(f)


def f_from_pwlinear(params, K):
    vals = np.exp(params)
    knots = np.linspace(-0.25, 0.25, K)
    f = np.interp(xs, knots, vals)
    return normalize_f(f)


def load_BL():
    with open('delsarte_dual/restricted_holder/coeffBL.txt') as fh:
        s = fh.read().strip().strip('{}').split(',')
    return np.array([int(x) for x in s], dtype=float)


_BL_cached = None
def f_from_BL_scaled(scale_factor=1.0, head_keep=None):
    """BL witness mapped onto sub-interval. head_keep: if set, zero out tail beyond index."""
    global _BL_cached
    if _BL_cached is None:
        _BL_cached = load_BL()
    v = _BL_cached.copy()
    if head_keep is not None:
        v[head_keep:] = 0
    L = len(v)
    half = 0.25 * scale_factor
    # bin j of BL covers x in [-half + j*(2*half/L), -half + (j+1)*(2*half/L))
    # Find which BL bin each xs[i] falls into
    bin_idx = ((xs + half) / (2*half/L)).astype(int)
    bin_idx = np.clip(bin_idx, 0, L-1)
    in_support = (xs >= -half) & (xs < half)
    f = np.where(in_support, v[bin_idx], 0.0)
    return normalize_f(f)


# === Ascent ===
def neg_obj(params, K, parametrization, M_target, penalty=1e3):
    if parametrization == 'gauss':
        f = f_from_gaussians(params, K)
    elif parametrization == 'step':
        f = f_from_steps(params, K)
    elif parametrization == 'pwlin':
        f = f_from_pwlinear(params, K)
    if f is None:
        return 1e6
    Linf, L2sq, L1, c_emp = metrics(f)
    pen = penalty * max(0.0, Linf - M_target)**2
    return -c_emp + pen


def init_params(K, parametrization, asymmetric=False, seed=0):
    rng = np.random.default_rng(seed)
    if parametrization == 'gauss':
        lw = rng.normal(0.0, 0.5, K)
        if asymmetric:
            cs_raw = rng.uniform(-1, 1, K)
        else:
            half = (K+1)//2
            tmp = rng.uniform(-1, 1, half)
            cs_raw = np.concatenate([-tmp, tmp[:K-half]])
        lws = rng.normal(np.log(0.05), 0.3, K)
        return np.concatenate([lw, cs_raw, lws])
    elif parametrization == 'step':
        if asymmetric:
            return rng.normal(0.0, 1.0, K)
        else:
            half = (K+1)//2
            tmp = rng.normal(0.0, 1.0, half)
            return np.concatenate([tmp, tmp[:K-half][::-1]])
    elif parametrization == 'pwlin':
        if asymmetric:
            return rng.normal(0.0, 1.0, K)
        else:
            half = (K+1)//2
            tmp = rng.normal(0.0, 1.0, half)
            return np.concatenate([tmp, tmp[:K-half][::-1]])


def ascend(K, parametrization, M_target, n_restarts=4, asymmetric=True):
    best = (-np.inf, None, None, None)
    for r in range(n_restarts):
        x0 = init_params(K, parametrization, asymmetric=asymmetric, seed=r * 17 + 3)
        try:
            res = minimize(neg_obj, x0, args=(K, parametrization, M_target),
                           method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-8})
        except Exception:
            continue
        if parametrization == 'gauss':
            f = f_from_gaussians(res.x, K)
        elif parametrization == 'step':
            f = f_from_steps(res.x, K)
        elif parametrization == 'pwlin':
            f = f_from_pwlinear(res.x, K)
        if f is None:
            continue
        Linf, L2sq, L1, c_emp = metrics(f)
        if Linf > M_target * 1.01:
            continue
        if c_emp > best[0]:
            best = (c_emp, res.x, f, {'Linf': Linf, 'L2sq': L2sq, 'L1': L1, 'restart': r})
    return best


def asymmetry(f):
    f_rev = f[::-1]
    num = float(np.sum(np.abs(f - f_rev)) * DX)
    den = float(np.sum(np.abs(f + f_rev)) * DX)
    return num / den if den > 0 else 0


def support_extent(f, threshold_frac=0.001):
    thr = threshold_frac * np.max(f)
    mask = f > thr
    if not mask.any():
        return (0, 0)
    idx = np.where(mask)[0]
    return float(xs[idx[0]]), float(xs[idx[-1]])


def describe_f(f):
    Linf, L2sq, L1, c_emp = metrics(f)
    asym = asymmetry(f)
    sup_lo, sup_hi = support_extent(f)
    peak_x = float(xs[np.argmax(f)])
    g = conv_self(f)
    g_xs = np.linspace(-0.5, 0.5, 2*N-1)
    peak_g_x = float(g_xs[np.argmax(g)])
    return {
        'c_emp': c_emp, 'M': Linf, 'L2sq': L2sq, 'L1': L1,
        'asymmetry': asym, 'support_lo': sup_lo, 'support_hi': sup_hi,
        'support_width': sup_hi - sup_lo,
        'peak_f_x': peak_x, 'peak_g_x': peak_g_x,
        'f_max': float(np.max(f)),
        'norm_f_sq': float(np.sum(f*f) * DX),  # ||f||_2^2 = K
    }


def adversarial_BL(M_target):
    """Try BL truncated/scaled + mixtures with indicator."""
    results = []
    ind = np.full_like(xs, 2.0)
    for head_keep in [None, 50, 100, 150, 200, 300, 400]:
        for sf in [0.5, 0.7, 0.85, 1.0]:
            try:
                f_BL = f_from_BL_scaled(scale_factor=sf, head_keep=head_keep)
                if f_BL is None:
                    continue
                for alpha in [0.3, 0.5, 0.7, 0.85, 0.95, 1.0]:
                    f_mix = alpha * f_BL + (1-alpha) * ind
                    f_mix = normalize_f(f_mix)
                    if f_mix is None:
                        continue
                    Linf, L2sq, L1, c_emp = metrics(f_mix)
                    if Linf <= M_target * 1.005:
                        results.append((c_emp, f_mix,
                                        {'kind': 'BL_mix', 'head_keep': str(head_keep),
                                         'sf': sf, 'alpha': alpha, 'Linf': Linf}))
            except Exception:
                continue
    return results


def main():
    c_star = float(np.log(16) / np.pi)
    print(f"Threshold c_star = log(16)/pi = {c_star:.10f}")
    print(f"Grid: N={N}, dx={DX:.6e}")
    print()

    M_grid = [1.27, 1.30, 1.33, 1.35, 1.378]
    out = {'c_star': c_star, 'N': N, 'results': {}}
    t_start = time.time()

    configs = [
        ('gauss', 3), ('gauss', 5), ('gauss', 8),
        ('step', 8), ('step', 16), ('step', 32),
        ('pwlin', 12), ('pwlin', 24),
    ]

    for M in M_grid:
        print(f"=" * 60)
        print(f"M_target = {M}  (elapsed {time.time()-t_start:.1f}s)")
        print(f"=" * 60)
        best_overall = (-np.inf, None, None)
        for (param, K) in configs:
            for asym in [False, True]:
                t0 = time.time()
                c_emp, x, f, info = ascend(K, param, M, n_restarts=4, asymmetric=asym)
                if f is None:
                    continue
                tag = f"{param}-K{K}-{'asy' if asym else 'sym'}"
                print(f"  {tag:18s}: c={c_emp:.5f}, Linf={info['Linf']:.4f}, t={time.time()-t0:.1f}s")
                if c_emp > best_overall[0]:
                    best_overall = (c_emp, f, {'param': param, 'K': K, 'asym': asym, **info})

        # Adversarial
        adv = adversarial_BL(M)
        for (c_emp, f, info) in adv:
            if c_emp > best_overall[0]:
                best_overall = (c_emp, f, info)
                print(f"  ADV BL: c={c_emp:.5f}, Linf={info['Linf']:.4f}, head={info['head_keep']}, sf={info['sf']}, alpha={info['alpha']}")

        c_best, f_best, info_best = best_overall
        desc = describe_f(f_best)
        margin = c_star - c_best
        verdict = 'PROMISING' if margin > 0.01 else ('NEAR-CRITICAL' if margin > -0.001 else 'DISPROVED')
        print(f"\n  BEST at M={M}: c={c_best:.6f}, margin to c*={margin:+.6f}, {verdict}")
        print(f"  STRUCT: asym={desc['asymmetry']:.4f}, K=||f||_2^2={desc['norm_f_sq']:.3f}, "
              f"sup=[{desc['support_lo']:.3f},{desc['support_hi']:.3f}] (w={desc['support_width']:.3f}), "
              f"f_peak_x={desc['peak_f_x']:.3f}")
        out['results'][str(M)] = {
            'best_c_emp': float(c_best),
            'best_info': {k: (str(v)) for k, v in info_best.items()},
            'description': desc,
            'distance_to_threshold': float(margin),
            'verdict': verdict,
        }
        # Save f_best as numpy
        np.save(f'delsarte_dual/path_a_unconditional_holder/f_best_M{M}.npy', f_best)

    with open('delsarte_dual/path_a_unconditional_holder/phi_curve.json', 'w') as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\nSaved phi_curve.json (elapsed {time.time()-t_start:.1f}s)")

    print("\n" + "=" * 60)
    print("PHI(M) CURVE -- SUMMARY")
    print("=" * 60)
    print(f"  c_star (threshold) = {c_star:.6f}")
    for M in M_grid:
        cb = out['results'][str(M)]['best_c_emp']
        margin = c_star - cb
        v = out['results'][str(M)]['verdict']
        print(f"  M={M}: Phi_emp >= {cb:.6f}, margin = {margin:+.6f}, {v}")


if __name__ == '__main__':
    main()
