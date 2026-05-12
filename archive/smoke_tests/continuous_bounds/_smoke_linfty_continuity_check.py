"""Smoke test: verify the L^infty continuity bound

   ||f*f - f_a*f_a||_inf <= eps(d, m, M, ...)

where f_a is the canonical step-function L^1-projection at resolution (d, m).

For each test family, we:
  1. Sample f on a fine grid (N points across [-1/4, 1/4]).
  2. Compute f_a (cumulative-floor discretization, height = a_i / m).
  3. Compute ||f*f - f_a*f_a||_inf via FFT.
  4. Compute the analytic bounds:
       eps_inf       = M^2/2 + 2/m + 1/(2 m^2)             (under (H_M))
       eps_TV        = V M/d + 2/m + 1/(2 m^2)             (under (H_M, TV<=V))
       eps_lip       = (L/(4d) + 1/(2m)) * (2M + 2d/m)     (under (H_{M,L}))
       eps_quant     = 2/m + 1/(2 m^2)                     (step-function quant only)
  5. Verify (numerically) that the actual error is within each bound.

USAGE:
  python _smoke_linfty_continuity_check.py
"""
from __future__ import annotations

import json
import numpy as np


# ------------------------------------------------------------------
# Discretize f to a step function with quantized heights.
# ------------------------------------------------------------------
def _bin_masses(f_vals: np.ndarray, x: np.ndarray, d: int) -> np.ndarray:
    """Compute mu_i = integral of f over bin_i, i = 0,...,d-1.

    Bins partition [-1/4, 1/4] uniformly into d intervals of width w = 1/(2d).
    """
    w = 1.0 / (2 * d)
    edges = -0.25 + np.arange(d + 1) * w
    masses = np.zeros(d)
    for i in range(d):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if i == d - 1:  # include right endpoint
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        # trapezoidal integration over this bin
        if mask.any():
            xb = x[mask]
            fb = f_vals[mask]
            if len(xb) > 1:
                masses[i] = np.trapz(fb, xb)
            else:
                masses[i] = fb[0] * w
    return masses


def _cumulative_floor(mu: np.ndarray, m: int, d: int) -> np.ndarray:
    """Canonical cumulative-floor discretization of mu.
    Returns integer vector a with sum(a) = S = 2*d*m.
    """
    S = 2 * d * m
    M_cum = np.concatenate([[0.0], np.cumsum(mu)])
    D = np.floor(S * M_cum).astype(np.int64)
    D[0] = 0
    D[-1] = S  # enforce ending exactly
    a = np.diff(D)
    return a


def _step_eval(a: np.ndarray, m: int, d: int, x: np.ndarray) -> np.ndarray:
    """Evaluate the step function f_a(x) on the grid x.

    Heights: a_i / m on bin_i = [-1/4 + i*w, -1/4 + (i+1)*w], w = 1/(2d).
    """
    w = 1.0 / (2 * d)
    edges = -0.25 + np.arange(d + 1) * w
    out = np.zeros_like(x)
    for i in range(d):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if i == d - 1:
            mask = (x >= edges[i]) & (x <= edges[i + 1])
        out[mask] = a[i] / m
    return out


def _conv_inf_via_fft(f_vals: np.ndarray, x: np.ndarray) -> float:
    """Compute ||f*f||_inf via FFT-based convolution.
    Returns the sup of (f*f)(t) over t in [-1/2, 1/2].
    """
    dx = x[1] - x[0]
    # Numerical convolution on the grid.
    conv = np.convolve(f_vals, f_vals, mode='full') * dx
    return float(np.max(np.abs(conv)))


def _conv_diff_inf(f_vals: np.ndarray, fa_vals: np.ndarray, x: np.ndarray) -> float:
    """Compute ||f*f - f_a*f_a||_inf."""
    dx = x[1] - x[0]
    f_self = np.convolve(f_vals, f_vals, mode='full') * dx
    fa_self = np.convolve(fa_vals, fa_vals, mode='full') * dx
    return float(np.max(np.abs(f_self - fa_self)))


# ------------------------------------------------------------------
# Analytic bounds (the formulae from the proof).
# ------------------------------------------------------------------
def eps_inf(d: int, m: int, M: float) -> float:
    """L^infty-only bound (rigorous): M^2/2 + 2/m + 1/(2m^2).

    Derivation: ||f*f - fa*fa||_inf <= ||f*f - fbar*fbar||_inf + ||fbar*fbar - fa*fa||_inf
      <= M^2/2  + (2/m + 1/(2m^2)).
    See proof/linfty_continuity_bound.md sections 5.4 and 5.6.
    """
    return M * M / 2.0 + 2.0 / m + 0.5 / (m * m)


def eps_TV(d: int, m: int, M: float, V: float) -> float:
    """TV-based bound: V*M/d + 2/m + 1/(2m^2)."""
    return V * M / d + 2.0 / m + 0.5 / (m * m)


def eps_lip(d: int, m: int, M: float, L: float) -> float:
    """Lipschitz-based bound: (L/(4d) + 1/(2m)) * (2M + 1/m)."""
    return (L / (4.0 * d) + 1.0 / (2.0 * m)) * (2.0 * M + 1.0 / m)


def eps_quant(d: int, m: int, M: float) -> float:
    """C&S Lemma 3 step-function quantization-only bound."""
    return 2.0 / m + 0.5 / (m * m)


# ------------------------------------------------------------------
# Test families.
# ------------------------------------------------------------------
def family_uniform(N: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """f = uniform on [-1/4, 1/4]: f(x) = 2."""
    x = np.linspace(-0.25, 0.25, N)
    f = np.full_like(x, 2.0)
    info = {'name': 'uniform', 'M': 2.0, 'TV': 0.0, 'L': 0.0}
    return x, f, info


def family_centered_triangle(N: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """f = triangle peaked at 0: f(x) = 8(1 - 4|x|) on [-1/4, 1/4], integral = 1."""
    x = np.linspace(-0.25, 0.25, N)
    f = 8.0 * np.maximum(0.0, 1.0 - 4.0 * np.abs(x))
    info = {'name': 'triangle', 'M': 8.0, 'TV': 16.0, 'L': 32.0}  # f' jumps -32 to 0
    return x, f, info


def family_bump(N: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """f = smooth bump: cos^2(2*pi*x) * scale. Lipschitz."""
    x = np.linspace(-0.25, 0.25, N)
    raw = np.cos(2 * np.pi * x) ** 2  # peak at x=0 is 1, edges 0.
    f = raw / np.trapz(raw, x)  # normalize integral to 1.
    M = float(np.max(f))
    L = float(np.max(np.abs(np.gradient(f, x))))
    info = {'name': 'cos2_bump', 'M': M, 'TV': 2 * M, 'L': L}
    return x, f, info


def family_gauss(N: int, sig: float = 0.07) -> tuple[np.ndarray, np.ndarray, dict]:
    """f = truncated centered Gaussian, normalized."""
    x = np.linspace(-0.25, 0.25, N)
    raw = np.exp(-(x ** 2) / (2 * sig ** 2))
    f = raw / np.trapz(raw, x)
    M = float(np.max(f))
    L = float(np.max(np.abs(np.gradient(f, x))))
    info = {'name': f'gauss_sig{sig}', 'M': M, 'TV': 2 * M, 'L': L}
    return x, f, info


def family_double_bump(N: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """f = two bumps near +-1/4 (asymmetric)."""
    x = np.linspace(-0.25, 0.25, N)
    raw = np.exp(-((x + 0.18) ** 2) / 0.005) + 0.5 * np.exp(-((x - 0.15) ** 2) / 0.008)
    f = raw / np.trapz(raw, x)
    M = float(np.max(f))
    L = float(np.max(np.abs(np.gradient(f, x))))
    info = {'name': 'double_bump', 'M': M, 'TV': 4 * M, 'L': L}
    return x, f, info


def family_step_func(N: int, d: int, m: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """f = a step function (so f - f_a should be small if f matches f_a)."""
    x = np.linspace(-0.25, 0.25, N)
    rng = np.random.default_rng(seed=42)
    a_init = rng.integers(0, 2 * m + 1, size=d)
    a_init = a_init.astype(np.int64)
    needed = 2 * d * m
    while a_init.sum() != needed:
        diff = needed - a_init.sum()
        i = int(rng.integers(0, d))
        a_init[i] = max(0, a_init[i] + np.sign(diff))
    f = _step_eval(a_init, m, d, x)
    integ = np.trapz(f, x)
    if integ > 0:
        f = f / integ  # normalize
    M = float(np.max(f))
    info = {'name': 'random_step', 'M': M, 'TV': 4 * M, 'L': 1e9}
    return x, f, info


# ------------------------------------------------------------------
# Main driver.
# ------------------------------------------------------------------
def run_test(family_fn, d: int, m: int, N: int = 4001) -> dict:
    """Run a single (family, d, m) test."""
    x, f, info = family_fn(N) if family_fn != family_step_func else family_fn(N, d, m)

    # Compute discretization.
    mu = _bin_masses(f, x, d)
    # Re-normalize mu to sum to 1 (to handle small grid quadrature error).
    if mu.sum() > 0:
        mu = mu / mu.sum()
    a = _cumulative_floor(mu, m, d)
    fa = _step_eval(a, m, d, x)
    # Re-normalize fa to integrate to 1 (small grid effect).
    integ_fa = np.trapz(fa, x)
    if integ_fa > 0:
        fa = fa / integ_fa

    # Empirical errors.
    err_l1 = float(np.trapz(np.abs(f - fa), x))
    err_linf = float(np.max(np.abs(f - fa)))
    conv_diff = _conv_diff_inf(f, fa, x)

    # Analytic bounds.
    M = info['M']
    V = info['TV']
    L = info['L']
    bound_inf = eps_inf(d, m, M)
    bound_TV = eps_TV(d, m, M, V)
    bound_lip = eps_lip(d, m, M, L) if L < 1e6 else float('inf')
    bound_quant = eps_quant(d, m, M)

    # Triangle-inequality alternative: 2 * ||f - fa||_1 * max(||f||_inf, ||fa||_inf)
    M_actual = max(np.max(f), np.max(fa))
    bound_actual_l1 = 2.0 * err_l1 * M_actual

    return {
        'family': info['name'],
        'd': d,
        'm': m,
        'M_input': M,
        'M_actual': float(M_actual),
        'TV_input': V,
        'L_input': L,
        'err_l1': err_l1,
        'err_linf': err_linf,
        'conv_diff_linf': conv_diff,
        'bound_inf': bound_inf,
        'bound_TV': bound_TV,
        'bound_lip': bound_lip,
        'bound_quant': bound_quant,
        'bound_actual_l1': bound_actual_l1,
        'sat_inf': conv_diff <= bound_inf,
        'sat_TV': conv_diff <= bound_TV,
        'sat_lip': conv_diff <= bound_lip,
        'sat_quant': conv_diff <= bound_quant,
        'sat_actual_l1': conv_diff <= bound_actual_l1,
    }


def main() -> None:
    families = [
        family_uniform,
        family_centered_triangle,
        family_bump,
        family_gauss,
        family_double_bump,
        family_step_func,
    ]
    configs = [
        (2, 20),
        (4, 20),
        (8, 20),
        (16, 20),
    ]

    results = []
    for d, m in configs:
        for fam in families:
            try:
                r = run_test(fam, d, m)
                results.append(r)
            except Exception as e:
                results.append({'family': fam.__name__, 'd': d, 'm': m, 'error': str(e)})

    # Pretty print.
    print(f"\n{'='*100}")
    print(f"{'Family':<22} {'d':>3} {'m':>3} {'M_act':>7} {'L1err':>8} {'Linf':>8} "
          f"{'conv_d':>9} {'eps_inf':>9} {'eps_TV':>9} {'eps_quant':>9} {'sat_inf':>8}")
    print(f"{'='*100}")
    for r in results:
        if 'error' in r:
            print(f"{r['family']:<22} {r['d']:>3} {r['m']:>3} ERROR: {r['error']}")
            continue
        sat_str = 'OK' if r['sat_actual_l1'] else 'FAIL'
        print(f"{r['family']:<22} {r['d']:>3} {r['m']:>3} {r['M_actual']:>7.3f} "
              f"{r['err_l1']:>8.4f} {r['err_linf']:>8.4f} {r['conv_diff_linf']:>9.4f} "
              f"{r['bound_inf']:>9.4f} {r['bound_TV']:>9.4f} {r['bound_quant']:>9.4f} "
              f"{sat_str:>8}")

    # Aggregate sat counts.
    n_total = sum(1 for r in results if 'error' not in r)
    sat_inf = sum(r.get('sat_inf', False) for r in results)
    sat_TV = sum(r.get('sat_TV', False) for r in results)
    sat_quant = sum(r.get('sat_quant', False) for r in results)
    sat_actual = sum(r.get('sat_actual_l1', False) for r in results)
    print(f"\nSummary: out of {n_total} tests:")
    print(f"  conv_diff <= eps_inf  ({sat_inf}/{n_total})")
    print(f"  conv_diff <= eps_TV   ({sat_TV}/{n_total})")
    print(f"  conv_diff <= eps_quant({sat_quant}/{n_total}) (step-quantization-only)")
    print(f"  conv_diff <= 2*L1err*M_act ({sat_actual}/{n_total}) (true triangle ineq)")

    # Headroom analysis at (d=2, m=20).
    print(f"\n{'='*100}")
    print("HEADROOM ANALYSIS at d=2, m=20 (need eps <= 1.5 - 1.281 = 0.219):")
    print(f"{'='*100}")
    for M_try in [0.486, 1.0, 1.6, 2.0, 4.0]:
        e_inf = eps_inf(2, 20, M_try)
        e_quant = eps_quant(2, 20, M_try)
        for V_try in [0.5, 1.0, 2.0]:
            e_TV = eps_TV(2, 20, M_try, V_try)
            print(f"  M={M_try:5.2f}, V={V_try:4.1f}: eps_inf={e_inf:6.3f},  eps_TV={e_TV:6.3f},  "
                  f"eps_quant={e_quant:6.4f},  fits (<=0.219)? "
                  f"{'YES' if e_inf <= 0.219 else 'no(inf)'}/"
                  f"{'YES' if e_TV <= 0.219 else 'no(TV)'}")
        print()

    # Save raw.
    out = {'configs': configs, 'results': results}
    with open('_smoke_linfty_continuity_check.json', 'w') as fp:
        # Convert numpy floats to plain floats.
        def to_native(o):
            if isinstance(o, np.generic):
                return o.item()
            return o
        json.dump(out, fp, default=to_native, indent=2)
    print(f"\nRaw results in: _smoke_linfty_continuity_check.json")


if __name__ == '__main__':
    main()
