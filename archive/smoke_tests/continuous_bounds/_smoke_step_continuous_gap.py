"""Empirically measure the step-vs-continuous gap

  eps(d, m) := sup over continuous probability densities f on [-1/4, 1/4]
                with ||f||_inf <= M, integral f = 1, and bin averages = a_i/m
              of (||f_a * f_a||_inf - ||f * f||_inf)

where f_a is the step function with bin averages a, and a is fixed
(typically a "borderline" cell from the cascade).

Approach:
  1. Pick a step function with bin averages a (sum a = 4nm, n=d/2, so m fixes
     the cascade discretization).
  2. Parameterize CONTINUOUS f as piecewise-constant on a sub-grid (K subbins
     per big bin), with K large enough to approximate any continuous shape.
  3. Constraints (linear in heights):
       - h_l >= 0 for each subbin l
       - h_l <= M for each subbin l (the L^inf cap)
       - sum_{l in bin i} h_l / K = a_i / m (i.e., the bin average matches)
       - integral f = sum h_l * (hbin/K) = 1 (auto from sum of bin averages)
  4. Objective: minimize max over t-grid of (f * f)(t).  This is a non-convex
     QP since (f*f)(t) is a quadratic form in h (PSD only when f >= 0
     and t = 0 by Cauchy-Schwarz, but at general t the quadratic form
     in h is NOT PSD).  We solve via:
       (a) SLSQP from random starts (gradient-based local search)
       (b) Fourier-mode parameterization with scipy.optimize.differential_evolution

Then we report eps = ||f_a*f_a||_inf - min ||f*f||_inf.

We measure for (d, m) in {(2, 20), (4, 20), (8, 20)} (we drop d=16 to stay
within wall budget; piecewise local opt scales as O(d^2 K^2)).

We test across multiple step functions a and multiple ||f||_inf caps M.

USAGE:
  python _smoke_step_continuous_gap.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize, differential_evolution

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)


# =====================================================================
# Helpers
# =====================================================================
def step_max_ff(a, n_half, m):
    """||f_a * f_a||_inf for the step function with bin averages a.

    By the M-chain analysis: this equals max_k conv(a)[k] / (2 d m^2)
    where d = 2 n_half.
    """
    a = np.asarray(a, dtype=np.float64)
    d = len(a)
    conv = np.convolve(a, a)  # length 2d-1
    return float(np.max(conv)) / (2.0 * d * m * m)


def fff_sup_from_heights(h_sub, K, d, hbin, t_grid_size=None):
    """Compute ||f * f||_inf where f is piecewise constant on subgrid.

    h_sub is length d*K (heights on each subbin of width hbin/K).
    Returns max value of (f*f)(t) on a t-grid over [-1/2, 1/2].
    """
    hsub = hbin / K
    fff = np.convolve(h_sub, h_sub) * hsub  # length 2*d*K - 1
    return float(np.max(fff))


def fff_grid(h_sub, hbin, K):
    """Full convolution as numpy array (so we can extract max)."""
    hsub = hbin / K
    return np.convolve(h_sub, h_sub) * hsub


def heights_satisfying_avgs(a, m, K, d):
    """Return the step (uniform within bin) heights — initial point."""
    return np.repeat(np.asarray(a, dtype=np.float64) / m, K)


# =====================================================================
# Optimization: SLSQP with linear constraints
# =====================================================================
def project_to_avg_constraint(h, a, m, K, d):
    """Project h to satisfy bin averages, in-place version (returns new array).

    Each bin: (1/K) sum_{l in bin i} h_l = a_i / m
       => sum_{l in bin i} h_l = K * a_i / m
    Adjust by adding a constant to bin i so sum matches.
    """
    h = h.copy()
    for i in range(d):
        seg = h[i*K:(i+1)*K]
        cur = seg.sum()
        target = K * a[i] / m
        seg += (target - cur) / K
        h[i*K:(i+1)*K] = seg
    return h


def minimize_fff_inf_slsqp(
    a, n_half, m, M_cap, K=8, n_starts=8, seed=0, t_extra_grid=False, verbose=False
):
    """Minimize ||f*f||_inf over piecewise-constant f on K subbins/bin
    with bin avgs = a/m, 0 <= h <= M_cap.

    Method:
      Smoothed-max objective via log-sum-exp + L-BFGS-B with a sequence of
      sharpening betas; project onto bin-avg constraint after each phase.

    Returns: (best_fff_inf, best_h)
    """
    d = 2 * n_half
    hbin = 1.0 / (2.0 * d)
    n = d * K  # number of free heights
    hsub = hbin / K

    rng = np.random.default_rng(seed)

    def obj_max(h):
        fff = np.convolve(h, h) * hsub
        return float(np.max(fff))

    def obj_smooth(h, beta):
        fff = np.convolve(h, h) * hsub
        fmax = np.max(fff)
        return fmax + np.log(np.sum(np.exp(beta * (fff - fmax)))) / beta

    def obj_pen(h, beta, lam_avg):
        # Penalty for bin avg violations.
        val = obj_smooth(h, beta)
        pen = 0.0
        for i in range(d):
            e = np.sum(h[i*K:(i+1)*K]) / K - a[i] / m
            pen += lam_avg * e * e
        return val + pen

    bounds = [(0.0, M_cap)] * n

    best_val = np.inf
    best_h = None
    starts = []
    # Start 1: step function (uniform within bin)
    h0 = heights_satisfying_avgs(a, m, K, d)
    if np.max(h0) <= M_cap + 1e-9:
        starts.append(h0.copy())

    # Start 2-N: perturbed step
    for s in range(n_starts):
        h_pert = h0 + rng.normal(0, 1.0, size=n)
        h_pert = np.clip(h_pert, 0, M_cap)
        h_pert = project_to_avg_constraint(h_pert, a, m, K, d)
        h_pert = np.clip(h_pert, 0, M_cap)
        for _ in range(3):
            h_pert = project_to_avg_constraint(h_pert, a, m, K, d)
            h_pert = np.clip(h_pert, 0, M_cap)
        starts.append(h_pert)

    # Also try sin perturbations as starts (often found in early experiments):
    x_local = (np.arange(K) + 0.5) / K  # bin-fraction
    for amp in (0.7, 1.5):
        if amp > M_cap - max(a)/m:
            continue
        for freq in (1,):
            h_sin = h0.copy()
            for i in range(d):
                h_sin[i*K:(i+1)*K] += amp * np.sin(2 * np.pi * freq * x_local)
            h_sin = np.clip(h_sin, 0, M_cap)
            h_sin = project_to_avg_constraint(h_sin, a, m, K, d)
            h_sin = np.clip(h_sin, 0, M_cap)
            starts.append(h_sin)
    # Out-of-phase per-bin sin
    for amp in (0.7,):
        if amp > M_cap - max(a)/m:
            continue
        h_sin = h0.copy()
        for i in range(d):
            phase = i * np.pi  # alternating phase between bins
            h_sin[i*K:(i+1)*K] += amp * np.sin(2 * np.pi * x_local + phase)
        h_sin = np.clip(h_sin, 0, M_cap)
        h_sin = project_to_avg_constraint(h_sin, a, m, K, d)
        h_sin = np.clip(h_sin, 0, M_cap)
        starts.append(h_sin)

    for s_idx, h_init in enumerate(starts):
        try:
            h_cur = h_init.copy()
            for beta in (40.0, 200.0, 800.0):
                res = minimize(
                    obj_pen, h_cur, args=(beta, 1e4),
                    method='L-BFGS-B', bounds=bounds,
                    options={'maxiter': 150, 'ftol': 1e-9},
                )
                h_cur = res.x
            # Final projection + clip
            h_cur = project_to_avg_constraint(h_cur, a, m, K, d)
            h_cur = np.clip(h_cur, 0, M_cap)
            v = obj_max(h_cur)
            if verbose:
                print(f'  start {s_idx}: ||f*f||_inf = {v:.5f}')
            if v < best_val:
                best_val = v
                best_h = h_cur.copy()
        except Exception as e:
            if verbose:
                print(f'  start {s_idx}: FAIL {e}')
    return best_val, best_h


# =====================================================================
# Optimization: differential evolution on Fourier-mode parameterization
# =====================================================================
def heights_from_fourier(coefs, a, m, K, d):
    """Build heights from per-bin Fourier coefficients on top of step function.

    coefs has shape (d, 2 * F) for F frequencies (sin/cos pairs),
    representing sum over freq f=1..F of a_f * cos(2*pi*f*x/hbin) +
    b_f * sin(2*pi*f*x/hbin), evaluated at subbin centers.

    Note: cos has zero average over the bin only if the bin spans an integer
    number of periods, which is true here (period = hbin/f).  Same for sin.
    So bin averages are unaffected.

    Final heights = clip(step + perturbation, 0, M_cap) but here we just
    return raw; clipping is done outside.
    """
    h = np.repeat(a / m, K)
    F = coefs.shape[1] // 2
    # Subbin centers within one bin: (-0.5 + (l+0.5)/K) * hbin
    # i.e., from -hbin/2 + hbin/(2K) ... to hbin/2 - hbin/(2K)
    # Actually in absolute coords: bin i center = -1/4 + (i+0.5)*hbin
    #   subbin l in bin i has center = bin center + (-hbin/2 + (l+0.5) * hbin/K)
    # For Fourier we want a phase relative to the bin start:
    # x_local = (l + 0.5) / K  in [0, 1] (fraction of bin width)
    x_local = (np.arange(K) + 0.5) / K  # [0, 1]
    for i in range(d):
        for ff in range(1, F + 1):
            ac = coefs[i, 2 * (ff - 1)]
            bc = coefs[i, 2 * (ff - 1) + 1]
            h[i*K:(i+1)*K] += ac * np.cos(2 * np.pi * ff * x_local) + \
                              bc * np.sin(2 * np.pi * ff * x_local)
    return h


def minimize_fff_inf_fourier(
    a, n_half, m, M_cap, K=8, F=2, popsize=10, maxiter=20, seed=0, verbose=False
):
    """Differential evolution over per-bin Fourier coefficients.

    Returns: (best_fff_inf, best_h)
    """
    d = 2 * n_half
    hbin = 1.0 / (2.0 * d)
    a_arr = np.asarray(a, dtype=np.float64)

    # Bound coefs: each at most M_cap (but actually amp + step <= M_cap)
    # We'll use a soft penalty for overshoot, but also constrain coefs
    # to [-M_cap, M_cap].
    n_params = d * 2 * F

    def fobj(x):
        coefs = x.reshape(d, 2 * F)
        h = heights_from_fourier(coefs, a_arr, m, K, d)
        # Soft-penalty for h < 0 or h > M_cap:
        # We strictly clip to [0, M_cap] but record violation for penalty.
        viol_low = np.maximum(-h, 0).sum() * 100
        viol_high = np.maximum(h - M_cap, 0).sum() * 100
        h = np.clip(h, 0, M_cap)
        # After clipping, bin avgs may shift; project back.
        # That destroys the Fourier structure.  Instead just penalize.
        # Actually: enforce avgs by projection -> clipping cycle.
        for _ in range(3):
            for i in range(d):
                seg = h[i*K:(i+1)*K]
                cur = seg.sum()
                target = K * a_arr[i] / m
                seg += (target - cur) / K
                h[i*K:(i+1)*K] = seg
            h = np.clip(h, 0, M_cap)
        # Verify avgs
        avg_err = 0.0
        for i in range(d):
            avg_err += abs(np.sum(h[i*K:(i+1)*K]) / K - a_arr[i] / m)
        # Compute objective
        fff = np.convolve(h, h) * (hbin / K)
        val = float(np.max(fff))
        return val + viol_low + viol_high + avg_err * 100

    # Bounds: coefs in [-cap, cap] where cap = min(M_cap, 1.5*max(a)/m) is enough
    cap = min(M_cap, 3.0 * float(max(a_arr)) / m + 1.0)
    bounds = [(-cap, cap)] * n_params

    res = differential_evolution(
        fobj, bounds,
        popsize=popsize, maxiter=maxiter, seed=seed,
        polish=True, tol=1e-6, mutation=(0.4, 1.0), recombination=0.8,
        disp=verbose,
    )
    coefs = res.x.reshape(d, 2 * F)
    h = heights_from_fourier(coefs, a_arr, m, K, d)
    h = np.clip(h, 0, M_cap)
    for _ in range(5):
        for i in range(d):
            seg = h[i*K:(i+1)*K]
            cur = seg.sum()
            target = K * a_arr[i] / m
            seg += (target - cur) / K
            h[i*K:(i+1)*K] = seg
        h = np.clip(h, 0, M_cap)
    fff = np.convolve(h, h) * (hbin / K)
    val = float(np.max(fff))
    return val, h


# =====================================================================
# Main driver: measure eps(d, m) for various step functions
# =====================================================================
def measure_eps_for_a(a, n_half, m, M_caps=(1.5, 4.0, 10.0, 100.0),
                     K=12, F=3, n_slsqp_starts=6, n_seeds=1,
                     de_pop=12, de_iter=15, seed=0, verbose=False):
    """For step function with bin averages a, measure eps = step - min(continuous)
    for each M_cap.

    Multi-seed: run SLSQP-based optimization with n_seeds distinct seeds,
    take the best (= min) ||f*f||_inf found.

    Returns dict with per-M_cap results.
    """
    a = np.asarray(a, dtype=np.float64)
    d = 2 * n_half
    step_val = step_max_ff(a, n_half, m)

    # max bin height of the step function:
    step_max_height = float(np.max(a) / m)
    # Useful M_caps must be >= step_max_height to make the cell feasible.

    out = {
        'a': a.tolist(),
        'n_half': int(n_half),
        'm': int(m),
        'd': int(d),
        'step_ff_inf': step_val,
        'step_max_height': step_max_height,
        'per_M': {},
    }

    for M_cap in M_caps:
        if M_cap < step_max_height - 1e-9:
            # Step itself violates the L^inf cap; skip
            out['per_M'][str(M_cap)] = {
                'M_cap': float(M_cap),
                'feasible': False,
                'reason': f'step_max_height={step_max_height} > M_cap={M_cap}',
            }
            continue

        t0 = time.time()
        # SLSQP with multiple seeds
        v_sl_best = np.inf
        for s_off in range(n_seeds):
            v_sl, h_sl = minimize_fff_inf_slsqp(
                a, n_half, m, M_cap, K=K, n_starts=n_slsqp_starts,
                seed=seed + s_off, verbose=False)
            if v_sl < v_sl_best:
                v_sl_best = v_sl
        # Fourier-DE
        v_de, h_de = minimize_fff_inf_fourier(
            a, n_half, m, M_cap, K=K, F=F, popsize=de_pop, maxiter=de_iter,
            seed=seed + 1, verbose=False)
        # Best
        best_v = min(v_sl_best, v_de)
        eps = step_val - best_v
        elapsed = time.time() - t0
        out['per_M'][str(M_cap)] = {
            'M_cap': float(M_cap),
            'min_continuous_ff_inf': float(best_v),
            'eps': float(eps),
            'slsqp_val': float(v_sl_best),
            'fourier_de_val': float(v_de),
            'wall_s': float(elapsed),
        }
        if verbose:
            print(f'    M_cap={M_cap}: step={step_val:.4f}, SLSQP={v_sl_best:.4f}, '
                  f'DE={v_de:.4f}, best={best_v:.4f}, eps={eps:.4f} '
                  f'({elapsed:.1f}s)')
    return out


def main():
    t_start = time.time()
    print('=' * 72)
    print('SMOKE: empirical step-vs-continuous gap measurement')
    print('=' * 72)

    out = {'config': 'eps(d,m) for various (d,m) and step functions'}

    # M_cap series: include a low cap (close to step_max_height), normal,
    # and high (continuous limit).
    M_caps = (1.5, 4.0, 10.0, 100.0)
    out['M_caps_tested'] = list(M_caps)

    # ---------------- (d=2, m=20) ---------------------
    print('\n[d=2, m=20]')
    print('-' * 72)
    out['d2_m20'] = []
    cells_d2 = [
        np.array([40, 40]),    # central palindromic
        np.array([35, 45]),    # mildly asymmetric
        np.array([30, 50]),    # more asymmetric
        np.array([20, 60]),    # very asymmetric (= boundary case)
    ]
    for c in cells_d2:
        print(f'  cell c={c.tolist()}, S=4nm=80')
        r = measure_eps_for_a(
            c, n_half=1, m=20, M_caps=M_caps,
            K=24, F=2, n_slsqp_starts=4, n_seeds=3,
            de_pop=8, de_iter=10,
            seed=0, verbose=True)
        out['d2_m20'].append(r)

    # ---------------- (d=4, m=20) ---------------------
    print('\n[d=4, m=20]')
    print('-' * 72)
    out['d4_m20'] = []
    # n_half=2 -> S = 4*2*20 = 160
    cells_d4 = [
        np.array([40, 40, 40, 40]),    # uniform
        np.array([20, 60, 60, 20]),    # dual-peak symmetric
        np.array([20, 50, 50, 40]),    # mildly asymmetric
        np.array([0, 80, 80, 0]),      # extreme
    ]
    for c in cells_d4:
        print(f'  cell c={c.tolist()}, S=4nm=160')
        r = measure_eps_for_a(
            c, n_half=2, m=20, M_caps=M_caps,
            K=8, F=2, n_slsqp_starts=3, n_seeds=2,
            de_pop=6, de_iter=8,
            seed=0, verbose=True)
        out['d4_m20'].append(r)

    # ---------------- (d=8, m=20) ---------------------
    print('\n[d=8, m=20]')
    print('-' * 72)
    out['d8_m20'] = []
    # n_half=4 -> S = 4*4*20 = 320
    cells_d8 = [
        np.array([40, 40, 40, 40, 40, 40, 40, 40]),    # uniform
        np.array([20, 60, 40, 40, 40, 40, 60, 20]),    # symmetric perturbed
        np.array([0, 60, 80, 60, 60, 80, 60, 0]),      # peaked
    ]
    for c in cells_d8:
        print(f'  cell c={c.tolist()}, S=4nm=320')
        r = measure_eps_for_a(
            c, n_half=4, m=20, M_caps=M_caps,
            K=4, F=1, n_slsqp_starts=2, n_seeds=2,
            de_pop=4, de_iter=6,
            seed=0, verbose=True)
        out['d8_m20'].append(r)

    # ---------------- summary ---------------------
    print('\n' + '=' * 72)
    print('SUMMARY: eps(d, m) for various cells, at M_cap = 100')
    print('=' * 72)

    summary = {}
    for tag, lst in [('d2_m20', out['d2_m20']),
                      ('d4_m20', out['d4_m20']),
                      ('d8_m20', out['d8_m20'])]:
        max_eps_high_M = -np.inf
        eps_uniform_cell_high_M = None
        for r in lst:
            a = r['a']
            is_uniform = all(abs(x - a[0]) < 1e-9 for x in a)
            for M_str, res in r['per_M'].items():
                if not res.get('feasible', True):
                    continue
                if 'eps' in res:
                    if res['M_cap'] >= 50:
                        max_eps_high_M = max(max_eps_high_M, res['eps'])
                        if is_uniform:
                            eps_uniform_cell_high_M = res['eps']
        summary[tag] = {
            'max_eps_high_M_cap': float(max_eps_high_M),
            'eps_uniform_cell_high_M_cap': (
                float(eps_uniform_cell_high_M)
                if eps_uniform_cell_high_M is not None else None),
        }
        print(f'  {tag}: '
              f'max eps (M=large) = {max_eps_high_M:.4f}, '
              f'uniform-cell eps = '
              f'{eps_uniform_cell_high_M:.4f}'
              if eps_uniform_cell_high_M is not None else
              f'  {tag}: max eps = {max_eps_high_M:.4f}')
    out['summary_eps'] = summary

    # Trend on UNIFORM cell (apples-to-apples across d, since step_value=2)
    eps_d2 = summary['d2_m20']['eps_uniform_cell_high_M_cap']
    eps_d4 = summary['d4_m20']['eps_uniform_cell_high_M_cap']
    eps_d8 = summary['d8_m20']['eps_uniform_cell_high_M_cap']
    if eps_d2 is None or eps_d4 is None or eps_d8 is None:
        trend = 'UNDEFINED'
    elif eps_d4 < 0.9 * eps_d2 and eps_d8 < 0.9 * eps_d4:
        trend = 'SHRINKING'
    elif eps_d4 > 1.1 * eps_d2 or eps_d8 > 1.1 * eps_d4:
        trend = 'GROWING'
    else:
        trend = 'CONSTANT'
    out['trend'] = trend

    elapsed = time.time() - t_start
    print(f'\nTotal wall: {elapsed:.1f}s')
    print(f'EPSILON_EMPIRICAL: '
          f'eps(2,20)={eps_d2:.3f}, eps(4,20)={eps_d4:.3f}, '
          f'eps(8,20)={eps_d8:.3f}; trend = {trend}')

    out['elapsed_s'] = float(elapsed)
    out_path = os.path.join(_dir, '_smoke_step_continuous_gap.json')
    with open(out_path, 'w') as fp:
        json.dump(out, fp, indent=2, default=float)
    print(f'\n[saved] {out_path}')
    return out


if __name__ == '__main__':
    main()
