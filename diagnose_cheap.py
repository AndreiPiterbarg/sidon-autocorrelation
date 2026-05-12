"""Diagnose cheap-tier (natural/autoconv/McCormick) coverage by box width and d.

The actual BnB cascade tries cheap rigor BEFORE the LP. Most cert work is done
there. The earlier diagnostic missed this. Goal: identify whether the cheap
tier is failing at d=30 (and why), since that's where most certs happen.
"""
import numpy as np
import sys
import time

sys.path.insert(0, '.')
from interval_bnb.windows import build_windows
from interval_bnb.bound_eval import (
    batch_bounds_full, window_tensor,
    bound_natural_int_ge, bound_autoconv_int_ge,
    bound_mccormick_sw_int_ge, bound_mccormick_ne_int_ge,
    bound_mccormick_joint_face_dual_cert_int_ge,
)
from interval_bnb.box import SCALE as _SCALE


def to_int(arr):
    return [int(round(float(x) * _SCALE)) for x in arr]


def random_box(mu_c, hw, d, rng):
    offset = rng.standard_normal(d) * hw * 0.5
    center = np.maximum(0.0, np.minimum(0.5, mu_c + offset))
    lo = np.maximum(0.0, center - hw); hi = np.minimum(1.0, center + hw)
    hi[0] = min(hi[0], 0.5)
    return lo, hi


def main():
    target = 1.281
    target_num, target_den = 1281, 1000
    rng = np.random.default_rng(42)

    print('='*80)
    print('CHEAP-TIER cert rate by box width × position (50 random boxes per cell)')
    print('Bounds tested: natural, autoconv, McCormick SW, McCormick NE, joint-face')
    print('='*80)

    for d in [22, 24, 30]:
        data = np.load(f'mu_star_d{d}.npz', allow_pickle=True)
        mu = np.asarray(data['mu'])
        sigma_mu = mu[::-1].copy()
        windows = build_windows(d)
        A_tensor, scales = window_tensor(windows, d)

        print(f'\n=== d={d} (|W|={len(windows)}, mu_0={mu[0]:.3f}>{mu[-1]:.3f} so sigma_mu in H_d) ===')
        print(f'  hw     | natural% autoconv%   SW%    NE%   joint%  | ANY_cheap%   batch_full_pass%')

        for hw in [0.10, 0.05, 0.025, 0.01, 0.005]:
            n_trials = 30
            counts = {'nat':0, 'auto':0, 'sw':0, 'ne':0, 'joint':0, 'any':0, 'fastpass':0}
            n_valid = 0
            for _ in range(n_trials):
                lo, hi = random_box(sigma_mu, hw, d, rng)
                if lo.sum() > 1.0 or hi.sum() < 1.0:
                    continue
                n_valid += 1
                # batch_bounds_full to find the binding window
                lb_fast, w_idx, which, _, _ = batch_bounds_full(lo, hi, A_tensor, scales, target)
                pass_fast = (lb_fast >= target and w_idx >= 0)
                counts['fastpass'] += int(pass_fast)
                if not pass_fast:
                    continue
                w = windows[w_idx]
                lo_int, hi_int = to_int(lo), to_int(hi)
                if bound_natural_int_ge(lo_int, hi_int, w, target_num, target_den):
                    counts['nat'] += 1; counts['any'] += 1; continue
                if bound_autoconv_int_ge(lo_int, hi_int, w, d, target_num, target_den):
                    counts['auto'] += 1; counts['any'] += 1; continue
                if bound_mccormick_sw_int_ge(lo_int, hi_int, w, d, target_num, target_den):
                    counts['sw'] += 1; counts['any'] += 1; continue
                if bound_mccormick_ne_int_ge(lo_int, hi_int, w, d, target_num, target_den):
                    counts['ne'] += 1; counts['any'] += 1; continue
                if bound_mccormick_joint_face_dual_cert_int_ge(lo_int, hi_int, w, d, target_num, target_den):
                    counts['joint'] += 1; counts['any'] += 1; continue
            n = max(n_valid, 1)
            pct = lambda x: 100.0*x/n
            print(f'  {hw:.4f} |  {pct(counts["nat"]):5.1f}    {pct(counts["auto"]):5.1f}    {pct(counts["sw"]):5.1f}  {pct(counts["ne"]):5.1f}  {pct(counts["joint"]):5.1f}   | {pct(counts["any"]):5.1f}        {pct(counts["fastpass"]):5.1f}')

    print()
    print('='*80)
    print('What does this tell us?')
    print('  - "fastpass%" = boxes where lb_fast >= target (cheap bounds even attempted)')
    print('  - "ANY_cheap%" = boxes that pass the cheap rigor tier')
    print('  - The diff (fastpass - ANY_cheap) = boxes that lb_fast says yes but rigor refuses')
    print('  - If d=30 has lower fastpass% at moderate hw, the CHEAP tier is the binder')
    print('='*80)


if __name__ == '__main__':
    main()
