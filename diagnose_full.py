"""Test cert rates on RANDOM H_d boxes (not just near sigma(mu*)).

The BnB tree visits boxes throughout H_d, not just the hard region near
sigma(mu*). Most easy boxes (concentrated mass, high f(center)) get certified
by cheap rigor. The relative balance is what determines throughput.
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
from interval_bnb.bound_epigraph import _solve_epigraph_lp
from interval_bnb.box import SCALE as _SCALE


def to_int(arr):
    return [int(round(float(x) * _SCALE)) for x in arr]


def random_simplex_box(hw, d, rng):
    """Random box of half-width hw with center sampled from H_d simplex."""
    # Center: random Dirichlet, projected to H_d (mu_0 <= mu_{d-1})
    for _ in range(20):
        mu_c = rng.dirichlet(np.ones(d))
        if mu_c[0] <= mu_c[-1]:
            break
        mu_c = mu_c[::-1]  # flip if not in H_d
    lo = np.maximum(0.0, mu_c - hw); hi = np.minimum(1.0, mu_c + hw)
    hi[0] = min(hi[0], 0.5)
    return lo, hi


def main():
    target = 1.281
    target_num, target_den = 1281, 1000
    rng = np.random.default_rng(0)

    print('='*80)
    print('CERT RATES on RANDOM H_d boxes (full search space, not near sigma(mu*))')
    print('  100 random boxes per cell')
    print('  Cheap tier: natural / autoconv / SW / NE / joint-face')
    print('  Plus epi LP if cheap fails')
    print('='*80)

    for d in [22, 24, 30]:
        windows = build_windows(d)
        A_tensor, scales = window_tensor(windows, d)

        print(f'\n=== d={d} (|W|={len(windows)}) ===')
        print(f'  hw     | cheap_pass%   LP_pass%   ANY_pass%  | LP_ms_avg')

        for hw in [0.10, 0.05, 0.025, 0.01]:
            n_trials = 100 if hw >= 0.025 else 50
            n_valid = 0
            counts = {'cheap':0, 'lp':0, 'any':0}
            lp_ms = 0
            for _ in range(n_trials):
                lo, hi = random_simplex_box(hw, d, rng)
                if lo.sum() > 1.0 or hi.sum() < 1.0:
                    continue
                n_valid += 1
                # cheap tier
                lo_int, hi_int = to_int(lo), to_int(hi)
                lb_fast, w_idx, _, _, _ = batch_bounds_full(lo, hi, A_tensor, scales, target)
                cheap_cert = False
                if lb_fast >= target and w_idx >= 0:
                    w = windows[w_idx]
                    if (bound_natural_int_ge(lo_int, hi_int, w, target_num, target_den) or
                        bound_autoconv_int_ge(lo_int, hi_int, w, d, target_num, target_den) or
                        bound_mccormick_sw_int_ge(lo_int, hi_int, w, d, target_num, target_den) or
                        bound_mccormick_ne_int_ge(lo_int, hi_int, w, d, target_num, target_den) or
                        bound_mccormick_joint_face_dual_cert_int_ge(lo_int, hi_int, w, d, target_num, target_den)):
                        cheap_cert = True
                if cheap_cert:
                    counts['cheap'] += 1; counts['any'] += 1
                    continue
                # LP
                t0 = time.time()
                lp_val, *_ = _solve_epigraph_lp(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64), windows, d)
                lp_ms += (time.time()-t0)*1000
                if lp_val >= target:
                    counts['lp'] += 1; counts['any'] += 1
            n = max(n_valid, 1)
            pct = lambda x: 100.0*x/n
            avg_lp = lp_ms / max(n_valid - counts['cheap'], 1)
            print(f'  {hw:.4f} |  {pct(counts["cheap"]):5.1f}        {pct(counts["lp"]):5.1f}      {pct(counts["any"]):5.1f}     | {avg_lp:.0f}')


if __name__ == '__main__':
    main()
