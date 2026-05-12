"""Comprehensive diagnostic of d=30 BnB failure mode.

Tests:
  A) Cert rate by box width and position (Monte Carlo around sigma(mu*))
  B) Per-tier breakdown (multi-anchor, epi LP, centroid)
  C) Compare to d=22, d=24 cert profiles
  D) Lasserre SDP cert at wide d=30 boxes (sanity check on Fix 1)
  E) Per-axis split contribution analysis on stuck d=30 box
  F) Tree-size simulation using observed cert rates
"""
import numpy as np
import sys
import time
import json

sys.path.insert(0, '.')
from interval_bnb.windows import build_windows
from interval_bnb.bound_epigraph import _solve_epigraph_lp
from interval_bnb.bound_anchor import (
    build_multi_anchor_data, bound_anchor_multi_int_ge,
    build_centroid_anchor_cache, bound_anchor_centroid_int_ge,
)
from interval_bnb.box import SCALE as _SCALE


def make_box(mu_c, hw, d, rng=None):
    """Random box of half-width hw around mu_c, projected to box-feasible."""
    if rng is not None:
        offset = rng.standard_normal(d) * hw * 0.5
    else:
        offset = np.zeros(d)
    center = np.maximum(0.0, np.minimum(0.5, mu_c + offset))
    lo = np.maximum(0.0, center - hw)
    hi = np.minimum(1.0, center + hw)
    hi[0] = min(hi[0], 0.5)
    if lo.sum() > 1.0:
        lo = np.maximum(0.0, mu_c - hw); lo[0] = min(lo[0], 0.5)
    return lo, hi


def to_int(arr):
    return [int(round(float(x) * _SCALE)) for x in arr]


def run_cert_tiers(lo, hi, windows, d, target, anchors, cache):
    """Run all cert tiers on a single box, return dict of which fired."""
    results = {}
    lo_int = to_int(lo); hi_int = to_int(hi)
    target_num = int(round(target * 10000)); target_den = 10000

    # Multi-anchor
    t0 = time.time()
    cert_anchor = bound_anchor_multi_int_ge(lo_int, hi_int, anchors, target_num, target_den)
    results['anchor_cert'] = cert_anchor
    results['anchor_ms'] = (time.time()-t0)*1000

    # Centroid
    t0 = time.time()
    cert_centroid = bound_anchor_centroid_int_ge(lo_int, hi_int, target_num, target_den, cache)
    results['centroid_cert'] = cert_centroid
    results['centroid_ms'] = (time.time()-t0)*1000

    # Epi LP
    t0 = time.time()
    lp_val, *_ = _solve_epigraph_lp(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64), windows, d)
    results['lp_val'] = float(lp_val)
    results['lp_cert'] = bool(lp_val >= target)
    results['lp_ms'] = (time.time()-t0)*1000

    return results


def main():
    target = 1.281
    rng = np.random.default_rng(42)

    print('='*80)
    print('TEST A: Cert rate by box width × position (50 random boxes per cell)')
    print('='*80)

    summary = {}
    for d in [22, 24, 30]:
        print(f'\n=== d={d} ===')
        data = np.load(f'mu_star_d{d}.npz', allow_pickle=True)
        mu = np.asarray(data['mu'])
        sigma_mu = mu[::-1].copy()
        f_max = float(data['f'])
        margin = f_max - target

        windows = build_windows(d)
        anchors = build_multi_anchor_data(d, mu, windows=windows)
        cache = build_centroid_anchor_cache(d, windows=windows)

        print(f'f(mu*)={f_max:.4f}, margin={margin:.4f}, |W|={len(windows)}')
        print(f'  hw     |  anchor%  centroid%  LP%  any_cert%  | LP_ms_avg')

        d_summary = {}
        for hw in [0.05, 0.025, 0.01, 0.005, 0.002]:
            n_trials = 30 if hw >= 0.025 else 15  # cheap LP at small boxes
            stats = {'anchor':0, 'centroid':0, 'lp':0, 'any':0}
            lp_ms_sum = 0
            for _ in range(n_trials):
                lo, hi = make_box(sigma_mu, hw, d, rng)
                if lo.sum() > 1.0 or hi.sum() < 1.0:
                    continue
                r = run_cert_tiers(lo, hi, windows, d, target, anchors, cache)
                stats['anchor'] += int(r['anchor_cert'])
                stats['centroid'] += int(r['centroid_cert'])
                stats['lp'] += int(r['lp_cert'])
                stats['any'] += int(r['anchor_cert'] or r['centroid_cert'] or r['lp_cert'])
                lp_ms_sum += r['lp_ms']
            n = n_trials
            print(f'  {hw:.4f} |  {100*stats["anchor"]/n:5.1f}%   {100*stats["centroid"]/n:5.1f}%   {100*stats["lp"]/n:5.1f}%   {100*stats["any"]/n:5.1f}%   | {lp_ms_sum/n:.0f}')
            d_summary[hw] = {k: stats[k]/n for k in stats}
        summary[d] = d_summary

    print()
    print('='*80)
    print('TEST B: Tree-size simulation (random walk, expected nodes to drain)')
    print('='*80)
    print('  Assuming uniform box width as simulation parameter, track in_flight')
    print('  random walk with cert prob p (from Test A "any_cert%" at width hw)')
    print()

    for d in [22, 24, 30]:
        print(f'  d={d}:')
        for hw in [0.05, 0.025, 0.01]:
            p = summary[d].get(hw, {}).get('any', 0)
            if p < 0.5 + 1e-3:
                drift = 1 - 2*p
                print(f'    hw={hw}: cert_rate={100*p:5.1f}%, drift=+{drift:.3f}/box → tree GROWS')
            else:
                drift = 2*p - 1
                expected_drain_per_500 = int(500 / drift)
                print(f'    hw={hw}: cert_rate={100*p:5.1f}%, drain rate={drift:.3f}/box → drain {expected_drain_per_500} nodes per 500 in_flight')

    print()
    print('='*80)
    print('TEST C: WHERE does d=30 fail vs d=22? Check failed-LP boxes structurally.')
    print('='*80)

    for d in [22, 30]:
        data = np.load(f'mu_star_d{d}.npz', allow_pickle=True)
        mu = np.asarray(data['mu'])
        sigma_mu = mu[::-1].copy()
        windows = build_windows(d)

        # Find a box at hw=0.025 where LP fails
        for trial in range(20):
            lo, hi = make_box(sigma_mu, 0.025, d, rng)
            if lo.sum() > 1.0 or hi.sum() < 1.0:
                continue
            lp_val, *_ = _solve_epigraph_lp(np.asarray(lo,dtype=np.float64), np.asarray(hi,dtype=np.float64), windows, d)
            if lp_val < target:
                # Found a failing LP. Compute the McCormick gap per axis.
                center = (lo + hi) / 2
                f_center = max(w.scale*sum(float(center[i])*float(center[j]) for (i,j) in w.pairs_all) for w in windows)
                print(f'\n  d={d} failing box: hw=0.025, f(center)={f_center:.4f}, LP={lp_val:.4f}, gap={f_center-lp_val:.4f}')
                # Per-axis "would-LP-go-up-if-tightened" estimate
                w_axis = (hi - lo)
                axis_score = w_axis  # widest first
                worst = np.argsort(-axis_score)[:5]
                print(f'    widest axes (top 5): {worst.tolist()}, widths: {[float(w_axis[i]) for i in worst]}')
                break

    print()
    print('='*80)
    print('TEST D: Effective dimension — how many axes are "important"?')
    print('='*80)

    for d in [22, 24, 30]:
        data = np.load(f'mu_star_d{d}.npz', allow_pickle=True)
        mu = np.asarray(data['mu'])
        zeros_strict = (np.abs(mu) < 1e-6).sum()
        eps_neighbors = (np.abs(mu) < 0.001).sum()
        large_entries = (np.abs(mu) > 0.05).sum()
        print(f'  d={d}: # axes with |mu_i| < 1e-6: {zeros_strict}, < 1e-3: {eps_neighbors}, > 0.05: {large_entries}')
        print(f'    effective dim (axes with > 1e-3 mass): {d - eps_neighbors}')

    print()
    print('='*80)
    print('SUMMARY')
    print('='*80)
    for d in [22, 24, 30]:
        print(f'\n  d={d} cert profile:')
        for hw, p in summary[d].items():
            tag = '✓ DRAINS' if p['any'] > 0.5 else '✗ GROWS' if p['any'] < 0.45 else '⚠ MARGINAL'
            print(f'    hw={hw}: any-cert={100*p["any"]:5.1f}% (anchor={100*p["anchor"]:.1f}, centroid={100*p["centroid"]:.1f}, LP={100*p["lp"]:.1f}) {tag}')

    # Save full data
    json_summary = {str(d): {str(hw): v for hw, v in s.items()} for d, s in summary.items()}
    with open('diagnose_d30_summary.json', 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f'\nFull data → diagnose_d30_summary.json')


if __name__ == '__main__':
    main()
