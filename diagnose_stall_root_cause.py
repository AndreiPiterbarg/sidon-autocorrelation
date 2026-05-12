"""COMPREHENSIVE root-cause diagnostic for d=22 BnB stall.

Seven tests to determine EXACTLY what's holding us back:

  Test 1: Margin scan — find empirical drain threshold over target ∈ [1.260, 1.285]
  Test 2: LP residual distribution at hw=0.005 (surviving box width)
  Test 3: Per-tier deep dive on 10 LP-failing boxes
  Test 4: Lasserre-2 SDP feasibility at d=22 (smaller than d=30, may work)
  Test 5: Per-axis "if-tightened" sensitivity analysis
  Test 6: LP value as function of box position (level-set valley mapping)
  Test 7: Geographic distribution of stuck boxes
"""
import numpy as np
import sys
import time
import json

sys.path.insert(0, '.')
from interval_bnb.windows import build_windows
from interval_bnb.bound_eval import (
    batch_bounds_full, window_tensor,
    bound_natural_int_ge, bound_autoconv_int_ge,
    bound_mccormick_sw_int_ge, bound_mccormick_ne_int_ge,
    bound_mccormick_joint_face_dual_cert_int_ge,
)
from interval_bnb.bound_epigraph import _solve_epigraph_lp
from interval_bnb.bound_anchor import (
    build_multi_anchor_data, bound_anchor_multi_int_ge,
    build_centroid_anchor_cache, bound_anchor_centroid_int_ge,
)
from interval_bnb.box import SCALE as _SCALE


def to_int(arr):
    return [int(round(float(x) * _SCALE)) for x in arr]


def random_box_around(mu_c, hw, d, rng):
    """Random box of half-width hw with center sampled near mu_c (Gaussian noise)."""
    offset = rng.standard_normal(d) * hw * 1.0
    center = np.maximum(0.0, np.minimum(0.5, mu_c + offset))
    lo = np.maximum(0.0, center - hw); hi = np.minimum(1.0, center + hw)
    hi[0] = min(hi[0], 0.5)
    return lo, hi


def f_at(mu, windows):
    return max(w.scale * sum(float(mu[i])*float(mu[j]) for (i,j) in w.pairs_all) for w in windows)


def load_d22():
    data = np.load('mu_star_d22.npz', allow_pickle=True)
    mu = np.asarray(data['mu'])
    sigma_mu = mu[::-1].copy()
    f_max = float(data['f'])
    windows = build_windows(22)
    A_tensor, scales = window_tensor(windows, 22)
    anchors = build_multi_anchor_data(22, mu, windows=windows)
    cache = build_centroid_anchor_cache(22, windows=windows)
    return dict(mu=mu, sigma_mu=sigma_mu, f_max=f_max, windows=windows,
                A_tensor=A_tensor, scales=scales, anchors=anchors, cache=cache)


def cert_box(lo, hi, ctx, target, target_num=1281, target_den=1000):
    """Run ALL cert tiers, return dict of which fired and per-tier LBs."""
    windows = ctx['windows']
    d = 22
    lo_int = to_int(lo); hi_int = to_int(hi)

    # Cheap tier
    lb_fast, w_idx, _, _, _ = batch_bounds_full(lo, hi, ctx['A_tensor'], ctx['scales'], target)
    cheap_path = None
    if lb_fast >= target and w_idx >= 0:
        w = windows[w_idx]
        if bound_natural_int_ge(lo_int, hi_int, w, target_num, target_den): cheap_path = 'natural'
        elif bound_autoconv_int_ge(lo_int, hi_int, w, d, target_num, target_den): cheap_path = 'autoconv'
        elif bound_mccormick_sw_int_ge(lo_int, hi_int, w, d, target_num, target_den): cheap_path = 'sw'
        elif bound_mccormick_ne_int_ge(lo_int, hi_int, w, d, target_num, target_den): cheap_path = 'ne'
        elif bound_mccormick_joint_face_dual_cert_int_ge(lo_int, hi_int, w, d, target_num, target_den): cheap_path = 'joint'

    # Multi-anchor
    anchor_cert = bound_anchor_multi_int_ge(lo_int, hi_int, ctx['anchors'], target_num, target_den)

    # Centroid
    centroid_cert = bound_anchor_centroid_int_ge(lo_int, hi_int, target_num, target_den, ctx['cache'])

    # Epi LP
    lp_val, *_ = _solve_epigraph_lp(np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64), windows, d)

    return dict(
        lb_fast=float(lb_fast),
        cheap_path=cheap_path,
        cheap_cert=cheap_path is not None,
        anchor_cert=bool(anchor_cert),
        centroid_cert=bool(centroid_cert),
        lp_val=float(lp_val),
        lp_cert=bool(lp_val >= target),
        any_cert=bool((cheap_path is not None) or anchor_cert or centroid_cert or (lp_val >= target)),
    )


def main():
    rng = np.random.default_rng(0)
    ctx = load_d22()
    d = 22
    print(f'\n{"="*80}')
    print(f'd=22 stall root-cause diagnostic. f(mu*) = {ctx["f_max"]:.6f}')
    print(f'{"="*80}\n')

    # ========================================================================
    # TEST 1: Margin scan — what target makes the cascade drain?
    # ========================================================================
    print(f'\n{"="*80}\nTEST 1: Margin scan (cert rate at hw=0.005, sigma(mu*)-region, 100 boxes)\n{"="*80}')
    targets = [1.260, 1.265, 1.270, 1.275, 1.278, 1.280, 1.281, 1.2805, 1.282]
    for target in targets:
        target_num = int(round(target * 10000))
        target_den = 10000
        # pre-generate same boxes
        rng_t = np.random.default_rng(0)
        certs = {'cheap':0, 'anchor':0, 'centroid':0, 'lp':0, 'any':0}
        n_valid = 0
        for _ in range(100):
            lo, hi = random_box_around(ctx['sigma_mu'], 0.005, d, rng_t)
            if lo.sum() > 1.0 or hi.sum() < 1.0:
                continue
            n_valid += 1
            r = cert_box(lo, hi, ctx, target, target_num, target_den)
            for k in ('cheap_cert','anchor_cert','centroid_cert','lp_cert','any_cert'):
                key = k.replace('_cert','')
                if r[k]: certs[key] += 1
        n = max(n_valid, 1)
        pct = lambda x: 100*x/n
        drift_str = '✓ DRAINS' if certs['any']/n > 0.55 else '✗ STALLS' if certs['any']/n < 0.45 else '⚠ MARGINAL'
        print(f'  target={target:.4f}: cheap={pct(certs["cheap"]):5.1f}%  anchor={pct(certs["anchor"]):5.1f}%  centroid={pct(certs["centroid"]):5.1f}%  LP={pct(certs["lp"]):5.1f}%  ANY={pct(certs["any"]):5.1f}% {drift_str}')

    # ========================================================================
    # TEST 2: LP residual distribution at hw=0.005 (surviving box width)
    # ========================================================================
    print(f'\n{"="*80}\nTEST 2: LP residual distribution (200 boxes, sigma(mu*)-region, hw=0.005)\n{"="*80}')
    lp_vals = []
    rng_t = np.random.default_rng(1)
    for _ in range(200):
        lo, hi = random_box_around(ctx['sigma_mu'], 0.005, d, rng_t)
        if lo.sum() > 1.0 or hi.sum() < 1.0: continue
        lp_val, *_ = _solve_epigraph_lp(np.asarray(lo, dtype=np.float64),
                                          np.asarray(hi, dtype=np.float64),
                                          ctx['windows'], d)
        lp_vals.append(float(lp_val))
    lp_vals = np.array(lp_vals)
    print(f'  n_boxes = {len(lp_vals)}')
    print(f'  LP_value: min={lp_vals.min():.6f}, p10={np.percentile(lp_vals,10):.6f}, '
          f'p25={np.percentile(lp_vals,25):.6f}, p50={np.percentile(lp_vals,50):.6f}, '
          f'p75={np.percentile(lp_vals,75):.6f}, p90={np.percentile(lp_vals,90):.6f}, max={lp_vals.max():.6f}')
    for target in [1.270, 1.275, 1.278, 1.280, 1.281]:
        cert_pct = 100 * (lp_vals >= target).mean()
        print(f'  target={target:.4f}: LP cert rate={cert_pct:.1f}%')

    # ========================================================================
    # TEST 3: Per-tier deep dive on 10 LP-failing boxes at target=1.281
    # ========================================================================
    print(f'\n{"="*80}\nTEST 3: Per-tier deep dive (10 LP-failing boxes at target=1.281)\n{"="*80}')
    target = 1.281
    rng_t = np.random.default_rng(2)
    failing = []
    for _ in range(50):
        if len(failing) >= 10: break
        lo, hi = random_box_around(ctx['sigma_mu'], 0.005, d, rng_t)
        if lo.sum() > 1.0 or hi.sum() < 1.0: continue
        r = cert_box(lo, hi, ctx, target)
        if not r['lp_cert']:
            r['box_lo'] = lo; r['box_hi'] = hi
            failing.append(r)
    print(f'  Found {len(failing)} LP-failing boxes')
    for k, r in enumerate(failing):
        ctr = (r['box_lo'] + r['box_hi'])/2
        f_c = f_at(ctr, ctx['windows'])
        gap = f_c - r['lp_val']
        margin_to_t = r['lp_val'] - target
        print(f'  Box #{k}: f(center)={f_c:.4f}, LP={r["lp_val"]:.4f}, gap={gap:.4f}, LP_residual_to_target={margin_to_t:+.4f}')
        print(f'    cheap={r["cheap_path"]}  anchor={r["anchor_cert"]}  centroid={r["centroid_cert"]}')

    # ========================================================================
    # TEST 4: Lasserre-2 SDP feasibility at d=22 hw=0.005
    # ========================================================================
    print(f'\n{"="*80}\nTEST 4: Lasserre-2 SDP via CVXPY+MOSEK at d=22 hw=0.005 (5 boxes, 60s timeout)\n{"="*80}')
    try:
        from interval_bnb.lasserre_cert import lasserre_box_lb_float
        n_test = 5
        sdp_results = []
        for k in range(min(n_test, len(failing))):
            r = failing[k]
            t0 = time.time()
            try:
                sdp_lb = lasserre_box_lb_float(
                    r['box_lo'], r['box_hi'], ctx['windows'], d,
                    order=2, solver='MOSEK', verbose=False
                )
                elapsed = time.time() - t0
                cert = sdp_lb >= target
                sdp_results.append((k, sdp_lb, cert, elapsed, None))
                print(f'  Box #{k}: LP={r["lp_val"]:.4f} → SDP_LB={sdp_lb:.4f} cert={cert} ({elapsed:.1f}s)')
            except Exception as e:
                print(f'  Box #{k}: SDP FAILED: {type(e).__name__}: {str(e)[:80]}')
                sdp_results.append((k, None, False, time.time()-t0, str(e)))
            if time.time() - t0 > 60:
                print('  Timeout, aborting SDP test'); break
        sdp_certs = sum(1 for r in sdp_results if r[2])
        print(f'  Lasserre-2 SDP cert rate at d=22 hw=0.005: {sdp_certs}/{len(sdp_results)}')
    except Exception as e:
        print(f'  SDP test setup failed: {e}')

    # ========================================================================
    # TEST 5: Per-axis sensitivity (which axis would help most if tightened?)
    # ========================================================================
    print(f'\n{"="*80}\nTEST 5: Per-axis "if-tightened" sensitivity (3 LP-failing boxes)\n{"="*80}')
    for k, r in enumerate(failing[:3]):
        lo, hi = r['box_lo'], r['box_hi']
        print(f'\n  Box #{k}: original LP={r["lp_val"]:.4f}, target={target}')
        deltas = []
        for axis in range(d):
            mid = (lo[axis]+hi[axis])/2
            # tighten this axis by half (low half)
            lo_half = lo.copy(); hi_half = hi.copy(); hi_half[axis] = mid
            if lo_half.sum() <= 1.0 and hi_half.sum() >= 1.0:
                lp_val_half, *_ = _solve_epigraph_lp(np.asarray(lo_half,dtype=np.float64),
                                                       np.asarray(hi_half,dtype=np.float64), ctx['windows'], d)
                deltas.append((axis, float(lp_val_half - r['lp_val']), 'L'))
            # high half
            lo_half = lo.copy(); hi_half = hi.copy(); lo_half[axis] = mid
            if lo_half.sum() <= 1.0 and hi_half.sum() >= 1.0:
                lp_val_half, *_ = _solve_epigraph_lp(np.asarray(lo_half,dtype=np.float64),
                                                       np.asarray(hi_half,dtype=np.float64), ctx['windows'], d)
                deltas.append((axis, float(lp_val_half - r['lp_val']), 'H'))
        deltas.sort(key=lambda x:-x[1])
        print(f'    Top-5 best axes to split:')
        for ax, dlt, side in deltas[:5]:
            print(f'      axis {ax} ({side}-half): LP improves by {dlt:+.4f}')

    # ========================================================================
    # TEST 6: LP value as function of box position (level-set sweep)
    # ========================================================================
    print(f'\n{"="*80}\nTEST 6: Level-set sweep (LP value along path from sigma(mu*) to random direction)\n{"="*80}')
    rng_t = np.random.default_rng(3)
    direction = rng_t.standard_normal(d)
    direction[0] = -abs(direction[0])  # keep mu_0 small for H_d
    direction /= np.linalg.norm(direction)
    print(f'  Distance along direction:')
    for t_step in [0, 0.005, 0.01, 0.02, 0.05]:
        center = ctx['sigma_mu'] + t_step * direction
        center = np.maximum(0.0, np.minimum(0.5, center))
        # Tighten to feasibility
        if center.sum() < 1.0:
            center = center / center.sum()
        hw = 0.005
        lo = np.maximum(0.0, center - hw); hi = np.minimum(1.0, center + hw)
        hi[0] = min(hi[0], 0.5)
        if lo.sum() > 1.0 or hi.sum() < 1.0:
            print(f'    t={t_step}: infeasible'); continue
        lp_val, *_ = _solve_epigraph_lp(np.asarray(lo,dtype=np.float64), np.asarray(hi,dtype=np.float64), ctx['windows'], d)
        f_c = f_at(center, ctx['windows'])
        print(f'    t={t_step:.4f}: f(center)={f_c:.4f}, LP={float(lp_val):.4f}, f-LP_gap={f_c-float(lp_val):.4e}')

    # ========================================================================
    # TEST 7: Geographic distribution — where do stuck boxes cluster?
    # ========================================================================
    print(f'\n{"="*80}\nTEST 7: Geographic distribution (200 boxes, distance to sigma(mu*) vs cert)\n{"="*80}')
    target = 1.281
    rng_t = np.random.default_rng(4)
    by_distance = {0: {'n':0, 'cert':0}, 1: {'n':0, 'cert':0}, 2: {'n':0, 'cert':0}, 3: {'n':0, 'cert':0}}
    for _ in range(200):
        # random box anywhere in H_d
        for _r in range(20):
            mu_c = rng_t.dirichlet(np.ones(d))
            if mu_c[0] <= mu_c[-1]: break
            mu_c = mu_c[::-1]
        lo = np.maximum(0.0, mu_c - 0.005); hi = np.minimum(1.0, mu_c + 0.005)
        hi[0] = min(hi[0], 0.5)
        if lo.sum() > 1.0 or hi.sum() < 1.0: continue
        dist = np.linalg.norm(mu_c - ctx['sigma_mu'])
        bucket = min(3, int(dist / 0.05))  # 0=close, 3=far
        by_distance[bucket]['n'] += 1
        r = cert_box(lo, hi, ctx, target)
        if r['any_cert']: by_distance[bucket]['cert'] += 1
    for b in sorted(by_distance.keys()):
        d_lo, d_hi = b*0.05, (b+1)*0.05 if b<3 else float('inf')
        n = by_distance[b]['n']
        rate = 100*by_distance[b]['cert']/max(n,1)
        print(f'  distance to sigma(mu*) ∈ [{d_lo:.2f}, {d_hi}): n={n}, cert rate={rate:.1f}%')

    print(f'\n{"="*80}\nDIAGNOSTIC COMPLETE\n{"="*80}\n')


if __name__ == '__main__':
    main()
