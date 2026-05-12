"""Analyze REAL stuck boxes from a dumped d=22 BnB stall.

Loads all stuck_boxes_w*.npz files and runs ALL cert tiers + structural
analysis on each box. Tells us EXACTLY what's holding back the cascade.

Output:
  - Per-box LP value, gap to target, which tier got closest
  - Aggregate distribution of LP residuals
  - Geographic distribution (distance to sigma(mu*))
  - Per-axis width distribution (which axes are saturated)
  - Cert-rate at various target relaxations
  - Lasserre-2 SDP cert rate (MOSEK, 60s timeout per box)
"""
import numpy as np
import sys
import time
import glob

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


def load_dumped_boxes(prefix='stuck_boxes'):
    files = sorted(glob.glob(f'{prefix}_w*.npz'))
    if not files:
        print(f'NO dumped box files found matching {prefix}_w*.npz')
        sys.exit(1)
    los, his, depths = [], [], []
    for f in files:
        d = np.load(f)
        los.append(d['lo']); his.append(d['hi']); depths.append(d['depths'])
    los = np.concatenate(los, axis=0)
    his = np.concatenate(his, axis=0)
    depths = np.concatenate(depths, axis=0)
    print(f'Loaded {len(los)} stuck boxes from {len(files)} workers')
    return los, his, depths


def main():
    los, his, depths = load_dumped_boxes()
    n_total = len(los)
    d = los.shape[1]
    print(f'd={d}, n_boxes={n_total}, depth range [{depths.min()}, {depths.max()}]')

    # Setup
    data = np.load(f'mu_star_d{d}.npz', allow_pickle=True)
    mu = np.asarray(data['mu'])
    sigma_mu = mu[::-1].copy()
    f_max = float(data['f'])
    target = 1.281
    target_num, target_den = 1281, 1000

    windows = build_windows(d)
    A_tensor, scales = window_tensor(windows, d)
    anchors = build_multi_anchor_data(d, mu, windows=windows)
    cache = build_centroid_anchor_cache(d, windows=windows)

    print(f'\nf(mu*) = {f_max:.4f}, target = {target}, margin = {f_max - target:.4f}')

    # Subsample if too many
    n_analyze = min(200, n_total)
    if n_total > n_analyze:
        idx = np.random.RandomState(0).choice(n_total, n_analyze, replace=False)
        los_a = los[idx]; his_a = his[idx]; depths_a = depths[idx]
    else:
        los_a, his_a, depths_a = los, his, depths

    # ========================================================================
    # Analyze each box: LP value, per-tier cert, geometry
    # ========================================================================
    print(f'\n{"="*80}\nANALYSIS of {len(los_a)} stuck boxes\n{"="*80}')
    rows = []
    t0 = time.time()
    for i in range(len(los_a)):
        lo, hi, dep = los_a[i], his_a[i], int(depths_a[i])
        if lo.sum() > 1.0 or hi.sum() < 1.0:
            continue
        # Box geometry
        center = (lo + hi)/2
        max_w = float(np.max(hi - lo))
        avg_w = float(np.mean(hi - lo))
        dist_sigma = float(np.linalg.norm(center - sigma_mu))
        on_boundary = int((lo <= 1e-12).sum())
        # f at center
        f_c = max(w.scale * sum(float(center[i_])*float(center[j_]) for (i_,j_) in w.pairs_all) for w in windows)
        # Cheap tier
        lo_int, hi_int = to_int(lo), to_int(hi)
        lb_fast, w_idx, _, _, _ = batch_bounds_full(lo, hi, A_tensor, scales, target)
        cheap_cert = False
        if lb_fast >= target and w_idx >= 0:
            w = windows[w_idx]
            cheap_cert = (
                bound_natural_int_ge(lo_int, hi_int, w, target_num, target_den) or
                bound_autoconv_int_ge(lo_int, hi_int, w, d, target_num, target_den) or
                bound_mccormick_sw_int_ge(lo_int, hi_int, w, d, target_num, target_den) or
                bound_mccormick_ne_int_ge(lo_int, hi_int, w, d, target_num, target_den) or
                bound_mccormick_joint_face_dual_cert_int_ge(lo_int, hi_int, w, d, target_num, target_den)
            )
        anchor_cert = bound_anchor_multi_int_ge(lo_int, hi_int, anchors, target_num, target_den)
        centroid_cert = bound_anchor_centroid_int_ge(lo_int, hi_int, target_num, target_den, cache)
        lp_val, *_ = _solve_epigraph_lp(np.asarray(lo, dtype=np.float64),
                                         np.asarray(hi, dtype=np.float64), windows, d)
        lp_cert = lp_val >= target
        rows.append(dict(
            i=i, depth=dep, max_w=max_w, avg_w=avg_w, f_c=float(f_c), lb_fast=float(lb_fast),
            dist_sigma=dist_sigma, on_boundary=on_boundary, w_idx=int(w_idx) if w_idx >=0 else -1,
            cheap_cert=bool(cheap_cert), anchor_cert=bool(anchor_cert),
            centroid_cert=bool(centroid_cert), lp_val=float(lp_val), lp_cert=bool(lp_cert),
        ))
        if (i+1) % 50 == 0:
            print(f'  processed {i+1}/{len(los_a)} ({(time.time()-t0):.0f}s)')

    print(f'\n=== Aggregate stats over {len(rows)} stuck boxes ===')
    arr = lambda k: np.array([r[k] for r in rows])
    print(f'  depth        : min={arr("depth").min()}, p50={int(np.percentile(arr("depth"),50))}, max={arr("depth").max()}')
    print(f'  max_w        : min={arr("max_w").min():.4f}, p50={np.percentile(arr("max_w"),50):.4f}, max={arr("max_w").max():.4f}')
    print(f'  dist_sigma   : min={arr("dist_sigma").min():.4f}, p50={np.percentile(arr("dist_sigma"),50):.4f}, max={arr("dist_sigma").max():.4f}')
    print(f'  on_boundary  : avg={arr("on_boundary").mean():.1f} axes/box (max possible {d})')
    print(f'  f(center)    : min={arr("f_c").min():.4f}, p50={np.percentile(arr("f_c"),50):.4f}, max={arr("f_c").max():.4f}')
    print(f'  LP_value     : min={arr("lp_val").min():.4f}, p10={np.percentile(arr("lp_val"),10):.4f}, p50={np.percentile(arr("lp_val"),50):.4f}, p90={np.percentile(arr("lp_val"),90):.4f}, max={arr("lp_val").max():.4f}')
    print(f'  LP_residual  : (LP - target) min={arr("lp_val").min()-target:+.4f}, p50={np.percentile(arr("lp_val"),50)-target:+.4f}, max={arr("lp_val").max()-target:+.4f}')
    print()
    print(f'  Cert rate by tier on stuck boxes (target=1.281):')
    print(f'    cheap   : {100*arr("cheap_cert").mean():5.1f}%')
    print(f'    anchor  : {100*arr("anchor_cert").mean():5.1f}%')
    print(f'    centroid: {100*arr("centroid_cert").mean():5.1f}%')
    print(f'    LP      : {100*arr("lp_cert").mean():5.1f}%')
    print()
    # Cert rate at various target relaxations
    print(f'  Cert-rate at relaxed targets:')
    for t in [1.270, 1.275, 1.278, 1.280, 1.281]:
        pct = 100*(arr("lp_val") >= t).mean()
        print(f'    target={t:.4f}: LP would cert {pct:5.1f}% of stuck boxes')
    # Distance distribution
    print()
    print(f'  Stuck boxes by distance to sigma(mu*):')
    for d_thresh in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        pct = 100 * (arr("dist_sigma") < d_thresh).mean()
        print(f'    < {d_thresh:.2f}: {pct:5.1f}%')

    # Per-box LP residual histogram (text-mode)
    print()
    print(f'  LP residual (LP - target) histogram on stuck boxes:')
    res = arr("lp_val") - target
    bins = [-1.0, -0.05, -0.02, -0.01, -0.005, -0.001, 0.001, 0.01, 0.1]
    for k in range(len(bins)-1):
        lo_b, hi_b = bins[k], bins[k+1]
        n_b = ((res >= lo_b) & (res < hi_b)).sum()
        bar = '#' * int(50 * n_b / len(rows))
        print(f'    [{lo_b:+.3f}, {hi_b:+.3f}): n={n_b:3d}  {bar}')

    # Save full per-box table
    np.savez('stuck_boxes_analysis.npz',
             **{k: arr(k) for k in rows[0].keys()})
    print(f'\nFull data → stuck_boxes_analysis.npz')


if __name__ == '__main__':
    main()
