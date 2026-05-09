"""Coarse-grid cascade headroom analysis.

For each (d, S, c_target), enumerate ALL compositions, and bucket each:
  - 'pass': not pruned at grid (TV_W(c) <= c_target for all W) — uncertain
  - 'cert_tri': pruned at grid AND triangle-certified (margin > cell_var + quad_corr)
  - 'pruned_uncert': pruned at grid but NOT triangle-certified (HEADROOM)

The headroom = pruned_uncert / (pruned_uncert + cert_tri).
This is the maximum improvement any sound box-cert tightening can achieve.

Output: table per config + json file.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))


def enum_compositions(d, S):
    if d == 1:
        yield (S,)
        return
    for v in range(S + 1):
        for rest in enum_compositions(d - 1, S - v):
            yield (v,) + rest


def build_A(d, ell, s_lo):
    A = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        for j in range(d):
            if s_lo <= i + j <= s_lo + ell - 2:
                A[i, j] = 1.0
    return A


def quad_corr(d, S, ell, s_lo):
    n_pairs = 0
    self_terms = 0  # M_W
    for k in range(s_lo, s_lo + ell - 1):
        n_k = min(k + 1, d, 2 * d - 1 - k)
        n_pairs += n_k
        if k % 2 == 0 and k // 2 < d:
            self_terms += 1
    cross_W = n_pairs - self_terms
    d_sq = d * d
    pair_bound = min(cross_W, d_sq - n_pairs)
    if pair_bound <= 0:
        return 0.0, n_pairs
    return (2.0 * d / ell) * pair_bound / (4.0 * S * S), n_pairs


def cell_cert_tri(c, d, S, c_target):
    """Returns (max_TV_at_grid, best_net_box_cert).
    box_cert = margin - cell_var - quad_corr (positive => certified).
    Search across all windows."""
    h = 1.0 / (2.0 * S)
    mu = np.array(c, dtype=np.float64) / S
    best_net = -np.inf
    max_tv = 0.0
    for ell in range(2, 2 * d + 1):
        n_cv = ell - 1
        n_windows = 2 * d - 1 - n_cv + 1
        if n_windows <= 0:
            continue
        for s_lo in range(n_windows):
            A = build_A(d, ell, s_lo)
            tv = (2.0 * d / ell) * float(mu @ A @ mu)
            if tv > max_tv:
                max_tv = tv
            if tv <= c_target:
                continue
            margin = tv - c_target
            grad = (4.0 * d / ell) * (A @ mu)
            grad_sorted = np.sort(grad)
            cell_var = sum(grad_sorted[d - 1 - k] - grad_sorted[k]
                           for k in range(d // 2)) / (2.0 * S)
            qc, _ = quad_corr(d, S, ell, s_lo)
            net = margin - cell_var - qc
            if net > best_net:
                best_net = net
    return max_tv, best_net


def headroom(d, S, c_target):
    n_pass = 0  # not pruned at grid
    n_cert_tri = 0  # triangle certified
    n_pruned_uncert = 0  # the headroom
    n_total = 0
    t0 = time.time()
    for c in enum_compositions(d, S):
        n_total += 1
        max_tv, net = cell_cert_tri(c, d, S, c_target)
        if max_tv <= c_target + 1e-12:
            n_pass += 1
        elif net > 0:
            n_cert_tri += 1
        else:
            n_pruned_uncert += 1
    elapsed = time.time() - t0
    return {
        'd': d, 'S': S, 'c_target': c_target,
        'n_total': n_total, 'n_pass': n_pass,
        'n_cert_tri': n_cert_tri, 'n_pruned_uncert': n_pruned_uncert,
        'pct_pruned_certified': 100.0 * n_cert_tri /
            max(1, n_cert_tri + n_pruned_uncert),
        'pct_headroom': 100.0 * n_pruned_uncert /
            max(1, n_cert_tri + n_pruned_uncert),
        'wall_sec': elapsed,
    }


if __name__ == '__main__':
    configs = [
        (4, 20, 1.20), (4, 30, 1.20),
        (4, 30, 1.25), (6, 12, 1.20),
        (6, 15, 1.20), (8, 10, 1.20),
        (8, 12, 1.20),
    ]
    rows = []
    for d, S, c in configs:
        r = headroom(d, S, c)
        rows.append(r)
        print(f"  d={d}, S={S}, c={c}: total={r['n_total']:,}  "
              f"pass={r['n_pass']:,}  cert_tri={r['n_cert_tri']:,}  "
              f"uncert={r['n_pruned_uncert']:,}  "
              f"headroom={r['pct_headroom']:.1f}%  "
              f"({r['wall_sec']:.1f}s)")
    out = os.path.join(_dir, '_coarse_headroom_results.json')
    with open(out, 'w') as f:
        json.dump(rows, f, indent=2)
    print(f"\nSaved: {out}")
