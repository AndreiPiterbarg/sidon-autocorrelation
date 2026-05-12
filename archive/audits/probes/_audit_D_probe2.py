"""Agent D — follow-up probe.

Extend the (20,20,20,20) and (19,21,21,19) d=4 cells:
  (a) very deep splits, depth 6..10 with gradient_weighted axis;
  (b) 2-axis (k-d tree) splits — 4 subcells per level;
  (c) report at each level whether ALL leaves close.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v3 as v3
import _coarse_bnb_v6 as v6

C_TARGET = 1.281

CELLS = [
    ('d4_S80_a', dict(d=4, S=80, c=(20, 20, 20, 20))),
    ('d4_S80_b', dict(d=4, S=80, c=(19, 21, 21, 19))),
]


def make_cell(p):
    c = np.array(p['c'], dtype=np.float64)
    return v3.Cell.from_integer_composition(c, p['S'])


# ----------------------------------------------------------------------
# 4-way split: split along the top-2 axes by gradient_weighted score
# ----------------------------------------------------------------------

def split_top2(cell, cache, best_W):
    """Pick top-2 axes by gradient-weighted score, mid-split both → 4 subcells."""
    A = best_W.A
    grad_W = best_W.grad_coef * (A @ cache.mu_star)
    scores = (np.abs(grad_W) + best_W.Q_coef * (A @ cache.h)) * cache.h
    top2 = list(np.argsort(-scores)[:2])
    s1, s2 = cell.split(int(top2[0]))
    out = []
    for s in (s1, s2):
        a, b = s.split(int(top2[1]))
        out.extend([a, b])
    return out


def cert_one(cell, windows, bundle, c_target):
    """Single-cell cascade (no splitting)."""
    if v6.is_cell_empty(cell):
        return True
    if not cell.is_simplex_feasible():
        return True
    cache = v3.CellCache.build(cell)
    if float(np.max(v6.tier_B1_vec(cell, bundle, c_target))) > 0:
        return True
    if float(np.max(v6.tier_B1u_vec(cell, bundle, c_target))) > 0:
        return True
    v3.compute_F_all_windows(cache, windows, c_target)
    if cache.f_best_bound > 0:
        return True
    mw_vec = v6.mwQ_vec(cell, bundle, c_target, cache.mu_star)
    if v6.HAS_CVXPY:
        cand = v6.select_L_candidates_v6(mw_vec, bundle, K=3, include_widest=2)
        for w_idx in cand:
            lb, _ = v6.tier_L_single_v6(cache, windows[w_idx], c_target)
            if lb > 0:
                return True
        cand_j = v6.select_L_candidates_v6(mw_vec, bundle, K=4, include_widest=2)
        top_j = [windows[i] for i in cand_j]
        lb_j, _ = v6.tier_L_joint_lagrangian_v6(cache, top_j, c_target,
                                                  max_fw_iters=6)
        if lb_j > 0:
            return True
    return False


def deep_split_2way(cell, windows, bundle, c_target, max_depth, depth=0,
                     n_calls=None):
    """Standard 2-way mid-split using gradient_weighted axis."""
    if n_calls is None:
        n_calls = [0]
    n_calls[0] += 1
    if v6.is_cell_empty(cell):
        return True
    if not cell.is_simplex_feasible():
        return True
    cache = v3.CellCache.build(cell)
    if cert_one(cell, windows, bundle, c_target):
        return True
    if depth >= max_depth:
        return False
    best_W = windows[cache.f_best_W_idx] if cache.f_best_W_idx >= 0 else windows[0]
    axis = int(v3.split_axis_gradient_weighted(cache, best_W))
    sub1, sub2 = cell.split(axis)
    if not deep_split_2way(sub1, windows, bundle, c_target, max_depth,
                            depth + 1, n_calls):
        return False
    return deep_split_2way(sub2, windows, bundle, c_target, max_depth,
                            depth + 1, n_calls)


def deep_split_4way(cell, windows, bundle, c_target, max_depth, depth=0,
                     n_calls=None):
    """4-way split along top-2 axes."""
    if n_calls is None:
        n_calls = [0]
    n_calls[0] += 1
    if v6.is_cell_empty(cell):
        return True
    if not cell.is_simplex_feasible():
        return True
    cache = v3.CellCache.build(cell)
    if cert_one(cell, windows, bundle, c_target):
        return True
    if depth >= max_depth:
        return False
    best_W = windows[cache.f_best_W_idx] if cache.f_best_W_idx >= 0 else windows[0]
    subs = split_top2(cell, cache, best_W)
    for s in subs:
        if not deep_split_4way(s, windows, bundle, c_target, max_depth,
                                depth + 1, n_calls):
            return False
    return True


def main():
    windows = v3.build_all_windows(4)
    bundle = v6.get_bundle(windows)
    out = {}
    for name, p in CELLS:
        cell = make_cell(p)
        row = {}
        # 2-way depth sweep
        for md in (3, 4, 5, 6, 7, 8):
            t0 = time.time()
            n_calls = [0]
            ok = deep_split_2way(cell, windows, bundle, C_TARGET, max_depth=md,
                                  n_calls=n_calls)
            row[f'2way_d{md}'] = dict(ok=ok, n_calls=n_calls[0],
                                       wall=time.time() - t0)
            print(f"{name} 2way d={md}: ok={ok} n_calls={n_calls[0]} "
                  f"wall={row[f'2way_d{md}']['wall']:.2f}s")
            if ok:
                break  # no need for deeper
        # 4-way depth sweep
        for md in (2, 3, 4, 5):
            t0 = time.time()
            n_calls = [0]
            ok = deep_split_4way(cell, windows, bundle, C_TARGET, max_depth=md,
                                  n_calls=n_calls)
            row[f'4way_d{md}'] = dict(ok=ok, n_calls=n_calls[0],
                                       wall=time.time() - t0)
            print(f"{name} 4way d={md}: ok={ok} n_calls={n_calls[0]} "
                  f"wall={row[f'4way_d{md}']['wall']:.2f}s")
            if ok:
                break
        out[name] = row
    with open(os.path.join(_dir, '_audit_D_probe2.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == '__main__':
    main()
