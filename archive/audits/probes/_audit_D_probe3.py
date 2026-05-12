"""Agent D — probe 3: characterise the d=4 hard cells.

For (20,20,20,20) and (19,21,21,19) at S=80:
 - Run B1, B1u, B1diag, F, L_single, L_joint on the root cell.
 - Report each tier's LB (negative means uncertified).
 - Walk one split path and report LBs along it to understand WHERE the
   structural barrier is.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v3 as v3
import _coarse_bnb_v5 as v5
import _coarse_bnb_v6 as v6

C_TARGET = 1.281
CELLS = [('a_central', (20, 20, 20, 20)),
         ('b_offset',  (19, 21, 21, 19))]


def evaluate_cell(cell, windows, bundle, label, c_target=C_TARGET):
    out = {'label': label, 'lo': list(cell.lo), 'hi': list(cell.hi)}
    cache = v3.CellCache.build(cell)
    out['B1_max'] = float(np.max(v6.tier_B1_vec(cell, bundle, c_target)))
    out['B1u_max'] = float(np.max(v6.tier_B1u_vec(cell, bundle, c_target)))
    # B1diag (per-W)
    b1diag = -np.inf
    for w in windows:
        b1diag = max(b1diag, v5.tier_B1diag_amgm(cell, w, c_target))
    out['B1diag_max'] = float(b1diag)
    v3.compute_F_all_windows(cache, windows, c_target)
    out['F_best'] = float(cache.f_best_bound)
    out['F_best_W'] = int(cache.f_best_W_idx)
    mw_vec = v6.mwQ_vec(cell, bundle, c_target, cache.mu_star)
    L_best = -np.inf
    if v6.HAS_CVXPY:
        cand = v6.select_L_candidates_v6(mw_vec, bundle, K=3, include_widest=2)
        L_vals = []
        for w_idx in cand:
            lb, _ = v6.tier_L_single_v6(cache, windows[w_idx], c_target)
            L_vals.append((int(w_idx), float(lb)))
            if lb > L_best:
                L_best = lb
        out['L_single_vals'] = L_vals
        out['L_single_best'] = float(L_best)
        cand_j = v6.select_L_candidates_v6(mw_vec, bundle, K=4, include_widest=2)
        top_j = [windows[i] for i in cand_j]
        lb_j, info_j = v6.tier_L_joint_lagrangian_v6(cache, top_j, c_target,
                                                       max_fw_iters=10)
        out['L_joint_best'] = float(lb_j)
        out['L_joint_iters'] = info_j.get('iters_used', -1)
    return out, cache


def walk_split(cell, windows, bundle, depth=4, label_prefix=''):
    """Greedy depth-walk: split, follow the worse child each time."""
    trace = []
    cur = cell
    for d in range(depth):
        info, cache = evaluate_cell(cur, windows, bundle, f'{label_prefix}d{d}')
        info['depth'] = d
        trace.append(info)
        if info['L_joint_best'] > 0:
            info['certified'] = True
            break
        info['certified'] = False
        best_W = (windows[cache.f_best_W_idx]
                  if cache.f_best_W_idx >= 0 else windows[0])
        axis = int(v3.split_axis_gradient_weighted(cache, best_W))
        info['split_axis'] = axis
        s1, s2 = cur.split(axis)
        # Evaluate both children's F bound to pick worse one
        c1 = v3.CellCache.build(s1)
        c2 = v3.CellCache.build(s2)
        v3.compute_F_all_windows(c1, windows, C_TARGET)
        v3.compute_F_all_windows(c2, windows, C_TARGET)
        info['child1_F'] = float(c1.f_best_bound)
        info['child2_F'] = float(c2.f_best_bound)
        cur = s1 if c1.f_best_bound < c2.f_best_bound else s2
        info['took_child'] = 1 if c1.f_best_bound < c2.f_best_bound else 2
    return trace


def main():
    windows = v3.build_all_windows(4)
    bundle = v6.get_bundle(windows)
    out = {}
    for name, c in CELLS:
        cell = v3.Cell.from_integer_composition(np.array(c, float), 80)
        trace = walk_split(cell, windows, bundle, depth=6, label_prefix=name)
        out[name] = trace
        print(f"\n=== {name} c={c} ===")
        for r in trace:
            print(f" d={r['depth']:>2d} F={r['F_best']:+.4f}  "
                  f"L1={r.get('L_single_best', np.nan):+.4f}  "
                  f"Lj={r.get('L_joint_best', np.nan):+.4f}  "
                  f"cert={r.get('certified', False)}  "
                  f"ax={r.get('split_axis', '-')}  "
                  f"chF={(r.get('child1_F', np.nan), r.get('child2_F', np.nan))}")
    with open(os.path.join(_dir, '_audit_D_probe3.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == '__main__':
    main()
