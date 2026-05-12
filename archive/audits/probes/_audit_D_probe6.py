"""Agent D — probe 6: lift B45 (depth cap) on all sub1_fail cells found in
probe5 to determine actually-closeable cells via deeper splitting.

Also: characterise root-gap L_joint LB on hard cells to determine the
"structural floor".
"""
from __future__ import annotations
import os, sys, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v3 as v3
import _coarse_bnb_v6 as v6

C_TARGET = 1.281
RNG = np.random.default_rng(20260511)


def random_composition(d, S, rng):
    cuts = np.sort(rng.integers(0, S + 1, size=d - 1))
    parts = np.diff(np.concatenate([[0], cuts, [S]]))
    return parts.astype(int)


def lift_run(cell, windows, bundle, max_depth=6, depth=0, n_calls=None):
    if n_calls is None:
        n_calls = [0]
    n_calls[0] += 1
    if v6.is_cell_empty(cell):
        return True
    if not cell.is_simplex_feasible():
        return True
    cache = v3.CellCache.build(cell)
    if float(np.max(v6.tier_B1_vec(cell, bundle, C_TARGET))) > 0:
        return True
    if float(np.max(v6.tier_B1u_vec(cell, bundle, C_TARGET))) > 0:
        return True
    v3.compute_F_all_windows(cache, windows, C_TARGET)
    if cache.f_best_bound > 0:
        return True
    mw_vec = v6.mwQ_vec(cell, bundle, C_TARGET, cache.mu_star)
    if v6.HAS_CVXPY:
        cand = v6.select_L_candidates_v6(mw_vec, bundle, K=3, include_widest=2)
        for w_idx in cand:
            lb, _ = v6.tier_L_single_v6(cache, windows[w_idx], C_TARGET)
            if lb > 0:
                return True
        cand_j = v6.select_L_candidates_v6(mw_vec, bundle, K=4, include_widest=2)
        top_j = [windows[i] for i in cand_j]
        lb_j, _ = v6.tier_L_joint_lagrangian_v6(cache, top_j, C_TARGET,
                                                  max_fw_iters=6)
        if lb_j > 0:
            return True
    if depth >= max_depth:
        return False
    best_W = windows[cache.f_best_W_idx] if cache.f_best_W_idx >= 0 else windows[0]
    axis = int(v3.split_axis_gradient_weighted(cache, best_W))
    s1, s2 = cell.split(axis)
    if not lift_run(s1, windows, bundle, max_depth, depth + 1, n_calls):
        return False
    return lift_run(s2, windows, bundle, max_depth, depth + 1, n_calls)


def root_gap(cell, windows, bundle):
    """Return (F, L_single, L_joint) root LBs."""
    cache = v3.CellCache.build(cell)
    v3.compute_F_all_windows(cache, windows, C_TARGET)
    F = cache.f_best_bound
    mw_vec = v6.mwQ_vec(cell, bundle, C_TARGET, cache.mu_star)
    cand = v6.select_L_candidates_v6(mw_vec, bundle, K=3, include_widest=2)
    L1 = -np.inf
    for w_idx in cand:
        lb, _ = v6.tier_L_single_v6(cache, windows[w_idx], C_TARGET)
        L1 = max(L1, lb)
    cand_j = v6.select_L_candidates_v6(mw_vec, bundle, K=4, include_widest=2)
    top_j = [windows[i] for i in cand_j]
    Lj, _ = v6.tier_L_joint_lagrangian_v6(cache, top_j, C_TARGET, max_fw_iters=10)
    return float(F), float(L1), float(Lj)


def sweep_open(d, S, n=200):
    windows = v3.build_all_windows(d)
    bundle = v6.get_bundle(windows)
    opens = []
    for i in range(n):
        c = random_composition(d, S, RNG)
        cell = v3.Cell.from_integer_composition(c.astype(float), S)
        mu = cell.center
        tv_at_center = max(w.Q_coef * float(mu @ w.A @ mu) for w in windows)
        if tv_at_center < C_TARGET:
            continue
        r = v6.cert_cell(cell, windows, C_TARGET, max_depth=6, bundle=bundle)
        if not r.certified:
            F, L1, Lj = root_gap(cell, windows, bundle)
            opens.append(dict(c=c.tolist(), tier=r.tier_used,
                                F=F, L1=L1, Lj=Lj))
    return opens


def main():
    out = {}
    for d, S in [(4, 80), (6, 30), (8, 16)]:
        print(f"\n=== d={d} S={S} ===")
        opens = sweep_open(d, S, n=200)
        print(f"open: {len(opens)}")
        windows = v3.build_all_windows(d)
        bundle = v6.get_bundle(windows)
        closed_by_lift = 0
        lift_results = []
        for r in opens:
            cell = v3.Cell.from_integer_composition(np.array(r['c'], float), S)
            n_calls = [0]
            ok = lift_run(cell, windows, bundle, max_depth=6, n_calls=n_calls)
            r['lift_ok'] = ok
            r['lift_n_calls'] = n_calls[0]
            if ok:
                closed_by_lift += 1
            print(f"  c={r['c']} tier={r['tier']} F={r['F']:+.4f} "
                  f"L1={r['L1']:+.4f} Lj={r['Lj']:+.4f}  "
                  f"lift_ok={ok} n_calls={n_calls[0]}")
            lift_results.append(r)
        out[f'd{d}_S{S}'] = dict(total_open=len(opens),
                                    closed_by_B45_lift=closed_by_lift,
                                    cells=lift_results)
        print(f"  lifted-cap closes {closed_by_lift}/{len(opens)}")
    with open(os.path.join(_dir, '_audit_D_probe6.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == '__main__':
    main()
