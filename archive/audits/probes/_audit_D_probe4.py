"""Agent D — probe 4: find d=8 cells that fire B46.

Scan all d=8 compositions at S=16 with first-coordinate ≥ second (orbit
canonicalisation skipped — we just want to find which compositions trigger
B46 to see whether lifting B46 closes them).
"""
from __future__ import annotations
import os, sys, json
from itertools import product
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v3 as v3
import _coarse_bnb_v6 as v6

C_TARGET = 1.281
# Sweep multiple (d,S) configs
SWEEPS = [(4, 80), (6, 30), (8, 16)]
D, S = 8, 16


def enumerate_compositions(d, S):
    out = []
    def rec(prefix, remaining, slots):
        if slots == 1:
            out.append(tuple(prefix + [remaining]))
            return
        for v in range(remaining + 1):
            rec(prefix + [v], remaining - v, slots - 1)
    rec([], S, d)
    return out


def canon(c):
    """Return canonical (sorted-desc) repr — orbit reduction."""
    return tuple(sorted(c, reverse=True))


def main():
    windows = v3.build_all_windows(D)
    bundle = v6.get_bundle(windows)
    seen = set()
    open_cells = []
    b46_cells = []
    total = 0
    for c in enumerate_compositions(D, S):
        k = canon(c)
        if k in seen:
            continue
        seen.add(k)
        cell = v3.Cell.from_integer_composition(np.array(c, float), S)
        # Quick TV check: skip cells whose centre clearly TV < c_target
        mu = cell.center
        # Approx TV: max over windows of Q · mu^T A mu
        tv_at_center = -np.inf
        for w in windows:
            val = w.Q_coef * float(mu @ w.A @ mu)
            if val > tv_at_center:
                tv_at_center = val
        if tv_at_center < C_TARGET:
            continue  # cell cannot achieve c_target
        total += 1
        # Run the v6 cascade with stock B46 to see what it returns
        r = v6.cert_cell(cell, windows, C_TARGET, max_depth=6, verbose=False,
                          bundle=bundle)
        if not r.certified:
            open_cells.append(dict(c=c, tier=r.tier_used,
                                     n_sub=r.n_subcells))
            if r.tier_used == 'B46_no_signal':
                b46_cells.append(c)
    print(f"d={D} S={S} c_target={C_TARGET}")
    print(f"  total feasible orbit reps = {total}")
    print(f"  open cells = {len(open_cells)}")
    print(f"  B46_no_signal cells = {len(b46_cells)}")
    for r in open_cells:
        print(f"   {r['c']}  tier={r['tier']}")

    # For each B46 cell, lift B46 (force deep split) and see what happens
    print("\n=== B46-lift behaviour ===")
    out = []
    for c in b46_cells:
        cell = v3.Cell.from_integer_composition(np.array(c, float), S)
        # Manual cascade: skip B46 and just split anyway
        ok, depth_used, n_calls = lift_b46_run(cell, windows, bundle)
        out.append(dict(c=c, lifted_ok=ok, depth=depth_used, n_calls=n_calls))
        print(f"  {c}: lifted_ok={ok} depth={depth_used} n_calls={n_calls}")

    with open(os.path.join(_dir, '_audit_D_probe4.json'), 'w') as f:
        json.dump({'open_count': len(open_cells), 'b46_count': len(b46_cells),
                    'open_cells': open_cells, 'b46_lift': out},
                  f, indent=2, default=str)


def lift_b46_run(cell, windows, bundle, max_depth=6, depth=0, n_calls=None):
    if n_calls is None:
        n_calls = [0]
    n_calls[0] += 1
    if v6.is_cell_empty(cell):
        return True, depth, n_calls[0]
    if not cell.is_simplex_feasible():
        return True, depth, n_calls[0]
    cache = v3.CellCache.build(cell)
    if float(np.max(v6.tier_B1_vec(cell, bundle, C_TARGET))) > 0:
        return True, depth, n_calls[0]
    if float(np.max(v6.tier_B1u_vec(cell, bundle, C_TARGET))) > 0:
        return True, depth, n_calls[0]
    v3.compute_F_all_windows(cache, windows, C_TARGET)
    if cache.f_best_bound > 0:
        return True, depth, n_calls[0]
    mw_vec = v6.mwQ_vec(cell, bundle, C_TARGET, cache.mu_star)
    if v6.HAS_CVXPY:
        cand = v6.select_L_candidates_v6(mw_vec, bundle, K=3, include_widest=2)
        for w_idx in cand:
            lb, _ = v6.tier_L_single_v6(cache, windows[w_idx], C_TARGET)
            if lb > 0:
                return True, depth, n_calls[0]
        cand_j = v6.select_L_candidates_v6(mw_vec, bundle, K=4, include_widest=2)
        top_j = [windows[i] for i in cand_j]
        lb_j, _ = v6.tier_L_joint_lagrangian_v6(cache, top_j, C_TARGET,
                                                  max_fw_iters=6)
        if lb_j > 0:
            return True, depth, n_calls[0]
    # SKIP B46 — always split
    if depth >= max_depth:
        return False, depth, n_calls[0]
    best_W = windows[cache.f_best_W_idx] if cache.f_best_W_idx >= 0 else windows[0]
    axis = int(v3.split_axis_gradient_weighted(cache, best_W))
    s1, s2 = cell.split(axis)
    ok1, d1, _ = lift_b46_run(s1, windows, bundle, max_depth, depth + 1, n_calls)
    if not ok1:
        return False, d1, n_calls[0]
    ok2, d2, _ = lift_b46_run(s2, windows, bundle, max_depth, depth + 1, n_calls)
    return (ok1 and ok2), max(d1, d2), n_calls[0]


if __name__ == '__main__':
    main()
