"""Agent D — probe 5: B46 detector via random sampling at d=6 S=30
and d=8 S=16.  We want to find any cell that fires B46 to test the lift.
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
    """Random nonneg integer composition Σ=S."""
    cuts = np.sort(rng.integers(0, S + 1, size=d - 1))
    parts = np.diff(np.concatenate([[0], cuts, [S]]))
    return parts.astype(int)


def sweep(d, S, n=400):
    windows = v3.build_all_windows(d)
    bundle = v6.get_bundle(windows)
    tier_counts = {}
    b46_examples = []
    open_examples = []
    for i in range(n):
        c = random_composition(d, S, RNG)
        cell = v3.Cell.from_integer_composition(c.astype(float), S)
        mu = cell.center
        tv_at_center = max(w.Q_coef * float(mu @ w.A @ mu) for w in windows)
        if tv_at_center < C_TARGET:
            tier_counts['below_thresh'] = tier_counts.get('below_thresh', 0) + 1
            continue
        r = v6.cert_cell(cell, windows, C_TARGET, max_depth=6, bundle=bundle)
        key = r.tier_used + ('|cert' if r.certified else '|open')
        tier_counts[key] = tier_counts.get(key, 0) + 1
        if not r.certified:
            open_examples.append(dict(c=c.tolist(), tier=r.tier_used,
                                        n_sub=r.n_subcells))
            if r.tier_used == 'B46_no_signal':
                b46_examples.append(c.tolist())
    return dict(d=d, S=S, tier_counts=tier_counts,
                 open_examples=open_examples,
                 b46_examples=b46_examples)


def lift_run(cell, windows, bundle, max_depth=6, depth=0, n_calls=None):
    """Cascade without B46, no d≤4 cap."""
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


def main():
    out = {}
    for d, S in [(4, 80), (6, 30), (8, 16)]:
        print(f"=== d={d} S={S} ===")
        r = sweep(d, S, n=200)
        print(r['tier_counts'])
        # try lift on each B46 example
        if r['b46_examples']:
            windows = v3.build_all_windows(d)
            bundle = v6.get_bundle(windows)
            lifted = []
            for c in r['b46_examples'][:10]:
                cell = v3.Cell.from_integer_composition(np.array(c, float), S)
                n_calls = [0]
                ok = lift_run(cell, windows, bundle, max_depth=6,
                                n_calls=n_calls)
                lifted.append(dict(c=c, ok=ok, n_calls=n_calls[0]))
                print(f"  B46 lift: c={c} ok={ok} n_calls={n_calls[0]}")
            r['b46_lifted'] = lifted
        out[f'd{d}_S{S}'] = r
    with open(os.path.join(_dir, '_audit_D_probe5.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == '__main__':
    main()
