"""AGENT F audit probe: Monte Carlo soundness check of v6 vs v3 cascades.

For 8 cells (varied d, S), run v6's root tiers (B1, B1u, B1diag, F, L_single,
L_joint). Then sample 5000 (μ_i, ε) inside each cell and compute
    mc_min_max = min_μ max_W f_W(ε(μ))
where ε = μ − μ* and X = εε^T.
Verify v6's bound ≤ mc_min_max + 1e-5 in every case (any violation = fatal).

Also tests two encoding equivalences via Monte Carlo:
  (1) cp.multiply(eps_col, lo_row) entrywise == np.outer-style on representative
      values, by comparing solve results to v3.
  (2) Degenerate lo == hi cell.
"""
from __future__ import annotations
import os, sys, json, time
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v3 as v3
import _coarse_bnb_v5 as v5
import _coarse_bnb_v6 as v6


def f_W_at_mu(W, mu, c_target):
    return W.Q_coef * float(mu @ W.A @ mu) - c_target


def mc_min_max_over_cell(cell, windows, c_target, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    d = cell.d
    lo, hi = cell.lo, cell.hi
    if np.all(hi == lo):
        mu = lo / (lo.sum() if lo.sum() > 0 else 1.0)
        max_f = max(f_W_at_mu(W, mu, c_target) for W in windows)
        return max_f, max_f, max_f, 1
    min_max = +np.inf
    accepted = 0
    for _ in range(n):
        # Sample uniform in box, then renormalize. To get a Σ=1 sample, project.
        u = rng.uniform(lo, hi)
        s = u.sum()
        if s <= 0:
            continue
        mu = u / s
        # Reject if not in box after rescaling
        if np.any(mu < lo - 1e-12) or np.any(mu > hi + 1e-12):
            continue
        accepted += 1
        max_f = max(f_W_at_mu(W, mu, c_target) for W in windows)
        if max_f < min_max:
            min_max = max_f
    # Also try corner sampling (Dirichlet-rejection)
    for _ in range(n):
        z = rng.dirichlet(np.ones(d))
        mu = lo + z * (hi - lo)
        s = mu.sum()
        if s <= 0:
            continue
        mu = mu / s
        if np.any(mu < lo - 1e-12) or np.any(mu > hi + 1e-12):
            continue
        accepted += 1
        max_f = max(f_W_at_mu(W, mu, c_target) for W in windows)
        if max_f < min_max:
            min_max = max_f
    return min_max, accepted


def root_tiers(cell, windows, c_target):
    """Run v6 root tiers and report the BEST (i.e., max) LB any tier produced.

    The cascade-certification logic uses bound > 0, so the maximum across tiers
    is the strongest LB that v6 could claim.
    """
    out = {}
    bundle = v6.get_bundle(windows)
    # B1
    b1_vec = v6.tier_B1_vec(cell, bundle, c_target)
    out['B1_max'] = float(np.max(b1_vec))
    # B1u
    b1u_vec = v6.tier_B1u_vec(cell, bundle, c_target)
    out['B1u_max'] = float(np.max(b1u_vec))
    # B1diag
    b1diag_max = -np.inf
    for W in windows:
        b = v5.tier_B1diag_amgm(cell, W, c_target)
        b1diag_max = max(b1diag_max, b)
    out['B1diag_max'] = float(b1diag_max)
    # F
    cache = v3.CellCache.build(cell)
    v3.compute_F_all_windows(cache, windows, c_target)
    out['F_max'] = float(cache.f_best_bound)
    # L_single (v6)
    if v3.HAS_CVXPY:
        mw_vec = v6.mwQ_vec(cell, bundle, c_target, cache.mu_star)
        candidates = v6.select_L_candidates_v6(mw_vec, bundle, K=3,
                                                  include_widest=2)
        L_max = -np.inf
        for w_idx in candidates:
            lb, _ = v6.tier_L_single_v6(cache, windows[w_idx], c_target)
            if lb > L_max:
                L_max = lb
        out['L_max_v6'] = float(L_max)
        # L_single (v3) on SAME candidates for direct compare
        L_max_v3 = -np.inf
        for w_idx in candidates:
            lb = v3.tier_L_single(cache, windows[w_idx], c_target)
            if lb > L_max_v3:
                L_max_v3 = lb
        out['L_max_v3'] = float(L_max_v3)
        # L_joint (v6 Lagrangian)
        top_j = [windows[i] for i in candidates]
        lbj, _ = v6.tier_L_joint_lagrangian_v6(cache, top_j, c_target,
                                                  max_fw_iters=6)
        out['Ljoint_v6'] = float(lbj)
    else:
        out['L_max_v6'] = -np.inf
        out['L_max_v3'] = -np.inf
        out['Ljoint_v6'] = -np.inf
    return out


def main():
    results = []
    c_target = 1.281
    test_cells = [
        # (d, S, c) — c is a fractional vector summing to ~S
        (4, 8, [2, 2, 2, 2]),
        (4, 8, [1, 3, 2, 2]),
        (6, 10, [2, 2, 2, 2, 1, 1]),
        (6, 10, [1, 2, 2, 2, 2, 1]),
        (8, 12, [2, 2, 1, 2, 1, 2, 1, 1]),
        (8, 12, [1, 1, 2, 2, 2, 2, 1, 1]),
        (4, 4, [1, 1, 1, 1]),     # symmetric, tight
        (4, 4, [4, 0, 0, 0]),      # degenerate corner
    ]
    for d, S, c in test_cells:
        c = np.array(c)
        windows = v3.build_all_windows(d)
        cell = v3.Cell.from_integer_composition(c, S)
        t0 = time.time()
        tiers = root_tiers(cell, windows, c_target)
        t1 = time.time()
        mc_min_max, n_acc = mc_min_max_over_cell(cell, windows, c_target,
                                                      n=5000, seed=42)
        t2 = time.time()
        v6_best = max(tiers['B1_max'], tiers['B1u_max'], tiers['B1diag_max'],
                       tiers['F_max'], tiers['L_max_v6'], tiers['Ljoint_v6'])
        violation = v6_best > mc_min_max + 1e-5
        v3_v6_diff = abs(tiers['L_max_v6'] - tiers['L_max_v3'])
        results.append({
            'd': d, 'S': S, 'c': c.tolist(),
            'tiers': tiers,
            'v6_best_lb': v6_best,
            'mc_min_max': mc_min_max,
            'n_mc_accepted': n_acc,
            'v6_minus_mc': v6_best - mc_min_max,
            'SOUND': not violation,
            'L_v6_vs_v3_diff': v3_v6_diff,
            'tier_time_s': t1 - t0,
            'mc_time_s': t2 - t1,
        })
        print(f"d={d} S={S} c={c.tolist()}: v6_best={v6_best:.5f} "
              f"mc={mc_min_max:.5f} diff={v6_best-mc_min_max:+.5f} "
              f"v3-v6_L_diff={v3_v6_diff:.2e}")

    # Degenerate cell: lo == hi
    print("\n--- Degenerate cell (lo == hi) ---")
    d = 4
    lo_eq = np.array([0.25, 0.25, 0.25, 0.25])
    cell_deg = v3.Cell(lo=lo_eq.copy(), hi=lo_eq.copy())
    windows = v3.build_all_windows(d)
    try:
        tiers_deg = root_tiers(cell_deg, windows, c_target)
        print(f"  v6 tiers on degenerate cell: {tiers_deg}")
        results.append({'d': d, 'cell': 'degenerate_lo_eq_hi',
                          'tiers': tiers_deg, 'CRASHED': False})
    except Exception as e:
        print(f"  CRASH: {e}")
        results.append({'d': d, 'cell': 'degenerate_lo_eq_hi',
                          'CRASHED': True, 'error': str(e)})

    with open(os.path.join(_dir, '_audit_F_probe.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    # Global verdict
    any_unsound = any(not r.get('SOUND', True) for r in results
                       if 'SOUND' in r)
    any_crash = any(r.get('CRASHED', False) for r in results)
    print(f"\nGLOBAL VERDICT: SOUND={not any_unsound} CRASHED={any_crash}")
    return results


if __name__ == '__main__':
    main()
