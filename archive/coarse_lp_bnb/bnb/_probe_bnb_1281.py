"""Quick probe: where does v4 fail at c_target=1.281?

Goals (run fast, <2 min):
  - Test a representative spread of compositions at d=4, S=160 (small enough)
  - And d=6, S=120 (a bit larger).
  - Track which tier closes / which tier the open cells fail at.
  - Measure per-tier timing.
  - Identify the hardest open compositions for targeted analysis.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v4 as v4
import _coarse_bnb_v3 as v3


def per_tier_breakdown(d, S, c_target, compositions, label, max_depth=3):
    windows = v4.build_all_windows(d)
    tier_counts = {}
    tier_times = {}
    open_cases = []
    deepest_open = None
    total_t0 = time.time()
    for c_tup in compositions:
        c = np.array(c_tup, dtype=np.float64)
        if abs(c.sum() - S) > 1e-9 or c.min() < 0:
            continue
        cell = v4.Cell.from_integer_composition(c, S)
        # Time each top-level tier separately on the ROOT cell only
        mu_star = cell.center
        cache = v3.CellCache.build(cell)
        # B1
        t0 = time.time()
        b1_max = -np.inf
        for W in windows:
            b = v4.tier_B1_mu_corner(cell, W, c_target)
            if b > b1_max:
                b1_max = b
        t_B1 = time.time() - t0
        # F all
        t0 = time.time()
        v3.compute_F_all_windows(cache, windows, c_target)
        t_F = time.time() - t0
        f_best = cache.f_best_bound
        # L (top 3 by F-rank)
        L_best = -np.inf
        t_L = 0.0
        if v3.HAS_CVXPY and cache.f_ranked_indices is not None:
            candidates = v4.select_L_candidates(cache, windows, K=3)
            for w_idx in candidates:
                t0 = time.time()
                lb = v3.tier_L_single(cache, windows[w_idx], c_target, False)
                t_L += time.time() - t0
                if lb > L_best:
                    L_best = lb
        # L_joint
        t0 = time.time()
        Lj = None
        if v3.HAS_CVXPY and cache.f_ranked_indices is not None:
            Lj = v3.tier_L_joint(cache, windows, c_target, K=4)
        t_Lj = time.time() - t0

        # Full cascade with depth
        t0 = time.time()
        result = v4.cert_cell(cell, windows, c_target, max_depth=max_depth)
        t_full = time.time() - t0

        tier = result.tier_used if result.certified else f'OPEN:{result.tier_used}'
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        tier_times[tier] = tier_times.get(tier, 0.0) + t_full

        if not result.certified:
            open_cases.append({
                'c': c_tup,
                'B1_best': float(b1_max),
                'F_best': float(f_best),
                'L_best': float(L_best),
                'Lj': float(Lj) if Lj is not None else None,
                't_B1_ms': t_B1 * 1000,
                't_F_ms': t_F * 1000,
                't_L_ms': t_L * 1000,
                't_Lj_ms': t_Lj * 1000,
                't_full_s': t_full,
                'depth': result.depth_used,
                'sub_cells': result.n_subcells,
            })
    total = time.time() - total_t0
    return {
        'label': label,
        'd': d, 'S': S, 'c_target': c_target,
        'n_total': len(compositions),
        'tier_counts': tier_counts,
        'tier_times_s': tier_times,
        'open_cases': open_cases,
        'total_s': total,
    }


def main():
    # Set d=4 first (fast)
    d4_S = 160
    d4_comps = [
        (40, 40, 40, 40),
        (39, 41, 41, 39),
        (30, 50, 50, 30),
        (20, 60, 60, 20),
        (50, 30, 30, 50),
        (10, 70, 70, 10),
        (45, 35, 35, 45),
        (80, 0, 0, 80),
        (60, 20, 20, 60),
        (35, 45, 45, 35),
        (30, 30, 50, 50),
        (30, 50, 30, 50),
        (20, 40, 60, 40),
        (10, 50, 70, 30),
        (15, 35, 55, 55),
        (25, 35, 50, 50),
        (10, 30, 70, 50),
        (20, 30, 60, 50),
        (30, 35, 45, 50),
        (37, 43, 43, 37),
        (38, 42, 42, 38),
        (36, 44, 44, 36),
        (33, 47, 47, 33),
        (28, 52, 52, 28),
    ]
    r1 = per_tier_breakdown(4, d4_S, 1.281, d4_comps, 'd4_S160_c1.281', max_depth=3)

    # d=6, S=60 (smaller for speed)
    d6_S = 60
    rng = np.random.default_rng(0)
    d6_comps = []
    for _ in range(15):
        u = rng.dirichlet(np.ones(6) * 1.2)
        c = np.round(u * d6_S).astype(int)
        c[-1] = d6_S - c[:-1].sum()
        if c.min() < 0:
            continue
        d6_comps.append(tuple(int(x) for x in c))
    d6_comps += [
        (10, 10, 10, 10, 10, 10),
        (15, 5, 10, 10, 5, 15),
        (8, 12, 10, 10, 12, 8),
    ]
    r2 = per_tier_breakdown(6, d6_S, 1.281, d6_comps, 'd6_S60_c1.281', max_depth=3)

    out = {'d4': r1, 'd6': r2}
    with open('_probe_bnb_1281.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)

    # Print summary
    for tag, r in out.items():
        print('=' * 70)
        print(f"{tag}: d={r['d']} S={r['S']} c_target={r['c_target']} n={r['n_total']}")
        print(f"  total: {r['total_s']:.1f}s")
        print(f"  tier counts: {r['tier_counts']}")
        print(f"  tier times (s): {r['tier_times_s']}")
        if r['open_cases']:
            print(f"  OPEN ({len(r['open_cases'])}):")
            for oc in r['open_cases'][:6]:
                print(f"    c={oc['c']}")
                print(f"      B1={oc['B1_best']:.4f} F={oc['F_best']:.4f} "
                      f"L={oc['L_best']:.4f} Lj={oc['Lj']}  "
                      f"depth={oc['depth']} subs={oc['sub_cells']}")
                print(f"      times(ms): B1={oc['t_B1_ms']:.1f} F={oc['t_F_ms']:.1f} "
                      f"L={oc['t_L_ms']:.1f} Lj={oc['t_Lj_ms']:.1f} "
                      f"full={oc['t_full_s']:.2f}s")


if __name__ == '__main__':
    main()
