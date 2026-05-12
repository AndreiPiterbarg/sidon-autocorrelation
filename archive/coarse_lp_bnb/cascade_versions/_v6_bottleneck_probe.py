"""Bottleneck probe for _coarse_bnb_v6.py at c_target=1.281.

Goals (fast, <3 min):
  P1. Per-tier wall clock on root cells AND with default cascade at depth=3.
  P2. SDP compile vs solve breakdown (first solve vs subsequent).
  P3. Closure histogram at d=4..d=8 with c_target=1.281.
  P4. Identify tightest open cells (which tier had the smallest negative gap),
      and where the depth budget is exhausted.
  P5. FW iter histogram — how often FW actually improves over best L_single.
  P6. Cache stats: how often the bundle and SDP template are rebuilt.
"""
from __future__ import annotations
import os, sys, time, json, logging, warnings
logging.getLogger('cvxpy').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

import _coarse_bnb_v3 as v3
import _coarse_bnb_v4 as v4
import _coarse_bnb_v5 as v5
import _coarse_bnb_v6 as v6


def root_tier_profile(d, S, c_target, compositions, label):
    """For each composition, time each individual tier on the ROOT cell."""
    windows = v3.build_all_windows(d)
    bundle = v6.get_bundle(windows)
    # Warm template
    cell0 = v3.Cell.from_integer_composition(
        np.asarray(compositions[0], dtype=np.float64), S)
    cache0 = v3.CellCache.build(cell0)
    if v6.HAS_CVXPY:
        v6.get_sdp_template_v6(d)
        # first solve to compile
        v6.tier_L_single_v6(cache0, windows[0], 1.0)

    rows = []
    for c_tup in compositions:
        c = np.asarray(c_tup, dtype=np.float64)
        if abs(c.sum() - S) > 1e-9 or c.min() < 0:
            continue
        cell = v3.Cell.from_integer_composition(c, S)
        if not cell.is_simplex_feasible():
            continue
        cache = v3.CellCache.build(cell)
        rec = {'c': c_tup, 'd': d, 'S': S}

        # B1 vectorized
        t0 = time.time()
        b1_vec = v6.tier_B1_vec(cell, bundle, c_target)
        rec['t_B1_ms'] = (time.time() - t0) * 1000
        rec['B1_max'] = float(b1_vec.max())

        # B1u vectorized
        t0 = time.time()
        b1u_vec = v6.tier_B1u_vec(cell, bundle, c_target)
        rec['t_B1u_ms'] = (time.time() - t0) * 1000
        rec['B1u_max'] = float(b1u_vec.max())

        # B1diag per-W (v5 path, kept in v6)
        t0 = time.time()
        b1d_max = -np.inf
        for W in windows:
            b = v5.tier_B1diag_amgm(cell, W, c_target)
            if b > b1d_max: b1d_max = b
        rec['t_B1diag_ms'] = (time.time() - t0) * 1000
        rec['B1diag_max'] = float(b1d_max)

        # F all
        t0 = time.time()
        v3.compute_F_all_windows(cache, windows, c_target)
        rec['t_F_ms'] = (time.time() - t0) * 1000
        rec['F_best'] = float(cache.f_best_bound)

        # m_W vec (S4)
        t0 = time.time()
        mw_vec = v6.mwQ_vec(cell, bundle, c_target, cache.mu_star)
        rec['t_mwvec_ms'] = (time.time() - t0) * 1000

        # L_single top-3
        t0 = time.time()
        L_best = -np.inf
        L_each = []
        if v6.HAS_CVXPY:
            K_L = min(3, len(windows))
            cands = v6.select_L_candidates_v6(mw_vec, bundle, K=K_L,
                                                include_widest=2)
            for w_idx in cands:
                t1 = time.time()
                lb, _ = v6.tier_L_single_v6(cache, windows[w_idx], c_target)
                L_each.append({'w_idx': int(w_idx), 'lb': float(lb),
                                't_ms': (time.time() - t1) * 1000})
                if lb > L_best: L_best = lb
        rec['t_L_ms'] = (time.time() - t0) * 1000
        rec['L_best'] = float(L_best)
        rec['L_each'] = L_each

        # L_joint
        t0 = time.time()
        Lj_best = -np.inf
        joint_info = None
        if v6.HAS_CVXPY:
            cands_j = v6.select_L_candidates_v6(mw_vec, bundle, K=4,
                                                include_widest=2)
            top_j = [windows[i] for i in cands_j]
            Lj_best, joint_info = v6.tier_L_joint_lagrangian_v6(
                cache, top_j, c_target, max_fw_iters=6)
        rec['t_Lj_ms'] = (time.time() - t0) * 1000
        rec['Lj_best'] = float(Lj_best)
        rec['Lj_iters'] = joint_info.get('iters_used', 0) if joint_info else 0
        rec['Lj_method'] = joint_info.get('method', '') if joint_info else ''

        # Bound gap to certification: max over tiers minus 0
        rec['root_best'] = max(rec['B1_max'], rec['B1u_max'], rec['B1diag_max'],
                                 rec['F_best'], rec['L_best'], rec['Lj_best'])
        rec['root_certified'] = rec['root_best'] > 0

        rows.append(rec)
    return rows


def cascade_profile(d, S, c_target, compositions, max_depth=3):
    windows = v3.build_all_windows(d)
    v6.get_bundle(windows)
    if v6.HAS_CVXPY:
        v6.get_sdp_template_v6(d)
        cell0 = v3.Cell.from_integer_composition(
            np.asarray(compositions[0], dtype=np.float64), S)
        cache0 = v3.CellCache.build(cell0)
        v6.tier_L_single_v6(cache0, windows[0], 1.0)
    counts = {}
    times = {}
    t0 = time.time()
    for c_tup in compositions:
        c = np.asarray(c_tup, dtype=np.float64)
        if abs(c.sum() - S) > 1e-9 or c.min() < 0:
            continue
        t1 = time.time()
        r = v6.certify_composition(c, S, d, c_target, windows=windows,
                                     max_depth=max_depth)
        dt = time.time() - t1
        tier = r.tier_used if r.certified else f'OPEN:{r.tier_used}'
        counts[tier] = counts.get(tier, 0) + 1
        times[tier] = times.get(tier, 0.0) + dt
    return {'counts': counts, 'times_s': times,
             'total_s': time.time() - t0, 'n': len(compositions)}


def main():
    print("=" * 72)
    print("v6 BOTTLENECK PROBE @ c_target=1.281")
    print("=" * 72)
    c_target = 1.281

    # ----- d=4 S=80 (near-uniform compositions) -----
    print("\n--- d=4, S=80, c=1.281 (root profile) ---")
    d4_comps = [
        (20, 20, 20, 20),  # exactly uniform
        (19, 21, 21, 19),
        (18, 22, 22, 18),
        (15, 25, 25, 15),
        (10, 30, 30, 10),
        (5, 35, 35, 5),
        (22, 18, 18, 22),
        (25, 15, 15, 25),
        (30, 10, 10, 30),
        (15, 15, 25, 25),
        (15, 25, 15, 25),
        (10, 20, 30, 20),
        (12, 18, 22, 28),
    ]
    rows4 = root_tier_profile(4, 80, c_target, d4_comps, 'd4S80')
    print(f"  n={len(rows4)}  certified at root: "
          f"{sum(r['root_certified'] for r in rows4)}/{len(rows4)}")
    # Per-tier average time
    if rows4:
        for k in ('t_B1_ms', 't_B1u_ms', 't_B1diag_ms', 't_F_ms',
                  't_mwvec_ms', 't_L_ms', 't_Lj_ms'):
            avg = np.mean([r[k] for r in rows4])
            print(f"    avg {k}: {avg:.1f}")

    casc4 = cascade_profile(4, 80, c_target, d4_comps, max_depth=3)
    print(f"  cascade d=4 max_depth=3: {casc4['total_s']:.2f}s  "
          f"{casc4['counts']}")

    # ----- d=6, S=30 -----
    print("\n--- d=6, S=30, c=1.281 (root profile) ---")
    rng = np.random.default_rng(20260511)
    d6_comps = []
    while len(d6_comps) < 12:
        u = rng.dirichlet(np.ones(6) * 1.5)
        c = np.round(u * 30).astype(int)
        c[-1] = 30 - c[:-1].sum()
        if c.min() < 0: continue
        d6_comps.append(tuple(int(x) for x in c))
    d6_comps += [(5, 5, 5, 5, 5, 5), (6, 4, 5, 5, 4, 6)]
    rows6 = root_tier_profile(6, 30, c_target, d6_comps, 'd6S30')
    print(f"  n={len(rows6)}  certified at root: "
          f"{sum(r['root_certified'] for r in rows6)}/{len(rows6)}")
    for k in ('t_B1_ms', 't_B1u_ms', 't_B1diag_ms', 't_F_ms',
              't_mwvec_ms', 't_L_ms', 't_Lj_ms'):
        avg = np.mean([r[k] for r in rows6])
        print(f"    avg {k}: {avg:.1f}")

    casc6 = cascade_profile(6, 30, c_target, d6_comps, max_depth=3)
    print(f"  cascade d=6 max_depth=3: {casc6['total_s']:.2f}s  "
          f"{casc6['counts']}")

    # ----- d=8, S=16 -----
    print("\n--- d=8, S=16, c=1.281 (root profile) ---")
    rng = np.random.default_rng(20260511)
    d8_comps = []
    while len(d8_comps) < 12:
        u = rng.dirichlet(np.ones(8) * 1.5)
        c = np.round(u * 16).astype(int)
        c[-1] = 16 - c[:-1].sum()
        if c.min() < 0: continue
        d8_comps.append(tuple(int(x) for x in c))
    d8_comps += [(2, 2, 2, 2, 2, 2, 2, 2)]
    rows8 = root_tier_profile(8, 16, c_target, d8_comps, 'd8S16')
    print(f"  n={len(rows8)}  certified at root: "
          f"{sum(r['root_certified'] for r in rows8)}/{len(rows8)}")
    for k in ('t_B1_ms', 't_B1u_ms', 't_B1diag_ms', 't_F_ms',
              't_mwvec_ms', 't_L_ms', 't_Lj_ms'):
        avg = np.mean([r[k] for r in rows8])
        print(f"    avg {k}: {avg:.1f}")

    casc8 = cascade_profile(8, 16, c_target, d8_comps, max_depth=3)
    print(f"  cascade d=8 max_depth=3: {casc8['total_s']:.2f}s  "
          f"{casc8['counts']}")

    # ----- Aggregate: top "hard" cells -----
    all_rows = rows4 + rows6 + rows8
    open_rows = [r for r in all_rows if not r['root_certified']]
    print(f"\n--- HARD CELLS (root open) ---  n={len(open_rows)}")
    open_rows.sort(key=lambda r: r['root_best'], reverse=True)
    for r in open_rows[:10]:
        print(f"  d={r['d']} S={r['S']} c={r['c']}")
        print(f"    B1={r['B1_max']:+.4f}  B1u={r['B1u_max']:+.4f}  "
              f"B1diag={r['B1diag_max']:+.4f}  F={r['F_best']:+.4f}  "
              f"L={r['L_best']:+.4f}  Lj={r['Lj_best']:+.4f}  "
              f"(gap to cert = {-r['root_best']:+.4f})")
        print(f"    times(ms): B1={r['t_B1_ms']:.1f} B1u={r['t_B1u_ms']:.1f} "
              f"B1d={r['t_B1diag_ms']:.1f} F={r['t_F_ms']:.1f} "
              f"L={r['t_L_ms']:.1f}({len(r['L_each'])} solves) "
              f"Lj={r['t_Lj_ms']:.1f}({r['Lj_iters']}+1 solves)")

    # ----- FW improvement histogram -----
    print("\n--- FW IMPROVEMENT HISTOGRAM ---")
    improvements = []
    for r in all_rows:
        if r['L_best'] > -np.inf and r['Lj_best'] > -np.inf:
            improvements.append(r['Lj_best'] - r['L_best'])
    if improvements:
        improvements = np.array(improvements)
        print(f"  n={len(improvements)}")
        print(f"  Lj - L:  min={improvements.min():.5f}  "
              f"median={np.median(improvements):.5f}  "
              f"max={improvements.max():.5f}")
        print(f"  improvements >=  1e-3: {(improvements >= 1e-3).sum()}")
        print(f"  improvements >=  1e-5: {(improvements >= 1e-5).sum()}")
        print(f"  improvements <= -1e-5: {(improvements <= -1e-5).sum()}  "
              "(L > Lj — Lj is suboptimal warm-start)")

    out = {
        'rows4': rows4, 'rows6': rows6, 'rows8': rows8,
        'cascade': {'d4': casc4, 'd6': casc6, 'd8': casc8},
    }
    with open('_v6_bottleneck.json', 'w') as f:
        json.dump(out, f, indent=2, default=lambda x:
                    float(x) if hasattr(x, 'dtype') else str(x))
    print("\nSaved: _v6_bottleneck.json")


if __name__ == '__main__':
    main()
