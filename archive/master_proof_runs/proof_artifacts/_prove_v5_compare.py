"""Run stage1 (numba F) then compare v4 vs v5 on the residue at d=8 S=16.

Goal: quantify how many of v4's L_joint closures (the unsound branch) actually
hold under v5's sound L_joint, and how many genuinely open up.

Designed for a single short run (≤10 minutes) — caps total stage-2 cells at
N_MAX for tractability and reports both v4 and v5 closure stats side by side.
"""
from __future__ import annotations
import os, sys, time, json, logging
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_canonical_compositions_batched
import _coarse_bnb_v4 as v4
import _coarse_bnb_v5 as v5
from _d16_F_bench import _prune_coarse_count_cell


def run(d=8, S=16, c_target=1.275, N_MAX=300, time_budget=400.0):
    print(f"\n{'='*72}")
    print(f"v4-vs-v5 prove compare:  d={d} S={S} c_target={c_target}")
    print(f"  cap on stage-2 cells: {N_MAX}  total budget: {time_budget}s")
    print(f"{'='*72}\n")

    # ----- Stage 1: numba F screen -----
    print("Stage 1: numba F screen + cell-net pass")
    warm = np.zeros((1, d), dtype=np.int32); warm[0, 0] = S
    _prune_coarse_count_cell(warm, d, S, c_target)
    t0 = time.time()
    residue = []
    n_total = 0
    n_grid_pruned = 0
    n_grid_surv = 0
    n_cell_uncertain = 0
    for batch in generate_canonical_compositions_batched(d, S,
                                                            batch_size=200_000):
        batch_i32 = batch.astype(np.int32)
        survived, neg_mask, n_neg, min_net = _prune_coarse_count_cell(
            batch_i32, d, S, c_target)
        n_total += len(batch)
        n_grid_pruned += int((~survived).sum())
        n_grid_surv += int(survived.sum())
        n_cell_uncertain += int(n_neg)
        residue_mask = (~survived) & neg_mask
        for idx in np.where(residue_mask)[0]:
            residue.append(batch[idx].astype(np.int64).copy())
        if time.time() - t0 > 60:
            break
    print(f"  total: {n_total:,}  grid_pruned: {n_grid_pruned:,}  "
          f"cell-uncertain residue: {n_cell_uncertain:,}  "
          f"grid_surv: {n_grid_surv:,}  ({time.time()-t0:.1f}s)")
    print(f"  residue collected: {len(residue):,}")

    # ----- Stage 2: compare v4 vs v5 on capped residue -----
    if N_MAX and len(residue) > N_MAX:
        # sample uniformly
        idx = np.linspace(0, len(residue) - 1, N_MAX).astype(int)
        sample = [residue[i] for i in idx]
    else:
        sample = residue
    print(f"\nStage 2 sample: {len(sample)} cells")

    windows = v4.build_all_windows(d)
    v4.get_sdp_template(d); v4.get_joint_template(d, 4)

    v4_counts = {}
    v5_counts = {}
    # tracks: per-cell (v4_tier, v5_tier)
    disagree = []  # cases where v4 says certified but v5 says open
    v4_total_t = v5_total_t = 0.0
    v4_open = 0
    v5_open = 0
    v4_unsound_count = 0  # v4 closed via L_joint while v5 did not certify

    t_start = time.time()
    for k, c in enumerate(sample):
        if time.time() - t_start > time_budget:
            print(f"  TIME BUDGET HIT at {k}/{len(sample)}")
            break
        cf = c.astype(np.float64)
        t0 = time.time()
        r4 = v4.certify_composition(cf, S, d, c_target, windows=windows,
                                       max_depth=3)
        v4_total_t += time.time() - t0
        v4_tier = r4.tier_used if r4.certified else f'OPEN:{r4.tier_used}'
        v4_counts[v4_tier] = v4_counts.get(v4_tier, 0) + 1
        if not r4.certified: v4_open += 1

        t0 = time.time()
        r5 = v5.certify_composition(cf, S, d, c_target, windows=windows,
                                       max_depth=3)
        v5_total_t += time.time() - t0
        v5_tier = r5.tier_used if r5.certified else f'OPEN:{r5.tier_used}'
        v5_counts[v5_tier] = v5_counts.get(v5_tier, 0) + 1
        if not r5.certified: v5_open += 1

        # Track v4 unsound: v4 closes via L_joint AND v5 does NOT close
        if (r4.certified and r4.tier_used == 'L_joint' and not r5.certified):
            v4_unsound_count += 1
            if len(disagree) < 20:
                disagree.append({'c': c.tolist(),
                                 'v4_tier': r4.tier_used,
                                 'v4_bound': float(r4.bound),
                                 'v5_tier': r5.tier_used,
                                 'v5_depth': r5.depth_used})

        if (k+1) % 50 == 0:
            print(f"  [{k+1}/{len(sample)}]  v4: {v4_counts}  v5_open={v5_open}  "
                  f"v4_unsound={v4_unsound_count}  ({time.time()-t_start:.1f}s)")

    out = {
        'd': d, 'S': S, 'c_target': c_target,
        'sample_n': len(sample),
        'residue_total': len(residue),
        'v4_counts': v4_counts,
        'v5_counts': v5_counts,
        'v4_open': v4_open,
        'v5_open': v5_open,
        'v4_unsound_count': v4_unsound_count,
        'v4_total_t': v4_total_t,
        'v5_total_t': v5_total_t,
        'disagree_samples': disagree,
    }
    print(f"\n{'='*72}")
    print(f"RESULTS  (sample of {len(sample)} residue cells)")
    print(f"{'='*72}")
    print(f"  v4 ({v4_total_t:.1f}s):  {v4_counts}")
    print(f"  v5 ({v5_total_t:.1f}s):  {v5_counts}")
    print(f"\n  v4 closed but v5 OPEN:  {v4_unsound_count} cells")
    if v4_unsound_count > 0:
        print(f"  Scaling to full residue ({len(residue):,}):  "
              f"estimated ~{int(round(v4_unsound_count / len(sample) * len(residue)))} "
              f"falsely-closed cells")
    print(f"  v4 open: {v4_open}  v5 open: {v5_open}")

    with open('_prove_v5_compare.json', 'w') as fp:
        json.dump(out, fp, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else str(x))
    print(f"\n  saved: _prove_v5_compare.json")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--d', type=int, default=8)
    ap.add_argument('--S', type=int, default=16)
    ap.add_argument('--c_target', type=float, default=1.275)
    ap.add_argument('--N_MAX', type=int, default=300)
    ap.add_argument('--time_budget', type=float, default=400.0)
    args = ap.parse_args()
    run(args.d, args.S, args.c_target, args.N_MAX, args.time_budget)
