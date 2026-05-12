"""Sweep (d, S) at c_target=1.25 with numba F to find configs where the
cell-net cert closes 100% of cells (min_net > 0)."""
import os, sys, time, logging
logging.getLogger('cvxpy').setLevel(logging.ERROR)
import numpy as np

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_dir, 'cloninger-steinerberger', 'cpu'))

from compositions import generate_canonical_compositions_batched
from _d16_F_bench import _prune_coarse_count_cell


def bench(d, S, c_target, time_budget=180.0):
    print(f"\n=== d={d}, S={S}, c_target={c_target} ===", flush=True)
    # JIT warm
    warm = np.zeros((1, d), dtype=np.int32)
    warm[0, 0] = S
    _prune_coarse_count_cell(warm, d, S, c_target)

    n_total = 0
    n_grid_pruned = 0
    n_grid_surv = 0
    n_cell_uncertain = 0
    min_net = np.inf
    t0 = time.time()
    for batch in generate_canonical_compositions_batched(d, S, batch_size=200_000):
        if time.time() - t0 > time_budget:
            print(f"  TIME BUDGET REACHED", flush=True)
            break
        survived, neg_mask, n_neg, mn = _prune_coarse_count_cell(
            batch.astype(np.int32), d, S, c_target)
        n_total += len(batch)
        n_grid_pruned += int((~survived).sum())
        n_grid_surv += int(survived.sum())
        n_cell_uncertain += int(n_neg)
        if mn < min_net:
            min_net = float(mn)
    elapsed = time.time() - t0
    pct_uncert = 100 * n_cell_uncertain / max(n_total, 1)
    pct_surv = 100 * n_grid_surv / max(n_total, 1)
    print(f"  total processed: {n_total:,}", flush=True)
    print(f"  grid-pruned:     {n_grid_pruned:,}", flush=True)
    print(f"  grid-surv:       {n_grid_surv:,}  ({pct_surv:.4f}%)", flush=True)
    print(f"  cell-uncertain:  {n_cell_uncertain:,}  ({pct_uncert:.4f}%)", flush=True)
    print(f"  min_net:         {min_net:.6f}", flush=True)
    print(f"  elapsed:         {elapsed:.1f}s", flush=True)
    if min_net > 0 and n_grid_surv == 0:
        print(f"  *** PROVEN @ this config: all cells F-certified ***", flush=True)
    return {'d': d, 'S': S, 'c_target': c_target,
             'n_total': n_total, 'n_grid_surv': n_grid_surv,
             'n_cell_uncertain': n_cell_uncertain, 'min_net': min_net,
             'elapsed_s': elapsed}


if __name__ == '__main__':
    c_target = 1.25
    # Search progressively for the sweet spot: high d, moderate S
    configs = [
        (8, 16),   # 245K  cells, very fast
        (8, 24),   # 2.6M  cells
        (10, 20),  # 10M   cells
        (12, 12),  # 1.3M  cells (already tested: 261K residue, bad)
        (12, 24),  # 417M  cells, too big
        (16, 16),  # 150M  cells (75 min) — likely sweet spot for tight cells
    ]
    results = []
    for d, S in configs:
        try:
            r = bench(d, S, c_target, time_budget=120.0)
            results.append(r)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)

    print(f"\n=== SUMMARY @ c=1.25 ===", flush=True)
    print(f"{'d':>3} {'S':>3} {'total':>14} {'grid_surv':>9} {'uncert%':>8} {'min_net':>10} {'elapsed':>8}", flush=True)
    for r in results:
        print(f"{r['d']:>3} {r['S']:>3} {r['n_total']:>14,} {r['n_grid_surv']:>9,} "
              f"{100*r['n_cell_uncertain']/max(r['n_total'],1):>7.4f}% "
              f"{r['min_net']:>+10.4f} {r['elapsed_s']:>7.1f}s", flush=True)
