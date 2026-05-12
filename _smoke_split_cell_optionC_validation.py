"""Validate Option C (trace + Cauchy-Schwarz cuts in Shor SDP) on the
14 known L-survivors at d=10 (n=5, m=5, c=1.28).

Prior baseline: 9/14 split-pruned (the optimal split-cell I built).
With Option C cuts ON in `l_direct.prune_L_direct` (default), check if
prune count goes up.

Soundness already verified:
  - 14/14 parent-cell L decisions match with vs without cuts (the cells
    were L-survivors before; they remain L-survivors).
  - At d=6, 6/6 cells match.
The cuts can only TIGHTEN the SDP (more `infeasible`, never less), so
this test should give ≥ 9/14.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(_HERE, 'cloninger-steinerberger', 'cpu'))

from post_filters import apply_split_cell_filter_parallel


def main():
    cache = os.path.join(_HERE, '_smoke_split_cell_l_survivors.json')
    with open(cache) as fp:
        data = json.load(fp)
    cells = np.array(data['l_survivors'], dtype=np.int32)
    print(f'Loaded {len(cells)} L-survivors (n_half=5, m=5, c=1.28, d=10).')

    n_workers = int(os.environ.get('N_WORKERS',
                                     max(1, min(8, (os.cpu_count() or 4) - 2))))
    print(f'n_workers={n_workers}, max_depth=1 (Option A baseline)')
    print(f'Trace+CS cuts (Option C): ALWAYS ON in prune_L_direct')

    # Run each cell individually to capture per-cell timing
    results = []
    n_split = 0
    t_global = time.time()
    for i, c_int in enumerate(cells):
        arr = np.array([c_int], dtype=np.int32)
        t0 = time.time()
        out = apply_split_cell_filter_parallel(
            arr, n_half_child=5, m=5, c_target=1.28,
            n_workers=n_workers, max_d=10, early_terminate=True,
            max_depth=1)
        t_one = time.time() - t0
        sp = (len(out) == 0)
        if sp: n_split += 1
        print(f'  [{i+1:>2}/{len(cells)}] {c_int.tolist()}: '
              f'split_pruned={sp}  wall={t_one:.1f}s')
        results.append({
            'c_int': c_int.tolist(),
            'split_pruned': bool(sp),
            'time_s': float(t_one),
        })

    total = time.time() - t_global
    print(f'\n========================================')
    print(f'OPTION C VALIDATION: {n_split}/{len(cells)} split-pruned in {total:.1f}s')
    print(f'Prior baseline (without trace cuts): 9/14 split-pruned')
    print(f'Delta: {n_split - 9:+d}')
    print(f'========================================')

    out_path = os.path.join(_HERE, '_smoke_split_cell_optionC_validation.json')
    with open(out_path, 'w') as fp:
        json.dump({
            'config': {'n_half': 5, 'm': 5, 'c_target': 1.28, 'd': 10,
                       'max_depth': 1, 'trace_cuts': True,
                       'n_workers': n_workers},
            'baseline_n_split': 9,
            'this_run_n_split': n_split,
            'delta': n_split - 9,
            'total_time_s': total,
            'results': results,
        }, fp, indent=2)
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    import multiprocessing as _mp
    _mp.freeze_support()
    main()
