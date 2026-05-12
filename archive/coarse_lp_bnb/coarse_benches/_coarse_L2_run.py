"""Driver: load cached hard cells, run Shor + L2, compare cert rates and times.

Bench targets:
  - 'shor-failing' cells: rare survivors where Shor fails. These are the
    cells that would benefit most from L2.
  - 'top hard' cells: the most-negative-tri-net cells (least likely savable
    by ANY tighter cert; this is the worst-case for both Shor and L2).
  - For each config, run both and report.
"""
from __future__ import annotations
import json, time, sys, os
import numpy as np
from typing import List, Tuple

from _coarse_L_bench import (build_A_matrix, tv_at, grad_at,
                              all_windows, cell_cert_shor)
from _coarse_L2_bench import cell_cert_lasserre2


CACHE = '_coarse_L2_hardcells_cache.json'


def _load_cache(d: int, S: int, c_target: float):
    with open(CACHE) as f:
        data = json.load(f)
    key = f'{d}_{S}_{c_target}'
    return data[key]


def find_truly_hard(rows, d: int, S: int, c_target: float,
                     max_screen: int = 200, solver: str = 'MOSEK'):
    """Screen the most-negative-tri-net cells with Shor (best_only mode);
    return the ones where Shor fails (truly_hard).
    """
    truly_hard = []
    rows_sorted = sorted(rows, key=lambda r: r['tri_net'])  # ASC = most negative first
    for k, row in enumerate(rows_sorted[:max_screen]):
        c = np.asarray(row['c'], dtype=np.int32)
        Wstar = tuple(row['tri_W'])
        lb, status = cell_cert_shor(c, S, d, c_target, Wstar, solver=solver)
        cert = lb >= c_target - 1e-9
        if not cert:
            truly_hard.append({**row, 'shor_lb': float(lb)})
    return truly_hard


def bench_compare(d: int, S: int, c_target: float, max_cells: int = 20,
                   solver: str = 'MOSEK', mode: str = 'top_hard',
                   screen_for_truly_hard: int = 200):
    """mode: 'top_hard' = most-negative-tri-net cells.
            'shor_failing' = filter to cells where Shor fails (best-W).
            'shor_failing_multi_w' = filter to cells where Shor fails for
                                     EVERY window (multi-W, more selective).
    """
    print(f"\n=== d={d} S={S} c={c_target} mode={mode} ===")
    cache = _load_cache(d, S, c_target)
    rows = cache['rows']
    n_grid_pass = cache['n_grid_pass']
    n_tri_cert = cache['n_tri_cert']
    print(f"    grid passers: {n_grid_pass:,}  tri_cert: {n_tri_cert:,}  hard: {len(rows):,}")

    if mode == 'shor_failing':
        print(f"    screening up to {screen_for_truly_hard} most-negative cells with Shor...")
        truly_hard = find_truly_hard(rows, d, S, c_target,
                                       max_screen=screen_for_truly_hard,
                                       solver=solver)
        print(f"    truly hard (Shor-fail at W*): {len(truly_hard)}")
        if not truly_hard:
            print("    No Shor-failing cells found in screen.")
            return None
        cells = truly_hard[:max_cells]
    else:  # 'top_hard'
        cells = sorted(rows, key=lambda r: r['tri_net'])[:max_cells]

    # Run both Shor and L2 on these cells, on best window
    n_shor_cert = 0
    n_l2_cert = 0
    n_l2_strict = 0
    n_l2_rescue = 0
    times_shor = []
    times_l2 = []
    detail = []
    for k, row in enumerate(cells):
        c = np.asarray(row['c'], dtype=np.int32)
        Wstar = tuple(row['tri_W'])

        t0 = time.time()
        shor_lb, shor_status = cell_cert_shor(c, S, d, c_target, Wstar, solver=solver)
        dt_shor = time.time() - t0

        t0 = time.time()
        l2_lb, l2_status = cell_cert_lasserre2(c, S, d, c_target, Wstar, solver=solver)
        dt_l2 = time.time() - t0

        shor_cert = shor_lb >= c_target - 1e-9
        l2_cert = l2_lb >= c_target - 1e-9

        if shor_cert: n_shor_cert += 1
        if l2_cert: n_l2_cert += 1
        if l2_lb > shor_lb + 1e-8: n_l2_strict += 1
        if l2_cert and not shor_cert: n_l2_rescue += 1

        times_shor.append(dt_shor)
        times_l2.append(dt_l2)
        detail.append({
            'k': k, 'c': row['c'],
            'tri_net': row['tri_net'],
            'tri_W': row['tri_W'],
            'shor_lb': float(shor_lb),
            'l2_lb': float(l2_lb),
            'gap': float(l2_lb - shor_lb),
            'shor_cert': shor_cert,
            'l2_cert': l2_cert,
            'l2_rescue': l2_cert and not shor_cert,
            't_shor_ms': 1000 * dt_shor,
            't_l2_ms': 1000 * dt_l2,
        })

        if k < 8:
            tag = 'RESCUE' if (l2_cert and not shor_cert) else (
                'cert' if shor_cert else 'fail')
            print(f"    [{k:3d}] c={row['c']}  tri_net={row['tri_net']:+.5f}  "
                  f"shor={shor_lb:.5f}({'C' if shor_cert else 'f'}) "
                  f"L2={l2_lb:.5f}({'C' if l2_cert else 'f'}) "
                  f"gap=+{l2_lb-shor_lb:.2e}  {tag}  "
                  f"T_s={dt_shor*1000:.0f}ms T_L2={dt_l2*1000:.0f}ms")

    times_shor = np.asarray(times_shor)
    times_l2 = np.asarray(times_l2)

    print(f"\n    --- Summary ({mode}) ---")
    print(f"    Cells tested            : {len(cells)}")
    print(f"    Shor certified          : {n_shor_cert}/{len(cells)}")
    print(f"    Lasserre-2 certified    : {n_l2_cert}/{len(cells)}")
    print(f"    L2 strictly > Shor LB   : {n_l2_strict}")
    print(f"    L2 rescues (Shor fail to L2 cert): {n_l2_rescue}")
    if len(times_shor):
        print(f"    Shor time/cell (ms)     : "
              f"med={1000*np.median(times_shor):.1f}  "
              f"p95={1000*np.percentile(times_shor,95):.1f}  "
              f"max={1000*np.max(times_shor):.1f}")
        print(f"    L2  time/cell (ms)      : "
              f"med={1000*np.median(times_l2):.1f}  "
              f"p95={1000*np.percentile(times_l2,95):.1f}  "
              f"max={1000*np.max(times_l2):.1f}")
        print(f"    Time ratio L2/Shor      : "
              f"med={np.median(times_l2)/max(1e-9,np.median(times_shor)):.2f}x")

    return {
        'd': d, 'S': S, 'c_target': c_target, 'mode': mode,
        'n_total_hard': len(rows),
        'n_cells_tested': len(cells),
        'n_shor_cert': n_shor_cert,
        'n_l2_cert': n_l2_cert,
        'n_l2_strict': n_l2_strict,
        'n_l2_rescue': n_l2_rescue,
        't_shor_med_ms': float(1000 * np.median(times_shor)) if len(times_shor) else None,
        't_l2_med_ms': float(1000 * np.median(times_l2)) if len(times_l2) else None,
        't_shor_p95_ms': float(1000 * np.percentile(times_shor, 95)) if len(times_shor) else None,
        't_l2_p95_ms': float(1000 * np.percentile(times_l2, 95)) if len(times_l2) else None,
        'detail': detail,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_cells_top', type=int, default=12)
    ap.add_argument('--max_cells_truly', type=int, default=12)
    ap.add_argument('--screen_truly', type=int, default=300)
    ap.add_argument('--out', default='_coarse_L2_compare.json')
    args = ap.parse_args()

    all_results = []

    # Three configs: d=4 S=20 c=1.20, d=6 S=15 c=1.20, d=8 S=12 c=1.20.
    for d, S, c in [(4, 20, 1.20), (6, 15, 1.20), (8, 12, 1.20)]:
        r_top = bench_compare(d, S, c, max_cells=args.max_cells_top,
                                mode='top_hard')
        if r_top:
            all_results.append(r_top)
        r_truly = bench_compare(d, S, c, max_cells=args.max_cells_truly,
                                  mode='shor_failing',
                                  screen_for_truly_hard=args.screen_truly)
        if r_truly:
            all_results.append(r_truly)

    # Trim 'detail' for output
    out_results = []
    for r in all_results:
        rcopy = dict(r)
        if 'detail' in rcopy and len(rcopy['detail']) > 25:
            rcopy['detail'] = rcopy['detail'][:25]
        out_results.append(rcopy)
    with open(args.out, 'w') as f:
        json.dump(out_results, f, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
