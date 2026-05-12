#!/usr/bin/env python
"""Benchmark: Lasserre highd solver across d and order combinations.

Usage:
    python tests/bench_lasserre_highd.py
"""
import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from lasserre_highd import solve_highd_sparse

val_d_known = {
    4: 1.102, 6: 1.171, 8: 1.205, 10: 1.241,
    12: 1.271, 14: 1.284, 16: 1.319,
    32: 1.336, 64: 1.384, 128: 1.420,
}

CONFIGS = [
    # (d, order, bw, max_cg_rounds, n_bisect)
    (4,  2, 3, 5, 8),
    (4,  3, 3, 5, 8),
    (6,  2, 4, 5, 8),
    (6,  3, 4, 5, 8),
    (8,  2, 4, 5, 8),
    (8,  3, 4, 5, 8),
    (10, 2, 4, 5, 8),
    (10, 3, 4, 5, 8),
    (12, 2, 4, 5, 8),
    (12, 3, 4, 5, 8),
    (16, 2, 4, 8, 8),
    (16, 3, 4, 8, 8),
    (24, 2, 6, 8, 8),
    (24, 3, 6, 8, 8),
]


def main():
    print(f"Lasserre Highd Benchmark — {datetime.now().isoformat()}")
    print(f"Python: {sys.version}")
    try:
        import mosek
        print(f"MOSEK: {mosek.Env.getversion()}")
    except Exception as e:
        print(f"MOSEK: {e}")
    print()

    results = []
    for d, order, bw, cg_rounds, n_bisect in CONFIGS:
        print(f"\n{'='*70}")
        print(f"CONFIG: d={d} O{order} bw={bw}")
        print(f"{'='*70}")
        t0 = time.time()
        try:
            r = solve_highd_sparse(
                d=d, order=order, bandwidth=bw,
                max_cg_rounds=cg_rounds, n_bisect=n_bisect,
                verbose=True,
            )
            elapsed = time.time() - t0
            vd = val_d_known.get(d, 0)
            gc = (r['lb'] - 1) / (vd - 1) * 100 if vd > 1 else 0
            sound = r['lb'] <= vd + 1e-6 if vd > 0 else True
            entry = {
                'd': d, 'order': order, 'bw': bw,
                'n_y': r['n_y'], 'lb': r['lb'],
                'gap_closure': gc,
                'n_active_windows': r.get('n_active_windows', 0),
                'elapsed': elapsed,
                'sound': sound,
            }
            results.append(entry)
            print(f"\n>>> d={d} O{order}: lb={r['lb']:.8f} gc={gc:.1f}% "
                  f"n_y={r['n_y']:,} time={elapsed:.1f}s "
                  f"{'SOUND' if sound else '*** UNSOUND ***'}")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"\n>>> d={d} O{order}: FAILED after {elapsed:.1f}s: {e}")
            results.append({
                'd': d, 'order': order, 'bw': bw,
                'error': str(e), 'elapsed': elapsed,
            })

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"{'d':>3} {'O':>2} {'bw':>3} {'n_y':>10} {'lb':>12} {'gc%':>7} "
          f"{'wins':>6} {'time':>9} {'sound':>6}")
    print(f"{'-'*80}")
    for r in results:
        if 'error' in r:
            print(f"{r['d']:>3} {r['order']:>2} {r['bw']:>3} "
                  f"{'ERROR':>10} {'-':>12} {'-':>7} "
                  f"{'-':>6} {r['elapsed']:>8.1f}s {'-':>6}")
        else:
            print(f"{r['d']:>3} {r['order']:>2} {r['bw']:>3} "
                  f"{r['n_y']:>10,} {r['lb']:>12.6f} "
                  f"{r['gap_closure']:>6.1f}% "
                  f"{r['n_active_windows']:>6} "
                  f"{r['elapsed']:>8.1f}s "
                  f"{'OK' if r['sound'] else 'FAIL':>6}")
    print(f"{'='*80}")

    # Save JSON
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'data', 'bench_lasserre_highd.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
