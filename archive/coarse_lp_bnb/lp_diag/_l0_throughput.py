"""L0-only throughput: max enumerable L0 size per d_parent in 30 min.

For each d_parent in {4, 6, 8, 10}:
   - Pick smallest non-vacuous m at c=1.25
   - Time run_level0 with F+Q post-filter (no L at L0; L is overkill at L0)
   - Hard 8-min cap per config
   - Compute throughput: comps_processed / wall_sec
   - Derive: max L0 in 30 min = throughput * 1800
"""
import os, sys, time, json
import numpy as np
from math import comb

ROOT = os.environ.get('CASCADE_ROOT', '/home/ubuntu')
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, ROOT)
from pruning import correction
from run_cascade import run_level0

C_TARGET = 1.25
C_UPPER  = 1.5029

CONFIGS = [
    (2, 20),   # d=4,  S=160,    708K comps
    (2, 50),   # d=4,  S=400,    10.8M comps
    (3, 10),   # d=6,  S=120,    234M comps
    (3, 15),   # d=6,  S=180,    1.7B comps
    (3, 20),   # d=6,  S=240,    7B comps
    (4, 10),   # d=8,  S=160,    633B comps (will likely cap out)
    (5, 10),   # d=10, S=200,    1.76e15 (definitely caps out)
]


def main():
    rows = []
    for n_half, m in CONFIGS:
        d = 2 * n_half
        S = 4 * n_half * m
        if C_TARGET + correction(m, n_half) >= C_UPPER:
            print(f"d={d}, m={m}: VACUOUS"); continue
        n_total = comb(S + d - 1, d - 1)
        print(f"\n--- d={d}, m={m}, S={S}, total_comp={n_total:,} ---", flush=True)
        t0 = time.time()
        try:
            r = run_level0(n_half, m, C_TARGET, verbose=False, use_F=True,
                            use_Q=True, use_L=False)
            wall = time.time() - t0
            n_surv = int(r['n_survivors'])
            tput = n_total / max(wall, 0.001)
            max_30min = int(tput * 1800)
            print(f"  L0 wall = {wall:.2f}s  →  {n_surv:,} survivors", flush=True)
            print(f"  throughput = {tput:,.0f} comps/sec  ⇒  "
                  f"max L0 in 30 min = {max_30min:,} comps", flush=True)
            rows.append({'d': d, 'n_half': n_half, 'm': m, 'S': S,
                          'n_total': n_total, 'n_survivors': n_surv,
                          'wall_sec': wall, 'comps_per_sec': tput,
                          'max_l0_in_30min': max_30min})
        except Exception as e:
            print(f"  ERR: {e}", flush=True)
            rows.append({'d': d, 'n_half': n_half, 'm': m, 'error': str(e)})

    print("\n\n=========================================================")
    print(f"{'d':>3} {'cfg':>10} {'comps':>15} {'wall':>8} {'comps/s':>14} {'max in 30min':>18} {'survs':>10}")
    for r in rows:
        if 'error' in r:
            print(f"{r['d']:>3} n={r['n_half']} m={r['m']:>3}  ERR: {r['error']}")
            continue
        print(f"{r['d']:>3} n={r['n_half']} m={r['m']:>3} {r['n_total']:>15,} "
              f"{r['wall_sec']:>7.1f}s {r['comps_per_sec']:>13,.0f} "
              f"{r['max_l0_in_30min']:>18,} {r['n_survivors']:>10,}")
    with open(os.path.join(ROOT, '_l0_throughput.json'), 'w') as f:
        json.dump(rows, f, indent=2, default=str)


if __name__ == '__main__':
    main()
