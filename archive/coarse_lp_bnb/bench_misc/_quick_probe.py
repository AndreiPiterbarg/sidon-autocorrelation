"""Quick representative sweep at c=1.25 to identify high-vs-low (d, m) winners.

Per (n_half, m) config:
   1. Run L0 with --use_F (full enumeration; sub-second at d0=4, ~minute at d0=6)
   2. If L0 closes: report PROVEN.
   3. Otherwise sample at most 2 L1 parents (fixed wall cap), report:
      - L0 survivor count
      - avg L1 survivors per parent (from the 2-sample)
      - rough projection: L1 total survivors = avg * L0_survivors
      - feasibility verdict for L2
Tight wall budget: 30s/config max (skip configs that exceed).
"""
import os
import signal
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger', 'cpu'))

from pruning import correction, count_compositions
from run_cascade import process_parent_fused, run_level0


C_UPPER = 1.5029
C_TARGET = 1.25
WALL_PER_CONFIG = 45.0   # hard cap per config
WALL_PER_PARENT = 8.0    # hard cap per L1 parent


class TimeoutErr(Exception):
    pass


def alarm_handler(signum, frame):
    raise TimeoutErr()


def probe_one(n_half: int, m: int) -> dict:
    """Run L0 then up to 2 L1 parents under wall cap."""
    d0 = 2 * n_half
    S0 = 4 * n_half * m
    corr = correction(m, n_half)
    if C_TARGET + corr >= C_UPPER:
        return {'status': 'VACUOUS'}

    nc = count_compositions(d0, S0)
    out = {
        'n_half': n_half, 'm': m, 'd0': d0, 'S0': S0,
        'n_compositions': int(nc), 'corr': float(corr),
        'thresh': float(C_TARGET + corr),
    }

    # L0
    t0 = time.time()
    try:
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(int(WALL_PER_CONFIG))
        result = run_level0(n_half, m, C_TARGET, verbose=False, use_F=True)
    except TimeoutErr:
        out['status'] = 'L0_TIMEOUT'
        out['l0_wall'] = round(time.time() - t0, 2)
        signal.alarm(0)
        return out
    finally:
        signal.alarm(0)
    out['l0_wall'] = round(time.time() - t0, 2)
    survivors = result['survivors']
    n_surv = int(result['n_survivors'])
    out['l0_survivors'] = n_surv

    if n_surv == 0:
        out['status'] = 'PROVEN_AT_L0'
        return out

    # L1 sample: up to 2 parents under wall cap
    d_child = 2 * d0
    n_half_child = d_child // 2
    if len(survivors) > 2:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(survivors), 2, replace=False)
        sample = survivors[idx]
    else:
        sample = survivors

    l1_results = []
    remaining = WALL_PER_CONFIG - (time.time() - t0)
    for i, parent in enumerate(sample):
        if remaining <= WALL_PER_PARENT:
            break
        try:
            signal.signal(signal.SIGALRM, alarm_handler)
            signal.alarm(int(min(WALL_PER_PARENT, remaining)))
            tp = time.time()
            surv_i, n_children_i = process_parent_fused(
                parent, m, C_TARGET, n_half_child,
                use_flat_threshold=False, use_F=True, skip_sdp_cert=True)
            tp_elapsed = time.time() - tp
            l1_results.append({
                'children': int(n_children_i),
                'survivors': int(len(surv_i)),
                'wall': round(tp_elapsed, 2),
            })
        except TimeoutErr:
            l1_results.append({'children': -1, 'survivors': -1, 'wall': -1})
            break
        finally:
            signal.alarm(0)
        remaining = WALL_PER_CONFIG - (time.time() - t0)

    out['l1_sample'] = l1_results
    if not l1_results:
        out['status'] = 'L1_NO_SAMPLE'
        return out

    avg_surv = sum(r['survivors'] for r in l1_results) / len(l1_results)
    out['l1_avg_survivors_per_parent'] = avg_surv
    out['l1_total_proj'] = int(avg_surv * n_surv)
    out['status'] = 'OK'
    return out


def main():
    configs = []
    # Wide grid, shallow probe
    for n_half in (2, 3):
        for m in (10, 15, 20, 30, 50):
            configs.append((n_half, m))
    # n_half=4 only with small m (d0=8, L0 expensive)
    for m in (10, 15):
        configs.append((4, m))

    print(f"{'n':>2} {'m':>3} {'d0':>3} {'L0_comps':>14} {'L0_surv':>10} "
          f"{'L0_wall':>7} {'L1_avg':>9} {'L1_total':>12} {'status'}")
    print('-' * 90)
    rows = []
    for n_half, m in configs:
        r = probe_one(n_half, m)
        if r.get('status') == 'VACUOUS':
            print(f"{n_half:>2} {m:>3} {2*n_half:>3} VACUOUS")
            continue
        if r.get('status') == 'L0_TIMEOUT':
            print(f"{n_half:>2} {m:>3} {2*n_half:>3} {r['n_compositions']:>14,} "
                  f"L0_TIMEOUT  ({r['l0_wall']}s)")
            continue
        l1_avg = r.get('l1_avg_survivors_per_parent', float('nan'))
        l1_total = r.get('l1_total_proj', -1)
        print(f"{n_half:>2} {m:>3} {r['d0']:>3} {r['n_compositions']:>14,} "
              f"{r['l0_survivors']:>10,} {r['l0_wall']:>6.2f}s "
              f"{l1_avg:>9.1f} {l1_total:>12,} {r['status']}")
        rows.append(r)
    print()
    print('Summary:')
    print('  PROVEN_AT_L0:', [(r['n_half'], r['m']) for r in rows
                               if r['status'] == 'PROVEN_AT_L0'])
    # rank by L1_total_proj ascending
    finite = [r for r in rows if r.get('l1_total_proj', -1) >= 0
                and r['status'] == 'OK']
    finite.sort(key=lambda r: r['l1_total_proj'])
    print('  Best non-trivial:')
    for r in finite[:5]:
        print(f"    n={r['n_half']} m={r['m']} d0={r['d0']}: "
              f"L0_surv={r['l0_survivors']:,} -> L1_total~{r['l1_total_proj']:,}")


if __name__ == '__main__':
    main()
