"""Quick parents/30min benchmark.  Just time real parents end-to-end."""
import os, sys, time, json
import numpy as np
ROOT = os.environ.get('CASCADE_ROOT', '/home/ubuntu')
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger'))
sys.path.insert(0, os.path.join(ROOT, 'cloninger-steinerberger', 'cpu'))
sys.path.insert(0, ROOT)
from pruning import correction
from run_cascade import process_parent_fused, run_level0
from post_filters import (apply_FN_filter_parallel,
                            apply_Q_filter_parallel,
                            apply_L_filter_parallel)

N_SAMPLE = 10           # parents per d_child
PER_PARENT_HARD_S = 30  # skip parents that exceed this wall

CONFIGS = [
    # d_child, n_parent, m, c
    (8,  2, 20, 1.25),
    (12, 3, 10, 1.25),
    (16, 4, 10, 1.25),
]


def time_parent(p, n_half_child, m, c):
    t = time.time()
    sF, n_ch = process_parent_fused(p, m, c, n_half_child,
                                       use_flat_threshold=False, use_F=True,
                                       use_Q=False, skip_sdp_cert=True)
    wF = time.time() - t
    sFN = apply_FN_filter_parallel(sF, n_half_child, m, c)
    sQ = apply_Q_filter_parallel(sFN, n_half_child, m, c, n_workers=64)
    sL = apply_L_filter_parallel(sQ, n_half_child, m, c, solver='MOSEK', n_workers=64)
    return time.time() - t, len(sF), len(sFN), len(sQ), len(sL), int(n_ch), wF


def main():
    rows = []
    for d_child, n_parent, m, c in CONFIGS:
        if c + correction(m, n_parent) >= 1.5029:
            print(f"d={d_child}: VACUOUS"); continue
        d_parent = 2 * n_parent
        print(f"\n--- d_child={d_child} (n_parent={n_parent} m={m} c={c}) ---", flush=True)
        # Get L0 parents
        try:
            r0 = run_level0(n_parent, m, c, verbose=False, use_F=True, use_Q=False)
            par = r0['survivors']
        except Exception as e:
            print(f"  L0 ERR: {e}"); continue
        if len(par) == 0:
            print(f"  L0 had 0 survivors at this c.  Skip."); continue
        rng = np.random.default_rng(7)
        idx = rng.choice(len(par), min(N_SAMPLE, len(par)), replace=False)
        sample = par[idx]
        n_half_child = d_child // 2
        walls = []
        slow = 0
        for i, parent in enumerate(sample):
            t = time.time()
            try:
                w, nF, nFN, nQ, nL, nch, wF = time_parent(parent, n_half_child, m, c)
            except Exception as e:
                print(f"  [{i+1}] EXC {e}"); continue
            walls.append(w)
            print(f"  [{i+1}/{len(sample)}] children={nch:,} F={nF} FN={nFN} Q={nQ} L={nL}  "
                  f"wall={w:.2f}s (Fkernel={wF:.2f}s)", flush=True)
            if w > PER_PARENT_HARD_S:
                slow += 1
        if not walls:
            print(f"  no walls"); continue
        walls = np.array(walls)
        mean = float(walls.mean())
        med = float(np.median(walls))
        p30 = int(1800.0 / max(mean, 1e-3))
        p30_med = int(1800.0 / max(med, 1e-3))
        rows.append({'d_child': d_child, 'cfg': f'n={n_parent} m={m}',
                      'n_sample': len(walls),
                      'mean_wall_s': mean, 'median_wall_s': med,
                      'p95_wall_s': float(np.percentile(walls, 95)),
                      'parents_per_30min_mean': p30,
                      'parents_per_30min_median': p30_med,
                      'slow_parents': slow})
        print(f"  ⇒ d={d_child}: mean={mean:.2f}s  median={med:.2f}s  "
              f"parents/30min(mean)={p30:,}  parents/30min(med)={p30_med:,}", flush=True)
    print("\n\n=========================================================")
    print(f"{'d':>4} {'cfg':>10} {'mean':>8} {'med':>8} {'p95':>8} {'p/30m(mean)':>12} {'p/30m(med)':>12}")
    for r in rows:
        print(f"{r['d_child']:>4} {r['cfg']:>10} {r['mean_wall_s']:>7.2f}s "
              f"{r['median_wall_s']:>7.2f}s {r['p95_wall_s']:>7.2f}s "
              f"{r['parents_per_30min_mean']:>12,} {r['parents_per_30min_median']:>12,}")
    with open(f'{ROOT}/_qbench_out.json', 'w') as f:
        json.dump(rows, f, indent=2, default=str)


if __name__ == '__main__':
    main()
