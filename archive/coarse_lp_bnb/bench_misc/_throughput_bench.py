"""Throughput benchmark: parents/30min per d_child for the cascade.

For each d_child, pick a representative (n_parent, m, c_target), enumerate
the L0 d_parent compositions (= sample N), sample N_BENCH parents, and
time the full F → FN → Q → L cascade kernel + parallel post-filters on
each.  Report:
   - per-parent wall (median, mean, p95)
   - children/parent (median)
   - F / FN / Q / L survivor counts (median)
   - parents/30min  =  1800 / mean_wall   (the planning metric)

Output: JSON + console summary.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from math import comb

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


N_WORKERS = 64       # pod cores
N_BENCH_PARENTS = 30      # sample size per d_child
PER_PARENT_HARD_S = 90    # hard cap per parent (subprocess kill at this wall)
LEVEL_BUDGET_S = 1800.0


# (d_child, n_parent, m, c_target) — choose configs that REACH d_child with a
# reasonable L0 enumeration cost AND have non-trivial F-survivor count at L0.
# c_target = 1.25 (just below published 1.2802 — the regime where cascade is
# borderline).
CONFIGS = [
    # d_child, n_parent, m, c_target, L0_max_keep
    (8,   2, 15, 1.25, 5000),     # L0 d=4 m=15 — 302K comps, ~15K survivors
    (8,   2, 20, 1.25, 5000),     # L0 d=4 m=20 — 708K comps, ~34K survivors
    (12,  3, 10, 1.25, 5000),     # L0 d=6 m=10 — 234M comps; cap survivors
    (16,  4, 10, 1.25, 200),      # L0 d=8 m=10 — 633B comps; very expensive,
                                   # we'll sample a small batch instead of full L0
    (16,  4, 5,  1.25, 200),      # L0 d=8 m=5 vacuous? corr=0.44, c+corr=1.69 > 1.5 -> vacuous
                                   # leave for runtime check
]


def n_full_compositions(d, S):
    return comb(S + d - 1, d - 1)


def sample_l0_survivors(n_parent, m, c_target, max_keep, l0_timeout=600,
                          fast_partial=False):
    """Run L0 enumeration with F+Q post-filter, return up to max_keep survivors.

    For huge L0 (e.g., d=8 m=10), partial=True samples a partial enumeration:
    we run only the first ~10M compositions to seed the bench with real-shape
    parents.
    """
    d = 2 * n_parent
    S = 4 * n_parent * m
    n_total = n_full_compositions(d, S)
    print(f"  [L0] (n={n_parent}, m={m}, c={c_target})  d={d}  S={S}  "
            f"total_comp={n_total:,}", flush=True)

    if n_total > 10_000_000_000 or fast_partial:
        # Too big for full L0 enum.  Use partial sampling.
        return _sample_l0_partial(n_parent, m, c_target, max_keep)

    t0 = time.time()
    r = run_level0(n_parent, m, c_target, verbose=False, use_F=True,
                    use_Q=True, use_L=False)
    survivors = r['survivors']
    n_surv = int(r['n_survivors'])
    wall = time.time() - t0
    print(f"  [L0] done: {n_surv:,} survivors in {wall:.1f}s", flush=True)
    if n_surv > max_keep:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_surv, max_keep, replace=False)
        survivors = survivors[idx]
    return survivors


def _sample_l0_partial(n_parent, m, c_target, max_keep):
    """For huge L0 (d=8 m=10): enumerate first ~M comps via the L0 batched
    iterator until we collect >= max_keep F-survivors.  Apply F kernel only
    (not Q+L — those are slow on raw F-survivors at this scale)."""
    from compositions import generate_canonical_compositions_batched
    from _M1_bench import prune_F
    from pruning import correction
    d = 2 * n_parent
    S = 4 * n_parent * m
    threshold = c_target + correction(m, n_parent)
    print(f"  [L0 partial] sampling for {max_keep} F-survivors at d={d}, m={m}", flush=True)
    # Warm JIT
    warm = np.zeros((1, d), dtype=np.int32)
    prune_F(warm, n_parent, m, c_target)
    survivors_list = []
    n_seen = 0
    t0 = time.time()
    for batch in generate_canonical_compositions_batched(d, S, batch_size=200_000):
        n_seen += len(batch)
        sF = prune_F(batch, n_parent, m, c_target)
        f_idx = np.where(sF)[0]
        if len(f_idx) > 0:
            survivors_list.extend(batch[f_idx])
        if len(survivors_list) >= max_keep:
            break
        if time.time() - t0 > 180.0:
            print(f"  [L0 partial] time budget hit; collected {len(survivors_list)}", flush=True)
            break
    survivors = np.array(survivors_list[:max_keep], dtype=np.int32)
    print(f"  [L0 partial] found {len(survivors)} F-survivors after scanning "
          f"{n_seen:,} comps in {time.time()-t0:.1f}s", flush=True)
    return survivors


def time_one_parent(parent, n_half_child, m, c_target):
    """F-kernel + parallel FN + Q + L; return per-stage walls and survivor counts."""
    timing = {}
    tp = time.time()
    surv_F, n_children = process_parent_fused(
        parent, m, c_target, n_half_child,
        use_flat_threshold=False, use_F=True, use_Q=False,
        skip_sdp_cert=True)
    timing['wall_F'] = time.time() - tp
    timing['children'] = int(n_children)
    timing['F_survivors'] = int(len(surv_F))

    t = time.time()
    surv_FN = apply_FN_filter_parallel(surv_F, n_half_child, m, c_target)
    timing['wall_FN'] = time.time() - t
    timing['FN_survivors'] = int(len(surv_FN))

    t = time.time()
    surv_Q = apply_Q_filter_parallel(surv_FN, n_half_child, m, c_target,
                                       n_workers=N_WORKERS)
    timing['wall_Q'] = time.time() - t
    timing['Q_survivors'] = int(len(surv_Q))

    t = time.time()
    surv_L = apply_L_filter_parallel(surv_Q, n_half_child, m, c_target,
                                       solver='MOSEK', n_workers=N_WORKERS)
    timing['wall_L'] = time.time() - t
    timing['L_survivors'] = int(len(surv_L))

    timing['wall_total'] = time.time() - tp
    return timing


def bench_one_config(d_child, n_parent, m, c_target, max_keep):
    """Time per-parent throughput at d_child."""
    print(f"\n========== d_child={d_child}  (n_parent={n_parent}, m={m}, c={c_target}) ==========",
          flush=True)
    if c_target + correction(m, n_parent) >= 1.5029:
        print(f"  VACUOUS (c+corr={c_target+correction(m, n_parent):.4f} >= 1.5029)", flush=True)
        return {'d_child': d_child, 'verdict': 'VACUOUS'}

    survivors = sample_l0_survivors(n_parent, m, c_target, max_keep)
    if len(survivors) == 0:
        return {'d_child': d_child, 'verdict': 'NO_L0_SURVIVORS'}

    # Sample N_BENCH_PARENTS for timing
    rng = np.random.default_rng(123)
    n_sample = min(N_BENCH_PARENTS, len(survivors))
    idx = rng.choice(len(survivors), n_sample, replace=False)
    sample = survivors[idx]
    n_half_child = d_child // 2

    print(f"\n  Timing F+FN+Q+L on {n_sample} parents at d_child={d_child}...",
          flush=True)
    timings = []
    for i, p in enumerate(sample):
        t = time_one_parent(p, n_half_child, m, c_target)
        timings.append(t)
        print(f"    [{i+1}/{n_sample}] children={t['children']:,}  "
              f"F={t['F_survivors']} FN={t['FN_survivors']} "
              f"Q={t['Q_survivors']} L={t['L_survivors']}  "
              f"wall={t['wall_total']:.1f}s "
              f"(F={t['wall_F']:.1f} FN={t['wall_FN']:.1f} "
              f"Q={t['wall_Q']:.1f} L={t['wall_L']:.1f})",
              flush=True)

    walls = np.array([t['wall_total'] for t in timings])
    f_survs = np.array([t['F_survivors'] for t in timings])
    l_survs = np.array([t['L_survivors'] for t in timings])
    children = np.array([t['children'] for t in timings])

    summary = {
        'd_child': d_child,
        'n_parent': n_parent, 'm': m, 'c_target': c_target,
        'n_l0_survivors_used': int(len(survivors)),
        'n_sample_timed': int(n_sample),
        'wall_per_parent_median_s': float(np.median(walls)),
        'wall_per_parent_mean_s': float(np.mean(walls)),
        'wall_per_parent_p95_s': float(np.percentile(walls, 95)),
        'wall_per_parent_max_s': float(np.max(walls)),
        'children_per_parent_median': int(np.median(children)),
        'F_survivors_median': int(np.median(f_survs)),
        'L_survivors_median': int(np.median(l_survs)),
        'F_survivors_mean': float(np.mean(f_survs)),
        'L_survivors_mean': float(np.mean(l_survs)),
        'parents_per_30min_via_mean': int(LEVEL_BUDGET_S / max(np.mean(walls), 1e-3)),
        'parents_per_30min_via_median': int(LEVEL_BUDGET_S / max(np.median(walls), 1e-3)),
        'parents_per_30min_via_p95': int(LEVEL_BUDGET_S / max(np.percentile(walls, 95), 1e-3)),
        'per_parent': timings,
    }
    print(f"\n  ── d_child={d_child} summary ──", flush=True)
    print(f"     wall/parent: median={summary['wall_per_parent_median_s']:.1f}s  "
          f"mean={summary['wall_per_parent_mean_s']:.1f}s  "
          f"p95={summary['wall_per_parent_p95_s']:.1f}s  "
          f"max={summary['wall_per_parent_max_s']:.1f}s", flush=True)
    print(f"     L_surv/parent: median={summary['L_survivors_median']}  "
          f"mean={summary['L_survivors_mean']:.1f}", flush=True)
    print(f"     parents/30min: by_mean={summary['parents_per_30min_via_mean']}  "
          f"by_median={summary['parents_per_30min_via_median']}  "
          f"by_p95={summary['parents_per_30min_via_p95']}", flush=True)
    return summary


def main():
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(ROOT, f'throughput_bench_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'summary.json')
    full = {'started': ts, 'configs': CONFIGS, 'results': []}
    for cfg in CONFIGS:
        d_child, n_parent, m, c_target, max_keep = cfg
        try:
            r = bench_one_config(d_child, n_parent, m, c_target, max_keep)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = {'d_child': d_child, 'verdict': f'ERROR: {e}'}
        full['results'].append(r)
        with open(out_path, 'w') as f:
            json.dump(full, f, indent=2, default=str)

    # Final table
    print("\n\n=========================================================")
    print("THROUGHPUT TABLE — parents per 30 min")
    print("=========================================================")
    print(f"{'d_child':>7} {'config':>14} {'wall_med':>9} {'wall_p95':>9} "
          f"{'p/30min(mean)':>14} {'p/30min(p95)':>14}  L_med")
    for r in full['results']:
        if 'verdict' in r and r['verdict'] != 'OK':
            print(f"{r['d_child']:>7}  {r.get('verdict','?'):>30}")
            continue
        cfg_s = f"n={r['n_parent']}m={r['m']}"
        print(f"{r['d_child']:>7} {cfg_s:>14}  "
              f"{r['wall_per_parent_median_s']:>7.1f}s  "
              f"{r['wall_per_parent_p95_s']:>7.1f}s  "
              f"{r['parents_per_30min_via_mean']:>13,}  "
              f"{r['parents_per_30min_via_p95']:>13,}  "
              f"{r['L_survivors_median']}")
    print(f"\nFull JSON: {out_path}")


if __name__ == '__main__':
    main()
