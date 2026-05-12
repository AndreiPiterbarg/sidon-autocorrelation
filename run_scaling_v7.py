"""Scaling test v7 — exercise all improvements in their proper regimes.

P0/P2 always on. P1-LITE/P4 enabled at d>=14 (depth=22, K=3).
P3 (gap-weighted split) enabled at d>=14 with depth=10.
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interval_bnb.parallel import parallel_branch_and_bound


def run_one(d, target, time_budget_s, workers=64, enable_topk=False,
             enable_gap=False):
    print(f"\n{'='*70}")
    print(f"d={d} target={target} workers={workers} budget={time_budget_s}s "
          f"topk={enable_topk} gap={enable_gap}")
    print(f"{'='*70}", flush=True)
    if enable_topk:
        os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '22'
        os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
    else:
        os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '999'
    if enable_gap:
        os.environ['INTERVAL_BNB_GAP_DEPTH'] = '10'
    else:
        os.environ['INTERVAL_BNB_GAP_DEPTH'] = '999'

    t0 = time.time()
    result = parallel_branch_and_bound(
        d=d, target_c=target, workers=workers,
        init_split_depth=14,
        donate_threshold_floor=4,
        time_budget_s=time_budget_s,
        verbose=True,
    )
    elapsed = time.time() - t0
    result['d'] = d
    result['target'] = target
    result['workers'] = workers
    result['wall_time_s'] = elapsed
    return result


def main():
    workers = 64
    results = []

    # Each test uses slack 0.017 (matching d=20 problem).
    # For d <= 12 P1-LITE/P4 disabled (already easy enough).
    # For d >= 14 enable all.
    tests = [
        (8,  "1.184", 60,   False, False),
        (10, "1.208", 180,  False, False),
        (12, "1.238", 600,  False, False),
        (14, "1.253", 900,  True,  False),  # P1-LITE+P4 on
        (16, "1.257", 1800, True,  False),  # P1-LITE+P4 on
    ]

    for d, target, budget, topk, gap in tests:
        try:
            r = run_one(d, target, time_budget_s=budget, workers=workers,
                        enable_topk=topk, enable_gap=gap)
            results.append(r)
            print(f"\n[SUMMARY d={d}] success={r['success']}, "
                  f"nodes={r['total_nodes']:,}, "
                  f"depth={r['max_depth']}, "
                  f"time={r['wall_time_s']:.1f}s, "
                  f"coverage={100*r['coverage_fraction']:.2f}%", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({'d': d, 'target': target, 'error': str(e)})

    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool, list, dict))}
        serializable.append(sr)
    with open('scaling_results_v7.json', 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\n{'='*70}")
    print(f"SCALING SUMMARY v7 (slack=0.017)")
    print(f"{'='*70}")
    print(f"{'d':<4}{'target':<8}{'success':<10}{'nodes':<14}{'time(s)':<10}{'rate(/s)':<12}{'depth':<8}{'cover%':<8}")
    for r in results:
        if 'error' in r:
            print(f"{r['d']:<4}{r['target']:<8}{'ERROR':<10}{r.get('error','')[:40]}")
        else:
            rate = r['total_nodes'] / max(1, r['wall_time_s'])
            print(f"{r['d']:<4}{r['target']:<8}{str(r['success']):<10}"
                  f"{r['total_nodes']:<14,}"
                  f"{r['wall_time_s']:<10.1f}{rate:<12,.0f}"
                  f"{r['max_depth']:<8}{100*r['coverage_fraction']:<8.2f}")


if __name__ == "__main__":
    main()
