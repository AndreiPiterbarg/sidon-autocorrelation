"""d=20 feasibility — aggressive parameters + longer budget + precise progress.

Settings:
- topk_joint_depth = 14 (much earlier P1-LITE)
- topk_joint_K = 5 (more alternates)
- init_split_depth = 16 (more starter boxes)
- 2 hour budget
- Custom progress tracking with full-precision coverage
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Aggressive settings
os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '14'
os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '5'
# Bypass Shor pre-filter for now (Shor too loose at this scale)
# Keep gap-weighted off (caused stall at small d).

from interval_bnb.parallel import parallel_branch_and_bound


def main():
    d = 20
    target = '1.281'
    workers = 64
    time_budget_s = 7200  # 2 hours

    print(f"\n{'#'*72}")
    print(f"# d={d} target={target} workers={workers} budget={time_budget_s}s")
    print(f"# Aggressive: topk_depth=14, K=5, init_split_depth=16")
    print(f"{'#'*72}", flush=True)

    t0 = time.time()
    result = parallel_branch_and_bound(
        d=d, target_c=target, workers=workers,
        init_split_depth=16,
        donate_threshold_floor=4,
        time_budget_s=time_budget_s,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"d={d} t={target} RESULT")
    print(f"{'='*72}")
    print(f"  success: {result['success']}")
    print(f"  nodes: {result['total_nodes']:,}")
    print(f"  certified leaves: {result['total_leaves_certified']:,}")
    print(f"  max depth: {result['max_depth']}")
    print(f"  coverage: {100*result['coverage_fraction']:.6f}%")
    print(f"  in_flight final: {result['in_flight_final']}")
    print(f"  elapsed: {elapsed:.1f}s = {elapsed/60:.1f}min = {elapsed/3600:.2f}h")

    serializable = {k: v for k, v in result.items()
                    if isinstance(v, (int, float, str, bool, list, dict))}
    serializable['wall_time_s'] = elapsed
    with open('d20_aggressive_result.json', 'w') as f:
        json.dump(serializable, f, indent=2)


if __name__ == "__main__":
    main()
