"""d=20 feasibility with full CCTR cascade (SW + NE + joint + RLT).

Uses pre-computed mu* loaded from mu_star_d20.npz.
Time budget: 2 hours.
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Disable old P1-LITE/P3 (they're now subsumed/replaced).
os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '999'
os.environ['INTERVAL_BNB_GAP_DEPTH'] = '999'

from interval_bnb.parallel import parallel_branch_and_bound


def main():
    d = 20
    target = '1.281'
    workers = 64
    time_budget_s = 7200  # 2h

    print(f"\n{'#'*72}")
    print(f"# d={d} target={target} workers={workers} budget={time_budget_s}s")
    print(f"# CCTR cascade: SW + NE + joint-face + RLT (Sherali-Adams)")
    print(f"{'#'*72}", flush=True)

    t0 = time.time()
    result = parallel_branch_and_bound(
        d=d, target_c=target, workers=workers,
        init_split_depth=14,
        donate_threshold_floor=4,
        time_budget_s=time_budget_s,
        enable_cctr=True,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"d={d} t={target} CCTR FEASIBILITY RESULT")
    print(f"{'='*72}")
    print(f"  success: {result['success']}")
    print(f"  nodes: {result['total_nodes']:,}")
    print(f"  certified leaves: {result['total_leaves_certified']:,}")
    print(f"  max depth: {result['max_depth']}")
    print(f"  coverage: {100*result['coverage_fraction']:.6f}%")
    print(f"  in_flight final: {result['in_flight_final']}")
    print(f"  CCTR stats: {result.get('cctr_stats')}")
    print(f"  elapsed: {elapsed:.1f}s = {elapsed/60:.1f}min = {elapsed/3600:.2f}h")

    serializable = {k: v for k, v in result.items()
                    if isinstance(v, (int, float, str, bool, list, dict))}
    serializable['wall_time_s'] = elapsed
    with open('d20_cctr_result.json', 'w') as f:
        json.dump(serializable, f, indent=2)


if __name__ == "__main__":
    main()
