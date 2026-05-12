"""d=20 feasibility test for target c=1.281 with all improvements enabled.

P0 (donate fix): always on.
P2 (skip rigor on safe margin): always on.
P3 (gap-weighted split): off (caused thin-slab stall at small d).
P1-LITE+P4 (top-K joint cert with Shor pre-filter): ON at depth >= 22.
Init split depth: 14 to seed plenty of starter boxes.
Time budget: 1 hour.

Reports node throughput, depth, coverage % over time.
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Enable P1-LITE+P4 for d=20.
os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '22'
os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
# P3 left off (env var not set => default 999 = disabled).

from interval_bnb.parallel import parallel_branch_and_bound


def main():
    d = 20
    target = '1.281'
    workers = 64
    time_budget_s = 3600  # 1 hour

    print(f"\n{'#'*72}")
    print(f"# d={d} target={target} workers={workers} budget={time_budget_s}s")
    print(f"# All improvements: P0+P2 on. P1-LITE+P4 enabled at depth>=22.")
    print(f"{'#'*72}", flush=True)

    t0 = time.time()
    result = parallel_branch_and_bound(
        d=d, target_c=target, workers=workers,
        init_split_depth=14,
        donate_threshold_floor=4,
        time_budget_s=time_budget_s,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\n{'='*72}")
    print(f"d=20 FEASIBILITY RESULT")
    print(f"{'='*72}")
    print(f"  success: {result['success']}")
    print(f"  nodes: {result['total_nodes']:,}")
    print(f"  certified leaves: {result['total_leaves_certified']:,}")
    print(f"  max depth: {result['max_depth']}")
    print(f"  coverage: {100*result['coverage_fraction']:.4f}%")
    print(f"  in_flight final: {result['in_flight_final']}")
    print(f"  elapsed: {elapsed:.1f}s = {elapsed/60:.1f}min = {elapsed/3600:.2f}h")

    serializable = {k: v for k, v in result.items()
                    if isinstance(v, (int, float, str, bool, list, dict))}
    serializable['wall_time_s'] = elapsed
    with open('d20_feasibility_result.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved d20_feasibility_result.json")


if __name__ == "__main__":
    main()
