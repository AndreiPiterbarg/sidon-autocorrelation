"""Sweep d=22, 24 to test if larger margin overcomes bound stall.

For each d, briefly runs to see coverage progression in 5 minutes.
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Same aggressive settings as d=20 test
os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '14'
os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '5'

from interval_bnb.parallel import parallel_branch_and_bound


def main():
    workers = 64
    target = '1.281'
    budget = 300  # 5 min per d to see if progress is happening

    # Hypothesis: larger d gives larger margin, may close stall
    for d in [22, 24]:
        print(f"\n{'#'*72}")
        print(f"# d={d} target={target} workers={workers} budget={budget}s")
        print(f"{'#'*72}", flush=True)

        t0 = time.time()
        result = parallel_branch_and_bound(
            d=d, target_c=target, workers=workers,
            init_split_depth=14,
            donate_threshold_floor=4,
            time_budget_s=budget,
            verbose=True,
        )
        elapsed = time.time() - t0
        print(f"\n[d={d} VERDICT]")
        print(f"  success: {result['success']}")
        print(f"  nodes: {result['total_nodes']:,}")
        print(f"  certs: {result['total_leaves_certified']:,}")
        print(f"  coverage: {100*result['coverage_fraction']:.6f}%")
        print(f"  in_flight: {result['in_flight_final']}")
        print(f"  max_depth: {result['max_depth']}")
        print(f"  elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
