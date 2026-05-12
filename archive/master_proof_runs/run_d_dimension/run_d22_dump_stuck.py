"""Run d=22 t=1.281 BnB on the pod, then dump in_flight boxes when budget expires.

The dumped boxes ARE the actual stall-zone boxes (not synthetic). After this
finishes, analyze_stuck_boxes.py reports the structural pathology.

Output: stuck_boxes_w{worker_id}.npz files in cwd.
"""
import os
import sys
import time

os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '14'
os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
os.environ['INTERVAL_BNB_EPIGRAPH_DEPTH'] = '24'
os.environ['INTERVAL_BNB_EPIGRAPH_FILTER'] = '0.02'
os.environ['INTERVAL_BNB_ANCHOR_DEPTH'] = '24'
os.environ['INTERVAL_BNB_CENTROID_DEPTH'] = '60'
os.environ['INTERVAL_BNB_LP_SPLIT_DEPTH'] = '26'
os.environ['INTERVAL_BNB_DUMP_BOXES'] = 'stuck_boxes'  # dump prefix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interval_bnb.parallel import parallel_branch_and_bound


def main():
    d = 22
    target = '1.281'
    workers = 16  # use 16 of pod's 48 cores
    time_budget_s = 1800  # 30 minutes — let it stall
    init_split_depth = 22

    print("#" * 72, flush=True)
    print(f"# d={d} t={target} BnB with STUCK-BOX DUMP enabled")
    print(f"# 30 min budget; expects to stall like the original d=22 t=1.281 run")
    print(f"# Dumps in_flight boxes to stuck_boxes_w*.npz on exit")
    print(f"# workers={workers}/48 (kept at 16 to match local cascade timings)")
    print("#" * 72, flush=True)

    t0 = time.time()
    r = parallel_branch_and_bound(
        d=d, target_c=target,
        workers=workers,
        init_split_depth=init_split_depth,
        donate_threshold_floor=2,
        time_budget_s=time_budget_s,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n=== RESULT: success={r['success']} cov={100*r['coverage_fraction']:.4f}% in_flight={r['in_flight_final']} elapsed={elapsed:.0f}s ===")
    print(f"=== STATS: anchor={r.get('anchor_stats')} centroid={r.get('centroid_stats')} epi={r.get('epi_stats')} ===")


if __name__ == "__main__":
    main()
