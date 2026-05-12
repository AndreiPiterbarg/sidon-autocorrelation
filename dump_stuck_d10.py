"""Run d=10 t=1.2 BnB for 90s with stuck-box dumping enabled.

On time-budget exhaustion, the master drains the queue and worker
SIGINT handlers dump local stacks. Result: stuck_d10_master_queue.npz
plus stuck_d10_w*.npz (one per worker).
"""
import os
import sys
import time

# Same cascade config as run_d10_local_t1p2.py
os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '8'
os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
os.environ['INTERVAL_BNB_ANCHOR_DEPTH'] = '12'
os.environ['INTERVAL_BNB_EPIGRAPH_DEPTH'] = '12'
os.environ['INTERVAL_BNB_EPIGRAPH_FILTER'] = '0.02'
os.environ['INTERVAL_BNB_PC_DEPTH'] = '14'
os.environ['INTERVAL_BNB_LP_SPLIT_DEPTH'] = '14'
os.environ['INTERVAL_BNB_CENTROID_DEPTH'] = '999'   # off (was breaking the run)
os.environ['INTERVAL_BNB_BOUNDARY_SPLIT_DEPTH'] = '999'  # off
os.environ['INTERVAL_BNB_SDP_DEPTH'] = '999'  # off (we want to capture LP-stuck)

# Activate dumping
os.environ['INTERVAL_BNB_DUMP_BOXES'] = 'stuck_d10'
os.environ['INTERVAL_BNB_INSTANT_DUMP'] = '1'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interval_bnb.parallel import parallel_branch_and_bound


def main():
    t0 = time.time()
    r = parallel_branch_and_bound(
        d=10, target_c='1.2',
        workers=8,
        init_split_depth=12,
        donate_threshold_floor=2,
        time_budget_s=90,
        verbose=True,
    )
    print(f"\nDone. success={r['success']}  in_flight_final={r['in_flight_final']}  "
          f"coverage={100*r['coverage_fraction']:.5f}%  elapsed={time.time()-t0:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
