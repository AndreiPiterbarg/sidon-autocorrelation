"""d=22 t=1.281 BnB with init_split_depth=30 (Idea 2 — brute force starter partition).

The hypothesis: with init=30 (vs baseline 22), starter boxes are pre-shrunk per
axis, so fewer "wide boundary" boxes form during the cascade. The same cert rate
per box might then DRAIN since fewer boundary-stuck regions exist.
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
os.environ['INTERVAL_BNB_BOUNDARY_SPLIT_DEPTH'] = '30'
os.environ['INTERVAL_BNB_BOUNDARY_AXIS_COUNT'] = '11'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interval_bnb.parallel import parallel_branch_and_bound


def main():
    print("#"*72)
    print("# d=22 t=1.281 BnB with init_split_depth=30 (vs baseline 22)")
    print("# 30 min budget; expect ~10K starter boxes after H_d filter")
    print("#"*72, flush=True)

    t0 = time.time()
    r = parallel_branch_and_bound(
        d=22, target_c='1.281',
        workers=16,
        init_split_depth=30,
        donate_threshold_floor=2,
        time_budget_s=1800,
        verbose=True,
    )
    elapsed = time.time() - t0
    cert_rate = r['total_leaves_certified'] / max(r['total_nodes'], 1)
    print(f"\n=== RESULT ===")
    print(f"  success: {r['success']}")
    print(f"  nodes: {r['total_nodes']:,}, cert: {r['total_leaves_certified']:,}, cert_rate: {100*cert_rate:.2f}%")
    print(f"  in_flight: {r['in_flight_final']}, init_boxes: {r.get('init_boxes')}")
    print(f"  elapsed: {elapsed:.0f}s")
    if cert_rate > 0.5:
        print(f"  ✓ DRAINS")
    else:
        print(f"  ⚠ STILL STALLS at cert_rate={100*cert_rate:.1f}%")


if __name__ == '__main__':
    main()
