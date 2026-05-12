"""d=22 t=1.281 BnB with new boundary-split + smart-W centroid fixes ENABLED.

Compare cert rate to the baseline run_d22_dump_stuck.py result (49.7% over 30 min).
If cert rate clears 50%+, the cascade now drains.
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
# NEW: boundary-aware split heuristic. Triggers at depth >= 30 when
# >= 11 (= d/2) axes are on simplex boundary.
os.environ['INTERVAL_BNB_BOUNDARY_SPLIT_DEPTH'] = '30'
os.environ['INTERVAL_BNB_BOUNDARY_AXIS_COUNT'] = '11'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interval_bnb.parallel import parallel_branch_and_bound


def main():
    print("#"*72)
    print("# d=22 t=1.281 with BOUNDARY-SPLIT + SMART-W CENTROID enabled")
    print("# Baseline (no fixes): cert rate 49.7% over 30 min, in_flight=703")
    print("# Goal: cert rate > 50% → drains")
    print("#"*72, flush=True)

    t0 = time.time()
    r = parallel_branch_and_bound(
        d=22, target_c='1.281',
        workers=16,
        init_split_depth=22,
        donate_threshold_floor=2,
        time_budget_s=1800,
        verbose=True,
    )
    elapsed = time.time() - t0

    cert_rate = r['total_leaves_certified'] / max(r['total_nodes'], 1)
    print(f"\n=== RESULT ===")
    print(f"  success: {r['success']}")
    print(f"  nodes: {r['total_nodes']:,}, cert: {r['total_leaves_certified']:,}, cert_rate: {100*cert_rate:.2f}%")
    print(f"  in_flight: {r['in_flight_final']}")
    print(f"  coverage: {100*r['coverage_fraction']:.4f}%")
    print(f"  elapsed: {elapsed:.0f}s")
    print(f"  CENTROID: {r.get('centroid_stats')}")
    print(f"  ANCHOR: {r.get('anchor_stats')}")
    print(f"  EPIGRAPH: {r.get('epi_stats')}")
    if cert_rate > 0.5:
        print(f"\n  ✓ DRAINS (cert rate > 50%) — fixes work!")
    else:
        print(f"\n  ⚠ STILL STALLS (cert rate {100*cert_rate:.1f}% < 50%) — need more")


if __name__ == '__main__':
    main()
