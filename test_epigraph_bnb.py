"""d=10 t=1.208 BnB with epigraph enabled — moment-of-truth test.

Without epigraph: stalls at 99.06% (per empirical data).
With epigraph: should fully close.
"""
import os
import sys
import time

# Combined cascade per agent diagnostics:
#   - Joint-face top-3 (P1-LITE) at depth >= 12: closes 95.5%
#   - Epigraph LP at depth >= 24: closes residual 4.5% (the minimax gap)
os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '12'
os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
os.environ['INTERVAL_BNB_EPIGRAPH_DEPTH'] = '24'
os.environ['INTERVAL_BNB_EPIGRAPH_FILTER'] = '0.02'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interval_bnb.parallel import parallel_branch_and_bound


def main():
    print("=" * 70, flush=True)
    print("d=10 t=1.208 BnB — EPIGRAPH FIX TEST", flush=True)
    print("Goal: break the 99.06% bound stall.", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    r = parallel_branch_and_bound(
        d=10, target_c='1.208',
        workers=8, init_split_depth=14, donate_threshold_floor=4,
        time_budget_s=600,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\n{'=' * 70}", flush=True)
    print(f"FINAL RESULT", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"  success: {r['success']}", flush=True)
    print(f"  nodes: {r['total_nodes']:,}", flush=True)
    print(f"  coverage: {100 * r['coverage_fraction']:.6f}%", flush=True)
    print(f"  time: {elapsed:.1f}s", flush=True)
    print(f"  in_flight: {r['in_flight_final']}", flush=True)
    print(f"  CCTR stats: {r.get('cctr_stats', {})}", flush=True)
    print(f"  EPIGRAPH stats: {r.get('epi_stats', {})}", flush=True)


if __name__ == "__main__":
    main()
