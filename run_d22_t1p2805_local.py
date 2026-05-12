"""d=22 t=1.2805 BnB locally on 16-core machine.

Path A: target relaxed from 1.281 → 1.2805 (still beats CS 2017's 1.2802 by 0.0003).
This gives the cascade an extra ~0.0005 of margin past the d=22 t=1.281 stall point
(reached 99.99999849% there with 1500 surviving boxes at hw 0.003-0.005).

Cascade (in order):
  cheap -> tighten(>=4) -> standard -> P1-LITE(>=14)
  -> multi-anchor (mu*, sigma(mu*)) cheap filter (>=24)
  -> CCTR (multi-α) -> epigraph LP+RLT (>=24, pub-rigor cushion)
  -> per-box centroid anchor (>=60, only if epi LP failed)
  -> H_d half-simplex pre-filter on every box pop
  -> LP-binding split (>=26) + variance split (>=25)

Expected: clean drain in 1-2h, no stall.
"""
import os
import sys
import time
import json

# CASCADE THRESHOLDS for d=22:
os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '14'
os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
os.environ['INTERVAL_BNB_EPIGRAPH_DEPTH'] = '24'
os.environ['INTERVAL_BNB_EPIGRAPH_FILTER'] = '0.02'
os.environ['INTERVAL_BNB_ANCHOR_DEPTH'] = '24'
os.environ['INTERVAL_BNB_CENTROID_DEPTH'] = '60'
os.environ['INTERVAL_BNB_LP_SPLIT_DEPTH'] = '26'
# Boundary-aware split heuristic (Agent B fix):
# at depth >= 30 with >= 11 axes on lo=0 boundary, force split onto a free axis.
os.environ['INTERVAL_BNB_BOUNDARY_SPLIT_DEPTH'] = '30'
os.environ['INTERVAL_BNB_BOUNDARY_AXIS_COUNT'] = '11'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interval_bnb.parallel import parallel_branch_and_bound


def main():
    d = 22
    target = '1.2805'
    workers = 16
    time_budget_s = 43200  # 12 hours — option A, fire-and-forget, check at end
    init_split_depth = 22

    print("#" * 72, flush=True)
    print(f"# d={d} t={target} BnB on 16-core local machine", flush=True)
    print(f"# Margin = val({d})_UB - target = 1.30933 - 1.2805 = +0.0288", flush=True)
    print(f"# (vs +0.0283 at target=1.281 where cascade stalled)", flush=True)
    print(f"# Cascade: cheap -> tighten(>=4) -> standard -> P1-LITE(>=14)", flush=True)
    print(f"#          -> multi-anchor (>=24) -> CCTR -> epi LP (>=24)", flush=True)
    print(f"#          -> centroid anchor (>=60) -> H_d cut, splits", flush=True)
    print(f"# init_split_depth={init_split_depth}, "
          f"workers={workers}, budget={time_budget_s}s ({time_budget_s/3600:.1f}h)",
          flush=True)
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

    print("\n" + "=" * 72, flush=True)
    print(f"FINAL RESULT", flush=True)
    print("=" * 72, flush=True)
    print(f"  success: {r['success']}", flush=True)
    print(f"  total nodes: {r['total_nodes']:,}", flush=True)
    print(f"  certified leaves: {r['total_leaves_certified']:,}", flush=True)
    print(f"  max depth: {r['max_depth']}", flush=True)
    print(f"  coverage: {100 * r['coverage_fraction']:.6f}%", flush=True)
    print(f"  in_flight final: {r['in_flight_final']}", flush=True)
    print(f"  elapsed: {elapsed:.1f}s = {elapsed/60:.2f}min = {elapsed/3600:.2f}h", flush=True)
    print(f"  CCTR stats: {r.get('cctr_stats', {})}", flush=True)
    print(f"  EPIGRAPH stats: {r.get('epi_stats', {})}", flush=True)
    print(f"  ANCHOR stats: {r.get('anchor_stats', {})}", flush=True)
    print(f"  CENTROID stats: {r.get('centroid_stats', {})}", flush=True)

    serializable = {k: v for k, v in r.items()
                    if isinstance(v, (int, float, str, bool, list, dict))}
    serializable['wall_time_s'] = elapsed
    with open('d22_t1p2805_result.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved d22_t1p2805_result.json", flush=True)


if __name__ == "__main__":
    main()
