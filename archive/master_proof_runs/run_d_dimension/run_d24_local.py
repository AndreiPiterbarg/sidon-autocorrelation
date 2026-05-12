"""d=24 t=1.281 BnB locally on 16-core machine.

Cascade (in order, with new fixes integrated):
  cheap -> tighten(>=4) -> standard -> P1-LITE(>=14)
  -> multi-anchor (mu*, sigma(mu*)) + per-box centroid anchor (>=24)
  -> CCTR (multi-α) -> epigraph LP+RLT (>=24, with publication-rigor cushion)
  -> H_d half-simplex pre-filter on every box pop
  -> LP-binding split heuristic (>=26) + cross-box variance split (>=25)

Why this should close where d=22 stalled:
  - val(24) > val(22) ≈ 1.30933 (monotone increase with d), so margin to 1.281 grows.
  - sigma(mu*) is automatically built into the multi-anchor cascade — no
    H_d-orientation issue like at d=22.
  - bound_epigraph_int_ge now has Neumaier-Shcherbina cushion (1e-7) — sound but slightly stricter than the d=22 run.

init_split_depth=24 (gives ~5-15K starter boxes for 16 workers).
"""
import os
import sys
import time
import json

# CASCADE THRESHOLDS for d=24:
os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '14'
os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
os.environ['INTERVAL_BNB_EPIGRAPH_DEPTH'] = '24'
os.environ['INTERVAL_BNB_EPIGRAPH_FILTER'] = '0.02'
# Multi-anchor (cheap, 10us): runs at every box at depth >= 24.
os.environ['INTERVAL_BNB_ANCHOR_DEPTH'] = '24'
# Per-box centroid anchor (expensive, 98ms): only fires AFTER epi LP
# fails AND box is deep enough (hw < ~0.003). Empirically centroid
# cert rate is 0% for hw>0.003. Gate at depth 60 (hw ~ 0.5/2^(60/24)
# ≈ 0.027 — too wide for centroid to fire, but a few more splits
# inside the worker will reach 0.001 region where it fires 100%).
os.environ['INTERVAL_BNB_CENTROID_DEPTH'] = '60'
os.environ['INTERVAL_BNB_LP_SPLIT_DEPTH'] = '26'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interval_bnb.parallel import parallel_branch_and_bound


def main():
    d = 24
    target = '1.281'
    workers = 16
    time_budget_s = 21600  # 6 hours
    init_split_depth = 24

    print("#" * 72, flush=True)
    print(f"# d={d} t={target} BnB on 16-core local machine", flush=True)
    print(f"# Cascade: cheap -> tighten(>=4) -> standard -> P1-LITE(>=14)", flush=True)
    print(f"#          -> multi-anchor + centroid (>=24)", flush=True)
    print(f"#          -> CCTR -> epigraph LP+RLT (>=24, pub cushion)", flush=True)
    print(f"#          -> H_d cut on every box pop", flush=True)
    print(f"#          -> LP-binding(>=26) + variance split(>=25)", flush=True)
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
    with open('d24_t1281_result.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved d24_t1281_result.json", flush=True)


if __name__ == "__main__":
    main()
