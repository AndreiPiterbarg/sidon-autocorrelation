"""d=30 t=1.281 BnB locally on 16-core machine.

Cascade (in order):
  cheap -> tighten(>=4) -> standard -> P1-LITE(>=14)
  -> multi-anchor (mu*, sigma(mu*)) cheap filter (>=24)
  -> CCTR (multi-α) -> epigraph LP+RLT (>=24, pub-rigor cushion)
  -> per-box centroid anchor LAST-RESORT (>=80, only if epi LP failed)
  -> H_d half-simplex pre-filter on every box pop
  -> LP-binding split heuristic (>=28) + cross-box variance split (>=27)

Why d=30 should close where d=24 stalled:
  - val(30) projected ≈ 1.32-1.327 (extrapolating Δ ≈ +0.0044 per +2 d
    from val(22)=1.30933 → val(24)=1.31369). Margin to 1.281 ≈ 0.04+.
  - At ~0.04 margin, expected epi LP cert rate ≥ 55%; in_flight has
    NEGATIVE drift, comfortable drain.

Per-box cost scaling (audited):
  - LP solve at d=30 ≈ 1.3s/call (vs 0.4s at d=24, 5x slower)
  - 16 workers ≈ 12 LPs/sec aggregate
  - Tree must be small enough that 12/s * 6h = ~250K nodes covers it.

init_split_depth=26 (between d=24's 24 and d=30; gives ~1000-2000
starter boxes, balances parallelism vs starter overhead).

Centroid threshold raised to 80 from d=24's 60: at d=30 boxes are wider
per worker depth (more axes share splits), so centroid only fires
usefully at depth ~80+.
"""
import os
import sys
import time
import json

# CASCADE THRESHOLDS for d=30:
os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '14'
os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
os.environ['INTERVAL_BNB_EPIGRAPH_DEPTH'] = '24'
os.environ['INTERVAL_BNB_EPIGRAPH_FILTER'] = '0.02'
# Multi-anchor (cheap, ~10us): runs at every box at depth >= 24.
os.environ['INTERVAL_BNB_ANCHOR_DEPTH'] = '24'
# Centroid anchor (~55ms at d=30): only fires AFTER epi LP fails AND
# box is deep enough. d=30 needs deeper threshold than d=24's 60 because
# boxes are wider at the same worker depth (more axes per BnB depth).
os.environ['INTERVAL_BNB_CENTROID_DEPTH'] = '80'
# LP-binding split (chooses split axis from McCormick face duals):
# slightly later than d=24's 26, since each LP costs more.
os.environ['INTERVAL_BNB_LP_SPLIT_DEPTH'] = '28'
# Variance-weighted split heuristic (in-box mid · (1-mid) · width):
# kicks in at depth 27 (one before LP-binding so it has data when needed).
os.environ['INTERVAL_BNB_PC_DEPTH'] = '27'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interval_bnb.parallel import parallel_branch_and_bound


def main():
    d = 30
    target = '1.281'
    workers = 16
    time_budget_s = 21600  # 6 hours
    init_split_depth = 26

    print("#" * 72, flush=True)
    print(f"# d={d} t={target} BnB on 16-core local machine", flush=True)
    print(f"# Cascade: cheap -> tighten(>=4) -> standard -> P1-LITE(>=14)", flush=True)
    print(f"#          -> multi-anchor cheap filter (>=24)", flush=True)
    print(f"#          -> CCTR -> epigraph LP+RLT (>=24, pub-rigor cushion)", flush=True)
    print(f"#          -> CENTROID anchor LAST-RESORT (>=80)", flush=True)
    print(f"#          -> H_d cut on every box pop", flush=True)
    print(f"#          -> LP-binding(>=28) + variance split(>=27)", flush=True)
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
    with open('d30_t1281_result.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved d30_t1281_result.json", flush=True)


if __name__ == "__main__":
    main()
