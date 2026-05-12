"""d=22 t=1.281 BnB locally on 16-core machine, autopilot.

Cascade: cheap -> tighten(>=4) -> standard -> P1-LITE(>=14)
         -> mu*-anchor cut(>=24) -> epigraph LP(>=24, with new RLT/SOS/tangent cuts)
         -> H_d half-simplex pre-filter on every box pop
         -> cross-box variance split heuristic(>=25)

val(22) UB from KKT mu* will be loaded; expected >= 1.298 (val grows with d),
so target=1.281 should give margin >= 0.017 like d=20.

Tuned for 16-core local with all 5 fixes active:
  init_split_depth=22 (gives ~4K starter boxes — workers have ~250 each)
  donate_threshold_floor=2
  3-hour budget initially; can extend if making progress
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
# NEW: anchor cut around mu* — fires at depth 24 alongside epigraph LP.
# Cheap (greedy sort, ~1 us) and closes deep boxes near mu*.
os.environ['INTERVAL_BNB_ANCHOR_DEPTH'] = '24'
# NEW: LP-binding-axis split heuristic — uses just-solved epigraph LP duals
# to pick the axis whose tightening will most improve the LP next time.
# Activates at depth 26 (after epigraph LP is regularly solved).
os.environ['INTERVAL_BNB_LP_SPLIT_DEPTH'] = '26'
# Defaults active: TIGHTEN_DEPTH=4, PC_DEPTH=25, H_d cut on every box pop.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interval_bnb.parallel import parallel_branch_and_bound


def main():
    d = 22
    target = '1.281'
    workers = 16
    time_budget_s = 10800  # 3 hours

    print("#" * 72, flush=True)
    print(f"# d={d} t={target} BnB on 16-core local machine", flush=True)
    print(f"# Cascade: cheap -> tighten(>=4) -> standard -> P1-LITE(>=14)", flush=True)
    print(f"#          -> anchor cut(>=24) -> epigraph LP+RLT(>=24)", flush=True)
    print(f"#          -> H_d cut on every box pop", flush=True)
    print(f"#          -> cross-box variance split(>=25)", flush=True)
    print(f"# Budget: {time_budget_s}s ({time_budget_s/3600:.1f}h)", flush=True)
    print("#" * 72, flush=True)

    t0 = time.time()
    r = parallel_branch_and_bound(
        d=d, target_c=target,
        workers=workers,
        init_split_depth=22,
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
    print(f"  ANCHOR stats: attempts={r.get('anchor_attempts', 'N/A')}, certs={r.get('anchor_certs', 'N/A')}", flush=True)

    serializable = {k: v for k, v in r.items()
                    if isinstance(v, (int, float, str, bool, list, dict))}
    serializable['wall_time_s'] = elapsed
    with open('d22_t1281_result.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved d22_t1281_result.json", flush=True)


if __name__ == "__main__":
    main()
