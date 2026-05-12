"""d=20 t=1.281 BnB on the 192-core pod.

Cascade: cheap bounds -> early simplex tightening (depth>=4)
         -> standard rigor -> joint-face top-3 (depth>=14)
         -> epigraph LP (depth>=24) -> cross-box variance split (depth>=25).

val(20) UB from KKT mu* = 1.298333, target = 1.281, margin = +0.017.

Tuned for high core count (192) and d=20 LP cost (~125 ms each):
  * init_split_depth=26 - lots of starter boxes for parallelism
  * donate_threshold_floor=2 - eager donation
  * 12-hour budget
  * tighten_depth=4 (default) - tightens hi/lo box bounds early
  * topk_joint_depth=14, K=3 - P1-LITE alternates kick in earlier than d=14
  * epigraph_depth=24 - per-box LP closes minimax-maximin gap
"""
import os
import sys
import time
import json

# CASCADE THRESHOLDS for d=20:
os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '14'
os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
os.environ['INTERVAL_BNB_EPIGRAPH_DEPTH'] = '24'
os.environ['INTERVAL_BNB_EPIGRAPH_FILTER'] = '0.02'
# tighten_depth=4 default (lowered from 15 in this commit) - faster bound
# tightening on shallow boxes; ~zero cost, monotone tighter.
# pc_split_depth=25 default - cross-box variance EMA split heuristic.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interval_bnb.parallel import parallel_branch_and_bound


def main():
    d = 20
    target = '1.281'
    workers = 192
    time_budget_s = 43200  # 12 hours

    print("#" * 72, flush=True)
    print(f"# d={d} t={target} BnB on 192-core pod", flush=True)
    print(f"# Cascade: cheap -> tighten(>=4) -> standard -> P1-LITE(>=14)", flush=True)
    print(f"#          -> epigraph LP(>=24) -> cross-box variance split(>=25)", flush=True)
    print(f"# Budget: {time_budget_s}s ({time_budget_s/3600:.1f}h)", flush=True)
    print("#" * 72, flush=True)

    t0 = time.time()
    r = parallel_branch_and_bound(
        d=d, target_c=target,
        workers=workers,
        init_split_depth=24,  # was 28 -> some axes saturated past D_SHIFT
                              # combined with deep BnB tree. 24 keeps
                              # cumulative axis depth comfortably <60.
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

    serializable = {k: v for k, v in r.items()
                    if isinstance(v, (int, float, str, bool, list, dict))}
    serializable['wall_time_s'] = elapsed
    with open('d20_t1281_result.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved d20_t1281_result.json", flush=True)


if __name__ == "__main__":
    main()
