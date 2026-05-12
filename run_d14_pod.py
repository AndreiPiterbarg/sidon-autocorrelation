"""d=14 t=1.253 BnB on the 192-core pod.

Cascade: cheap bounds → standard rigor → joint-face top-3 (depth>=12)
         → epigraph LP (depth>=24).

Tuned for high core count (192):
  * init_split_depth=14 — gives plenty of starter boxes for parallelism
  * donate_threshold_floor=4 — eager donation (workers steal early)
  * 4-hour budget (should finish in <30 min based on profiling)
"""
import os
import sys
import time
import json

# CASCADE THRESHOLDS (validated at d=10, expected to scale to d=14):
os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '12'
os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
os.environ['INTERVAL_BNB_EPIGRAPH_DEPTH'] = '24'
os.environ['INTERVAL_BNB_EPIGRAPH_FILTER'] = '0.02'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interval_bnb.parallel import parallel_branch_and_bound


def main():
    d = 14
    target = '1.253'
    workers = 192  # full pod
    time_budget_s = 14400  # 4 hours

    print("#" * 72, flush=True)
    print(f"# d={d} t={target} BnB on 192-core pod", flush=True)
    print(f"# Cascade: cheap → joint-face top-3 (≥12) → epigraph LP (≥24)", flush=True)
    print(f"# Budget: {time_budget_s}s", flush=True)
    print("#" * 72, flush=True)

    t0 = time.time()
    r = parallel_branch_and_bound(
        d=d, target_c=target,
        workers=workers,
        init_split_depth=26,  # Aggressive starter split for 192 cores;
                              # ~10K-50K feasible starter boxes so every
                              # worker has 50-200 to chew on locally.
        donate_threshold_floor=2,  # More aggressive donation
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
    print(f"  elapsed: {elapsed:.1f}s = {elapsed/60:.2f}min", flush=True)
    print(f"  CCTR stats: {r.get('cctr_stats', {})}", flush=True)
    print(f"  EPIGRAPH stats: {r.get('epi_stats', {})}", flush=True)

    serializable = {k: v for k, v in r.items()
                    if isinstance(v, (int, float, str, bool, list, dict))}
    serializable['wall_time_s'] = elapsed
    with open('d14_t1253_result.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved d14_t1253_result.json", flush=True)


if __name__ == "__main__":
    main()
