"""Scaling test for interval_bnb.parallel_branch_and_bound.

Runs at d=10, 12, 14, 16 with TARGET SET FOR MARGIN ~0.017 (matching d=20 problem).
Reports node count, wall time, depth, and rate.

Use known val(d) UB from our pipeline:
- d=8:  val=1.20060 -> target=1.184 (slack 0.017)
- d=10: val=1.22492 -> target=1.208 (slack 0.017)
- d=12: estimate val~1.255 -> target=1.238 (slack 0.017)
- d=14: estimate val~1.270 -> target=1.253 (slack 0.017)
- d=16: val=1.27427 -> target=1.257 (slack 0.017)
"""
import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interval_bnb.parallel import parallel_branch_and_bound


def run_one(d, target, time_budget_s, workers=64):
    print(f"\n{'='*70}")
    print(f"d={d} target={target} workers={workers} time_budget={time_budget_s}s")
    print(f"{'='*70}", flush=True)
    t0 = time.time()
    result = parallel_branch_and_bound(
        d=d, target_c=target, workers=workers,
        init_split_depth=10,
        time_budget_s=time_budget_s,
        verbose=True,
    )
    elapsed = time.time() - t0
    result['d'] = d
    result['target'] = target
    result['workers'] = workers
    result['wall_time_s'] = elapsed
    return result


def main():
    workers = 64
    time_budget = 600  # 10 min budget per test
    results = []

    # Match slack 0.017 (the d=20 problem's slack)
    tests = [
        (8, "1.184", 60),    # easy, 1 min budget
        (10, "1.208", 120),  # 2 min
        (12, "1.238", 300),  # 5 min
        (14, "1.253", 600),  # 10 min
        (16, "1.257", 1800), # 30 min
    ]

    for d, target, budget in tests:
        try:
            r = run_one(d, target, time_budget_s=budget, workers=workers)
            results.append(r)
            print(f"\n[SUMMARY d={d}] success={r['success']}, "
                  f"nodes={r['total_nodes']:,}, "
                  f"depth={r['max_depth']}, "
                  f"time={r['wall_time_s']:.1f}s, "
                  f"coverage={100*r['coverage_fraction']:.2f}%", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({'d': d, 'target': target, 'error': str(e)})

    # Save and print summary
    out_path = 'scaling_results.json'
    # Strip non-serializable
    serializable = []
    for r in results:
        sr = {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool, list, dict))}
        serializable.append(sr)
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved {out_path}")

    print(f"\n{'='*70}")
    print(f"SCALING SUMMARY (slack=0.017)")
    print(f"{'='*70}")
    print(f"{'d':<4}{'target':<8}{'success':<10}{'nodes':<14}{'time(s)':<10}{'rate(/s)':<10}{'depth':<8}{'cover%':<8}")
    for r in results:
        if 'error' in r:
            print(f"{r['d']:<4}{r['target']:<8}{'ERROR':<10}{r.get('error','')[:40]}")
        else:
            rate = r['total_nodes'] / max(1, r['wall_time_s'])
            print(f"{r['d']:<4}{r['target']:<8}{str(r['success']):<10}"
                  f"{r['total_nodes']:<14,}"
                  f"{r['wall_time_s']:<10.1f}{rate:<10,.0f}"
                  f"{r['max_depth']:<8}{100*r['coverage_fraction']:<8.2f}")


if __name__ == "__main__":
    main()
