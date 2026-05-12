"""Quick parallel scaling test: d=10 t=1.208 at workers ∈ {1, 4, 8, 16}.

Verifies the cascade-fix scales near-linearly with cores.
"""
import os
import sys
import time

os.environ['INTERVAL_BNB_TOPK_JOINT_DEPTH'] = '12'
os.environ['INTERVAL_BNB_TOPK_JOINT_K'] = '3'
os.environ['INTERVAL_BNB_EPIGRAPH_DEPTH'] = '24'
os.environ['INTERVAL_BNB_EPIGRAPH_FILTER'] = '0.02'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from interval_bnb.parallel import parallel_branch_and_bound


def main():
    results = []
    for nw in [4, 8, 16]:
        print(f"\n=== workers={nw} ===", flush=True)
        t0 = time.time()
        r = parallel_branch_and_bound(
            d=10, target_c='1.208', workers=nw,
            init_split_depth=14, donate_threshold_floor=4,
            time_budget_s=900, verbose=False,
        )
        elapsed = time.time() - t0
        results.append((nw, elapsed, r['success'], r['total_nodes']))
        print(f"  workers={nw}: success={r['success']}, "
              f"nodes={r['total_nodes']:,}, time={elapsed:.1f}s", flush=True)

    # Speedup table
    base_time = results[0][1] if results else 1
    print(f"\n{'='*60}")
    print(f"{'workers':>8} {'time(s)':>10} {'speedup':>10} {'eff':>8}")
    print(f"{'='*60}")
    for nw, t, succ, nodes in results:
        speedup = base_time / t
        eff = speedup * results[0][0] / nw  # efficiency = speedup / (nw/base_nw)
        print(f"{nw:>8} {t:>10.1f} {speedup:>10.2f} {eff:>8.2f}")


if __name__ == "__main__":
    main()
