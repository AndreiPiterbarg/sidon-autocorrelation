"""Profile Lasserre Farkas SDP solve time at increasing (d, order, bandwidth).

Goal: produce an EVIDENCE-BACKED time estimate for the target config
(d=16, order=3, bandwidth=16) before committing to a long run.

Methodology:
  1. Run small configs (d ∈ {4, 6, 8}, order ∈ {2, 3}, b ∈ {3, 5, 7})
     and record build + solve time.
  2. Use n_y (number of y variables) and PSD-block sizes as predictive
     features.  SDP solve time in MOSEK scales roughly as O(n_y * (size of
     largest PSD block)^3) per IPM iteration, with O(log(1/ε)) iterations.
  3. Fit a power-law model: log(time) = α + β * log(n_y) + γ * log(b).
  4. Extrapolate to (d=16, order=3, b=16).

Outputs `_smoke_lasserre_timing.json` with the full timing matrix.
"""
from __future__ import annotations
import os, sys, time, json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'lasserre'))

from lasserre.d64_solver import solve_sparse_farkas_at_t


def time_one(d: int, order: int, bandwidth: int, t_test: float = 1.281,
             timeout_s: float = 300.0):
    """Run one (d, order, bandwidth) config, return timing breakdown."""
    print(f"\n=== d={d} order={order} bandwidth={bandwidth} t={t_test} ===",
          flush=True)
    t0 = time.time()
    try:
        result = solve_sparse_farkas_at_t(
            d=d, order=order, bandwidth=bandwidth,
            t_test=t_test, n_threads=0, mosek_tol=1e-9, verbose=False)
        wall = time.time() - t0
        out = {
            'd': d, 'order': order, 'bandwidth': bandwidth, 't_test': t_test,
            'status': str(result.status),
            'n_y': int(result.n_y),
            'n_eq': int(result.n_eq),
            'n_clique': int(result.n_clique),
            'wall_s': round(wall, 3),
            'completed': True,
        }
        # Extract block sizes if available
        if hasattr(result, 'meta') and result.meta:
            mom = result.meta.get('mom_blocks', [])
            if mom:
                out['mom_block_sizes'] = [m for _, _, m in mom]
                out['mom_block_size_max'] = max(out['mom_block_sizes'])
        print(f"  -> status={out['status']}  n_y={out['n_y']}  wall={wall:.2f}s",
              flush=True)
        return out
    except Exception as e:
        wall = time.time() - t0
        print(f"  -> EXCEPTION after {wall:.2f}s: {e}", flush=True)
        return {
            'd': d, 'order': order, 'bandwidth': bandwidth, 't_test': t_test,
            'wall_s': round(wall, 3),
            'completed': False,
            'error': str(e),
        }


def main():
    # Configs to time, in increasing cost order.
    configs = [
        (4, 2, 3),
        (4, 3, 3),
        (6, 2, 5),
        (6, 3, 5),
        (8, 2, 7),
        (8, 3, 7),
        (10, 2, 9),
        (10, 3, 9),
        (12, 2, 11),
        (12, 3, 11),
        (14, 2, 14),
        (14, 3, 14),
        (16, 2, 16),  # the original pod plan config
        # do NOT try (16, 3, 16) here — that's the target, may take hours
    ]

    results = []
    t_global = time.time()
    for d, order, b in configs:
        elapsed_global = time.time() - t_global
        if elapsed_global > 600:
            print(f"\n  Wall budget (10 min) exhausted; stopping sweep.",
                  flush=True)
            break
        r = time_one(d, order, b, t_test=1.281)
        results.append(r)
        # Save incrementally
        with open(os.path.join(_HERE, '_smoke_lasserre_timing.json'), 'w') as fp:
            json.dump({
                'configs_attempted': len(results),
                'total_wall_s': round(time.time() - t_global, 2),
                'results': results,
            }, fp, indent=2)

    print(f"\n========================================")
    print(f"All times (wall_s):")
    print(f"  d  order  b   n_y    wall_s   status")
    for r in results:
        if r.get('completed'):
            print(f"  {r['d']:2d}  {r['order']:5d}  {r['bandwidth']:2d}  "
                  f"{r['n_y']:5d}  {r['wall_s']:7.2f}   {r['status']}")
        else:
            print(f"  {r['d']:2d}  {r['order']:5d}  {r['bandwidth']:2d}  "
                  f"  -    {r['wall_s']:7.2f}   FAILED: {r.get('error', '?')[:40]}")

    # Naive extrapolation for (d=16, order=3, b=16)
    completed = [r for r in results if r.get('completed')]
    if len(completed) >= 4:
        # Fit log(wall) vs (n_y, max_block, order, b)
        # Power law: wall ~ n_y^a * b^c (rough)
        try:
            import numpy as np
            n_ys = np.array([r['n_y'] for r in completed if r['wall_s'] > 0.1])
            walls = np.array([r['wall_s'] for r in completed if r['wall_s'] > 0.1])
            if len(n_ys) >= 3:
                logn, logw = np.log(n_ys), np.log(walls)
                a, b_int = np.polyfit(logn, logw, 1)
                # For (d=16, order=3, b=16), n_y is approximately the d=16 order=2 n_y times an order-3/2 factor
                # Estimate n_y at order=3, b=16, d=16: similar to (d=14, order=3, b=14) extrapolated
                d16o3 = [r for r in completed if r['d'] == 14 and r['order'] == 3]
                if d16o3:
                    n_y_est = d16o3[-1]['n_y'] * 2  # very rough
                    wall_est = np.exp(a * np.log(n_y_est) + b_int)
                    print(f"\n  POWER-LAW EXTRAPOLATION (rough):")
                    print(f"    log(wall) = {b_int:.3f} + {a:.3f} * log(n_y)")
                    print(f"    estimated n_y for (d=16, order=3, b=16): ~{n_y_est}")
                    print(f"    estimated wall: {wall_est:.1f}s ({wall_est/60:.1f} min)")
        except Exception as e:
            print(f"  extrapolation failed: {e}")

    print(f"\nTotal wall: {time.time() - t_global:.1f}s")


if __name__ == '__main__':
    main()
