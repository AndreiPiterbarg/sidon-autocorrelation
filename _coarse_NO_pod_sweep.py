"""Larger-config sweep for N+O combined coarse-cascade pruning on the pod.

Run on pod (66.201.4.121) — 64 cores, 250 GB RAM.
"""
import os, sys, time, json, argparse
import numpy as np
import numba

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)

from _coarse_NO_bench import (precompute_op_rest, prune_coarse_NO,
                                prune_coarse_baseline, prune_coarse_O,
                                prune_coarse_N, enum_compositions)


def run_one(d, S, c_target):
    op_rest, _ = precompute_op_rest(d, 2 * d)
    op_rest_d = op_rest * d
    print(f"\n=== d={d}, S={S}, c={c_target} ===  precompute done", flush=True)
    t0 = time.time()
    batch = np.array(list(enum_compositions(d, S)), dtype=np.int32)
    enum_t = time.time() - t0
    print(f"  enumerated {len(batch):,} comps in {enum_t:.1f}s", flush=True)

    warm = batch[:1]
    _ = prune_coarse_baseline(warm, d, S, c_target)
    _ = prune_coarse_NO(warm, d, S, c_target, op_rest_d)

    t0 = time.time()
    sBL = prune_coarse_baseline(batch, d, S, c_target)
    tBL = time.time() - t0

    t0 = time.time()
    sNO = prune_coarse_NO(batch, d, S, c_target, op_rest_d)
    tNO = time.time() - t0

    nBL = int(sBL.sum())
    nNO = int(sNO.sum())
    extra = nBL - nNO
    pct = 100.0 * extra / max(1, nBL)
    print(f"  BL: {nBL:,} ({tBL:.1f}s)  |  NO: {nNO:,} ({tNO:.1f}s)  "
          f"|  -{extra:,} = {pct:.2f}%", flush=True)
    return {
        'd': d, 'S': S, 'c': c_target, 'n_total': len(batch),
        'BL': nBL, 'NO': nNO, 'extra': extra, 'pct': pct,
        'time_BL': tBL, 'time_NO': tNO,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        configs = [(8, 15, 1.20), (8, 20, 1.20), (10, 12, 1.20)]
    else:
        configs = [
            (8, 15, 1.20), (8, 20, 1.20), (8, 25, 1.20),
            (10, 10, 1.20), (10, 12, 1.20), (10, 15, 1.20),
            (12, 8, 1.20), (12, 10, 1.20),
        ]
    results = []
    for d, S, c in configs:
        try:
            results.append(run_one(d, S, c))
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            results.append({'d': d, 'S': S, 'c': c, 'error': str(e)})
    out = os.path.join(_dir, '_coarse_NO_pod_sweep_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSUMMARY:")
    for r in results:
        if 'pct' in r:
            print(f"  d={r['d']}, S={r['S']}, c={r['c']}: "
                  f"BL={r['BL']:,} -> NO={r['NO']:,}  "
                  f"-{r['extra']:,} ({r['pct']:.2f}%)  "
                  f"BL={r['time_BL']:.1f}s NO={r['time_NO']:.1f}s")


if __name__ == '__main__':
    main()
