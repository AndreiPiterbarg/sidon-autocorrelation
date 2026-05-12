"""K-sweep on a sample of iter-4 survivors.

Tests selective-SDP at K in {0, 8, 16, 32} on 100 random boxes from
iter_004_survivors.npz. Reports close rate and wall time per K.

Designed to coexist with the running cascade: 20 workers, ~40 GB RAM,
no file write conflicts.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from cert_pipeline.k_ladder import survivors_from_npz


# Worker globals
_W_CACHE = None
_W_WINDOWS = None
_W_D = None


def _w_init(d, target):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    sys.path.insert(0, str(_HERE))
    global _W_CACHE, _W_WINDOWS, _W_D
    from interval_bnb.windows import build_windows
    from interval_bnb.bound_sdp_escalation_fast import (
        build_sdp_escalation_cache_fast,
    )
    _W_D = d
    _W_WINDOWS = build_windows(d)
    _W_CACHE = build_sdp_escalation_cache_fast(d, _W_WINDOWS, target=target)


def _w_solve(args):
    """args = (lo_int_tuple, hi_int_tuple, K, target_num, target_den).
    Returns (cert: bool, wall_s: float)."""
    lo_int, hi_int, K, target_num, target_den = args
    from interval_bnb.bound_sdp_escalation_fast import (
        bound_sdp_escalation_int_ge_fast,
    )
    t0 = time.time()
    try:
        cert = bound_sdp_escalation_int_ge_fast(
            list(lo_int), list(hi_int), _W_WINDOWS, _W_D,
            target_num=target_num, target_den=target_den,
            cache=_W_CACHE,
            n_window_psd_cones=K,
            n_threads=1,
            time_limit_s=300.0,
            early_stop=True,
        )
        wall = time.time() - t0
        return (bool(cert), wall)
    except Exception as e:
        wall = time.time() - t0
        return (False, wall)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='runs/d22_deep_split/iter_004_survivors.npz')
    ap.add_argument('--n-boxes', type=int, default=100)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--K-values', type=str, default='0,8,16,32')
    ap.add_argument('--d', type=int, default=22)
    ap.add_argument('--target', type=float, default=1.2805)
    ap.add_argument('--target-num', type=int, default=12805)
    ap.add_argument('--target-den', type=int, default=10000)
    ap.add_argument('--out', default='_k_sweep_iter4_results.json')
    args = ap.parse_args()

    print(f'[load] {args.input} (numpy direct read)', flush=True)
    t0 = time.time()
    z = np.load(args.input, allow_pickle=True)
    n_total = int(z['hash'].size)
    print(f'[load] file has {n_total} boxes (numpy keys: {list(z.keys())}) '
          f'in {time.time()-t0:.1f}s', flush=True)

    rng = np.random.default_rng(2026)
    sample_idx = rng.choice(n_total, args.n_boxes, replace=False)
    # Build minimal box tuples directly (no SurvivorBox needed for the test)
    sample = []
    for i in sample_idx:
        lo_int = tuple(int(x) for x in z['lo_int'][int(i)])
        hi_int = tuple(int(x) for x in z['hi_int'][int(i)])
        sample.append((lo_int, hi_int))
    depth_in = int(z['depth'][int(sample_idx[0])])
    print(f'[sample] {args.n_boxes} boxes (depth={depth_in})', flush=True)

    K_list = [int(x) for x in args.K_values.split(',')]
    print(f'[plan] K values: {K_list}; workers: {args.workers}', flush=True)

    print(f'[init] spawning {args.workers} SDP workers...', flush=True)
    t0 = time.time()
    ctx = mp.get_context('fork')
    pool = ctx.Pool(args.workers, initializer=_w_init,
                    initargs=(args.d, args.target))
    print(f'[init] pool ready in {time.time()-t0:.1f}s', flush=True)

    results = {}
    for K in K_list:
        print(f'\n{"="*70}', flush=True)
        print(f'K = {K}', flush=True)
        print(f'{"="*70}', flush=True)
        work = [(lo, hi, K, args.target_num, args.target_den)
                for (lo, hi) in sample]
        t_start = time.time()
        per_box = []
        n_cert = 0
        n_done = 0
        next_print = max(1, args.n_boxes // 10)
        for cert, wall_s in pool.imap(_w_solve, work, chunksize=1):
            n_done += 1
            per_box.append({'cert': cert, 'wall_s': wall_s})
            if cert:
                n_cert += 1
            if n_done >= next_print:
                pct = 100 * n_done / args.n_boxes
                rate = n_done / max(1e-3, time.time() - t_start)
                print(f'  K={K} {n_done}/{args.n_boxes} ({pct:.0f}%) '
                      f'cert={n_cert} ({100*n_cert/n_done:.0f}%) '
                      f'rate={rate:.1f} box/s', flush=True)
                next_print += max(1, args.n_boxes // 10)
        t_total = time.time() - t_start
        avg_box_s = np.mean([r['wall_s'] for r in per_box])
        med_box_s = np.median([r['wall_s'] for r in per_box])
        print(f'  K={K} DONE  cert={n_cert}/{args.n_boxes} '
              f'({100*n_cert/args.n_boxes:.1f}%)  '
              f'wall={t_total:.0f}s  avg/box={avg_box_s:.1f}s  '
              f'med/box={med_box_s:.1f}s', flush=True)
        results[str(K)] = {
            'n_boxes': args.n_boxes,
            'n_cert': n_cert,
            'close_rate': n_cert / args.n_boxes,
            'wall_s_total': t_total,
            'avg_box_s': avg_box_s,
            'med_box_s': med_box_s,
            'per_box': per_box,
        }

    pool.close(); pool.join()

    print(f'\n{"="*70}', flush=True)
    print(f'SUMMARY (depth={depth_in})', flush=True)
    print(f'{"="*70}', flush=True)
    print(f'{"K":>4}  {"close%":>7}  {"avg_s/box":>10}  '
          f'{"close/sec":>10}  {"verdict":>20}', flush=True)
    for K in K_list:
        r = results[str(K)]
        close_per_s = r['close_rate'] / max(0.001, r['avg_box_s'])
        if r['close_rate'] >= 0.9375:
            verdict = 'CONTRACTS!'
        elif r['close_rate'] >= 0.85:
            verdict = 'near-contracts'
        elif r['close_rate'] >= 0.7:
            verdict = 'helpful'
        else:
            verdict = 'plateau'
        print(f'{K:>4}  {100*r["close_rate"]:>6.1f}%  '
              f'{r["avg_box_s"]:>9.2f}s  '
              f'{close_per_s:>9.4f}  {verdict:>20}', flush=True)

    with open(args.out, 'w') as f:
        json.dump({'depth_in': depth_in,
                   'n_boxes': args.n_boxes,
                   'K_list': K_list,
                   'results': {k: {kk: vv for kk, vv in v.items()
                                    if kk != 'per_box'}
                                for k, v in results.items()}},
                  f, indent=2)
    print(f'\nwrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
