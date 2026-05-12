"""Audit: does order-2 Lasserre SDP cert the 748 stuck d=10 boxes?

Sample 50, run fast-MOSEK at K=32 (selective windows) and K=999
(full-PSD baseline). Report cert rate + median time. Failure triage.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from interval_bnb.windows import build_windows
from interval_bnb.box import Box
from interval_bnb.bound_sdp_escalation_fast import (
    build_sdp_escalation_cache_fast,
    bound_sdp_escalation_int_ge_fast,
)


def main():
    npz = np.load('stuck_d10_master_queue.npz')
    los, his = npz['lo'], npz['hi']
    N = len(los)
    print(f"loaded {N} stuck boxes", flush=True)

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(N, 50, replace=False)
    sample_idx = sorted(int(k) for k in sample_idx)

    d = 10
    windows = build_windows(d)
    print(f"built {len(windows)} windows for d={d}", flush=True)
    cache = build_sdp_escalation_cache_fast(d, windows, target=1.2)
    print("built cache", flush=True)

    results = {32: [], 999: []}
    for ki, k in enumerate(sample_idx):
        lo = los[k].astype(np.float64)
        hi = his[k].astype(np.float64)
        B = Box(lo=lo, hi=hi)
        lo_int, hi_int = B.to_ints()
        for K in (32, 999):
            t0 = time.time()
            try:
                cert = bound_sdp_escalation_int_ge_fast(
                    lo_int, hi_int, windows, d,
                    target_num=12, target_den=10,
                    cache=cache, n_window_psd_cones=K,
                    n_threads=1, time_limit_s=15.0,
                )
            except Exception as e:
                cert = False
                print(f"    EXC k={k} K={K}: {type(e).__name__}: {e}", flush=True)
            dt = time.time() - t0
            results[K].append({'idx': int(k), 'cert': bool(cert), 'sec': float(dt)})
        max_w = float((hi - lo).max())
        print(
            f"  [{ki+1}/50] idx={k} maxw={max_w:.2e}  "
            f"K=32 cert={results[32][-1]['cert']} t={results[32][-1]['sec']:.2f}s  "
            f"K=999 cert={results[999][-1]['cert']} t={results[999][-1]['sec']:.2f}s",
            flush=True,
        )

    print("\n=== SUMMARY ===", flush=True)
    for K in (32, 999):
        n_cert = sum(1 for r in results[K] if r['cert'])
        med_t = float(np.median([r['sec'] for r in results[K]]))
        max_t = float(np.max([r['sec'] for r in results[K]]))
        print(
            f"K={K}: cert={n_cert}/{len(results[K])} "
            f"({100*n_cert/len(results[K]):.1f}%) "
            f"median_t={med_t:.2f}s max_t={max_t:.2f}s",
            flush=True,
        )

    with open('sdp_stuck_audit.json', 'w') as f:
        json.dump(
            {'sample_idx': sample_idx, 'results': {str(k): v for k, v in results.items()}},
            f, indent=2,
        )

    fails_999 = [r['idx'] for r in results[999] if not r['cert']]
    print(f"\nfailed-at-K=999 (n={len(fails_999)}): {fails_999[:10]}", flush=True)
    for fi in fails_999[:5]:
        max_w = float((his[fi] - los[fi]).max())
        sum_w = float((his[fi] - los[fi]).sum())
        print(f"  box idx={fi}: max_width={max_w:.3e} sum_width={sum_w:.3e}", flush=True)


if __name__ == '__main__':
    main()
